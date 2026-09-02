"""Training loop: DDP + SupCon + aux CE, bf16 autocast, EMA, checkpoints, tqdm."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .config import AppConfig
from .data.dataset import GarmentViewDataset, BalancedMultiViewSampler, build_label_map
from .data.eval_dataset import EvalDataset
from .engine import (
    EMA, build_optimizer, CosineWarmupScheduler, init_distributed, is_main_process,
    world_size, barrier, destroy_distributed,
)
from .losses.supcon import SupConLoss
from .model.vit import ViTConfig, build_model
from .evaluate import evaluate_retrieval, encode_catalog


class Trainer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        force_cpu = os.environ.get("OUTFIT_MATCHER_FORCE_CPU", "0") == "1"
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{local_rank}")
        self.rank, self.world = init_distributed()
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        cfg = self.cfg
        mcfg = ViTConfig(
            img_size=cfg.model.img_size, patch_size=cfg.model.patch_size, dim=cfg.model.dim,
            depth=cfg.model.depth, heads=cfg.model.heads, mlp_ratio=cfg.model.mlp_ratio,
            drop_rate=cfg.model.drop_rate, drop_path_rate=cfg.model.drop_path_rate,
        )
        model = build_model(mcfg, proj_dim=cfg.model.proj_dim)
        model.to(self.device)
        print(f"[rank {self.rank}] params: {sum(p.numel() for p in model.parameters()):,}")

        if self.world > 1:
            self.model = DDP(model, device_ids=[self.device] if self.device.type == "cuda" else None)
        else:
            self.model = model

        # datasets ------------------------------------------------------
        self.train_ds = GarmentViewDataset(cfg.data.manifest, cfg.data)
        # front-only prototype etc.: restrict TRAINING to configured angles,
        # keep eval/catalog on all views so cross-view retrieval stays meaningful
        if cfg.train.views:
            from .data.dataset import load_manifest as _lm
            self.train_ds.rows = _lm(cfg.data.manifest, views=list(cfg.train.views))
            self.train_ds.garment_to_rows = {}
            for i, r in enumerate(self.train_ds.rows):
                self.train_ds.garment_to_rows.setdefault(r["garment_id"], []).append(i)
            self.train_ds.garment_ids_sorted = sorted(self.train_ds.garment_to_rows.keys())
            assert self.train_ds.rows, f"view filter {cfg.train.views} matched no rows"
        labels = build_label_map(self.train_ds.rows)
        self.num_classes = len(labels)
        # val garments are held out -> val ds builds its OWN label map
        # (retrieval metrics only need consistent IDs within val)
        self.val_ds = EvalDataset(cfg.data.val_manifest, cfg.data) \
            if Path(cfg.data.val_manifest).exists() else None

        self.sampler = BalancedMultiViewSampler(
            self.train_ds.garment_to_rows,
            cfg.train.garments_per_batch, cfg.train.views_per_garment,
            seed=cfg.train.seed + 1000 * self.rank,
        )
        self.loader = DataLoader(
            self.train_ds, batch_sampler=self.sampler, num_workers=cfg.train.workers,
            pin_memory=(self.device.type == "cuda"), persistent_workers=cfg.train.workers > 0,
            drop_last=False,
        )

        # loss / optimizer ----------------------------------------------
        self.supcon = SupConLoss(temperature=cfg.train.temperature, gather_distributed=self.world > 1)
        self.aux_ce_weight = cfg.train.aux_ce_weight
        self.classifier = torch.nn.Linear(cfg.model.dim, self.num_classes).to(self.device)

        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        self.optimizer = build_optimizer(
            torch.nn.ModuleList([raw_model, self.classifier]), cfg.train.lr, cfg.train.weight_decay
        )
        steps_per_epoch = len(self.sampler)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=cfg.train.warmup_epochs * steps_per_epoch,
            total_steps=cfg.train.epochs * steps_per_epoch,
            min_lr=cfg.train.min_lr,
        )
        self.ema = EMA(raw_model, decay=cfg.train.ema_decay)

        self.start_epoch = 0
        self.global_step = 0
        self.out_dir = Path(cfg.train.out_dir)
        if is_main_process():
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._log_config()

        if cfg.train.resume:
            self._load_checkpoint(cfg.train.resume)

    def _log_config(self):
        cfg = self.cfg
        info = {
            "model": cfg.model.__dict__, "data": {k: str(v) for k, v in cfg.data.__dict__.items()},
            "train": {**{k: str(v) for k, v in cfg.train.__dict__.items()},
                       "num_classes": self.num_classes, "steps_per_epoch": len(self.sampler)},
        }
        with open(self.out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, default=str)

    # ------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        history = []
        for epoch in range(self.start_epoch, cfg.train.epochs):
            self.sampler.set_epoch(epoch)
            epoch_t0 = time.time()
            stats = self._train_epoch(epoch)
            stats["epoch_time_s"] = round(time.time() - epoch_t0, 1)
            history.append(stats)
            if is_main_process():
                print(f"[epoch {epoch}] {stats}", flush=True)
                self._save_checkpoint(epoch, stats)
                self._append_history(history)
            if self.val_ds is not None and (epoch + 1) % cfg.train.eval_every == 0:
                self._run_eval(epoch)
        if is_main_process():
            self._save_checkpoint(cfg.train.epochs - 1, history[-1] if history else {}, final=True)
        destroy_distributed()

    def _train_epoch(self, epoch: int) -> dict:
        cfg = self.cfg
        self.model.train()
        totals = {"supcon": 0.0, "ce": 0.0, "n": 0}
        if torch.cuda.is_available():
            amp_dtype = torch.bfloat16 if (cfg.train.bf16 and torch.cuda.is_bf16_supported()) else torch.float32
        else:
            amp_dtype = torch.float32
        if is_main_process():
            pbar = tqdm.tqdm(self.loader, desc=f"epoch {epoch}", ncols=120)
        else:
            pbar = self.loader

        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.global_step += 1

            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                z_proj, cls_feats = self.model(images, return_cls=True)   # single forward
                z = F.normalize(z_proj.float(), dim=-1)
                loss_sc = self.supcon(z, targets)

                logits = self.classifier(cls_feats.float())
                loss_ce = F.cross_entropy(logits, targets)
                loss = loss_sc + self.aux_ce_weight * loss_ce

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # classifier is outside DDP -> manually sync its grads across ranks
            if self.world > 1 and dist.is_initialized():
                for p in self.classifier.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= self.world
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad] +
                    [p for p in self.classifier.parameters() if p.requires_grad],
                    cfg.train.grad_clip,
                )
            self.optimizer.step()
            self.scheduler.step(self.global_step)
            with torch.no_grad():
                raw = self.model.module if isinstance(self.model, DDP) else self.model
                self.ema.update(raw)

            bs = targets.numel()
            totals["supcon"] += loss_sc.item() * bs
            totals["ce"] += loss_ce.item() * bs
            totals["n"] += bs
            if is_main_process() and self.global_step % cfg.train.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(sc=f"{totals['supcon']/max(totals['n'],1):.4f}",
                                 ce=f"{totals['ce']/max(totals['n'],1):.4f}", lr=f"{lr:.2e}")
        return {k: round(v / max(totals["n"], 1), 5) if k != "n" else v for k, v in totals.items()}

    def _cls_features(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone CLS features (before projection head) for the aux classifier."""
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        return raw.forward_features(images)[:, 0]

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, stats: dict, final: bool = False):
        if not is_main_process():
            return
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt = {
            "model": raw.state_dict(),
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": {"global_step": self.global_step},
            "ema": self.ema.state_dict(),
            "epoch": epoch,
            "num_classes": self.num_classes,
            "model_cfg": self.cfg.model.__dict__,
            "proj_dim": self.cfg.model.proj_dim,
        }
        name = "checkpoint_final.pt" if final else "checkpoint_last.pt"
        torch.save(ckpt, self.out_dir / name)
        if (epoch % 10 == 0) or final:
            torch.save(ckpt, self.out_dir / f"checkpoint_e{epoch:04d}.pt")
        with open(self.out_dir / "last_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)

    def _append_history(self, history: list):
        with open(self.out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            for row in history[-1:]:
                f.write(json.dumps(row, default=str) + "\n")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        raw.load_state_dict(ckpt["model"])
        if self.classifier is not None and ckpt.get("classifier"):
            self.classifier.load_state_dict(ckpt["classifier"])
        if ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("ema"):
            self.ema.load_state_dict(ckpt["ema"])
        self.start_epoch = ckpt.get("epoch", -1) + 1
        self.global_step = ckpt.get("scheduler", {}).get("global_step", 0)

    # ------------------------------------------------------------------
    def _run_eval(self, epoch: int):
        if not is_main_process():
            return  # rank 1 skips; DDP allreduce keeps ranks aligned next epoch
        raw = self.model.module if isinstance(self.model, DDP) else self.model
        # swap in EMA weights for eval
        backup = {k: v.detach().clone() for k, v in raw.state_dict().items()}
        self.ema.copy_to(raw)
        metrics = evaluate_retrieval(raw, self.val_ds, self.cfg, device=self.device)
        raw.load_state_dict(backup)  # restore training weights
        print(f"[eval epoch {epoch}] {metrics}", flush=True)
        with open(self.out_dir / "eval_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}, default=str) + "\n")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Train outfit-matcher ViT with SupCon")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-override", default=None, help="override manifest dir (contains manifest_train/val.jsonl)")
    args = ap.parse_args()

    from .config import load_config
    cfg = load_config(args.config)
    if args.data_override:
        cfg.data.manifest = str(Path(args.data_override) / "manifest_train.jsonl")
        cfg.data.val_manifest = str(Path(args.data_override) / "manifest_val.jsonl")
    trainer = Trainer(cfg)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
