"""Distributed helpers + EMA + optimizer/schedule for from-scratch ViT training."""

from __future__ import annotations

import math
import os
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# DDP (gloo on Windows)
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int]:
    """Init process group from env vars. Returns (rank, world_size). No-op if not launched by torchrun."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend=backend, init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# EMA (exponential moving average of model weights)
# ---------------------------------------------------------------------------

class EMA:
    """Shadow weights updated as ema = decay*ema + (1-decay)*param."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                s = self.shadow[k]
                if v.is_floating_point():
                    s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
                else:
                    self.shadow[k] = v.detach().clone()

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]

    def copy_to(self, model: torch.nn.Module) -> None:
        """Load EMA weights into a model (for eval / export)."""
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=True)


# ---------------------------------------------------------------------------
# optimizer + cosine schedule with warmup
# ---------------------------------------------------------------------------

def param_groups_weight_decay(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on bias and norm params (standard ViT recipe)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(param_groups_weight_decay(model, weight_decay), lr=lr, betas=(0.9, 0.999))


class CosineWarmupScheduler:
    """Step-based: linear warmup then cosine decay to min_lr. Call per optimizer step."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, step: int) -> float:
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.min_lr + (self.base_lrs[0] - self.min_lr) * factor
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.min_lr + (base - self.min_lr) * factor
        return lr
