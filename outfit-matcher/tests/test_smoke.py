"""Smoke tests: tiny model, synthetic data, CPU. Run: pytest tests/ -v"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from outfit_matcher.config import load_config
from outfit_matcher.data.dataset import (BalancedMultiViewSampler, GarmentViewDataset,
                                          build_label_map)
from outfit_matcher.data.prepare_data import scan_garments, split_train_val, write_manifest
from outfit_matcher.data.synth_data import main as synth_main
from outfit_matcher.losses.supcon import SupConLoss
from outfit_matcher.model.vit import ViTConfig, build_model, count_parameters
from outfit_matcher.evaluate import encode_catalog


SMOKE_DIR = ROOT / "tests" / "_smoke_data"
MANIFEST_DIR = SMOKE_DIR / "manifests"


@pytest.fixture(scope="session")
def smoke_env():
    """Generate synthetic garments + manifests once for the whole test session."""
    if not (SMOKE_DIR / "shirt_000").exists():
        sys.argv = ["synth", "--out-dir", str(SMOKE_DIR), "--garments", "12", "--size", "128"]
        synth_main()
    records = scan_garments(SMOKE_DIR, min_views=2)
    assert len(records) == 12
    train, val, _ = split_train_val(records, val_fraction=0.25, seed=42)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(train, MANIFEST_DIR / "manifest_train.jsonl")
    garment_var_manifest_path = MANIFEST_DIR / "manifest_val.jsonl"
    write_manifest(val, garment_var_manifest_path)
    return records, train, val


def test_model_forward():
    cfg = ViTConfig(img_size=64, patch_size=16, dim=64, depth=2, heads=2, mlp_ratio=2.0, drop_rate=0.0, drop_path_rate=0.0)
    model = build_model(cfg, proj_dim=32)
    x = torch.randn(2, 3, 64, 64)
    z = model(x)
    assert z.shape == (2, 32)
    # attention sanity: model with all-zero input should still be finite
    assert torch.isfinite(z).all()
    assert count_parameters(model) > 0


def test_supcon_loss():
    torch.manual_seed(0)
    loss_fn = SupConLoss(temperature=0.1)
    # 3 garments x 2 views; within-class sims set high artificially
    feats = F.normalize(torch.randn(6, 8), dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = loss_fn(feats, labels)
    assert loss.item() >= 0 and torch.isfinite(loss)

    # perfect clustering -> near-zero loss (each anchor duplicated as its positive)
    anchors = F.normalize(torch.randn(3, 8), dim=-1)
    feats = torch.cat([anchors, anchors], 0)  # rows: a0,a1,a2,a0,a1,a2
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    loss0 = loss_fn(feats, labels)
    assert loss0.item() < 0.01, f"perfect clustering loss too high: {loss0.item()}"


def test_supcon_loss_zero_when_one_class_one_view():
    loss_fn = SupConLoss(temperature=0.1)
    feats = F.normalize(torch.randn(3, 8), dim=-1)
    labels = torch.tensor([0, 1, 2])
    loss = loss_fn(feats, labels)
    assert loss.item() == 0.0


def test_sampler_balance(smoke_env):
    records, train, val = smoke_env
    ds = GarmentViewDataset(str(MANIFEST_DIR / "manifest_train.jsonl"), load_config(str(ROOT / "configs" / "smoke.yaml")).data)
    sampler = BalancedMultiViewSampler(ds.garment_to_rows, garments_per_batch=4, views_per_garment=2)
    assert sampler.num_batches == 3  # 9 train garments / 4 per batch -> 3 batches (last partially resampled)
    batches = list(iter(sampler))
    for batch in batches:
        assert len(batch) == 8
        labels = [ds.rows[i]["garment_id"] for i in batch]
        # each garment appears exactly V=2 times
        c = Counter(labels)
        assert all(v == 2 for v in c.values()) and len(c) == 4
    # set_epoch changes permutation but structure holds
    sampler.set_epoch(1)
    assert len(list(iter(sampler))) == 3


def test_sampler_epoch_reseed():
    """Sampler must produce different orders across epochs."""
    g = {"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]}
    s = BalancedMultiViewSampler(g, garments_per_batch=4, views_per_garment=2)
    s.set_epoch(0)
    a0 = [tuple(b) for b in iter(s)]
    s.set_epoch(1)
    a1 = [tuple(b) for b in iter(s)]
    assert a0 != a1


def test_transforms_keying():
    from outfit_matcher.data.transforms import key_white_background
    img = torch.ones(3, 32, 32)
    img[:, 10:20, 10:20] = torch.tensor([0.8, 0.2, 0.2]).view(3, 1, 1)
    garment, mask = key_white_background(img, tolerance=0.08, grow=0)
    assert mask[15, 15].item() and not mask[0, 0].item()
    assert garment[:, 0, 0].sum() == 0  # background zeroed
    assert garment[:, 15, 15].sum() > 0  # garment pixels kept
    from outfit_matcher.data.dataset import _load_image
    from outfit_matcher.data.transforms import RenderAugment
    cfg = load_config(str(ROOT / "configs" / "smoke.yaml")).data
    img = _load_image(str(SMOKE_DIR / "shirt_000" / "front.png"))
    out = RenderAugment(cfg)(img)
    assert out.shape == (3, cfg.img_size, cfg.img_size)
    assert 0.0 <= out.min() and out.max() <= 1.0001
    assert torch.isfinite(out).all()


def test_evaluate_and_match_flow(smoke_env):
    from outfit_matcher.evaluate import evaluate_retrieval
    cfg = load_config(str(ROOT / "configs" / "smoke.yaml"))
    ds = GarmentViewDataset(str(MANIFEST_DIR / "manifest_val.jsonl"), cfg.data)
    model = build_model(ViTConfig(img_size=64, patch_size=16, dim=64, depth=2, heads=2, mlp_ratio=2.0, drop_rate=0.0, drop_path_rate=0.0), proj_dim=32)
    metrics = evaluate_retrieval(model, ds, cfg, device=torch.device("cpu"))
    assert "recall@1" in metrics and "mrr" in metrics and metrics["n_queries"] == len(ds)
    catalog = encode_catalog(model, str(MANIFEST_DIR / "manifest_val.jsonl"), cfg, device=torch.device("cpu"))
    assert catalog["features"].shape[0] == len(ds)
    assert len(catalog["garment_ids"]) == len(ds)


def test_end_to_end_train(smoke_env):
    """Run the actual Trainer on synthetic data for 2 epochs on CPU."""
    from outfit_matcher.train import Trainer
    cfg = load_config(str(ROOT / "configs" / "smoke.yaml"))
    cfg.train.workers = 0
    cfg.train.out_dir = str(ROOT / "tests" / "_smoke_runs" / "e2e")
    trainer = Trainer(cfg)
    trainer.train()
    ckpt_path = Path(cfg.train.out_dir) / "checkpoint_final.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model" in ckpt and "ema" in ckpt

    # match CLI flow: load checkpoint, encode catalog, retrieve
    from outfit_matcher.match import load_model, query_catalog
    model, _ = load_model(str(ckpt_path), torch.device("cpu"))
    from outfit_matcher.data.eval_dataset import EvalDataset
    qds = EvalDataset(str(MANIFEST_DIR / "manifest_val.jsonl"), cfg.data)
    img, label = qds[0]
    with torch.no_grad():
        z = model(img[None]).float()
    z = torch.nn.functional.normalize(z, dim=-1)
    catalog = encode_catalog(model, str(MANIFEST_DIR / "holdout_manifest.jsonl") if False else str(MANIFEST_DIR / "manifest_val.jsonl"), cfg, device=torch.device("cpu"))
    results = query_catalog(model, z[0], catalog, topk=3)
    assert len(results) == 3
    assert all("garment_id" in r and "similarity" in r for r in results)


def test_prepare_data_cli(smoke_env):
    """prepare_data CLI end-to-end on the synthetic tree."""
    result = subprocess.run(
        [sys.executable, "-m", "outfit_matcher.data.prepare_data",
         "--data-root", str(SMOKE_DIR), "--out-dir", str(MANIFEST_DIR / "cli"),
         "--val-fraction", "0.2", "--seed", "7"],
        cwd=str(ROOT), capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode == 0, result.stderr
    out_train = MANIFEST_DIR / "cli" / "manifest_train.jsonl"
    assert out_train.exists()
    lines = out_train.read_text().strip().splitlines()
    assert len(lines) > 0
    json.loads(lines[0])


def test_view_filter_front_only(smoke_env):
    """train.views=['front']: training restricted to front, eval/catalog keep all angles."""
    from outfit_matcher.data.eval_dataset import EvalDataset
    cfg = load_config(str(ROOT / "configs" / "smoke.yaml"))
    cfg.train.views = ["front"]

    # simulate what Trainer._build does: filter train rows only
    from outfit_matcher.data.dataset import load_manifest
    rows = load_manifest(cfg.data.manifest, views=cfg.train.views)
    assert rows and all(r["angle"] == "front" for r in rows)

    # train dataset garment grouping: exactly 1 front row per garment
    from outfit_matcher.data.dataset import GarmentViewDataset as GVD
    ds = GVD(cfg.data.manifest, cfg.data)
    ds.rows = rows
    g2r = {}
    for i, r in enumerate(ds.rows):
        g2r.setdefault(r["garment_id"], []).append(i)
    assert len(g2r) == 9
    assert all(len(v) == 1 for v in g2r.values())

    # sampler: 1 view/garment still yields balanced P x V batches (repeats row)
    sampler = BalancedMultiViewSampler(g2r, garments_per_batch=4, views_per_garment=2)
    for batch in iter(sampler):
        assert len(batch) == 8
        labels = [ds.rows[i]["garment_id"] for i in batch]
        c = Counter(labels)
        assert all(v == 2 for v in c.values()) and len(c) == 4

    # eval dataset + catalog encoding UNCHANGED: all 4 angles
    val = EvalDataset(str(MANIFEST_DIR / "manifest_val.jsonl"), cfg.data)
    assert len({r["angle"] for r in val.rows}) == 4
    model = build_model(ViTConfig(img_size=64, patch_size=16, dim=64, depth=2, heads=2,
                                   mlp_ratio=2.0, drop_rate=0.0, drop_path_rate=0.0), proj_dim=32)
    catalog = encode_catalog(model, str(MANIFEST_DIR / "manifest_val.jsonl"), cfg,
                             device=torch.device("cpu"))
    assert catalog["features"].shape[0] == len(val.rows)
    assert len(set(catalog["angles"])) == 4

    # prepare_data --views front still available (manifest-level alternative)
    result = subprocess.run(
        [sys.executable, "-m", "outfit_matcher.data.prepare_data",
         "--data-root", str(SMOKE_DIR), "--out-dir", str(MANIFEST_DIR / "cli_front"),
         "--val-fraction", "0.2", "--seed", "7", "--views", "front"],
        cwd=str(ROOT), capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode == 0, result.stderr
    mrows = [json.loads(l) for l in
             (MANIFEST_DIR / "cli_front" / "manifest_train.jsonl").read_text().strip().splitlines()]
    assert mrows and all(r["angle"] == "front" for r in mrows)
    split = json.loads((MANIFEST_DIR / "cli_front" / "split.json").read_text())
    assert len(mrows) == split["train_garments"]  # exactly one front view per garment
