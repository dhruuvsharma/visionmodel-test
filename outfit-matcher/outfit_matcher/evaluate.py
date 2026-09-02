"""Retrieval evaluation: leave-one-view-out Recall@K / MRR, plus catalog encoding.

Protocol (garment-level holdout, angle-level split):
- Query set: one view angle of each held-out garment (e.g. front view only).
- Gallery set: remaining views (other angles) of the same garments + all views of
  distractor garments... simplified: gallery = all val views except the query's angle.

Since all val garments are seen classes... no wait — val garments are NOT in train.
Protocol: for each val garment, query = each view; gallery = all other val views.
A correct retrieval = gallery view from the same garment (any angle).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import AppConfig
from .data.dataset import build_label_map, load_manifest
from .data.eval_dataset import EvalDataset


@torch.no_grad()
def encode_dataset(model: torch.nn.Module, dataset, cfg: AppConfig, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode all images of a dataset -> (features (N, proj_dim) L2-normalized, labels)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    feats, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        z = model(images)
        if z.dtype != torch.float32:
            z = z.float()
        feats.append(F.normalize(z.cpu(), dim=-1))
        labels.append(targets)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def evaluate_retrieval(model: torch.nn.Module, val_ds, cfg: AppConfig, device) -> Dict[str, float]:
    """Leave-one-view-out retrieval on val garments (all angles as queries)."""
    feats, labels = encode_dataset(model, val_ds, cfg, device)

    # leave-one-view-out: same-garment views are positives; the query view itself excluded
    n = feats.shape[0]
    sim = feats @ feats.T
    sim.fill_diagonal_(float("-inf"))
    order = sim.argsort(dim=1, descending=True)

    ks = cfg.eval.recall_k
    metrics = {}
    ranks = []
    for k in ks:
        metrics[f"recall@{k}"] = 0.0
    for i in range(n):
        correct_rank = None
        # iterate gallery in similarity order, skipping the query itself;
        # ranks count gallery positions (self row excluded) so Recall@K is honest
        rank = -1
        for j in order[i].tolist():
            if j == i:
                continue
            rank += 1
            if labels[j] == labels[i]:
                correct_rank = rank
                break
        if correct_rank is None:
            ranks.append(n)
            continue
        ranks.append(correct_rank)
        for k in ks:
            if correct_rank < k:
                metrics[f"recall@{k}"] += 1
    for k in ks:
        metrics[f"recall@{k}"] = round(metrics[f"recall@{k}"] / n, 4)
    metrics["mrr"] = round(sum(1.0 / (r + 1) for r in ranks) / n, 4)
    metrics["n_queries"] = n
    return metrics


# ---------------------------------------------------------------------------
# catalog embedding (for production retrieval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_catalog(model: torch.nn.Module, manifest_path: str, cfg: AppConfig, device, batch_size: int = 64) -> Dict:
    """Embed all catalog views -> saved .pt artifact for nearest-neighbor lookup.

    Returns dict with keys: features (Tensor), garment_ids (list), paths (list),
    angles (list), label_map (garment_id -> int).
    """
    rows = load_manifest(manifest_path)
    ds = EvalDataset(manifest_path, cfg.data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    feats, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        z = model(images).float()
        feats.append(F.normalize(z.cpu(), dim=-1))
        labels.append(targets)
    feats = torch.cat(feats).cpu()
    labels = torch.cat(labels).cpu()
    inv = {v: k for k, v in ds.label_map.items()}
    return {
        "features": feats,
        "garment_ids": [inv[int(l)] for l in labels],
        "paths": [r["path"] for r in rows],
        "angles": [r["angle"] for r in rows],
        "label_map": ds.label_map,
    }
