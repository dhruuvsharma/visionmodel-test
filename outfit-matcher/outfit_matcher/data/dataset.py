"""Dataset over manifest jsonl + balanced multi-view SupCon batch sampler."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import DataConfig
from .transforms import RenderAugment


def load_manifest(path: str, views: Optional[List[str]] = None) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if views and row.get("angle") not in views:
                    continue
                rows.append(row)
    return rows


def build_label_map(rows: List[Dict]) -> Dict[str, int]:
    ids = sorted({r["garment_id"] for r in rows})
    return {g: i for i, g in enumerate(ids)}


def _load_image(path: str) -> torch.Tensor:
    """Load image -> (3,H,W) float in [0,1]."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)  # (H, W, 3)
    img = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
    return img


class GarmentViewDataset(Dataset):
    """Each item = one rendered view of one garment, independently augmented."""

    def __init__(self, manifest_path: str, cfg: DataConfig, label_map: Optional[Dict[str, int]] = None):
        self.rows = load_manifest(manifest_path)
        self.cfg = cfg
        self.label_map = label_map or build_label_map(self.rows)
        self.garment_to_rows: Dict[str, List[int]] = {}
        for i, r in enumerate(self.rows):
            self.garment_to_rows.setdefault(r["garment_id"], []).append(i)
        self.garment_ids_sorted: List[str] = sorted(self.garment_to_rows.keys())
        self.augment = RenderAugment(cfg)

    def __len__(self) -> int:
        return len(self.rows)

    def garment_count(self) -> int:
        return len(self.garment_to_rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[idx]
        img = _load_image(row["path"])
        img = self.augment(img)
        return img, self.label_map[row["garment_id"]]


class BalancedMultiViewSampler(torch.utils.data.Sampler):
    """Batch sampler yielding P garments x V views per batch as dataset row indices.

    Every batch has exactly V positives per class -> ideal for SupCon.
    Epoch length = ceil(num_garments / P) batches.
    """

    def __init__(
        self,
        garment_to_rows: Dict[str, List[int]],
        garments_per_batch: int,
        views_per_garment: int,
        seed: int = 42,
    ):
        self.garment_to_rows = {g: list(v) for g, v in garment_to_rows.items() if len(v) > 0}
        self.garment_ids = list(self.garment_to_rows.keys())
        self.P = max(1, garments_per_batch)
        self.V = max(1, views_per_garment)
        self.rng = random.Random(seed)
        self.num_batches = max(1, (len(self.garment_ids) + self.P - 1) // self.P)

    def set_epoch(self, epoch: int) -> None:
        self.rng = random.Random(1234 + epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        ids = self.garment_ids[:]
        self.rng.shuffle(ids)
        for b in range(self.num_batches):
            batch_ids = ids[b * self.P:(b + 1) * self.P]
            while len(batch_ids) < self.P:  # wrap-around resample for the last batch
                in_batch = set(batch_ids)
                pool = [g for g in ids if g not in in_batch] or ids
                batch_ids.append(pool[self.rng.randrange(len(pool))])
            indices: List[int] = []
            for g in batch_ids:
                rows = self.garment_to_rows[g]
                if self.V >= len(rows):
                    picked = [rows[i % len(rows)] for i in range(self.V)]
                else:
                    picked = self.rng.sample(rows, self.V)
                indices.extend(picked)
            yield indices
