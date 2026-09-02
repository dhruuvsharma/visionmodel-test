"""Deterministic eval dataset (no augmentation)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import DataConfig
from .dataset import build_label_map, load_manifest, _load_image


class EvalDataset(Dataset):
    """Plain resize + center-crop eval dataset over a manifest."""

    def __init__(self, manifest_path: str, cfg: DataConfig, label_map: Optional[Dict[str, int]] = None):
        self.rows = load_manifest(manifest_path)
        self.cfg = cfg
        self.label_map = label_map or build_label_map(self.rows)
        self.garment_to_rows: Dict[str, List[int]] = {}
        for i, r in enumerate(self.rows):
            self.garment_to_rows.setdefault(r["garment_id"], []).append(i)
        self.garment_ids_sorted: List[str] = sorted(self.garment_to_rows.keys())

    def __len__(self) -> int:
        return len(self.rows)

    def garment_count(self) -> int:
        return len(self.garment_to_rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[idx]
        img = _load_image(row["path"])
        img = self._resize_center_crop(img, self.cfg.img_size)
        return img, self.label_map[row["garment_id"]]

    @staticmethod
    def _resize_center_crop(img: torch.Tensor, size: int) -> torch.Tensor:
        _, h, w = img.shape
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = F.interpolate(img[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
        top = (nh - size) // 2
        left = (nw - size) // 2
        return img[:, top:top + size, left:left + size]
