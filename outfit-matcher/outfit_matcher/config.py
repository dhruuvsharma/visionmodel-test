"""YAML config loading -> typed dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    img_size: int = 224
    patch_size: int = 16
    dim: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.0
    embed_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    proj_dim: int = 256


@dataclass
class DataConfig:
    manifest: str = "data/manifest_train.jsonl"
    val_manifest: str = "data/manifest_val.jsonl"
    img_size: int = 224
    # background keying for renders (None disables compositing)
    bg_key_tolerance: float = 0.08  # near-white threshold for border flood fill
    bg_key_grow: int = 2  # px dilation of garment mask
    p_composite: float = 0.8  # probability of replacing white bg per sample
    min_scale: float = 0.5
    max_scale: float = 1.0
    color_jitter: float = 0.4
    p_gray: float = 0.05
    p_perspective: float = 0.3
    p_occlusion: float = 0.5
    occlusion_scale: tuple = (0.02, 0.15)
    p_blur: float = 0.2
    p_noise: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 100
    garments_per_batch: int = 32  # P
    views_per_garment: int = 2  # V (SupCon positives per class per GPU)
    workers: int = 8
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    temperature: float = 0.1
    aux_ce_weight: float = 0.5  # garment-classification aux loss on CLS features
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    bf16: bool = True
    log_every: int = 20
    eval_every: int = 5  # epochs
    out_dir: str = "runs/shirts"
    resume: Optional[str] = None
    seed: int = 42
    channels_last: bool = False
    # restrict TRAINING to these render angles (e.g. ["front"] for a front-only
    # prototype); null = all views. Eval/catalog encoding always use all views,
    # so leave-one-view-out retrieval stays meaningful.
    views: Optional[list] = None


@dataclass
class EvalConfig:
    batch_size: int = 64
    recall_k: tuple = (1, 5, 10, 20)


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _build(cls, raw: Optional[Dict[str, Any]]) -> Any:
    if not raw:
        return cls()
    valid = {f.name: v for f in dataclasses.fields(cls) for k, v in raw.items() if f.name == k}
    return cls(**valid)


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(
        model=_build(ModelConfig, raw.get("model")),
        data=_build(DataConfig, raw.get("data")),
        train=_build(TrainConfig, raw.get("train")),
        eval=_build(EvalConfig, raw.get("eval")),
    )
