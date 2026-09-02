"""Production retrieval: embed a real photo -> nearest catalog garments.

Usage:
    python -m outfit_matcher.match --config configs/shirts.yaml --checkpoint runs/shirts/checkpoint_final.pt \
        --catalog manifest_val.jsonl --query photo.jpg --topk 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from .config import load_config
from .data.eval_dataset import EvalDataset
from .evaluate import encode_catalog
from .model.vit import ViTConfig, build_model


def load_model(checkpoint_path: str, device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mcfg = ckpt["model_cfg"]
    vit_cfg = ViTConfig(**{k: v for k, v in mcfg.items() if k in ViTConfig().__dict__})
    proj_dim = int(ckpt.get("proj_dim", 256))
    model = build_model(vit_cfg, proj_dim=proj_dim)
    raw = ckpt.get("ema", {}).get("shadow")
    if raw is None:
        raw = ckpt["model"]
    model.load_state_dict(raw, strict=True)
    model.to(device).eval()
    return model, ckpt


def embed_query_image(model, image_path: str, cfg, device) -> torch.Tensor:
    """Embed a single image with the SAME eval preprocessing (resize+center-crop)."""
    from .data.dataset import _load_image
    img = _load_image(image_path)
    img = EvalDataset._resize_center_crop(img, cfg.model.img_size)
    img = img[None].to(device)
    with torch.no_grad():
        z = model(img).float()
    return F.normalize(z, dim=-1)[0]


def query_catalog(model, query_feat: torch.Tensor, catalog: Dict, topk: int = 5) -> List[Dict]:
    sims = catalog["features"] @ query_feat.cpu()
    order = sims.argsort(descending=True)
    results = []
    seen_garments = set()
    for j in order.tolist():
        gid = catalog["garment_ids"][j]
        if gid in seen_garments:
            continue
        seen_garments.add(gid)
        results.append({
            "garment_id": gid,
            "view_path": catalog["paths"][j],
            "angle": catalog["angles"][j],
            "similarity": round(float(sims[j]), 4),
        })
        if len(results) >= topk:
            break
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Match a photo to catalog garments")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--catalog", required=True, help="manifest jsonl of catalog views")
    ap.add_argument("--query", required=True, help="photo to match")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--catalog-cache", default=None, help="optional .pt cache of catalog embeddings")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)

    if args.catalog_cache and Path(args.catalog_cache).exists():
        catalog = torch.load(args.catalog_cache, map_location="cpu", weights_only=False)
    else:
        catalog = encode_catalog(model, args.catalog, cfg, device)
        if args.catalog_cache:
            torch.save(catalog, args.catalog_cache)
            print(f"catalog cached -> {args.catalog_cache}")

    query_feat = embed_query_image(model, args.query, cfg, device)
    results = query_catalog(model, query_feat, catalog, topk=args.topk)
    print(json.dumps({"query": str(args.query), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
