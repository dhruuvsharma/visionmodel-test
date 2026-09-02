"""Scan a garment-render dataset directory into train/val manifests.

Expected layout (one folder per shirt):
    data_root/
        shirt_001/           # garment identity (folder name = label)
            front.png
            back.png
            left.png
            right.png

Angle files are matched case-insensitively by stem; unmatched files are ignored.
Usage:
    python -m outfit_matcher.data.prepare_data --data-root <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ANGLE_ALIASES = {
    "front": "front", "f": "front",
    "back": "back", "b": "back",
    "left": "left", "l": "left", "side": "left", "side_l": "left",
    "right": "right", "r": "right", "side2": "right", "side_r": "right",
}
CANONICAL_ANGLES = ("front", "back", "left", "right")


def find_angle(stem: str) -> Tuple[bool, str]:
    s = stem.strip().lower()
    for alias, canon in ANGLE_ALIASES.items():
        if s == alias or s.endswith("_" + alias) or s.endswith("-" + alias):
            return True, canon
    return False, ""


def scan_garments(data_root: Path, min_views: int = 2, strict_angles: bool = False) -> List[Dict]:
    garment_dirs = sorted(
        [p for p in data_root.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )
    records: List[Dict] = []
    for gdir in garment_dirs:
        views: Dict[str, str] = {}
        for f in sorted(gdir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMG_EXTS:
                continue
            ok, angle = find_angle(f.stem)
            if not ok:
                continue
            if angle not in views or f.name < Path(views[angle]).name:
                views[angle] = str(f.resolve())
        if strict_angles and not all(a in views for a in CANONICAL_ANGLES):
            continue
        if len(views) < min_views:
            continue
        records.append({"garment_id": gdir.name, "views": views})
    return records


def split_train_val(
    records: List[Dict], val_fraction: float = 0.05, val_count: int | None = None, seed: int = 42
) -> Tuple[List[Dict], List[Dict], set]:
    rng = random.Random(seed)
    ids = sorted(r["garment_id"] for r in records)
    if val_count is None:
        n_val = max(1, int(round(len(ids) * val_fraction)))
    else:
        n_val = val_count
    n_val = min(n_val, max(len(ids) - 1, 0))
    val_ids = set(rng.sample(ids, n_val)) if n_val > 0 else set()
    train = [r for r in records if r["garment_id"] not in val_ids]
    val = [r for r in records if r["garment_id"] in val_ids]
    return train, val, val_ids


def write_manifest(records: List[Dict], path: Path, views: Tuple[str, ...] = ()) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fwd:
        for r in records:
            for angle, fp in sorted(r["views"].items()):
                if views and angle not in views:
                    continue
                fwd.write(json.dumps({
                    "path": fp,
                    "garment_id": r["garment_id"],
                    "angle": angle,
                }) + "\n")
                count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan garment render directory into manifests")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--val-count", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--strict-angles", action="store_true")
    ap.add_argument("--views", nargs="*", default=None,
                    help="restrict manifests to these angles (e.g. --views front); default all")
    args = ap.parse_args()

    if not args.data_root.is_dir():
        print(f"error: data root not found: {args.data_root}", flush=True)
        return 1

    records = scan_garments(args.data_root, args.min_views, args.strict_angles)
    if not records:
        print("error: no garments matched (check folder layout / angle names)", flush=True)
        return 1

    total_views = sum(len(r["views"]) for r in records)
    views_filter = tuple(args.views) if args.views else ()
    if views_filter:
        kept = sum(1 for r in records for a in r["views"] if a in views_filter)
        print(f"found {len(records)} garments, {total_views} images "
              f"({kept} after view filter {list(views_filter)})", flush=True)
    else:
        print(f"found {len(records)} garments, {total_views} images", flush=True)

    train, val, val_ids = split_train_val(records, args.val_fraction, args.val_count, args.seed)
    n_train = write_manifest(train, args.out_dir / "manifest_train.jsonl", views_filter)
    n_val = write_manifest(val, args.out_dir / "manifest_val.jsonl", views_filter)
    with open(args.out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump({"train_garments": len(train), "val_garments": len(val),
                   "val_ids": sorted(val_ids)}, f, indent=2)
    print(f"train: {len(train)} garments / {n_train} images -> {args.out_dir / 'manifest_train.jsonl'}", flush=True)
    print(f"val:   {len(val)} garments / {n_val} images -> {args.out_dir / 'manifest_val.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
