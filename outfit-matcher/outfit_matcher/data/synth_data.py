"""Generate a tiny synthetic garment-render dataset for smoke tests.

Creates N garments, each a colored T-pose shape on a white background at 4 angles.
Deterministic (seeded). Layout: one folder per garment, front/back/left/right.png.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def draw_tpose_garment(size: int, color: tuple, angle: str, rng: random.Random) -> Image.Image:
    """Draw a crude T-posed garment silhouette on white, angle-warped slightly."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    s = size
    cx = s / 2
    body_w = s * rng.uniform(0.16, 0.24)
    body_h = s * rng.uniform(0.34, 0.44)
    arm_span = s * rng.uniform(0.5, 0.7)
    sleeve_h = s * rng.uniform(0.07, 0.11)
    top = s * 0.22

    # torso
    d.rectangle([cx - body_w / 2, top, cx + body_w / 2, top + body_h], fill=color)
    # arms (T-pose horizontal bar)
    d.rectangle([cx - arm_span / 2, top + 0.02 * s, cx + arm_span / 2, top + 0.02 * s + sleeve_h], fill=color)
    # subtle per-angle shading so views are distinguishable
    shade = {"front": 1.0, "back": 0.88, "left": 0.94, "right": 0.91}[angle]
    px = img.load()
    arr = np.array(img, dtype=np.int16)
    arr = np.clip(arr * shade, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic garment data for smoke tests")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--garments", type=int, default=12)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    palettes = [
        (200, 60, 60), (60, 120, 200), (90, 170, 90), (210, 160, 50), (150, 80, 170),
        (70, 70, 90), (230, 120, 60), (50, 150, 150), (180, 70, 120), (120, 200, 80),
        (240, 200, 70), (100, 60, 200),
    ]
    for i in range(args.garments):
        gdir = args.out_dir / f"shirt_{i:03d}"
        gdir.mkdir(parents=True, exist_ok=True)
        color = palettes[i % len(palettes)]
        color = tuple(min(255, int(c * rng.uniform(0.8, 1.2))) for c in color)
        for angle in ("front", "back", "left", "right"):
            img = draw_tpose_garment(args.size, color, angle, rng)
            img.save(gdir / f"{angle}.png")
    print(f"generated {args.garments} synthetic garments -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
