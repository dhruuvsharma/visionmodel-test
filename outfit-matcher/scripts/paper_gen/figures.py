"""Paper figures: architecture, SupCon embedding space, LR schedule, augmentation panels, retrieval.

All figures are generated with matplotlib; the augmentation panels run the REAL
outfit_matcher transforms on a synthetic render, so the paper shows true pipeline output.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["axes.linewidth"] = 0.6


def _box(ax, x, y, w, h, label, sub=None, fc="#eef3fb", ec="#31527a", fs=8.0, lw=0.8):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                            fc=fc, ec=ec, lw=lw)
    ax.add_patch(patch)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center", fontsize=fs, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 2.0, color="#444444")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)
    return (x, y, w, h)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.8, color="#31527a"))


def figure_architecture(path, figsize=(7.4, 4.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _box(ax, 0.1, 2.4, 1.15, 1.2, "Input", "224 x 224 x 3", fc="#f2f2f2", ec="#666666")
    _box(ax, 1.7, 2.4, 1.5, 1.2, "Patch Stem", "Conv2d k=s=16", fc="#e7f0e4", ec="#3a6b3a")
    _box(ax, 3.65, 2.4, 1.55, 1.2, "Tokens", "196 + CLS = 197", fc="#f7f2dd", ec="#8a7420")
    ax.text(4.42, 2.18, r"$\mathbf{z}_0 = \mathrm{patches} + \mathbf{E}_{pos}$", ha="center", fontsize=8)
    _box(ax, 5.65, 2.4, 2.1, 1.2, "12 Transformer", "blocks (pre-norm)", fc="#eef3fb")
    ax.text(6.7, 2.18, r"$\times$ 12 depth", ha="center", fontsize=8, color="#444444")
    _box(ax, 8.15, 2.4, 1.1, 1.2, "Layer", "Norm", fc="#f2f2f2", ec="#666666")

    _arrow(ax, 1.25, 3.0, 1.7, 3.0)
    _arrow(ax, 3.2, 3.0, 3.65, 3.0)
    _arrow(ax, 5.2, 3.0, 5.65, 3.0)
    _arrow(ax, 7.75, 3.0, 8.15, 3.0)

    # block internals above
    _box(ax, 5.5, 4.35, 2.4, 1.35, "", fc="#ffffff")
    ax.text(6.7, 5.42, "Block internals", ha="center", fontsize=8, weight="bold")
    ax.text(6.7, 5.12, r"$\mathrm{LN} \rightarrow$ MSA(6 heads, $d_k=64$)", ha="center", fontsize=7.5)
    ax.text(6.7, 4.88, r"$\mathrm{LN} \rightarrow$ MLP(384 $\rightarrow$ 1536 $\rightarrow$ 384)", ha="center", fontsize=7.5)
    ax.text(6.7, 4.64, "residual + DropPath (stochastic depth)", ha="center", fontsize=7.5, color="#444444")
    ax.add_patch(FancyArrowPatch((5.65, 4.35), (5.65, 3.6), arrowstyle="-", lw=0.6, color="#999999"))

    # head below
    _box(ax, 8.15, 0.9, 1.1, 1.1, "CLS", "token 384-d", fc="#f7e7e7", ec="#8a3a3a")
    _box(ax, 6.6, 0.9, 1.2, 1.1, "Projection", "384-384-256", fc="#f7e7e7", ec="#8a3a3a")
    _box(ax, 4.7, 0.9, 1.5, 1.1, "L2 normalize", r"$\|f(x)\|_2 = 1$", fc="#e7f0e4", ec="#3a6b3a")
    _box(ax, 2.4, 0.9, 1.9, 1.1, "Embedding", r"$f_\theta(x) \in \mathbb{R}^{256}$", fc="#eef3fb")
    ax.text(8.7, 2.15, "pool", ha="center", fontsize=7, color="#444444")
    _arrow(ax, 8.7, 2.4, 8.7, 2.0)
    _arrow(ax, 8.15, 1.45, 7.8, 1.45)
    _arrow(ax, 6.6, 1.45, 6.2, 1.45)
    _arrow(ax, 4.7, 1.45, 4.3, 1.45)

    ax.text(0.1, 5.6, "Outfit Matcher encoder (from scratch, 21.9M parameters)",
            fontsize=9, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def figure_embedding_space(path, figsize=(7.2, 2.9)):
    """Schematic: per-garment view embeddings before vs after SupCon training."""
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    n_g, n_v = 5, 4
    markers = ["o", "s", "^", "D"]

    # left: untrained - random directions, no structure
    for g in range(n_g):
        pts = rng.normal(size=(n_v, 2)) * 1.1
        for v in range(n_v):
            axes[0].scatter(*pts[v], marker=markers[v], s=26,
                            color=f"C{g}", edgecolor="k", linewidth=0.3)
    axes[0].set_title("Untrained encoder", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].text(0.5, -0.14, "views of one garment scattered", transform=axes[0].transAxes,
                 ha="center", fontsize=8, color="#444444")

    # right: trained - tight clusters, views separated by marker
    for g in range(n_g):
        center = np.array([np.cos(g / n_g * 2 * np.pi), np.sin(g / n_g * 2 * np.pi)]) * 1.6
        for v in range(n_v):
            off = rng.normal(size=2) * 0.13
            axes[1].scatter(*(center + off), marker=markers[v], s=26,
                            color=f"C{g}", edgecolor="k", linewidth=0.3)
    axes[1].set_title("After SupCon training (schematic)", fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].text(0.5, -0.14, "colors = garments, markers = view angles", transform=axes[1].transAxes,
                 ha="center", fontsize=8, color="#444444")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def figure_lr_schedule(path, figsize=(5.4, 2.5)):
    epochs, warmup, peak, floor = 100, 5, 1e-3, 1e-5
    xs = np.arange(0, epochs, 0.1)
    ys = []
    for t in xs:
        if t < warmup:
            ys.append(peak * t / warmup)
        else:
            p = (t - warmup) / (epochs - warmup)
            ys.append(floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * p)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, np.array(ys) * 1e3, lw=1.4, color="#31527a")
    ax.axvspan(0, warmup, color="#f7f2dd", zorder=0)
    ax.text(warmup / 2, peak * 1e3 * 0.55, "linear\nwarmup", ha="center", fontsize=8, color="#8a7420")
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("learning rate (1e-3)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def figure_augmentation(path, synth_dir, img_size=224):
    """Run the REAL pipeline stages on a synthetic render and show panels."""
    import random
    from outfit_matcher.config import DataConfig
    from outfit_matcher.data.dataset import _load_image
    from outfit_matcher.data import transforms as T

    random.seed(11)
    torch.manual_seed(11)

    cfg = DataConfig(img_size=img_size, p_composite=0.9)
    img = _load_image(str(Path(synth_dir) / "shirt_000" / "front.png"))

    raw = img.clone()
    garment, mask = T.key_white_background(img, cfg.bg_key_tolerance, cfg.bg_key_grow)

    keyed = img.clone()
    keyed = keyed * mask[None].float()

    comp = T.composite_onto_background(keyed.clone(), mask)
    warped, wmask = T._perspective_warp(comp, mask)
    warped, wmask = T._rotate_affine(warped, wmask, angle_deg=6.0, scale=1.03)
    occ, _ = T.add_occluders(warped.clone(), wmask, cfg.occlusion_scale)
    crop = T._garment_crop(occ, wmask, cfg.min_scale, cfg.max_scale)
    final = torch.nn.functional.interpolate(crop[None], size=(img_size, img_size),
                                            mode="bilinear", align_corners=False)[0]

    panels = [
        (raw, "1. Render input"),
        (img * (0.35 + 0.65 * mask[None].float()), "2. Flood-fill keying"),
        (comp, "3. Procedural background"),
        (warped, "4. Perspective + affine"),
        (occ, "5. Occlusion"),
        (final, "6. Crop + resize (training tensor)"),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(7.6, 1.75))
    for ax, (t, title) in zip(axes, panels):
        arr = t.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(arr)
        ax.set_title(title, fontsize=6.6, pad=3)
        ax.axis("off")
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def figure_retrieval(path, figsize=(5.6, 2.7)):
    """Schematic top-K retrieval bars after per-garment view max-pooling."""
    garments = ["shirt_014", "shirt_007", "shirt_063", "shirt_231", "shirt_005", "shirt_102"]
    sims = [0.93, 0.88, 0.84, 0.81, 0.79, 0.74]
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#3a6b3a" if i == 0 else "#8aa4c0" for i in range(len(garments))]
    ax.barh(range(len(garments))[::-1], sims, color=colors, height=0.62)
    ax.set_yticks(range(len(garments))[::-1])
    ax.set_yticklabels(garments, fontsize=8)
    ax.set_xlabel("cosine similarity  s(f(query), f(view))", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 1.0)
    ax.axvline(sims[0], color="#3a6b3a", lw=0.6, ls="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Top-K retrieval (schematic scores)", fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def build_all(fig_dir: Path, synth_dir: str) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    out["arch"] = fig_dir / "fig_architecture.png";        figure_architecture(out["arch"])
    out["embed"] = fig_dir / "fig_embedding.png";         figure_embedding_space(out["embed"])
    out["lr"] = fig_dir / "fig_lr.png";                   figure_lr_schedule(out["lr"])
    out["aug"] = fig_dir / "fig_augmentation.png";        figure_augmentation(out["aug"], synth_dir)
    out["retrieval"] = fig_dir / "fig_retrieval.png";      figure_retrieval(out["retrieval"])
    return out
