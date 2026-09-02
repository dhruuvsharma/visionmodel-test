"""Domain-randomization transforms: render -> real-photo simulation.

Pipeline per sample:
1. Load RGBA/RGB render, convert to RGB.
2. Key out the near-white background via border flood fill -> binary garment mask.
3. Optionally composite garment onto a procedural background.
4. Geometric: bbox crop with scale jitter, horizontal flip, small affine, occasional perspective.
5. Photometric: brightness/contrast/saturation/hue jitter, grayscale, blur, noise.
6. Random occluder rectangles.
7. Resize to model input, normalize.

Torch-only ops (deterministic given seed). Images are float tensors in [0, 1].
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# bright-ish procedural background palettes (R, G, B means)
_BG_PALETTES = [
    (0.55, 0.65, 0.75), (0.80, 0.78, 0.70), (0.45, 0.55, 0.45), (0.70, 0.60, 0.55),
    (0.60, 0.70, 0.80), (0.85, 0.85, 0.88), (0.35, 0.40, 0.50), (0.75, 0.70, 0.80),
]


# ---------------------------------------------------------------------------
# background keying
# ---------------------------------------------------------------------------

def _binary_dilate(mask: torch.Tensor, k: int) -> torch.Tensor:
    """Max-pool dilation for a bool mask."""
    if k <= 0:
        return mask
    pad = k // 2
    m = mask[None, None].float()
    m = F.max_pool2d(m, kernel_size=2 * pad + 1, stride=1, padding=pad)
    return m[0, 0] > 0.5


def _flood_fill_keying(img: torch.Tensor, tolerance: float = 0.08, grow: int = 2) -> torch.Tensor:
    """Segment garment from near-white background via border flood fill.

    img: (3, H, W) float in [0, 1]. Returns bool mask (H, W); True = garment.
    """
    _, h, w = img.shape
    near_white = (img >= (1.0 - tolerance)).all(dim=0)  # (H, W)
    if near_white.all():
        return torch.zeros((h, w), dtype=torch.bool, device=img.device)

    visited = torch.zeros((h, w), dtype=torch.bool, device=img.device)

    seeds: List[Tuple[int, int]] = []
    for x in range(w):
        for y in (0, h - 1):
            if near_white[y, x]:
                visited[y, x] = True
                seeds.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if near_white[y, x] and not visited[y, x]:
                visited[y, x] = True
                seeds.append((y, x))

    # BFS over 4-neighborhood
    idx = 0
    while idx < len(seeds):
        y, x = seeds[idx]
        idx += 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and near_white[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                seeds.append((ny, nx))

    garment = ~visited
    if grow > 0:
        garment = _binary_dilate(garment, grow)
    return garment


def key_white_background(img: torch.Tensor, tolerance: float, grow: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (garment_pixels_rgb, mask). Garment pixels keep color; background set to 0."""
    mask = _flood_fill_keying(img, tolerance, grow)
    return img * mask[None], mask


# ---------------------------------------------------------------------------
# procedural backgrounds
# ---------------------------------------------------------------------------

def _paint_background(size: Tuple[int, int], palette: Tuple[float, float, float], device) -> torch.Tensor:
    """Smooth random background: base color + low-frequency blobby variation."""
    h, w = size
    c = torch.tensor(palette, device=device).view(3, 1, 1)
    low = torch.randn(3, max(h // 16, 1), max(w // 16, 1), device=device) * 0.08
    low = F.interpolate(low[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    fine = torch.randn(3, h, w, device=device) * 0.02
    bg = (c + low + fine).clamp(0.0, 1.0)
    return bg


def composite_onto_background(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """img: (3,H,W) garment pixels (bg zeroed). mask: (H,W) bool."""
    palette = random.choice(_BG_PALETTES)
    bg = _paint_background(img.shape[1:], palette, img.device)
    return torch.where(mask[None], img, bg)


# ---------------------------------------------------------------------------
# geometric ops
# ---------------------------------------------------------------------------

def _rotate_affine(img: torch.Tensor, mask: torch.Tensor, angle_deg: float, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate both img and mask around center using an affine grid."""
    theta = math.radians(angle_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    # inverse mapping (grid_sample convention)
    mat = torch.tensor([
        [cos / scale, sin / scale, 0.0],
        [-sin / scale, cos / scale, 0.0],
    ], dtype=img.dtype, device=img.device)
    grid = F.affine_grid(mat[None], img[None].shape, align_corners=False)
    img_r = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    mask_r = F.grid_sample(mask[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0] > 0.5
    return img_r, mask_r


def _perspective_warp(img: torch.Tensor, mask: torch.Tensor, max_shift: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random mild perspective: shift the four corners of the unit square."""
    h, w = img.shape[1:]
    device = img.device
    corners_src = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], device=device)
    shift = (torch.rand(4, 2, device=device) * 2 - 1) * max_shift
    corners_dst = (corners_src + shift).clamp(0.0, 1.0)
    H = _homography(corners_src, corners_dst)
    grid = _apply_homography_to_grid(H, h, w, device)
    img_w = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    mask_w = F.grid_sample(mask[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0] > 0.5
    return img_w, mask_w


def _homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Solve 4-point homography src -> dst (both (4,2) in [0,1] coords)."""
    A = src.new_zeros((8, 8))
    b = dst.reshape(-1)
    one = torch.ones((), device=src.device, dtype=src.dtype)
    zero = torch.zeros((), device=src.device, dtype=src.dtype)
    for i in range(4):
        x, y = src[i, 0], src[i, 1]
        A[2 * i] = torch.stack([x, y, one, zero, zero, zero, -x * dst[i, 0], -y * dst[i, 0]])
        A[2 * i + 1] = torch.stack([zero, zero, zero, x, y, one, -x * dst[i, 1], -y * dst[i, 1]])
    h = torch.linalg.solve(A, b)
    H = torch.cat([h, torch.ones(1, device=src.device)]).view(3, 3)
    return H


def _apply_homography_to_grid(H: torch.Tensor, h: int, w: int, device) -> torch.Tensor:
    """Generate sampling grid (1,h,w,2) for torch.grid_sample given 3x3 homography."""
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, w, device=device),
        torch.linspace(-1, 1, h, device=device),
        indexing="xy",
    )
    ones = torch.ones_like(xs)
    pts = torch.stack([xs, ys, ones], dim=-1)  # (h, w, 3)
    warped = pts @ H.T
    warped = warped[..., :2] / warped[..., 2:3].clamp(min=1e-6)
    return warped[None]


# ---------------------------------------------------------------------------
# photometric ops
# ---------------------------------------------------------------------------

def _color_jitter(img: torch.Tensor, strength: float) -> torch.Tensor:
    b = 1.0 + (random.random() * 2 - 1) * strength * 0.4   # brightness
    c = 1.0 + (random.random() * 2 - 1) * strength * 0.4   # contrast
    s = 1.0 + (random.random() * 2 - 1) * strength * 0.4   # saturation
    img = img * b
    mean = img.mean(dim=(1, 2), keepdim=True)
    img = (img - mean) * c + mean
    img = _adjust_saturation(img, s)
    return img.clamp(0.0, 1.0)


def _adjust_saturation(img: torch.Tensor, factor: float) -> torch.Tensor:
    gray = img.mean(dim=0, keepdim=True)
    return (gray + (img - gray) * factor).clamp(0.0, 1.0)


def _hue_shift(img: torch.Tensor, delta: float) -> torch.Tensor:
    """Shift hue by delta in [-0.5, 0.5] via RGB->HSV->RGB (used sparingly)."""
    hsv = _rgb_to_hsv(img)
    h = torch.frac(hsv[0:1] + delta)[0]
    h = torch.where(h < 0, h + 1, h)
    hsv = torch.cat([h[None], hsv[1:]])
    return _hsv_to_rgb(hsv)


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[0], img[1], img[2]
    mx, mn = img.max(0).values, img.min(0).values
    diff = mx - mn
    h = torch.zeros_like(mx)
    m = (mx == r) & (diff > 0)
    h[m] = ((g - b)[m] / diff[m]) / 6.0
    m = (mx == g) & (diff > 0)
    h[m] = (2.0 + ((b - r)[m] / diff[m])) / 6.0
    m = (mx == b) & (diff > 0)
    h[m] = (4.0 + ((r - g)[m] / diff[m])) / 6.0
    s = torch.where(mx > 0, diff / (mx + 1e-8), torch.zeros_like(mx))
    v = mx
    return torch.stack([h, s, v])


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[0], hsv[1], hsv[2]
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - f) * s
    case_idx = i.long() % 6
    r = torch.zeros_like(v); g = torch.zeros_like(v); b = torch.zeros_like(v)
    for idx, (rr, gg, bb) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,q),(v,p,q)]):
        m = case_idx == idx
        r[m] = rr[m]; g[m] = gg[m]; b[m] = bb[m]
    return torch.stack([r, g, b])


# ---------------------------------------------------------------------------
# occlusion
# ---------------------------------------------------------------------------

def add_occluders(img: torch.Tensor, mask: torch.Tensor, scale_range: Tuple[float, float], n_max: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random rectangle occluders (simulating arms/objects in front of garment)."""
    _, h, w = img.shape
    for _ in range(n_max):
        if random.random() > 0.5:
            continue
        frac = random.uniform(*scale_range)
        ow = max(4, int(w * frac))
        oh = max(4, int(h * random.uniform(*scale_range)))
        x0 = random.randint(0, max(w - ow - 1, 0))
        y0 = random.randint(0, max(h - oh - 1, 0))
        # occluder colors drawn from palette (skin/clothing tones)
        col = torch.tensor(random.choice(_BG_PALETTES), device=img.device)
        patch = torch.full((3, oh, ow), 0.0, device=img.device)
        for c in range(3):
            patch[c].fill_(float(col[c]))
        patch.mul_(0.8 + 0.4 * torch.rand(1, oh, ow, device=img.device)).clamp_(0.0, 1.0)
        img[:, y0:y0 + oh, x0:x0 + ow] = patch
        mask[y0:y0 + oh, x0:x0 + ow] = False
    return img, mask


# ---------------------------------------------------------------------------
# main augmentation pipeline
# ---------------------------------------------------------------------------

class RenderAugment:
    """Full render->real augmentation pipeline, applied per-sample on CPU tensors."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: (3,H,W) float [0,1] raw render -> augmented (3,H,W) in [0,1]."""
        cfg = self.cfg
        img = img.clone()

        # 1) key white background
        img0, mask = key_white_background(img, cfg.bg_key_tolerance, cfg.bg_key_grow)

        # 2) optional composite onto procedural background
        if random.random() < cfg.p_composite:
            img0 = composite_onto_background(img0, mask)

        # 3) geometric
        if random.random() < cfg.p_perspective:
            img0, mask = _perspective_warp(img0, mask)
        img0, mask = _rotate_affine(img0, mask, angle_deg=random.uniform(-7, 7), scale=random.uniform(0.95, 1.1))

        # 4) photometric
        img0 = _color_jitter(img0, cfg.color_jitter)
        if random.random() < cfg.p_gray:
            gray = img0.mean(dim=0, keepdim=True)
            img0 = gray.expand(3, -1, -1).clone()  # clone: expand shares memory, breaks later in-place writes
        if random.random() < cfg.p_blur:
            k = random.choice([3, 5])
            img0 = _gaussian_blur(img0, sigma=random.uniform(0.5, 1.5), ksize=k)
        if random.random() < cfg.p_noise:
            img0 = (img0 + torch.randn_like(img0) * random.uniform(0.01, 0.05)).clamp(0.0, 1.0)

        # 5) occluders
        if random.random() < cfg.p_occlusion:
            img0, mask = add_occluders(img0, mask, cfg.occlusion_scale)

        # 6) random crop around garment
        img0 = _garment_crop(img0, mask, cfg.min_scale, cfg.max_scale)

        # 7) horizontal flip (T-pose: left/right views become each other; safe regardless)
        if random.random() < 0.5:
            img0 = torch.flip(img0, dims=[2])

        # 8) resize to model input
        img0 = F.interpolate(img0[None], size=(cfg.img_size, cfg.img_size), mode="bilinear", align_corners=False)[0]
        return img0


def _gaussian_blur(img: torch.Tensor, sigma: float, ksize: int) -> torch.Tensor:
    x = torch.arange(ksize, dtype=img.dtype, device=img.device) - (ksize - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2 + 1e-8))
    g = (g / g.sum())
    img = img[None]  # (1, 3, H, W)
    img = F.conv2d(img, g.view(1, 1, 1, ksize).expand(3, -1, -1, -1), padding=(0, ksize // 2), groups=3)
    img = F.conv2d(img, g.view(1, 1, ksize, 1).expand(3, -1, -1, -1), padding=(ksize // 2, 0), groups=3)
    return img[0]


def _garment_crop(img: torch.Tensor, mask: torch.Tensor, min_scale: float, max_scale: float) -> torch.Tensor:
    """Random crop containing the garment (or full image if mask empty)."""
    _, h, w = img.shape
    if not mask.any():
        return img
    ys, xs = mask.nonzero(as_tuple=True)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    gh, gw = y1 - y0, x1 - x0
    # scale jitter on garment bbox
    s = random.uniform(min_scale, max_scale)
    gh2, gw2 = max(1, int(gh * (1 / s))), max(1, int(gw * (1 / s)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    # crop with random shift around center
    y0c = cy - gh2 // 2 + random.randint(-gh2 // 8, gh2 // 8)
    x0c = cx - gw2 // 2 + random.randint(-gw2 // 8, gw2 // 8)
    # ensure within bounds, padding with zeros if needed
    py0, py1 = max(0, -y0c), max(0, y0c + gh2 - h)
    px0, px1 = max(0, -x0c), max(0, x0c + gw2 - w)
    y0c, x0c = max(0, y0c), max(0, x0c)
    y1c, x1c = min(h, y0c + gh2), min(w, x0c + gw2)
    crop = img[:, y0c:y1c, x0c:x1c]
    if py0 or py1 or px0 or px1:
        crop = F.pad(crop, (px0, px1, py0, py1))
    return crop
