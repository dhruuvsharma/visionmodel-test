"""From-scratch Vision Transformer.

No pretrained weights anywhere. Components:
- Patchify: Conv2d stride-16 stem (patch embedding)
- Learnable 1D absolute positional embedding
- Pre-norm transformer encoder blocks (MultiheadAttention)
- CLS token for global representation
- MLP projection head for contrastive training

Reference: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViTConfig:
    """Hyperparameters of the vision transformer backbone."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        embed_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.embed_drop_rate = embed_drop_rate
        self.drop_path_rate = drop_path_rate

    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    def to_dict(self) -> dict:
        return dict(self.__dict__.items())


class DropPath(nn.Module):
    """Per-sample stochastic depth on a residual branch."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed(nn.Module):
    """Split image into patches and embed via Conv2d with stride=patch_size."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_chans,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, config: ViTConfig, drop_path: float):
        super().__init__()
        dim = config.dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.heads,
            dropout=config.attn_drop_rate,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(config.drop_rate),
            nn.Linear(hidden, dim),
            nn.Dropout(config.drop_rate),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path(attn_out)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class VisionTransformer(nn.Module):
    """ViT backbone returning CLS embedding, optional projection head for contrastive."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        num_patches = config.num_patches()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, config.dim))
        self.pos_drop = nn.Dropout(config.embed_drop_rate)

        drop_path_schedule = torch.linspace(0.0, config.drop_path_rate, config.depth)
        self.blocks = nn.ModuleList(
            [Block(config, float(drop_path_schedule[i])) for i in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        w = self.patch_embed.proj.weight
        nn.init.trunc_normal_(w.view(w.shape[0], -1), std=0.02)
        nn.init.zeros_(self.patch_embed.proj.bias)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.eps = 1e-6
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_tokens(self) -> int:
        return 1 + self.config.num_patches()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor, return_cls: bool = False):
        tokens = self.forward_features(x)
        cls = tokens[:, 0]
        proj = self.head(cls)
        if return_cls:
            return proj, cls
        return proj

    def embed_dim(self) -> int:
        return self.config.dim


class SupConHead(nn.Module):
    """Non-linear projection head (SimCLR/SupCon style)."""

    def __init__(self, dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(config: ViTConfig, proj_dim: int = 256) -> VisionTransformer:
    model = VisionTransformer(config)
    model.head = SupConHead(config.dim, proj_dim)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
