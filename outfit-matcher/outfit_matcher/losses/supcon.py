"""Supervised Contrastive Loss (Khosla et al., 2020) with DDP cross-GPU gathering.

L = (1/|P(i)|) * sum_{p in P(i)} -log( exp(z_i . z_p / tau) / sum_{a != i} exp(z_i . z_a / tau) )

Features must be L2-normalized before calling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, gather_distributed: bool = False):
        super().__init__()
        self.temperature = float(temperature)
        self.gather_distributed = gather_distributed
        self.world_size = dist.get_world_size() if gather_distributed and dist.is_initialized() else 1

    @staticmethod
    def _gather(features: torch.Tensor, labels: torch.Tensor):
        """All-gather across ranks; grad flows only through the local rank's slot."""
        ws = dist.get_world_size()
        rank = dist.get_rank()
        device = features.device

        if device.type == "cuda" and dist.get_backend() == "nccl":
            # NCCL: gather on GPU directly
            fs = [torch.zeros_like(features) for _ in range(ws)]
            ls = [torch.zeros_like(labels, device=device) for _ in range(ws)]
            dist.all_gather(fs, features.contiguous())
            dist.all_gather(ls, labels.to(device).contiguous())
            fs[rank] = features
            ls[rank] = labels
            return torch.cat(fs, 0), torch.cat(ls, 0)

        # gloo (Windows): gather on CPU, move back
        f_cpu = features.detach().to("cpu").contiguous()
        l_cpu = labels.detach().to("cpu").contiguous()
        fs = [torch.zeros_like(f_cpu) for _ in range(ws)]
        ls = [torch.zeros_like(l_cpu) for _ in range(ws)]
        dist.all_gather(fs, f_cpu)
        dist.all_gather(ls, l_cpu)
        out_f = [features if i == rank else fs[i].to(device) for i in range(ws)]
        out_l = [labels if i == rank else ls[i].to(device) for i in range(ws)]
        return torch.cat(out_f, 0), torch.cat(out_l, 0)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert features.ndim == 2, "features must be (N, D), L2-normalized"
        if self.world_size > 1:
            features, labels = self._gather(features, labels)
        device = features.device
        n = features.shape[0]
        if n < 2:
            return features.new_zeros(())

        sim = features @ features.T / self.temperature  # (N, N)

        # mask self-contrast
        eye = torch.eye(n, dtype=torch.bool, device=device)
        sim = sim.masked_fill(eye, float("-inf"))

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye  # (N, N)
        pos_counts = pos_mask.sum(dim=1)

        # rows with no positives contribute 0
        valid = pos_counts > 0
        if not valid.any():
            return features.new_zeros(())

        # log-softmax over all non-self entries
        log_prob = sim - sim.logsumexp(dim=1, keepdim=True)
        # select positive entries (avoid 0 * -inf = NaN): where-select, not multiply
        pos_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
        mean_log_prob_pos = pos_log_prob.sum(dim=1) / pos_counts.clamp(min=1)
        loss = -mean_log_prob_pos[valid].mean()
        return loss
