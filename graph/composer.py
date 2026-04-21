from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RelationalSemiringComposer(nn.Module):
    """Fuse [M, N, D] -> [N, D] via sum/max/agr semiring branches."""

    def __init__(self, d_model: int, n_relations_max: int, mode: str = "semiring"):
        super().__init__()
        self.d_model = d_model
        self.n_max = n_relations_max
        self.mode = mode

        self.W_beta = nn.Linear(d_model * n_relations_max, n_relations_max)
        self.W_max = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_relations_max)])
        self.phi = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_relations_max)])
        self.W_attn = nn.Linear(d_model, d_model)
        self.v_attn = nn.Parameter(torch.randn(d_model) * 0.02)

        if mode == "semiring":
            self.W_c = nn.Linear(3 * d_model, d_model)
        else:
            self.W_c = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _sum_branch(self, zs: torch.Tensor) -> torch.Tensor:
        M, N, D = zs.shape
        pad = self.n_max - M
        cat = zs.permute(1, 0, 2).reshape(N, M * D)
        if pad > 0:
            cat = F.pad(cat, (0, pad * D))
        logits = self.W_beta(cat)[:, :M]
        gate = F.softmax(logits, dim=-1)
        return (gate.t().unsqueeze(-1) * zs).sum(dim=0)

    def _max_branch(self, zs: torch.Tensor) -> torch.Tensor:
        M = zs.shape[0]
        projected = torch.stack([self.W_max[m](zs[m]) for m in range(M)], dim=0)
        return projected.max(dim=0).values

    def _agr_branch(self, zs: torch.Tensor) -> torch.Tensor:
        M, N, D = zs.shape
        phis = [self.phi[m](zs[m]) for m in range(M)]
        if M < 2:
            return torch.zeros(N, D, device=zs.device, dtype=zs.dtype)
        out = torch.zeros(N, D, device=zs.device, dtype=zs.dtype)
        for i in range(M):
            for j in range(i + 1, M):
                out = out + phis[i] * phis[j]
        return out

    def _attn_branch(self, zs: torch.Tensor) -> torch.Tensor:
        M = zs.shape[0]
        scores = torch.stack([
            (torch.tanh(self.W_attn(zs[m])) * self.v_attn).sum(dim=-1)
            for m in range(M)
        ], dim=0)
        alpha = F.softmax(scores, dim=0).unsqueeze(-1)
        return (alpha * zs).sum(dim=0)

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        if self.mode == "semiring":
            z_sum = self._sum_branch(zs)
            z_max = self._max_branch(zs)
            z_agr = self._agr_branch(zs)
            z = self.W_c(torch.cat([z_sum, z_max, z_agr], dim=-1))
        elif self.mode == "sum":
            z = self.W_c(self._sum_branch(zs))
        elif self.mode == "max":
            z = self.W_c(self._max_branch(zs))
        elif self.mode == "agr":
            z = self.W_c(self._agr_branch(zs))
        elif self.mode == "attn":
            z = self.W_c(self._attn_branch(zs))
        else:
            raise ValueError(f"unknown composer mode: {self.mode}")
        return self.norm(z)
