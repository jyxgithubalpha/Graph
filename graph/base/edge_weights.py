from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from domain.types import DayBatch


class CosineSimilarity(nn.Module):
    """Edge weight via cosine similarity on x_factor."""

    def forward(self, batch: DayBatch, ctx: dict) -> torch.Tensor:
        x = batch.x_factor
        x_norm = F.normalize(x, p=2, dim=-1)
        w = torch.mm(x_norm, x_norm.t())
        w = F.relu(w)
        w.fill_diagonal_(0.0)
        return w


class PearsonCorrelation(nn.Module):
    """Edge weight via Pearson correlation on ret_hist."""

    def forward(self, batch: DayBatch, ctx: dict) -> torch.Tensor:
        x = batch.ret_hist
        x_mean = x - x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True) + 1e-8
        x_norm = x_mean / x_std
        w = torch.mm(x_norm, x_norm.t()) / x.shape[-1]
        w = F.relu(w)
        w.fill_diagonal_(0.0)
        return w


class LearnedAttention(nn.Module):
    """Learnable attention edge weights based on node embedding h."""

    def __init__(self, d_model: int, prior_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.prior_scale = prior_scale
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)

    def forward(self, batch: DayBatch, ctx: dict) -> torch.Tensor:
        h = ctx["h"]
        Q = self.W_q(h)
        K = self.W_k(h)
        logits = torch.mm(Q, K.t()) / math.sqrt(self.d_model)

        prior_bias = ctx.get("prior_bias")
        if prior_bias is not None:
            logits = logits + self.prior_scale * prior_bias

        N = logits.shape[0]
        eye = torch.eye(N, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(eye, float("-inf"))

        w = F.softmax(logits, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        return w
