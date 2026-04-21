from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadEdgeAwareMessagePassing(nn.Module):
    """Multi-head attention with edge features for message passing."""

    def __init__(self, d_model: int, d_edge: int = 8, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_edge = d_edge

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_e = nn.Linear(1, d_edge * n_heads)
        self.u = nn.Parameter(torch.randn(n_heads, 2 * self.d_head + d_edge) * 0.02)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        N = h.shape[0]
        H, Dh = self.n_heads, self.d_head

        edge_feat = w.unsqueeze(-1)
        e = self.W_e(edge_feat).view(N, N, H, self.d_edge)

        q = self.W_q(h).view(N, H, Dh)
        k = self.W_k(h).view(N, H, Dh)
        v = self.W_v(h).view(N, H, Dh)

        qi = q.unsqueeze(1).expand(N, N, H, Dh)
        kj = k.unsqueeze(0).expand(N, N, H, Dh)
        feats = torch.cat([qi, kj, e], dim=-1)
        logits = torch.einsum('ijhf,hf->ijh', feats, self.u)

        mask = (w > 0).unsqueeze(-1)
        logits = logits.masked_fill(~mask, float('-inf'))
        no_nbr = (~mask.any(dim=1, keepdim=True)).expand_as(logits)
        logits = torch.where(no_nbr, torch.zeros_like(logits), logits)

        alpha = F.softmax(logits, dim=1)
        alpha = alpha * w.unsqueeze(-1)
        alpha = self.drop(alpha)

        vj = v.unsqueeze(0).expand(N, N, H, Dh)
        msg = (alpha.unsqueeze(-1) * vj).sum(dim=1).reshape(N, self.d_model)

        out = self.out_proj(msg)
        return self.norm(h + out)
