from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from domain.config import ModelDimConfig, TopkNeighborConfig
from .propagation import MultiHeadEdgeAwareMessagePassing


class TopkNeighborAggregator(nn.Module):
    """Aggregate node features using top-k neighbors with self-weight blending."""

    def __init__(self, dims: ModelDimConfig, cfg: TopkNeighborConfig):
        super().__init__()
        self.topk = cfg.topk
        self.n_layers = cfg.n_layers
        self.self_weight = cfg.self_weight
        self.self_weight_learnable = cfg.self_weight_learnable

        self.layers = nn.ModuleList([
            MultiHeadEdgeAwareMessagePassing(
                d_model=dims.d_model,
                d_edge=dims.d_edge,
                n_heads=dims.n_heads,
                dropout=dims.dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        if cfg.self_weight_learnable:
            self.gate = nn.Linear(dims.d_model, 1)
        else:
            self.gate = None

    def _topk_mask(self, w: torch.Tensor) -> torch.Tensor:
        N = w.shape[0]
        k = min(self.topk, N - 1)
        if k <= 0:
            return torch.zeros_like(w)
        _, idx = w.topk(k, dim=-1)
        mask = torch.zeros_like(w)
        mask.scatter_(-1, idx, 1.0)
        return w * mask

    def forward(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w_sparse = self._topk_mask(w)
        row_sum = w_sparse.sum(dim=-1, keepdim=True) + 1e-8
        w_norm = w_sparse / row_sum

        for layer in self.layers:
            msg = layer(h, w_norm)
            if self.gate is not None:
                alpha = torch.sigmoid(self.gate(h))
            else:
                alpha = self.self_weight
            h = alpha * h + (1 - alpha) * msg

        return h
