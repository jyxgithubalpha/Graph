from __future__ import annotations

import torch
from torch import nn

from domain.config import ModelConfig


class RankMLP(nn.Module):
    """MLP head that maps embeddings to ranking scores."""

    def __init__(self, d_in: int, cfg: ModelConfig):
        super().__init__()
        layers = []
        prev_dim = d_in
        for h in cfg.hidden:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LeakyReLU(0.01),
                nn.Dropout(cfg.dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
