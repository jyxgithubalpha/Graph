from __future__ import annotations

import torch
from torch import nn


def _kaiming(m: nn.Module) -> None:
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_uniform_(mod.weight, mode="fan_in", nonlinearity="relu")
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0.0)


class FactorEncoder(nn.Module):
    """[N, F_factor] -> [N, d_factor] via MLP with residual block."""

    def __init__(self, f_factor: int, d_factor: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(f_factor, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
        )
        self.block = nn.Sequential(
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
        )
        self.head = nn.Linear(128, d_factor)
        _kaiming(self)

    def forward(self, x_factor: torch.Tensor) -> torch.Tensor:
        h = self.proj(x_factor)
        h = h + self.block(h)
        return self.head(h)
