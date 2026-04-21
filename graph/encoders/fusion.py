from __future__ import annotations

import torch
from torch import nn

from domain.config import ModelDimConfig
from domain.types import DayBatch
from .factor import FactorEncoder
from .temporal import TCNTemporalEncoder, GRUTemporalEncoder


def _kaiming(m: nn.Module) -> None:
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_uniform_(mod.weight, mode="fan_in", nonlinearity="relu")
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0.0)


class NodeFeatureFusion(nn.Module):
    """Concat [h_factor, h_tmp, x_meta] -> d_model via MLP + LayerNorm."""

    def __init__(self, d_factor: int, d_tmp: int, f_meta: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        d_in = d_factor + d_tmp + f_meta
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_model * 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        _kaiming(self)

    def forward(self, h_factor: torch.Tensor, h_tmp: torch.Tensor, x_meta: torch.Tensor) -> torch.Tensor:
        h = torch.cat([h_factor, h_tmp, x_meta], dim=-1)
        return self.norm(self.mlp(h))


class NodeEncoder(nn.Module):
    """Encode batch features into node embeddings h [N, d_model]."""

    def __init__(self, dims: ModelDimConfig):
        super().__init__()
        self.factor_enc = FactorEncoder(dims.f_factor, dims.d_factor, dims.dropout)
        if dims.temporal_encoder == "tcn":
            self.tmp_enc = TCNTemporalEncoder(20, dims.d_tmp, dims.dropout)
        else:
            self.tmp_enc = GRUTemporalEncoder(20, dims.d_tmp, dims.dropout)
        self.fusion = NodeFeatureFusion(dims.d_factor, dims.d_tmp, dims.f_meta, dims.d_model, dims.dropout)

    def forward(self, batch: DayBatch) -> torch.Tensor:
        h_factor = self.factor_enc(batch.x_factor)
        h_tmp = self.tmp_enc(batch.ret_hist)
        return self.fusion(h_factor, h_tmp, batch.x_meta)


def build_node_encoder(dims: ModelDimConfig) -> NodeEncoder:
    return NodeEncoder(dims)
