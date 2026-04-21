from __future__ import annotations

import torch

from domain.config import ModelDimConfig, FactorLatentAttentionTopkNeighborConfig
from domain.types import DayBatch, Relation
from graph.base import BaseGraph, LearnedAttention, TopkNeighborAggregator


class FactorLatentAttentionTopkNeighborGraph(BaseGraph):
    """Graph using learned attention for edge weights."""

    name = "factor_latent_attention_topk_neighbor"

    def __init__(self, cfg: FactorLatentAttentionTopkNeighborConfig, dims: ModelDimConfig):
        edge_weight = LearnedAttention(dims.d_model, cfg.prior_scale)
        aggregator = TopkNeighborAggregator(dims, cfg.aggregator)
        super().__init__(edge_weight, aggregator)
        self.temporal_smooth = cfg.temporal_smooth

    def forward(self, batch: DayBatch, ctx: dict) -> tuple[torch.Tensor, Relation]:
        h = ctx["h"]
        w = self.edge_weight(batch, ctx)
        z = self.aggregator(h, w)
        return z, Relation(name=self.name, adj=w, edge_feat=w.unsqueeze(-1))
