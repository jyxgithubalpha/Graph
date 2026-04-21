from __future__ import annotations

import torch

from domain.config import ModelDimConfig, FactorReturnCorrelationTopkNeighborConfig
from domain.types import DayBatch, Relation
from graph.base import BaseGraph, PearsonCorrelation, TopkNeighborAggregator


class FactorReturnCorrelationTopkNeighborGraph(BaseGraph):
    """Graph using return history correlation for edge weights."""

    name = "factor_return_correlation_topk_neighbor"

    def __init__(self, cfg: FactorReturnCorrelationTopkNeighborConfig, dims: ModelDimConfig):
        edge_weight = PearsonCorrelation()
        aggregator = TopkNeighborAggregator(dims, cfg.aggregator)
        super().__init__(edge_weight, aggregator)

    def forward(self, batch: DayBatch, ctx: dict) -> tuple[torch.Tensor, Relation]:
        h = ctx["h"]
        w = self.edge_weight(batch, ctx)
        z = self.aggregator(h, w)
        return z, Relation(name=self.name, adj=w, edge_feat=w.unsqueeze(-1))
