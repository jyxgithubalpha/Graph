from __future__ import annotations

import torch

from domain.config import ModelDimConfig, FactorFactorSimilarityTopkNeighborConfig
from domain.types import DayBatch, Relation
from graph.base import BaseGraph, CosineSimilarity, TopkNeighborAggregator


class FactorFactorSimilarityTopkNeighborGraph(BaseGraph):
    """Graph using factor cosine similarity for edge weights."""

    name = "factor_factor_similarity_topk_neighbor"

    def __init__(self, cfg: FactorFactorSimilarityTopkNeighborConfig, dims: ModelDimConfig):
        edge_weight = CosineSimilarity()
        aggregator = TopkNeighborAggregator(dims, cfg.aggregator)
        super().__init__(edge_weight, aggregator)

    def forward(self, batch: DayBatch, ctx: dict) -> tuple[torch.Tensor, Relation]:
        h = ctx["h"]
        w = self.edge_weight(batch, ctx)
        z = self.aggregator(h, w)
        return z, Relation(name=self.name, adj=w, edge_feat=w.unsqueeze(-1))
