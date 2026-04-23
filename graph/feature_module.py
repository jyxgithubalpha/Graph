from __future__ import annotations

import torch
from torch import nn

from domain.config import GraphConfig
from domain.types import DayBatch, GraphOut
from .encoders import build_node_encoder
from .composer import RelationalSemiringComposer
from .factor_factor_similarity_topk_neighbor import FactorFactorSimilarityTopkNeighborGraph
from .factor_return_correlation_topk_neighbor import FactorReturnCorrelationTopkNeighborGraph
from .factor_latent_attention_topk_neighbor import FactorLatentAttentionTopkNeighborGraph


class GraphFeatureModule(nn.Module):
    """Encodes batch into node embeddings via multiple graphs."""

    def __init__(self, graph_cfg: GraphConfig):
        super().__init__()
        dims = graph_cfg.dims
        self.node_encoder = build_node_encoder(dims, graph_cfg.hist_len)

        graphs = []
        if graph_cfg.factor_factor_similarity_topk_neighbor.enabled:
            graphs.append(FactorFactorSimilarityTopkNeighborGraph(
                graph_cfg.factor_factor_similarity_topk_neighbor, dims
            ))
        if graph_cfg.factor_return_correlation_topk_neighbor.enabled:
            graphs.append(FactorReturnCorrelationTopkNeighborGraph(
                graph_cfg.factor_return_correlation_topk_neighbor, dims
            ))
        if graph_cfg.factor_latent_attention_topk_neighbor.enabled:
            graphs.append(FactorLatentAttentionTopkNeighborGraph(
                graph_cfg.factor_latent_attention_topk_neighbor, dims
            ))
        self.graphs = nn.ModuleList(graphs)

        self.composer = RelationalSemiringComposer(
            d_model=dims.d_model,
            n_relations_max=max(1, len(graphs)),
            mode=graph_cfg.composer,
        )

    def forward(self, batch: DayBatch) -> GraphOut:
        h = self.node_encoder(batch)

        if not self.graphs:
            return GraphOut(embedding=h, relations=[])

        ctx = {"h": h}
        zs, rels = [], []
        for g in self.graphs:
            z, rel = g(batch, ctx)
            zs.append(z)
            rels.append(rel)

        z_final = self.composer(torch.stack(zs, dim=0))
        return GraphOut(embedding=z_final, relations=rels)
