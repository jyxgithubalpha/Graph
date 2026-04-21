from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from domain.types import DayBatch, Relation


class BaseGraph(nn.Module, ABC):
    """Base class for all graph types."""

    name: str = "base"

    def __init__(self, edge_weight: nn.Module, aggregator: nn.Module):
        super().__init__()
        self.edge_weight = edge_weight
        self.aggregator = aggregator

    @abstractmethod
    def forward(self, batch: DayBatch, ctx: dict) -> tuple[torch.Tensor, Relation]:
        """Return (z [N, D], Relation)."""
        pass
