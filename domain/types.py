from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import polars as pl
import torch


@dataclass
class DayBatch:
    date: date
    codes: list[str]
    x_factor: torch.Tensor
    x_meta: torch.Tensor
    ret_hist: torch.Tensor
    norm_label: torch.Tensor
    origin_label: torch.Tensor
    liquid: torch.Tensor

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


@dataclass
class Relation:
    name: str
    adj: torch.Tensor
    edge_feat: torch.Tensor


@dataclass
class GraphOut:
    embedding: torch.Tensor
    relations: list[Relation] = field(default_factory=list)


@dataclass
class DataBundle:
    fac_df: pl.DataFrame
    origin_label_df: pl.DataFrame
    norm_label_df: pl.DataFrame
    liquid_df: pl.DataFrame
    common_keys: Optional[pl.DataFrame] = None