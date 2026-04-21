from dataclasses import dataclass, field
from datetime import date
from typing import List

import torch


@dataclass
class DayBatch:
    date: date
    codes: List[str]
    x_alpha: torch.Tensor
    x_style: torch.Tensor
    x_meta: torch.Tensor
    ret_hist: torch.Tensor
    industry: torch.Tensor
    label: torch.Tensor
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
class ForwardOut:
    score: torch.Tensor
    relations: List[Relation] = field(default_factory=list)
    reg_loss: torch.Tensor = field(default_factory=lambda: torch.zeros(()))


@dataclass
class DataBundle:
    common_keys: pl.DataFrame

    fac_df: pl.DataFrame
    origin_label_df: pl.DataFrame
    norm_label_df: pl.DataFrame
    liquid_df: pl.DataFrame