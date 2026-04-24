from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SourceConfig:
    fac_path: str = "/project/model_share/share_1/factor_data/fac20250212/fac20250212.fea"
    label_path: str = "/project/model_share/share_1/label_data/label1.fea"
    liquid_path: str = "/project/model_share/share_1/label_data/can_trade_amt1.fea"
    extra_fac_paths: dict[str, str] = field(default_factory=dict)
    industry_path: Optional[str] = None


@dataclass
class FeatureConfig:
    factor_cols: Optional[list[str]] = None
    factor_prefix: Optional[str] = None
    hist_len: int = 10


@dataclass
class ModelDimConfig:
    f_factor: int = 0
    f_meta: int = 2
    d_factor: int = 64
    d_tmp: int = 32
    d_model: int = 24
    d_edge: int = 4
    n_heads: int = 2
    dropout: float = 0.2
    temporal_encoder: Literal["tcn", "gru"] = "tcn"


@dataclass
class TopkNeighborConfig:
    topk: int = 20
    n_layers: int = 2
    self_weight_learnable: bool = True
    self_weight: float = 0.5


@dataclass
class FactorFactorSimilarityTopkNeighborConfig:
    enabled: bool = True
    aggregator: TopkNeighborConfig = field(default_factory=TopkNeighborConfig)


@dataclass
class FactorReturnCorrelationTopkNeighborConfig:
    enabled: bool = True
    aggregator: TopkNeighborConfig = field(default_factory=TopkNeighborConfig)


@dataclass
class FactorLatentAttentionTopkNeighborConfig:
    enabled: bool = True
    prior_scale: float = 1.0
    temporal_smooth: bool = True
    aggregator: TopkNeighborConfig = field(default_factory=TopkNeighborConfig)


@dataclass
class GraphConfig:
    dims: ModelDimConfig = field(default_factory=ModelDimConfig)
    hist_len: int = 10
    composer: Literal["semiring", "sum", "max", "agr", "attn"] = "semiring"
    factor_factor_similarity_topk_neighbor: FactorFactorSimilarityTopkNeighborConfig = field(
        default_factory=FactorFactorSimilarityTopkNeighborConfig
    )
    factor_return_correlation_topk_neighbor: FactorReturnCorrelationTopkNeighborConfig = field(
        default_factory=FactorReturnCorrelationTopkNeighborConfig
    )
    factor_latent_attention_topk_neighbor: FactorLatentAttentionTopkNeighborConfig = field(
        default_factory=FactorLatentAttentionTopkNeighborConfig
    )


@dataclass
class ModelConfig:
    hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.2


@dataclass
class TrainConfig:
    max_epochs: int = 2
    batch_size: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    early_stop_patience: int = 8
    w_rank: float = 1.0
    w_ic: float = 0.5
    w_reg: float = 1e-3


@dataclass
class EvalConfig:
    top_k: int = 500
    money: float = 1.5e9


@dataclass
class RunConfig:
    seasons: list[str] = field(default_factory=lambda: [
        "2023q1", "2023q2", "2023q3", "2023q4",
        "2024q1", "2024q2", "2024q3", "2024q4",
        "2025q1", "2025q2"
    ])
    valid_period: int = 1
    gap_days: int = 10
    seed: int = 42
    results_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "results"
    ))


@dataclass
class ExperimentConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)

