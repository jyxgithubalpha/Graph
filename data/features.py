from __future__ import annotations

import polars as pl

from domain.config import FeatureConfig


def resolve_factor_cols(fac_df: pl.DataFrame, cfg: FeatureConfig) -> list[str]:
    """Resolve factor column names from fac_df based on config."""
    all_cols = [c for c in fac_df.columns if c not in ["date", "Code"]]
    
    if cfg.factor_cols is not None:
        return [c for c in cfg.factor_cols if c in all_cols]
    
    if cfg.factor_prefix is not None:
        return [c for c in all_cols if c.startswith(cfg.factor_prefix)]
    
    return all_cols
