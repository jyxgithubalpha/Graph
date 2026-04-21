from __future__ import annotations

import polars as pl

from domain.config import DataConfig

def load_raw_dfs(cfg: DataConfig) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    fac_df = pl.read_ipc(cfg.fac_path, memory_map=False)
    fac_df = fac_df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"))
    fac_df = fac_df.with_columns(pl.col("Code").cast(pl.Categorical))

    label_df = pl.read_ipc(cfg.label_path, memory_map=False)
    label_df = label_df.rename({"index": "date"})
    label_df = label_df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"))
    label_df = label_df.unpivot(on=[c for c in label_df.columns if c != "date"], index="date", variable_name="Code", value_name="label")
    label_df = label_df.with_columns(pl.col("Code").cast(pl.Categorical))

    liquid_df = pl.read_ipc(cfg.liquid_path, memory_map=False)
    liquid_df = liquid_df.rename({"index": "date"})
    liquid_df = liquid_df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"))
    liquid_df = liquid_df.unpivot(on=[c for c in liquid_df.columns if c != "date"], index="date", variable_name="Code", value_name="liquid")
    liquid_df = liquid_df.with_columns(pl.col("Code").cast(pl.Categorical))
    
    return fac_df, label_df, liquid_df


def clean_raw_dfs(
    fac_df: pl.DataFrame,
    label_df: pl.DataFrame,
    liquid_df: pl.DataFrame,
    min_valid_ratio: float = 0.1,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # 筛选掉全为非有效值的列
    factor_cols = [c for c in fac_df.columns if c not in ["date", "Code"]]
    valid_cols = [c for c in factor_cols if fac_df[c].is_not_null().sum() > 0 and fac_df[c].is_not_nan().sum() > 0]
    fac_df = fac_df.select(["date", "Code"] + valid_cols)
    
    # 筛选掉有效值少于阈值的行
    factor_cols = [c for c in fac_df.columns if c not in ["date", "Code"]]
    valid_count_expr = sum(pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in factor_cols)
    fac_df = fac_df.filter(valid_count_expr >= len(factor_cols) * min_valid_ratio)
    
    # 按照截面（同一日期）对因子值进行中位数填充
    factor_cols = [c for c in fac_df.columns if c not in ["date", "Code"]]
    fac_df = fac_df.with_columns(
        pl.col(factor_cols)
        .fill_nan(None)
        .fill_null(
            pl.col(factor_cols).median().over("date")
        )
    )

    # 检查哪些因子在某些日期的所有股票上都是相同的值
    sample_dates = (
        fac_df.select("date")
        .unique()
        .sample(n=min(5, fac_df["date"].n_unique()), seed=42)
        .get_column("date")
        .to_list()
    )
    sample_df = fac_df.filter(pl.col("date").is_in(sample_dates))
    nuniq_df = sample_df.group_by("date").agg(
        pl.col(factor_cols).fill_nan(None).drop_nulls().n_unique()
    )
    constant_cols = [
        c for c in factor_cols
        if nuniq_df.get_column(c).max() <= 1
    ]
    varying_cols = [c for c in factor_cols if c not in constant_cols]

    # 对constant_cols进行时序zscore标准化
    if constant_cols:
        fac_df = fac_df.with_columns(
            (
                (pl.col(constant_cols) - pl.col(constant_cols).mean().over("Code")) /
                (pl.col(constant_cols).std().over("Code") + 1e-8)
            )
        )

    # 对varying_cols进行截面zscore标准化
    if varying_cols:
        fac_df = fac_df.with_columns([
            ((pl.col(c) - pl.col(c).mean().over("date")) / (pl.col(c).std().over("date") + 1e-8)).alias(c)
            for c in varying_cols
        ])
    
    # 对label_df首先dropna和dropnull,然后进行截面zscore
    label_df = label_df.filter(pl.col("label").is_not_null() & pl.col("label").is_not_nan())
    label_df = label_df.with_columns(
        ((pl.col("label") - pl.col("label").mean().over("date")) / (pl.col("label").std().over("date") + 1e-8)).alias("label")
    )

    # 对liquid_df进行fillna和fillnull
    liquid_df = liquid_df.with_columns(
        pl.col("liquid").fill_nan(0.0).fill_null(0.0).alias("liq")
    )

    return fac_df, label_df, liquid_df
