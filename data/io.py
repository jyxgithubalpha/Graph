from __future__ import annotations

import logging

import polars as pl

from domain.config import SourceConfig
from domain.types import DataBundle

logger = logging.getLogger(__name__)


def load_raw_dfs(cfg: SourceConfig) -> DataBundle:
    fac_df = pl.read_ipc(cfg.fac_path, memory_map=False)
    fac_df = fac_df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"))
    fac_df = fac_df.with_columns(pl.col("Code").cast(pl.Categorical))

    for batch_name, path in cfg.extra_fac_paths.items():
        extra_df = pl.read_ipc(path, memory_map=False)
        extra_df = extra_df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"))
        extra_df = extra_df.with_columns(pl.col("Code").cast(pl.Categorical))
        new_cols = [c for c in extra_df.columns if c not in ["date", "Code"]]
        fac_df = fac_df.join(extra_df, on=["date", "Code"], how="left")
        logger.info("merged extra factor batch=%s, new_cols=%d", batch_name, len(new_cols))

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
    
    return DataBundle(fac_df=fac_df, origin_label_df=label_df, norm_label_df=label_df, liquid_df=liquid_df)


def clean_raw_dfs(data_bundle: DataBundle, min_valid_ratio: float = 0.1,) -> DataBundle:
    fac_df = data_bundle.fac_df
    origin_label_df = data_bundle.origin_label_df
    liquid_df = data_bundle.liquid_df

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
    origin_label_df = origin_label_df.filter(pl.col("label").is_not_null() & pl.col("label").is_not_nan())
    norm_label_df = origin_label_df.with_columns(
        ((pl.col("label") - pl.col("label").mean().over("date")) / (pl.col("label").std().over("date") + 1e-8)).alias("label")
    )

    # 对liquid_df进行fillna和fillnull
    liquid_df = liquid_df.with_columns(
        pl.col("liquid").fill_nan(0.0).fill_null(0.0).alias("liq")
    )

    # 对齐
    # 对齐三个数据框：取它们的交集
    common_keys = (
        fac_df.select("date", "Code")
        .join(norm_label_df.select("date", "Code"), on=["date", "Code"], how="inner")
        .join(liquid_df.select("date", "Code"), on=["date", "Code"], how="inner")
    )
    
    fac_df = fac_df.join(common_keys, on=["date", "Code"], how="inner")
    origin_label_df = origin_label_df.join(common_keys, on=["date", "Code"], how="inner")
    norm_label_df = norm_label_df.join(common_keys, on=["date", "Code"], how="inner")
    liquid_df = liquid_df.join(common_keys, on=["date", "Code"], how="inner")
    
    return DataBundle(common_keys=common_keys, fac_df=fac_df, origin_label_df=origin_label_df, norm_label_df=norm_label_df, liquid_df=liquid_df)
