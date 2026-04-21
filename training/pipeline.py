from __future__ import annotations

import os
from dataclasses import replace
from datetime import date

import polars as pl
import pytorch_lightning as pl_lit

from data import (
    QuarterDataset,
    clean_raw_dfs,
    load_raw_dfs,
    make_dataloader,
)
from domain.config import ExperimentConfig, ModelConfig
from .lightning_module import GraphRankLit


def split_style_alpha_cols(
    fac_df: pl.DataFrame,
    label_df: pl.DataFrame,
    train_dates: list[date],
    top_k: int = 100,
) -> tuple[list[str], list[str]]:
    factor_cols = [c for c in fac_df.columns if c not in ["date", "Code"]]
    train_fac = fac_df.filter(pl.col("date").is_in(train_dates))
    train_label = label_df.filter(pl.col("date").is_in(train_dates))
    merged = train_fac.join(train_label, on=["date", "Code"], how="inner")

    corrs: dict[str, float] = {}
    for col in factor_cols:
        try:
            corr = merged.select(pl.corr(col, "label")).item()
            corrs[col] = abs(corr) if corr is not None else 0.0
        except Exception:
            corrs[col] = 0.0

    sorted_cols = sorted(factor_cols, key=lambda c: corrs.get(c, 0.0), reverse=True)
    alpha_cols = sorted_cols[:top_k]
    style_cols = sorted_cols[top_k:]
    return style_cols, alpha_cols


def get_train_date_split(
    label_df: pl.DataFrame,
    season: str,
    period: int,
    gap_days: int = 10,
    min_train_start: date = date(2021, 1, 1),
) -> tuple[list[date], list[date], list[date]]:
    year = int(season[:4])
    quarter = int(season.split("q")[1])
    test_start_month = quarter * 3 - 2
    test_start = date(year, test_start_month, 1)

    valid_date_split = []
    for i in [-3, 0, 6, 12, 18, 24]:
        split_date = test_start - relativedelta(months=i)
        valid_date_split.append(split_date)
    valid_date_split.reverse()

    train_start = valid_date_split[0]
    valid_start = valid_date_split[period - 1]
    valid_end = valid_date_split[period]
    test_end = valid_date_split[-1]

    if train_start < min_train_start:
        train_start = min_train_start

    date_list = label_df.select("date").unique().sort("date").to_series().to_list()

    valid_dates = [x for x in date_list if valid_start <= x < valid_end][:-gap_days]
    train_dates = (
        [x for x in date_list if train_start <= x < valid_start][:-gap_days]
        + [x for x in date_list if valid_end <= x < test_start][gap_days:-gap_days]
    )
    test_dates = [x for x in date_list if test_start <= x < test_end]

    # 极端行情不参与训练
    not_train_start = date(2024, 2, 1)
    not_train_end = date(2024, 2, 23)
    not_train_dates = [x for x in date_list if not_train_start <= x <= not_train_end]
    train_dates = [x for x in train_dates if x not in not_train_dates]

    return train_dates, valid_dates, test_dates


def train_from_config(
    exp_cfg: ExperimentConfig,
) -> dict[str, pl.DataFrame]:
    pl_lit.seed_everything(exp_cfg.run.seed, workers=True)
    
    outputs: dict[str, pl.DataFrame] = {}

    fac_df, label_df, liquid_df = load_raw_dfs(exp_cfg.data)
    fac_df, label_df, liquid_df = clean_raw_dfs(fac_df, label_df, liquid_df)

    for season in exp_cfg.run.seasons:

        train_dates, valid_dates, test_dates = get_train_date_split(
            label_df=label_df,
            season=season,
            period=exp_cfg.run.valid_period,
            gap_days=exp_cfg.split.gap_days,
            min_train_start=date(2017, 1, 1),
        )

        train_ds = QuarterDataset(fac_df, label_df, liquid_df, train_dates, style_cols, alpha_cols, model_cfg.hist_len, "train")
        valid_ds = QuarterDataset(fac_df, label_df, liquid_df, valid_dates, style_cols, alpha_cols, model_cfg.hist_len, "valid")
        test_ds = QuarterDataset(fac_df, label_df, liquid_df, test_dates, style_cols, alpha_cols, model_cfg.hist_len, "test")

        train_dl = make_dataloader(train_ds, batch_size=exp_cfg.train.batch_size, shuffle=True)
        valid_dl = make_dataloader(valid_ds, batch_size=1, shuffle=False)
        test_dl = make_dataloader(test_ds, batch_size=1, shuffle=False)

        lit = GraphRankLit(model_cfg=model_cfg, train_cfg=exp_cfg.train)
        trainer = pl_lit.Trainer(
            max_epochs=exp_cfg.train.max_epochs,
            accelerator="auto",
            devices=1,
            strategy="auto",
            log_every_n_steps=1,
            enable_checkpointing=True,
            num_sanity_val_steps=0,
        )
        trainer.fit(lit, train_dl, valid_dl)
        trainer.test(lit, test_dl, ckpt_path=trainer.checkpoint_callback.best_model_path or None)

        if exp_cfg.run.compat_output_root is not None and exp_cfg.run.valid_period_idx is not None:
            export_train_code_compatible_scores(lit.test_df, exp_cfg.run.compat_output_root, exp_cfg.run.market, exp_cfg.run.valid_period_idx)

    return outputs
