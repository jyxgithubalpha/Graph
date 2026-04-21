from __future__ import annotations

import logging
import os

import polars as pl
import pytorch_lightning as pl_lit

from data import (
    GraphDataset,
    clean_raw_dfs,
    load_raw_dfs,
    get_date_lists,
    resolve_factor_cols,
    build_ret_hist_cache,
    make_dataloader,
)
from domain.config import ExperimentConfig
from .lightning_module import GraphRankLit
from .export import dump_season_outputs

logger = logging.getLogger(__name__)


def run(exp_cfg: ExperimentConfig) -> None:
    """Run the full training pipeline for all seasons."""
    pl_lit.seed_everything(exp_cfg.run.seed, workers=True)

    bundle = load_raw_dfs(exp_cfg.source)
    bundle = clean_raw_dfs(bundle)

    factor_cols = resolve_factor_cols(bundle.fac_df, exp_cfg.feature)
    exp_cfg.graph.dims.f_factor = len(factor_cols)
    logger.info("Resolved %d factor columns", len(factor_cols))

    ret_hist_cache = build_ret_hist_cache(bundle.norm_label_df, exp_cfg.feature.hist_len)
    logger.info("Built ret_hist cache for %d dates", len(ret_hist_cache))

    for season in exp_cfg.run.seasons:
        logger.info("=== Season %s ===", season)

        train_df, valid_df, test_df = get_date_lists(
            season, exp_cfg.run.valid_period, bundle.common_keys.select("date")
        )
        logger.info("Train=%d, Valid=%d, Test=%d dates", train_df.height, valid_df.height, test_df.height)

        train_ds = GraphDataset(bundle, train_df, factor_cols, ret_hist_cache, exp_cfg.feature.hist_len)
        valid_ds = GraphDataset(bundle, valid_df, factor_cols, ret_hist_cache, exp_cfg.feature.hist_len)
        test_ds = GraphDataset(bundle, test_df, factor_cols, ret_hist_cache, exp_cfg.feature.hist_len)

        train_dl = make_dataloader(train_ds, batch_size=exp_cfg.train.batch_size, shuffle=True)
        valid_dl = make_dataloader(valid_ds, batch_size=1, shuffle=False)
        test_dl = make_dataloader(test_ds, batch_size=1, shuffle=False)

        lit = GraphRankLit(
            graph_cfg=exp_cfg.graph,
            model_cfg=exp_cfg.model,
            train_cfg=exp_cfg.train,
            eval_cfg=exp_cfg.eval,
        )

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

        ckpt_path = None
        if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.test(lit, test_dl, ckpt_path=ckpt_path)

        out_dir = os.path.join(exp_cfg.run.results_dir, season)
        dump_season_outputs(out_dir, lit.test_records, exp_cfg.graph.dims.d_model)
        logger.info("Dumped outputs to %s", out_dir)
