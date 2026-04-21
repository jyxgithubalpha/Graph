from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


def build_ret_hist_cache(norm_label_df: pl.DataFrame, hist_len: int) -> dict[date, tuple[list[str], np.ndarray]]:
    """Build a cache mapping each date to (codes, ret_hist_matrix[K, hist_len])."""
    ret_hist_cache: dict[date, tuple[list[str], np.ndarray]] = {}
    all_dates = sorted(norm_label_df["date"].unique().to_list())
    
    for d in all_dates:
        past_dates = [pd for pd in all_dates if pd < d][-hist_len:]
        if len(past_dates) == 0:
            ret_hist_cache[d] = ([], np.zeros((0, hist_len), dtype=np.float32))
            continue
        
        hist_df = norm_label_df.filter(pl.col("date").is_in(past_dates))
        codes = sorted(hist_df["Code"].unique().to_list())
        
        pivot = hist_df.pivot(index="Code", on="date", values="label").sort("Code")
        date_cols = sorted([c for c in pivot.columns if c != "Code"])
        
        hist_mat = pivot.select(date_cols).to_numpy().astype(np.float32)
        hist_mat = np.nan_to_num(hist_mat, nan=0.0)
        
        if hist_mat.shape[1] < hist_len:
            pad_width = hist_len - hist_mat.shape[1]
            hist_mat = np.pad(hist_mat, ((0, 0), (pad_width, 0)), mode='constant', constant_values=0.0)
        
        ret_hist_cache[d] = (pivot["Code"].to_list(), hist_mat)
    
    return ret_hist_cache
