from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from domain.types import DayBatch


def industry_id_from_code(code: str) -> int:
    """根据股票代码推断行业ID（简化版本：使用代码前缀映射）"""
    if not code:
        return 0
    # 根据A股代码规则简单分类
    prefix = code[:3] if len(code) >= 3 else code
    # 简化映射：使用代码前三位的哈希值作为行业ID
    return hash(prefix) % 30  # 假设30个行业


def precompute_ret_hist(label_long: pl.DataFrame, hist_len: int) -> dict:
    """预计算每个日期的历史收益率矩阵"""
    ret_hist_cache = {}
    all_dates = sorted(label_long["date"].unique().to_list())
    
    for d in all_dates:
        # 获取当前日期之前hist_len个交易日的数据
        past_dates = [pd for pd in all_dates if pd < d][-hist_len:]
        if len(past_dates) == 0:
            ret_hist_cache[d] = ([], np.zeros((0, hist_len), dtype=np.float32))
            continue
        
        hist_df = label_long.filter(pl.col("date").is_in(past_dates))
        codes = sorted(hist_df["Code"].unique().to_list())
        
        # 构建收益率矩阵 (n_codes, hist_len)
        pivot = hist_df.pivot(index="Code", columns="date", values="label").sort("Code")
        date_cols = [c for c in pivot.columns if c != "Code"]
        # 按日期排序
        date_cols_sorted = sorted(date_cols)
        
        hist_mat = pivot.select(date_cols_sorted).to_numpy().astype(np.float32)
        # 填充NaN为0
        hist_mat = np.nan_to_num(hist_mat, nan=0.0)
        
        # 如果历史长度不足，左侧填充0
        if hist_mat.shape[1] < hist_len:
            pad_width = hist_len - hist_mat.shape[1]
            hist_mat = np.pad(hist_mat, ((0, 0), (pad_width, 0)), mode='constant', constant_values=0.0)
        
        ret_hist_cache[d] = (pivot["Code"].to_list(), hist_mat)
    
    return ret_hist_cache


class QuarterDataset(Dataset):
    def __init__(
        self,
        fac_long: pl.DataFrame,
        label_long: pl.DataFrame,
        liquid_long: pl.DataFrame,
        dates: list,
        style_cols: list[str],
        alpha_cols: list[str],
        hist_len: int = 20,
        stage: str = "train",
    ):
        self.dates = list(dates)
        self.style_cols = list(style_cols)
        self.alpha_cols = list(alpha_cols)
        self.hist_len = hist_len
        self.stage = stage

        d0, d1 = min(self.dates), max(self.dates)
        hist_start = d0 - timedelta(days=hist_len * 4)

        self.fac_long = fac_long.filter((pl.col("date") >= d0) & (pl.col("date") <= d1)).with_columns(pl.col("Code").cast(pl.String))
        self.label_long = label_long.filter((pl.col("date") >= hist_start) & (pl.col("date") <= d1)).with_columns(pl.col("Code").cast(pl.String))
        self.liquid_long = liquid_long.filter((pl.col("date") >= d0) & (pl.col("date") <= d1)).with_columns(pl.col("Code").cast(pl.String))
        self.ret_hist_cache = precompute_ret_hist(self.label_long, hist_len)

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> DayBatch:
        d = self.dates[idx]
        all_cols = self.style_cols + self.alpha_cols
        day = self.fac_long.filter(pl.col("date") == d).sort("Code")

        codes = day["Code"].to_list()
        x_style = day.select(self.style_cols).to_numpy().astype(np.float32)
        x_alpha = day.select(self.alpha_cols).to_numpy().astype(np.float32)

        lab = self.label_long.filter(pl.col("date") == d).select(["Code", "label"]).rename({"label": "_label"})
        liq = self.liquid_long.filter(pl.col("date") == d).select(["Code", "liq"]).rename({"liq": "_liq"})
        joined = pl.DataFrame({"Code": codes}).join(lab, on="Code", how="left").join(liq, on="Code", how="left")
        label_np = joined["_label"].fill_null(0.0).to_numpy().astype(np.float32)
        liquid_np = joined["_liq"].fill_null(0.0).to_numpy().astype(np.float32)

        hist_codes, hist_mat = self.ret_hist_cache[d]
        code_to_row = {c: i for i, c in enumerate(hist_codes)}
        ret_hist = np.zeros((len(codes), self.hist_len), dtype=np.float32)
        for i, c in enumerate(codes):
            r = code_to_row.get(c)
            if r is not None:
                ret_hist[i] = hist_mat[r]

        x_meta = np.stack([np.log1p(np.clip(liquid_np, 0.0, None)), ret_hist[:, -5:].sum(axis=1)], axis=1).astype(np.float32)
        industry = np.array([industry_id_from_code(c) for c in codes], dtype=np.int64)

        return DayBatch(
            date=d,
            codes=codes,
            x_alpha=torch.from_numpy(np.nan_to_num(x_alpha)),
            x_style=torch.from_numpy(np.nan_to_num(x_style)),
            x_meta=torch.from_numpy(np.nan_to_num(x_meta)),
            ret_hist=torch.from_numpy(np.nan_to_num(ret_hist)),
            industry=torch.from_numpy(industry),
            label=torch.from_numpy(np.nan_to_num(label_np)),
            liquid=torch.from_numpy(np.nan_to_num(liquid_np)),
        )


def make_dataloader(dataset: QuarterDataset, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=lambda b: b, drop_last=False)
