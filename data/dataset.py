from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from domain.types import DataBundle, DayBatch


class GraphDataset(Dataset):
    """Dataset that yields one DayBatch per trading day."""

    def __init__(
        self,
        bundle: DataBundle,
        date_df: pl.DataFrame,
        factor_cols: list[str],
        ret_hist_cache: dict[date, tuple[list[str], np.ndarray]],
        hist_len: int,
    ):
        self.dates = date_df["date"].unique().sort().to_list()
        self.factor_cols = factor_cols
        self.hist_len = hist_len
        self.ret_hist_cache = ret_hist_cache

        self.fac_df = bundle.fac_df
        self.norm_label_df = bundle.norm_label_df
        self.liquid_df = bundle.liquid_df

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> DayBatch:
        d = self.dates[idx]
        day_fac = self.fac_df.filter(pl.col("date") == d).sort("Code")
        codes = day_fac["Code"].to_list()

        x_factor = day_fac.select(self.factor_cols).to_numpy().astype(np.float32)

        day_label = self.norm_label_df.filter(pl.col("date") == d).select(["Code", "label"]).rename({"label": "_label"})
        day_liq = self.liquid_df.filter(pl.col("date") == d).select(["Code", "liq"]).rename({"liq": "_liq"})
        joined = pl.DataFrame({"Code": codes}).join(day_label, on="Code", how="left").join(day_liq, on="Code", how="left")
        label_np = joined["_label"].fill_null(0.0).to_numpy().astype(np.float32)
        liquid_np = joined["_liq"].fill_null(0.0).to_numpy().astype(np.float32)

        hist_codes, hist_mat = self.ret_hist_cache.get(d, ([], np.zeros((0, self.hist_len), dtype=np.float32)))
        code_to_row = {c: i for i, c in enumerate(hist_codes)}
        ret_hist = np.zeros((len(codes), self.hist_len), dtype=np.float32)
        for i, c in enumerate(codes):
            r = code_to_row.get(c)
            if r is not None:
                ret_hist[i] = hist_mat[r]

        x_meta = np.stack([
            np.log1p(np.clip(liquid_np, 0.0, None)),
            ret_hist[:, -5:].sum(axis=1),
        ], axis=1).astype(np.float32)

        return DayBatch(
            date=d,
            codes=codes,
            x_factor=torch.from_numpy(np.nan_to_num(x_factor)),
            x_meta=torch.from_numpy(np.nan_to_num(x_meta)),
            ret_hist=torch.from_numpy(np.nan_to_num(ret_hist)),
            label=torch.from_numpy(np.nan_to_num(label_np)),
            liquid=torch.from_numpy(np.nan_to_num(liquid_np)),
        )


def make_dataloader(dataset: GraphDataset, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=lambda b: b, drop_last=False)
