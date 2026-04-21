from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl


def write_feather(df: pl.DataFrame, path: str) -> None:
    """Write DataFrame to feather file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_ipc(path)


def dump_season_outputs(out_dir: str, test_records: list, d_model: int) -> None:
    """Dump embeddings and scores to feather files."""
    os.makedirs(out_dir, exist_ok=True)

    rows_date, rows_code, rows_score = [], [], []
    emb_rows = []

    for date, codes, embeddings, scores in test_records:
        n = len(codes)
        rows_date.extend([date] * n)
        rows_code.extend(codes)
        rows_score.extend(scores.tolist())
        emb_rows.append(embeddings)

    score_df = pl.DataFrame({
        "date": rows_date,
        "Code": rows_code,
        "score": rows_score,
    }).sort(["date", "Code"])
    write_feather(score_df, os.path.join(out_dir, "score.feather"))

    if emb_rows:
        all_emb = np.vstack(emb_rows)
        emb_data = {"date": rows_date, "Code": rows_code}
        for i in range(d_model):
            emb_data[f"e_{i}"] = all_emb[:, i].tolist()
        emb_df = pl.DataFrame(emb_data).sort(["date", "Code"])
        write_feather(emb_df, os.path.join(out_dir, "embeddings.feather"))
