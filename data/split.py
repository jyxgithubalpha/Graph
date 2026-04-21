from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import polars as pl


def get_date_split(season: str, period: int) -> tuple[date, date, date, date, date]:
    year = int(season[:4])
    quarter = int(season.split("q")[1])
    test_month = quarter * 3 - 2
    test_start = date(year, test_month, 1)

    valid_date_split = [
        test_start - relativedelta(months=i)
        for i in [-3, 0, 6, 12, 18, 24]
    ]
    valid_date_split.sort()

    train_start = valid_date_split[0]
    valid_start = valid_date_split[period - 1]
    valid_end = valid_date_split[period]
    test_end = valid_date_split[-1]

    return train_start, valid_start, valid_end, test_start, test_end


def get_date_lists(
    season: str,
    period: int,
    date_df: pl.DataFrame,
    start_date: str = "20210101",
    gap_days: int = 10,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_start, valid_start, valid_end, test_start, test_end = get_date_split(season, period)
    start_date_obj = datetime.strptime(start_date, "%Y%m%d").date()

    if train_start < start_date_obj:
        train_start = start_date_obj

    # 验证集：valid_start <= date < valid_end，再去掉最后 gap_days 天
    valid_date_df = date_df.filter(
        (pl.col("date") >= pl.lit(valid_start)) &
        (pl.col("date") < pl.lit(valid_end))
    )
    if valid_date_df.height > gap_days:
        valid_date_df = valid_date_df.slice(0, valid_date_df.height - gap_days)
    else:
        valid_date_df = valid_date_df.clear()

    # 训练集分两段
    train_date_df_1 = date_df.filter(
        (pl.col("date") >= pl.lit(train_start)) &
        (pl.col("date") < pl.lit(valid_start))
    )
    if train_date_df_1.height > gap_days:
        train_date_df_1 = train_date_df_1.slice(0, train_date_df_1.height - gap_days)
    else:
        train_date_df_1 = train_date_df_1.clear()

    train_date_df_2 = date_df.filter(
        (pl.col("date") >= pl.lit(valid_end)) &
        (pl.col("date") < pl.lit(test_start))
    )
    if train_date_df_2.height > gap_days:
        train_date_df_2 = train_date_df_2.slice(0, train_date_df_2.height - gap_days)
    else:
        train_date_df_2 = train_date_df_2.clear()

    train_date_df = pl.concat([train_date_df_1, train_date_df_2])

    # 测试集
    test_date_df = date_df.filter(
        (pl.col("date") >= pl.lit(test_start)) &
        (pl.col("date") < pl.lit(test_end))
    )

    # 极端行情不参与训练
    extreme_start = date(2024, 2, 1)
    extreme_end = date(2024, 2, 23)
    train_date_df = train_date_df.filter(
        (pl.col("date") < pl.lit(extreme_start)) |
        (pl.col("date") > pl.lit(extreme_end))
    )

    return train_date_df, valid_date_df, test_date_df
