"""Daily forecasting features available at the end of the preceding day.

Calendar features refer to the forecast day. Sales, availability, promotions,
and weather are historical observations only. Recovery targets must be computed
with a recovery model fitted before the forecast origins under consideration.
"""

from __future__ import annotations

import numpy as np
import polars as pl


KEYS = ["store_id", "product_id"]
ID_FEATURES = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]
WEATHER_COLUMNS = [
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]


def build_features(
    frame: pl.DataFrame,
    target: np.ndarray,
    warmup_days: int = 28,
) -> tuple[pl.DataFrame, list[str]]:
    """Return the sorted panel, metadata, and an explicit feature allowlist.

    ``target`` follows the *input* row order and is attached before sorting.
    No rows are dropped. ``__warmup_complete`` marks dates at least
    ``warmup_days`` after the first panel date; callers decide which rows to fit.

    Lags use a complete daily calendar within each store-product pair. Missing
    calendar days and duplicate keys raise rather than silently changing a
    one-day lag into a lag of a different duration. Missing historical values
    remain null and are accepted by LightGBM. Rolling statistics use all
    available preceding observations up to the requested window size, with
    population standard deviation (ddof=0).
    """
    if warmup_days < 0:
        raise ValueError("warmup_days must be nonnegative")
    required = set(ID_FEATURES + WEATHER_COLUMNS + [
        "dt", "hours_sale", "hours_stock_status", "discount",
    ])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing feature input columns: {missing}")

    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 1 or len(values) != frame.height:
        raise ValueError("target must be one-dimensional and match frame height")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("target must contain finite nonnegative values")
    for column in ["hours_sale", "hours_stock_status"]:
        invalid = frame.select(
            (pl.col(column).is_null() | (pl.col(column).list.len() != 24)).any()
        ).item()
        if invalid:
            raise ValueError(f"{column} must contain a 24-hour list per row")

    date_expression = (
        pl.col("dt").str.to_date()
        if frame.schema["dt"] == pl.String
        else pl.col("dt").cast(pl.Date)
    )
    data = frame.with_columns(
        pl.Series("target", values, dtype=pl.Float64),
        date_expression.alias("__date"),
        pl.col("hours_sale").list.slice(6, 16).list.sum()
        .cast(pl.Float64).alias("raw_target"),
        (pl.col("hours_stock_status").list.slice(6, 16).list.sum() == 0)
        .alias("is_observed"),
        (pl.col("hours_stock_status").list.slice(6, 16).list.sum() / 16.0)
        .alias("__stockout_rate"),
    ).sort(KEYS + ["__date"])
    if data.select(pl.any_horizontal(
        [pl.col(column).is_null() for column in KEYS + ["__date"]]
    ).any()).item():
        raise ValueError("Store-product keys and dates cannot be null")
    if data.select(pl.struct(KEYS + ["__date"]).is_duplicated().any()).item():
        raise ValueError("Duplicate store-product-date rows")
    if data.select(
        (pl.col("__date").diff().over(KEYS).dt.total_days() != 1)
        .fill_null(False).any()
    ).item():
        raise ValueError("Store-product series must have a complete daily calendar")

    features = ID_FEATURES + ["weekday", "weekend", "day_index"]
    expressions: list[pl.Expr] = [
        (pl.col("__date").dt.weekday() - 1).alias("weekday"),
        (pl.col("__date").dt.weekday() >= 6).cast(pl.Int8).alias("weekend"),
        (pl.col("__date") - pl.col("__date").min()).dt.total_days()
        .cast(pl.Int32).alias("day_index"),
        ((pl.col("__date") - pl.col("__date").min()).dt.total_days()
         >= warmup_days).alias("__warmup_complete"),
    ]
    for lag in [1, 2, 7, 14]:
        name = f"target_lag_{lag}"
        expressions.append(pl.col("target").shift(lag).over(KEYS).alias(name))
        features.append(name)
    for window in [7, 14]:
        name = f"target_mean_{window}"
        expressions.append(
            pl.col("target").shift(1).rolling_mean(window, min_samples=1)
            .over(KEYS).alias(name)
        )
        features.append(name)
    expressions.extend([
        pl.col("target").shift(1).rolling_std(7, min_samples=1, ddof=0)
        .over(KEYS).alias("target_std_7"),
        pl.col("__stockout_rate").shift(1).over(KEYS)
        .alias("stockout_rate_lag_1"),
        pl.col("__stockout_rate").shift(1).rolling_mean(7, min_samples=1)
        .over(KEYS).alias("stockout_rate_mean_7"),
    ])
    features.extend(["target_std_7", "stockout_rate_lag_1", "stockout_rate_mean_7"])
    for lag in [1, 7]:
        name = f"discount_lag_{lag}"
        expressions.append(pl.col("discount").shift(lag).over(KEYS).alias(name))
        features.append(name)
    for column in WEATHER_COLUMNS:
        name = f"{column}_lag_1"
        expressions.append(pl.col(column).shift(1).over(KEYS).alias(name))
        features.append(name)

    return data.with_columns(expressions), features
