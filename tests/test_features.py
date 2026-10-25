from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from fresh_demand.features import build_features


def panel(n_days=20):
    rows = []
    targets = []
    for store_id in [0, 1]:
        for day in range(n_days):
            sales = [0.0] * 24
            sales[12] = float(day + 1)
            stock = [0] * 24
            if day == 0:
                stock[6:10] = [1] * 4
            rows.append({
                "store_id": store_id,
                "product_id": 7,
                "city_id": 0,
                "management_group_id": 1,
                "first_category_id": 2,
                "second_category_id": 3,
                "third_category_id": 4,
                "dt": (date(2024, 1, 1) + timedelta(days=day)).isoformat(),
                "hours_sale": sales,
                "hours_stock_status": stock,
                "discount": float(day) / 100,
                "precpt": float(day),
                "avg_temperature": float(day + 10),
                "avg_humidity": float(day + 20),
                "avg_wind_level": float(day + 30),
            })
            targets.append(float(store_id * 1000 + day + 1))
    return pl.DataFrame(rows), np.asarray(targets)


def test_lags_rolling_statistics_and_series_boundaries():
    frame, target = panel()
    result, features = build_features(frame, target, warmup_days=7)
    row = result.filter((pl.col("store_id") == 0) & (pl.col("day_index") == 8)).row(0, named=True)
    assert row["target_lag_1"] == 8
    assert row["target_lag_2"] == 7
    assert row["target_lag_7"] == 2
    assert row["target_lag_14"] is None
    assert row["target_mean_7"] == pytest.approx(np.mean(np.arange(2, 9)))
    assert row["target_mean_14"] == pytest.approx(np.mean(np.arange(1, 9)))
    assert row["target_std_7"] == pytest.approx(np.std(np.arange(2, 9)))
    assert row["discount_lag_1"] == pytest.approx(0.07)
    assert row["precpt_lag_1"] == 7
    assert row["raw_target"] == 9
    assert row["__warmup_complete"] is True
    first_second_store = result.filter(pl.col("store_id") == 1).row(0, named=True)
    assert first_second_store["target_lag_1"] is None
    assert first_second_store["target_mean_7"] is None
    assert "target" not in features
    assert "raw_target" not in features
    assert "discount" not in features
    assert "hours_stock_status" not in features
    assert "avg_temperature" not in features
    second_day = result.filter((pl.col("store_id") == 0) & (pl.col("day_index") == 1)).row(0, named=True)
    assert second_day["stockout_rate_lag_1"] == 0.25


def test_target_stays_aligned_when_input_rows_are_sorted():
    frame, target = panel()
    order = np.random.default_rng(9).permutation(frame.height)
    shuffled = frame[order.tolist()]
    result, _ = build_features(shuffled, target[order])
    assert result["target"].to_list() == target.tolist()


def test_future_changes_and_append_do_not_change_past_features():
    frame, target = panel()
    result, features = build_features(frame, target)
    cutoff = "2024-01-11"
    future = frame["dt"].to_numpy() >= cutoff
    changed = frame.with_columns(
        pl.when(pl.col("dt") >= cutoff).then(900.0).otherwise(pl.col("discount")).alias("discount"),
        pl.when(pl.col("dt") >= cutoff).then(900.0).otherwise(pl.col("precpt")).alias("precpt"),
    )
    changed_target = target.copy()
    changed_target[future] = 99999
    mutated, mutated_features = build_features(changed, changed_target)
    assert features == mutated_features
    assert result.filter(pl.col("dt") < cutoff).select(features).equals(
        mutated.filter(pl.col("dt") < cutoff).select(features)
    )
    # A forecast cannot use that day's target, weather, or promotion either.
    assert result.filter(pl.col("dt") == cutoff).select(features).equals(
        mutated.filter(pl.col("dt") == cutoff).select(features)
    )
    short = frame.filter(pl.col("dt") < cutoff)
    short_result, _ = build_features(short, target[~future])
    assert short_result.select(features).equals(result.filter(pl.col("dt") < cutoff).select(features))


def test_missing_calendar_days_duplicate_keys_and_invalid_targets_fail():
    frame, target = panel()
    keep = frame["dt"].to_numpy() != "2024-01-05"
    with pytest.raises(ValueError, match="complete daily calendar"):
        build_features(frame.filter(pl.col("dt") != "2024-01-05"), target[keep])
    with pytest.raises(ValueError, match="Duplicate"):
        build_features(pl.concat([frame, frame.head(1)]), np.r_[target, target[0]])
    with pytest.raises(ValueError, match="match frame height"):
        build_features(frame, target[:-1])
