import json
import pickle
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from fresh_demand.data import file_hash
from fresh_demand.pipeline import forecast
from fresh_demand.retrain import retrain


def history_frame(days=35):
    rows = []
    for store in [1, 2]:
        for day in range(days):
            value = float(1 + store + day % 7)
            rows.append({
                "city_id": 1,
                "store_id": store,
                "management_group_id": 1,
                "first_category_id": 1,
                "second_category_id": 1,
                "third_category_id": 1,
                "product_id": store,
                "dt": (date(2024, 1, 1) + timedelta(days=day)).isoformat(),
                "hours_sale": [0.0] * 6 + [value] * 16 + [0.0] * 2,
                "hours_stock_status": [0] * 24,
                "stock_hour6_22_cnt": 0,
                "sale_amount": value * 16,
                "discount": 0.0,
                "precpt": 0.0,
                "avg_temperature": 10.0,
                "avg_humidity": 40.0,
                "avg_wind_level": 1.0,
            })
    return pl.DataFrame(rows).with_columns(pl.col("stock_hour6_22_cnt").cast(pl.Int32))


@pytest.fixture
def inputs(tmp_path):
    history_path = tmp_path / "history.parquet"
    config_path = tmp_path / "config.json"
    selection_path = tmp_path / "frozen_selection.json"
    model_path = tmp_path / "model.pkl"
    history_frame().write_parquet(history_path)
    config_path.write_text(json.dumps({
        "warmup_days": 14,
        "quantiles": [0.5, 0.75, 0.9],
        "n_estimators": 3,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_child_samples": 2,
        "threads": 1,
        "seed": 42,
    }))
    selection_path.write_text(json.dumps({
        "selected_method": "profile",
        "config_sha256": file_hash(config_path),
    }))
    return history_path, selection_path, config_path, model_path


def test_retrain_bundle_predicts_next_day_reproducibly_and_without_crossing(inputs, tmp_path):
    history_path, _, _, model_path = inputs
    metadata = retrain(*inputs)
    first = forecast(model_path, history_path, tmp_path / "first.parquet")
    second = forecast(model_path, history_path, tmp_path / "second.parquet")
    assert first.equals(second)
    # A 28-day serving window must retain the training-origin day_index and
    # produce the same features and predictions as the complete history file.
    recent_path = tmp_path / "recent.parquet"
    pl.read_parquet(history_path).group_by("store_id", "product_id").tail(28).write_parquet(recent_path)
    assert first.equals(forecast(model_path, recent_path, tmp_path / "recent_prediction.parquet"))
    repeat_path = tmp_path / "repeat.pkl"
    retrain(*inputs[:3], repeat_path)
    assert first.equals(forecast(repeat_path, history_path, tmp_path / "repeat_prediction.parquet"))
    assert first.height == 2
    assert first["dt"].to_list() == ["2024-02-05"] * 2
    assert first["store_id"].to_list() == [1, 2]
    predictions = first.select("demand_q50", "demand_q75", "demand_q90").to_numpy()
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    assert (np.diff(predictions, axis=1) >= 0).all()
    assert metadata["official_evaluation_read"] is False
    assert metadata["history_rows"] == 70
    assert metadata["fitting_rows"] == 42
    assert json.loads(model_path.with_suffix(".metadata.json").read_text())["n_series"] == 2
    with model_path.open("rb") as stream:
        bundle = pickle.load(stream)
    assert set(bundle) == {
        "models", "method", "recovery", "features", "config", "train_end",
        "training_start", "series",
    }
    assert list(bundle["models"]) == ["q50", "q75", "q90"]


def test_forecast_requires_28_history_days_after_retraining(inputs, tmp_path):
    history_path, _, _, model_path = inputs
    retrain(*inputs)
    recent = pl.read_parquet(history_path).group_by("store_id", "product_id").tail(27)
    short_path = tmp_path / "short.parquet"
    recent.write_parquet(short_path)
    with pytest.raises(ValueError, match="at least 28"):
        forecast(model_path, short_path, tmp_path / "unused.parquet")


def test_retrain_rejects_config_changes_after_selection(inputs):
    _, _, config_path, _ = inputs
    config = json.loads(config_path.read_text())
    config["n_estimators"] = 4
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Config hash"):
        retrain(*inputs)


def test_retrain_rejects_unknown_selected_method(inputs):
    _, selection_path, _, _ = inputs
    selection = json.loads(selection_path.read_text())
    selection["selected_method"] = "future_method"
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="selected_method"):
        retrain(*inputs)


def test_retrain_rejects_short_or_gapped_history(inputs):
    history_path, _, _, _ = inputs
    history_frame(34).write_parquet(history_path)
    with pytest.raises(ValueError, match="at least 35"):
        retrain(*inputs)
    # Keep 35 rows in each pair, but remove an internal calendar day from a
    # 36-day panel. Row count alone cannot validate a daily lag pipeline.
    history_frame(36).filter(pl.col("dt") != "2024-01-10").write_parquet(history_path)
    with pytest.raises(ValueError, match="complete daily calendar"):
        retrain(*inputs)
