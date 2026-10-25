import numpy as np
import polars as pl
import pytest

from fresh_demand.evaluation import (
    inventory_metrics,
    make_censoring,
    paired_store_bootstrap,
    pinball,
    regression_metrics,
)


def test_numeric_forecast_and_inventory_metrics():
    assert regression_metrics([2, 4], [1, 6]) == pytest.approx({"mae": 1.5, "wape": 0.5, "bias": 1 / 6})
    assert pinball([2, 4], [1, 6], q=0.75) == pytest.approx(0.625)
    assert inventory_metrics([2, 4], [1, 6]) == pytest.approx({
        "normalized_cost": 5 / 6,
        "mean_shortage": 0.5,
        "mean_excess": 1,
        "fill_rate": 5 / 6,
        "cycle_service": 0.5,
    })


def test_zero_demand_relative_metrics_are_explicitly_undefined():
    assert np.isnan(regression_metrics([0], [1])["wape"])
    assert np.isnan(inventory_metrics([0], [1])["normalized_cost"])
    assert np.isnan(inventory_metrics([0], [1])["fill_rate"])
    assert inventory_metrics([0], [1])["mean_excess"] == 1


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])
    with pytest.raises(ValueError):
        pinball([1], [1], q=1)
    with pytest.raises(ValueError):
        inventory_metrics([1], [-1])


def test_paired_store_bootstrap_identical_policies_have_zero_difference():
    frame = pl.DataFrame({"store_id": [1, 1, 2], "truth": [1.0, 2.0, 10.0], "candidate": [2.0, 1.0, 8.0]})
    frame = frame.with_columns(pl.col("candidate").alias("baseline"))
    result = paired_store_bootstrap(frame, "truth", "candidate", "baseline", n_boot=200)
    assert result["estimate"] == result["ci_low"] == result["ci_high"] == 0
    assert result["n_stores"] == 2
    assert result["conditional_on_dates"] is True


def test_store_bootstrap_recomputes_ratio_from_all_store_rows():
    frame = pl.DataFrame({
        "store_id": [1, 1, 2],
        "truth": [1.0, 1.0, 100.0],
        "candidate": [1.0, 1.0, 100.0],
        "baseline": [0.0, 0.0, 90.0],
    })
    result = paired_store_bootstrap(frame, "truth", "candidate", "baseline", n_boot=500, seed=7)
    assert result["estimate"] == pytest.approx(-36 / 102)
    assert result["ci_low"] == pytest.approx(-3)
    assert result["ci_high"] == pytest.approx(-0.3)
    assert result == paired_store_bootstrap(frame, "truth", "candidate", "baseline", n_boot=500, seed=7)


def _hourly_frame():
    status = np.zeros((3, 24), dtype=np.int32)
    status[1, 12] = 1
    status[2, :6] = 1  # Closed-hour flags do not make an operational day censored.
    sales = np.ones((3, 24), dtype=float)
    return pl.DataFrame({
        "store_id": [1, 2, 3],
        "hours_sale": sales.tolist(),
        "hours_stock_status": status.tolist(),
        "sale_amount": [24.0] * 3,
        "stock_hour6_22_cnt": [0, 1, 0],
    })


@pytest.mark.parametrize("regime,hours", [("suffix", [18, 19, 20, 21]), ("peak", [9, 10, 16, 17])])
def test_fixed_masks_only_change_fully_available_operational_hours(regime, hours):
    original = _hourly_frame()
    censored = make_censoring(original, regime=regime)
    masks = np.array(censored["controlled_mask"].to_list())
    assert np.flatnonzero(masks[0]).tolist() == hours
    assert not masks[1].any()
    assert np.flatnonzero(masks[2]).tolist() == hours
    assert censored["sale_amount"].to_list() == [20.0, 24.0, 20.0]
    assert censored["stock_hour6_22_cnt"].to_list() == [4, 1, 4]
    assert original["sale_amount"].to_list() == [24.0, 24.0, 24.0]


def test_empirical_masks_come_only_from_training_not_target_sales():
    training_masks = np.zeros((3, 24), dtype=np.int8)
    training_masks[1, 8:11] = 1
    training_masks[2, 20:22] = 1
    original = _hourly_frame()
    changed_sales = original.with_columns(pl.Series("hours_sale", [[1000.0] * 24] * original.height))
    result = make_censoring(original, training_masks, regime="empirical", seed=11)
    other = make_censoring(changed_sales, training_masks, regime="empirical", seed=11)
    assert result["controlled_mask"].to_list() == other["controlled_mask"].to_list()
    for idx in [0, 2]:
        mask = result["controlled_mask"][idx].to_numpy()
        assert any(np.array_equal(mask, training_masks[j]) for j in [1, 2])
        assert mask.any()
    assert not result["controlled_censored"][1]


def test_empirical_masks_reject_empty_operational_pool():
    masks = np.zeros((2, 24), dtype=np.int8)
    masks[:, :6] = 1
    with pytest.raises(ValueError, match="nonempty operational"):
        make_censoring(_hourly_frame(), masks)


def test_existing_censored_day_keeps_original_daily_fields():
    frame = _hourly_frame().with_columns(pl.Series("sale_amount", [24.0, 23.99, 24.0]))
    result = make_censoring(frame, regime="suffix")
    assert result["sale_amount"][1] == 23.99
    assert result["hours_sale"][1].to_list() == frame["hours_sale"][1].to_list()
    assert result["hours_stock_status"][1].to_list() == frame["hours_stock_status"][1].to_list()
