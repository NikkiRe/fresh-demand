import json
import pickle

import numpy as np
import polars as pl
import pytest

from fresh_demand.recovery import HourlyRecovery


def frame(rows):
    return pl.DataFrame(rows)


def row(sales=None, stock=None, store=1, product=1, category=1):
    operational = [1.0] * 16 if sales is None else list(sales)
    return {
        "store_id": store,
        "product_id": product,
        "third_category_id": category,
        "hours_sale": [999.0] * 6 + operational + [999.0] * 2,
        "hours_stock_status": [1] * 6 + ([0] * 16 if stock is None else list(stock)) + [1] * 2,
        # Deliberately unrelated to the operational target.
        "sale_amount": 100_000.0,
    }


def test_profile_recovers_missing_peak_and_uses_operational_hours_only():
    pattern = [8.0] + [1.0] * 15
    recovery = HourlyRecovery().fit(frame([row(pattern), row(pattern)]))
    missing_peak = frame([row([0.0] + [1.0] * 15, [1] + [0] * 15)])
    assert recovery.transform(missing_peak, "raw")[0] == 15.0
    assert recovery.transform(missing_peak, "uniform")[0] == pytest.approx(16.0)
    assert recovery.transform(missing_peak, "profile")[0] == pytest.approx(23.0)
    json.dumps(recovery.diagnostics_)


@pytest.mark.parametrize("method", ["raw", "uniform", "profile"])
def test_zero_demand_is_finite_and_zero(method):
    data = frame([row([0.0] * 16), row([0.0] * 16, [1] * 16)])
    recovery = HourlyRecovery().fit(data)
    np.testing.assert_array_equal(recovery.transform(data, method), [0.0, 0.0])


@pytest.mark.parametrize("method", ["raw", "uniform", "profile"])
def test_complete_stock_is_identity_even_above_prior_cap(method):
    recovery = HourlyRecovery().fit(frame([row()]))
    data = frame([row(np.arange(16.0)), row([100.0] * 16)])
    expected = [sum(range(16)), 1600.0]
    np.testing.assert_array_equal(recovery.transform(data, method), expected)


def test_full_stockout_uses_prior_and_unknown_pair_falls_back():
    recovery = HourlyRecovery().fit(frame([row([2.0] * 16)]))
    data = frame([
        row([0.0] * 16, [1] * 16),
        row([0.0] * 16, [1] * 16, store=9),
        row([0.0] * 16, [1] * 16, product=9, category=9),
    ])
    for method in ["uniform", "profile"]:
        np.testing.assert_allclose(recovery.transform(data, method), [32.0] * 3)


def test_cap_and_share_floor_preserve_all_observed_sales():
    recovery = HourlyRecovery().fit(frame([row()]))
    # One available hour: dividing by floor(.2), then capping at 5*16.
    floor_case = row([2.0] + [0.0] * 15, [0] + [1] * 15)
    cap_case = row([20.0] + [0.0] * 15, [0] + [1] * 15)
    # Nonzero sales inside absent hours remain an unconditional lower bound.
    contradictory_case = row([1.0] + [10.0] * 15, [0] + [1] * 15)
    result = recovery.transform(frame([floor_case, cap_case, contradictory_case]), "uniform")
    np.testing.assert_allclose(result, [10.0, 80.0, 151.0])


def test_profile_falls_back_to_warmup_category_then_global():
    warmup = frame([
        row([8.0] + [1.0] * 15, product=1, category=1),
        row([1.0] * 15 + [8.0], product=2, category=2),
    ])
    recovery = HourlyRecovery().fit(warmup)
    data = frame([
        row([0.0] + [1.0] * 15, [1] + [0] * 15, product=99, category=1),
        row([0.0] + [1.0] * 15, [1] + [0] * 15, product=98, category=99),
    ])
    result = recovery.transform(data)
    assert result[0] > result[1] > 15.0


def test_frozen_profile_is_independent_of_future_input_rows_and_pickle_roundtrip():
    recovery = HourlyRecovery().fit(frame([row([8.0] + [1.0] * 15)]))
    past = frame([row([0.0] + [1.0] * 15, [1] + [0] * 15)])
    before = recovery.transform(past)
    future = frame([row([1000.0] * 16, product=100, category=100)])
    appended = pl.concat([past, future])
    np.testing.assert_array_equal(recovery.transform(appended)[:1], before)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(recovery)).transform(past), before)


def test_no_complete_warmup_day_uses_observed_mean_and_uniform_shape():
    data = frame([row([0.0] * 8 + [2.0] * 8, [1] * 8 + [0] * 8)])
    recovery = HourlyRecovery().fit(data)
    assert recovery.diagnostics_["positive_complete_days"] == 0
    assert recovery.diagnostics_["pairs_using_observed_prior"] == 1
    assert recovery.transform(data)[0] == pytest.approx(32.0)


def test_invalid_stock_status_and_unfitted_recovery_fail_explicitly():
    with pytest.raises(RuntimeError, match="fit"):
        HourlyRecovery().transform(frame([row()]))
    with pytest.raises(ValueError, match="only 0 or 1"):
        HourlyRecovery().fit(frame([row(stock=[2] + [0] * 15)]))
    with pytest.raises(ValueError, match="at least one"):
        HourlyRecovery().fit(pl.DataFrame())
