from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fresh_demand.active.environments import make_environment, sample_demands


def test_known_pmfs_shapes_support_seed_independence_and_regime_boundary():
    for name in ["heterogeneous", "geometric", "identified", "heavy_tail", "shift"]:
        environment = make_environment(name, seed=1, horizon=101)
        other_seed = make_environment(name, seed=999, horizon=101)
        pmfs = environment["pmfs"]
        assert pmfs.shape == (101, 4, 41)
        assert np.isfinite(pmfs).all() and (pmfs >= 0).all()
        np.testing.assert_allclose(pmfs.sum(axis=-1), 1, atol=1e-12)
        np.testing.assert_array_equal(pmfs, other_seed["pmfs"])
        if name == "shift":
            np.testing.assert_array_equal(pmfs[0], pmfs[49])
            assert not np.array_equal(pmfs[49], pmfs[50])
            assert environment["metadata"]["stationary_confidence_bounds_apply"] is False
        else:
            np.testing.assert_array_equal(pmfs[0], pmfs[-1])
    heterogeneous = make_environment("heterogeneous", seed=0)["metadata"]
    assert len(set(heterogeneous["context_q75_before_shift"])) >= 3


def test_geometric_control_uses_capped_tail_mass_not_truncated_density():
    pmfs = make_environment("geometric", seed=7, horizon=5, max_demand=12)["pmfs"]
    for context, probability in enumerate([0.10, 0.18, 0.30, 0.45]):
        np.testing.assert_allclose(pmfs[0, context, :12], probability * (1 - probability) ** np.arange(12))
        assert pmfs[0, context, 12] == pytest.approx((1 - probability) ** 12)


def test_inverse_cdf_common_random_numbers_and_exact_zero_mass():
    pmfs = np.zeros((1000, 2, 41))
    pmfs[..., 7] = 1
    draws = sample_demands(pmfs, seed=42)
    assert draws.shape == (1000, 2) and (draws == 7).all()
    np.testing.assert_array_equal(draws, sample_demands(pmfs, seed=42))
    mixed = pmfs.copy()
    mixed[..., 7] = 0.3
    mixed[..., 19] = 0.7
    uniforms = np.random.default_rng(17).random((1000, 2))
    np.testing.assert_array_equal(sample_demands(mixed, seed=17), np.where(uniforms < 0.3, 7, 19))
    with pytest.raises(ValueError, match="sum to one"):
        sample_demands(mixed * 0.5, seed=17)
    identified = make_environment("identified", seed=0)
    assert sample_demands(identified["pmfs"], seed=9).max() <= 1


def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_active_contexts.py"
    spec = importlib.util.spec_from_file_location("build_active_contexts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_context_model


def write_train(path, future_multiplier=1):
    rows = []
    for series in range(12):
        for day in range(75):
            value = float(1 + series + (day % 3))
            if day >= 69:
                value *= future_multiplier
            sales = [0.0] * 24
            sales[12] = value
            stock = [0] * 24
            if day % 9 == 0:
                stock[18] = 1
            rows.append({"store_id": series // 3, "product_id": series,
                         "dt": (date(2024, 1, 1) + timedelta(days=day)).isoformat(),
                         "hours_sale": sales, "hours_stock_status": stock})
    pl.DataFrame(rows).write_parquet(path)


def test_empirical_model_ignores_later_days_and_is_explicitly_semisynthetic(tmp_path):
    train = tmp_path / "train.parquet"
    write_train(train)
    model = builder()(train, max_series=12, n_contexts=4, threads=1)
    write_train(train, future_multiplier=10000)
    changed = builder()(train, max_series=12, n_contexts=4, threads=1)
    np.testing.assert_array_equal(model["pmfs"], changed["pmfs"])
    assert model["metadata"]["clusters"] == changed["metadata"]["clusters"]
    assert model["metadata"]["source_sha256"] != changed["metadata"]["source_sha256"]
    assert model["metadata"]["official_eval_file_read"] is False
    assert model["metadata"]["real_policy_counterfactual_identified"] is False
    scale = model["metadata"]["scale"]
    assert scale["mode"] == "tail_guard"
    assert scale["multiplier"] == pytest.approx(40 / max(scale["raw_q99_by_context"]))
    assert scale["maximum_context_clipped_row_fraction"] <= 0.011
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(model))
    environment = make_environment("fresh_empirical", seed=3, horizon=80, context_path=context_path)
    np.testing.assert_allclose(environment["pmfs"][0], model["pmfs"])
    np.testing.assert_array_equal(environment["pmfs"][0], environment["pmfs"][-1])
    assert np.count_nonzero(np.asarray(model["pmfs"]) == 0) > 0
    with pytest.raises(ValueError, match="shape"):
        make_environment("fresh_empirical", seed=3, context_path=context_path, n_contexts=3)
