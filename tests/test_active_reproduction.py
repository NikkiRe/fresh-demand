import json

import numpy as np

from fresh_demand.active.reproduction import (
    _fit_record_batch, analytic_identification, distributions, passive_plateau,
    population_band, run_reproduction,
)
from fresh_demand.active.statistics import RiskSetEstimator, cost_curve, minimax_regret


def test_observational_equivalence_and_cost_dependent_identification():
    result = analytic_identification()
    assert result["observation_laws"]["A"] == result["observation_laws"]["B"] == [
        {"sales": 5, "sold_out": False, "probability": 0.5},
        {"sales": 8, "sold_out": True, "probability": 0.5},
    ]
    low, high = result["cost_settings"]
    assert low["oracle_orders"] == {"A": 5, "B": 5}
    assert low["identified_quantile_bounds"] == [5, 5]
    assert low["full_cdf_band_minimax_order"] == 5
    assert low["full_cdf_band_minimax_regret"] == 0
    assert high["oracle_orders"] == {"A": 12, "B": 35}
    assert high["identified_quantile_bounds"] == [8, 40]
    assert high["full_cdf_band_minimax_order"] == 24
    assert high["full_cdf_band_minimax_regret"] == 16
    assert high["two_witness_worlds_minimax_orders"] == [23, 24]
    assert high["two_witness_worlds_minimax_regret"] == 12


def test_full_band_formula_against_independent_tail_vertex_enumeration():
    lower, upper = population_band()
    _, band_regret = minimax_regret(lower, upper, 3, 1)
    vertex_regrets = []
    for tail in range(8, 41):
        probability = np.zeros(41)
        probability[5] = probability[tail] = 0.5
        costs = cost_curve(probability, 3, 1)
        vertex_regrets.append(costs - costs.min())
    np.testing.assert_allclose(band_regret, np.max(vertex_regrets, axis=0), atol=1e-12)
    np.testing.assert_allclose(band_regret[8:], np.maximum(np.arange(8, 41) - 8, 40 - np.arange(8, 41)))


def test_batch_risk_sets_equal_sequential_observations():
    demand = np.asarray([5, 35, 5, 35, 35, 5, 35, 5])
    for audits in [0, 1, 4]:
        batched = _fit_record_batch(demand, audits, horizon=len(demand))
        sequential = RiskSetEstimator(40, n_contexts=1, horizon=len(demand))
        for index, value in enumerate(demand):
            order = 40 if index >= len(demand) - audits else 8
            sequential.observe(order, min(int(value), order), bool(value >= order))
        np.testing.assert_array_equal(batched.n, sequential.n)
        np.testing.assert_array_equal(batched.successes, sequential.successes)
        np.testing.assert_allclose(batched.point_pmf(), sequential.point_pmf())


def test_fixed_n_audits_and_raw_records_are_reproducible(tmp_path):
    first = passive_plateau(seed_count=3, sample_sizes=(20,), audit_counts=(0, 4))
    second = passive_plateau(seed_count=3, sample_sizes=(20,), audit_counts=(0, 4))
    assert first == second
    passive = [row for row in first["raw_records"] if row["audits"] == 0]
    for seed in range(3):
        for tail in ["lower", "uniform"]:
            for cu in [1, 3]:
                rows = [row for row in passive if row["seed"] == seed and row["tail_assumption"] == tail and row["underage_cost"] == cu]
                assert len(rows) == 2
                assert rows[0]["estimated_order"] == rows[1]["estimated_order"]
    result = run_reproduction(tmp_path, seed_count=2, sample_sizes=(20,), audit_counts=(0, 4))
    saved = json.loads((tmp_path / "identification.json").read_text())
    assert saved == result
    for row in result["monte_carlo"]["raw_records"]:
        assert row["n"] == 20
        curve = cost_curve(distributions()[row["world"]], row["underage_cost"], row["overage_cost"])
        expected = (20 - row["audits"]) * curve[8] + row["audits"] * curve[40]
        assert row["history_expected_cost"] == expected
