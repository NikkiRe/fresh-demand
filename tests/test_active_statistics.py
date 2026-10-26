import numpy as np
import pytest
from scipy.optimize import linprog

from fresh_demand.active.statistics import (
    RiskSetEstimator,
    cost_curve,
    minimax_regret,
    quantile_bounds,
)


def test_observation_uses_only_exposed_thresholds_and_handles_stockout_ties():
    estimator = RiskSetEstimator(4)
    estimator.observe(3, 1, False)
    np.testing.assert_array_equal(estimator.n, [1, 1, 1, 0])
    np.testing.assert_array_equal(estimator.successes, [0, 1, 1, 0])
    estimator.observe(2, 2, True)
    np.testing.assert_array_equal(estimator.n, [2, 2, 1, 0])
    np.testing.assert_array_equal(estimator.successes, [0, 1, 1, 0])
    estimator.observe(0, 0, True)
    assert estimator.rounds == 3
    with pytest.raises(ValueError, match="sold_out"):
        estimator.observe(2, 2, False)


def test_empty_bands_and_point_tail_assumptions():
    estimator = RiskSetEstimator(4)
    lower, upper = estimator.cdf_bounds()
    np.testing.assert_array_equal(lower, [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(upper, [1, 1, 1, 1, 1])
    np.testing.assert_allclose(estimator.point_pmf(), np.full(5, 0.2))
    np.testing.assert_array_equal(estimator.point_pmf("lower"), [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(estimator.point_pmf("upper"), [0, 0, 0, 0, 1])
    assert quantile_bounds(lower, upper, 0.75) == (0, 4)


def test_point_estimator_is_monotone_weighted_isotonic_and_normalized():
    estimator = RiskSetEstimator(4)
    # Unequal risk sets can yield empirical CDF reversals.
    estimator.observe(1, 0, False)
    estimator.observe(1, 0, False)
    estimator.observe(3, 3, True)
    p = estimator.point_pmf("uniform")
    # Weighted fit pools rates [2/3, 0, 0] at (2)/(3+1+1)=0.4.
    np.testing.assert_allclose(p, [0.4, 0, 0, 0.3, 0.3])
    assert np.isclose(p.sum(), 1)
    assert (p >= 0).all()
    assert estimator.quantile(0.75, "lower") == 3
    assert estimator.quantile(0.75, "upper") == 4


def test_numeric_cost_curve_and_degenerate_band():
    p = np.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(cost_curve(p, 3, 1), [3.9, 1.7, 0.7])
    cdf = p.cumsum()
    action, regret = minimax_regret(cdf, cdf, 3, 1)
    assert action == 2
    np.testing.assert_allclose(regret, cost_curve(p, 3, 1) - 0.7)
    assert quantile_bounds(cdf, cdf, 0.75) == (2, 2)


def _independent_lp_regret(lower, upper, cu, co):
    """Independent probability-simplex LP, without CDF-cost identities."""
    support = np.arange(len(lower))
    losses = cu * np.maximum(support[None, :] - support[:, None], 0)
    losses = losses + co * np.maximum(support[:, None] - support[None, :], 0)
    cdf_matrix = np.tril(np.ones((len(lower), len(lower))))[:-1]
    constraint = np.vstack([cdf_matrix, -cdf_matrix])
    rhs = np.r_[upper[:-1], -lower[:-1]]
    result = []
    for q in support:
        candidate_regrets = []
        for comparator in support:
            optimization = linprog(
                -(losses[q] - losses[comparator]),
                A_ub=constraint, b_ub=rhs,
                A_eq=np.ones((1, len(lower))), b_eq=[1],
                bounds=(0, None), method="highs",
            )
            assert optimization.success
            candidate_regrets.append(-optimization.fun)
        result.append(max(candidate_regrets))
    return np.array(result)


@pytest.mark.parametrize("seed", range(5))
def test_exact_regret_matches_independent_linear_program(seed):
    rng = np.random.default_rng(seed)
    first, second = rng.dirichlet(np.ones(5), size=2).cumsum(axis=1)
    lower, upper = np.minimum(first, second), np.maximum(first, second)
    lower[-1] = upper[-1] = 1
    cu, co = 3.7, 0.6
    action, regret = minimax_regret(lower, upper, cu, co)
    independent = _independent_lp_regret(lower, upper, cu, co)
    np.testing.assert_allclose(regret, independent, atol=1e-10)
    assert action == int(np.argmin(independent))


def test_indistinguishable_tails_under_cap_ten_change_economic_action():
    low_tail = np.zeros(21)
    high_tail = np.zeros(21)
    low_tail[[2, 10]] = [0.6, 0.4]
    high_tail[[2, 20]] = [0.6, 0.4]
    # Observable law at Q=10: sales=2, not sold out with mass .6;
    # sales=10, sold out with mass .4, even when D equals Q.
    def observable_law(p):
        result = {}
        for demand, mass in enumerate(p):
            observation = (min(demand, 10), demand >= 10)
            result[observation] = result.get(observation, 0.0) + mass
        return {key: value for key, value in result.items() if value}
    assert observable_law(low_tail) == observable_law(high_tail)
    assert np.argmin(cost_curve(low_tail, 3, 1)) == 10
    assert np.argmin(cost_curve(high_tail, 3, 1)) == 20
    # A lower critical ratio needs no knowledge of that tail.
    assert np.argmin(cost_curve(low_tail, 1, 1)) == 2
    assert np.argmin(cost_curve(high_tail, 1, 1)) == 2


def test_copy_and_vectorized_hypotheses_equal_explicit_observation_enumeration():
    estimator = RiskSetEstimator(5, horizon=100)
    for observation in [(3, 1, False), (2, 2, True), (5, 4, False)]:
        estimator.observe(*observation)
    p = np.array([0.1, 0.15, 0.2, 0.25, 0.2, 0.1])
    quantities = [0, 1, 3, 5]
    expected = []
    for q in quantities:
        value = 0.0
        for demand, mass in enumerate(p):
            clone = estimator.copy()
            clone.observe(q, min(demand, q), demand >= q)
            _, curve = minimax_regret(*clone.cdf_bounds(), 3, 1)
            value += mass * curve.min()
        expected.append(value)
    np.testing.assert_allclose(estimator.expected_minimax_regret_many(quantities, 3, 1, p=p), expected)
    assert estimator.expected_minimax_regret(3, 3, 1, p=p) == pytest.approx(expected[2])
    assert estimator.rounds == 3
    assert len(estimator.history) == 3
    clone = estimator.copy()
    clone.observe(5, 0, False)
    assert not np.array_equal(clone.n, estimator.n)


def test_horizon_and_invalid_probability_inputs():
    estimator = RiskSetEstimator(2, horizon=1)
    estimator.observe(1, 0, False)
    with pytest.raises(ValueError, match="horizon"):
        estimator.observe(1, 0, False)
    with pytest.raises(ValueError, match="horizon"):
        estimator.expected_minimax_regret(1, 3, 1)
    with pytest.raises(ValueError):
        cost_curve([0.2, 0.2], 3, 1)


def test_cdf_width_lookahead_matches_explicit_clones():
    estimator = RiskSetEstimator(3, horizon=100)
    estimator.observe(3, 1, False)
    p = np.array([0.1, 0.2, 0.3, 0.4])
    expected = []
    for q in [0, 2, 3]:
        widths = []
        for demand in range(4):
            clone = estimator.copy()
            clone.observe(q, min(demand, q), demand >= q)
            lower, upper = clone.cdf_bounds()
            widths.append((upper - lower).sum())
        expected.append(np.dot(p, widths))
    np.testing.assert_allclose(estimator.expected_cdf_width_many([0, 2, 3], p=p), expected)


def test_mean_field_planner_matches_manual_expected_count_updates_without_mutation():
    estimator = RiskSetEstimator(4, horizon=10)
    for observation in [(3, 1, False), (2, 2, True), (4, 3, False)]:
        estimator.observe(*observation)
    original_n, original_successes = estimator.n.copy(), estimator.successes.copy()
    original_history = list(estimator.history)
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    candidates, samples = [0, 2, 4], 64  # Virtual horizon exceeds real remaining horizon.
    regret_expected, width_expected = [], []
    for q in candidates:
        clone = estimator.copy()
        exposure = np.arange(4) < q
        clone.n = clone.n.astype(float) + samples * exposure
        clone.successes = clone.successes.astype(float) + samples * p.cumsum()[:-1] * exposure
        lower, upper = clone.cdf_bounds()
        _, curve = minimax_regret(lower, upper, 3, 1)
        regret_expected.append(curve.min())
        width_expected.append((upper - lower).sum())
    np.testing.assert_allclose(estimator.planned_minimax_regret_many(candidates, 3, 1, samples, p), regret_expected)
    np.testing.assert_allclose(estimator.planned_cdf_width_many(candidates, samples, p), width_expected)
    np.testing.assert_array_equal(estimator.n, original_n)
    np.testing.assert_array_equal(estimator.successes, original_successes)
    assert estimator.rounds == 3
    assert estimator.history == original_history
    with pytest.raises(ValueError, match="planning_samples"):
        estimator.planned_cdf_width_many(candidates, 0, p)


def test_single_sample_mean_field_is_not_exact_outcome_expectation():
    estimator = RiskSetEstimator(3, horizon=100, n_contexts=1)
    for demand in [0, 1, 2, 3] * 20:
        estimator.observe(3, min(demand, 3), demand >= 3)
    p = np.array([0.1, 0.2, 0.3, 0.4])
    mean_field = estimator.planned_minimax_regret_many([2, 3], 3, 1, planning_samples=1, p=p)
    exact = estimator.expected_minimax_regret_many([2, 3], 3, 1, p=p)
    assert not np.allclose(mean_field, exact, atol=1e-10, rtol=1e-10)
