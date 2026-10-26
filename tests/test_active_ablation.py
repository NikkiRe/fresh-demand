from itertools import product

import numpy as np
import pytest

from fresh_demand.active.ablation import allocate_myopic
from fresh_demand.active.statistics import RiskSetEstimator, cost_curve


class PointOnlyEstimator:
    def __init__(self, pmf):
        self.pmf = np.asarray(pmf, dtype=float)
        self.M = len(self.pmf) - 1

    def point_pmf(self, tail="uniform"):
        assert tail == "uniform"
        return self.pmf.copy()

    def __getattr__(self, name):
        raise AssertionError(f"Myopic control must not require {name}")


def test_single_context_moves_toward_point_quantile_with_available_budget():
    pmf = np.zeros(11)
    pmf[2] = pmf[6] = 0.5
    estimator = PointOnlyEstimator(pmf)
    base = np.array([2])
    for budget, expected in [(0, 2), (1, 3), (2, 4), (20, 6)]:
        result = allocate_myopic(base, budget, [estimator], 3, 1, np.random.default_rng(4))
        np.testing.assert_array_equal(result, [expected])
    np.testing.assert_array_equal(base, [2])


def test_never_buys_negative_value_or_tied_zero_value_extra_stock():
    pmf = np.zeros(11)
    pmf[2] = pmf[6] = 0.5
    estimator = PointOnlyEstimator(pmf)
    # At equal costs every quantity from two through six is optimal. The
    # at-most-budget tie rule must choose the first, using only two extras.
    np.testing.assert_array_equal(
        allocate_myopic(np.array([0]), 10, [estimator], 1, 1, np.random.default_rng(8)),
        [2],
    )
    # Baseline stock is already above the fitted critical quantile. Every
    # positive extra has negative immediate value, so the entire cap is unused.
    np.testing.assert_array_equal(
        allocate_myopic(np.array([8]), 10, [estimator], 3, 1, np.random.default_rng(8)),
        [8],
    )


def test_shared_small_budget_goes_to_larger_immediate_cost_reduction():
    first, second = np.zeros(9), np.zeros(9)
    first[0], first[4] = 0.2, 0.8
    second[0], second[6] = 0.6, 0.4
    result = allocate_myopic(
        np.array([1, 1]), 2, [PointOnlyEstimator(first), PointOnlyEstimator(second)],
        3, 1, np.random.default_rng(12),
    )
    # Each unit saves 2.2 in context zero versus 0.6 in context one.
    np.testing.assert_array_equal(result, [3, 1])


def test_myopic_matches_independent_bruteforce_and_preserves_histories():
    estimators = [RiskSetEstimator(6, horizon=100, n_contexts=2) for _ in range(2)]
    for estimator, observations in zip(estimators, [[0, 1, 2, 3, 3, 3], [0, 0, 1, 3, 3, 3]]):
        for sales in observations:
            estimator.observe(3, sales, sales == 3)
    before = [(e.n.copy(), e.successes.copy(), list(e.history)) for e in estimators]
    base = np.array([1, 2])
    curves = [cost_curve(e.point_pmf(tail="uniform"), 3, 1) for e in estimators]
    for budget in [0, 1, 3, 20]:
        result = allocate_myopic(base, budget, estimators, 3, 1, np.random.default_rng(42))
        repeat = allocate_myopic(base, budget, estimators, 3, 1, np.random.default_rng(42))
        np.testing.assert_array_equal(result, repeat)
        candidates = [q for q in product(range(base[0], 7), range(base[1], 7))
                      if sum(q) - base.sum() <= budget]
        costs = [sum(curve[q] for curve, q in zip(curves, quantities)) for quantities in candidates]
        achieved = sum(curve[q] for curve, q in zip(curves, result))
        assert achieved == pytest.approx(min(costs))
        least_extra = min(sum(q) - base.sum() for q, cost in zip(candidates, costs)
                          if np.isclose(cost, min(costs)))
        assert int((result - base).sum()) == least_extra
        for curve, start, selected in zip(curves, base, result):
            assert curve[selected] <= curve[start] + 1e-12
        for estimator, (counts, successes, history) in zip(estimators, before):
            np.testing.assert_array_equal(estimator.n, counts)
            np.testing.assert_array_equal(estimator.successes, successes)
            assert estimator.history == history
