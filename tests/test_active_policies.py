from itertools import product

import numpy as np
import pytest

from fresh_demand.active.policies import (
    _optimal_allocation,
    allocate_cost_aware,
    allocate_decision,
    allocate_predictive,
    allocate_uniform,
)


class ScoreEstimator:
    """Public-interface fixture with no accessible environment demand state."""

    def __init__(self, regret, width=None):
        self.regret = np.asarray(regret, dtype=float)
        self.width = np.asarray(width if width is not None else regret, dtype=float)
        self.M = len(regret) - 1

    @property
    def true_pmf(self):
        raise AssertionError("Allocation must never inspect latent demand")

    def point_pmf(self, tail="uniform"):
        assert tail == "uniform"
        return np.full(self.M + 1, 1 / (self.M + 1))

    def expected_minimax_regret_many(self, quantities, cu, co, p=None):
        np.testing.assert_allclose(p, self.point_pmf())
        return self.regret[np.asarray(quantities)]

    def expected_cdf_width_many(self, quantities, p=None):
        np.testing.assert_allclose(p, self.point_pmf())
        return self.width[np.asarray(quantities)]

    def planned_minimax_regret_many(self, quantities, cu, co, planning_samples=16, p=None):
        assert planning_samples == 16
        return self.expected_minimax_regret_many(quantities, cu, co, p=p)

    def planned_cdf_width_many(self, quantities, planning_samples=16, p=None):
        assert planning_samples == 16
        return self.expected_cdf_width_many(quantities, p=p)


def test_knapsack_matches_bruteforce_for_nonmonotone_and_negative_curves():
    curves = [
        np.array([0.0, 3.0, 2.0, 8.0]),
        np.array([0.0, -1.0, 5.0]),
        np.array([0.0, 1.0, 3.0, -2.0]),
    ]
    for budget in range(11):
        allocation = _optimal_allocation(curves, budget, np.random.default_rng(7))
        target = min(budget, sum(len(curve) - 1 for curve in curves))
        candidates = [
            amounts for amounts in product(*[range(len(curve)) for curve in curves])
            if sum(amounts) == target
        ]
        best = max(sum(curve[b] for curve, b in zip(curves, amounts)) for amounts in candidates)
        assert allocation.sum() == target
        assert sum(curve[b] for curve, b in zip(curves, allocation)) == pytest.approx(best)


def test_at_most_knapsack_matches_bruteforce_and_prefers_less_spending():
    curves = [np.array([0, 5, -2]), np.array([0, 0, 12]), np.array([0, -1, 0])]
    for budget in range(9):
        allocation = _optimal_allocation(curves, budget, np.random.default_rng(7), force_spend=False)
        candidates = [
            amounts for amounts in product(*[range(len(curve)) for curve in curves])
            if sum(amounts) <= budget
        ]
        scores = [sum(curve[b] for curve, b in zip(curves, amounts)) for amounts in candidates]
        best = max(scores)
        least_spend = min(sum(amounts) for amounts, score in zip(candidates, scores) if score == best)
        assert sum(curve[b] for curve, b in zip(curves, allocation)) == pytest.approx(best)
        assert allocation.sum() == least_spend
    np.testing.assert_array_equal(
        _optimal_allocation([np.zeros(4)] * 2, 5, np.random.default_rng(7), force_spend=False),
        [0, 0],
    )


def test_decision_scores_joint_allocation_rather_than_greedy_first_unit():
    # The first unit looks better in context 0, but context 1 only becomes
    # informative after two units. A marginal-gain greedy choice misses it.
    estimators = [ScoreEstimator([20, 15, 15]), ScoreEstimator([20, 20, 8])]
    result = allocate_decision(np.array([0, 0]), 2, estimators, 3, 1, np.random.default_rng(1))
    np.testing.assert_array_equal(result, [0, 2])


def test_uncertainty_ablation_can_differ_from_decision_relevance():
    estimators = [
        ScoreEstimator([10, 10, 9, 9], width=[10, 10, 1, 1]),
        ScoreEstimator([10, 10, 1, 1], width=[10, 10, 9, 9]),
    ]
    base = np.array([1, 1])
    predictive = allocate_predictive(base, 1, estimators, 3, 1, np.random.default_rng(1))
    decision = allocate_decision(base, 1, estimators, 3, 1, np.random.default_rng(1))
    np.testing.assert_array_equal(predictive, [2, 1])
    np.testing.assert_array_equal(decision, [1, 2])
    np.testing.assert_array_equal(base, [1, 1])


@pytest.mark.parametrize("method", [allocate_decision, allocate_predictive])
def test_planned_lookahead_preserves_exact_daily_budget(method):
    estimators = [ScoreEstimator([10, 10, 3, 2]), ScoreEstimator([10, 10, 6, 1])]
    orders = method(
        np.array([1, 1]), 3, estimators, 3, 1, np.random.default_rng(7),
        planning_samples=16,
    )
    assert (orders - 1).sum() == 3
    assert (orders <= 3).all()
    with pytest.raises(ValueError, match="planning_samples"):
        method(np.array([1, 1]), 3, estimators, 3, 1, np.random.default_rng(7), planning_samples=0)


@pytest.mark.parametrize("method", [allocate_decision, allocate_predictive])
@pytest.mark.parametrize("budget", [0, 1, 3, 20])
def test_fixed_budget_capacity_and_zero_information_control(method, budget):
    estimators = [ScoreEstimator([0] * 5) for _ in range(3)]
    base = np.array([0, 3, 4])
    first = method(base, budget, estimators, 3, 1, np.random.default_rng(3))
    second = method(base, budget, estimators, 3, 1, np.random.default_rng(3))
    np.testing.assert_array_equal(first, second)
    assert np.issubdtype(first.dtype, np.integer)
    assert (first >= base).all()
    assert (first <= 4).all()
    assert int((first - base).sum()) == min(budget, 5)


def test_uniform_budget_caps_reproducibility_and_empty_input():
    base = np.array([4, 3, 0])
    result = allocate_uniform(base, 20, 4, np.random.default_rng(1))
    np.testing.assert_array_equal(result, [4, 4, 4])
    np.testing.assert_array_equal(base, [4, 3, 0])
    np.testing.assert_array_equal(
        allocate_uniform(base, 3, 4, np.random.default_rng(12)),
        allocate_uniform(base, 3, 4, np.random.default_rng(12)),
    )
    assert allocate_uniform(np.array([], dtype=int), 4, 4, np.random.default_rng(1)).size == 0


def test_tied_scores_do_not_always_prefer_the_same_context_index():
    curves = [np.zeros(2)] * 4
    winners = {
        int(_optimal_allocation(curves, 1, np.random.default_rng(seed)).argmax())
        for seed in range(40)
    }
    assert winners == {0, 1, 2, 3}


@pytest.mark.parametrize("method", [allocate_decision, allocate_predictive])
def test_uninformative_scores_share_uniform_baseline_allocation_kernel(method):
    estimators = [ScoreEstimator(np.zeros(9)) for _ in range(4)]
    base = np.array([0, 3, 7, 4])
    for seed in range(10):
        uniform = allocate_uniform(base, 7, 8, np.random.default_rng(seed))
        scored = method(base, 7, estimators, 3, 1, np.random.default_rng(seed))
        np.testing.assert_array_equal(scored, uniform)


@pytest.mark.parametrize("base,budget,maximum", [([0.5], 1, 4), ([5], 1, 4), ([0], -1, 4), ([0], 0.5, 4)])
def test_invalid_order_or_budget_fails(base, budget, maximum):
    with pytest.raises(ValueError):
        allocate_uniform(np.asarray(base), budget, maximum, np.random.default_rng(1))


def test_real_estimator_allocation_does_not_change_observed_history():
    from fresh_demand.active.statistics import RiskSetEstimator

    estimators = [RiskSetEstimator(6, horizon=100, n_contexts=2) for _ in range(2)]
    for estimator in estimators:
        for sales in [0, 1, 2, 3, 3, 3]:
            estimator.observe(3, sales, sales >= 3)
    before = [(estimator.n.copy(), estimator.successes.copy()) for estimator in estimators]
    for method in [allocate_decision, allocate_predictive]:
        for planning_samples in [1, 16]:
            orders = method(
                np.array([2, 2]), 3, estimators, 3, 1, np.random.default_rng(1),
                planning_samples=planning_samples,
            )
            assert (orders - 2).sum() == 3
            for estimator, (n, successes) in zip(estimators, before):
                np.testing.assert_array_equal(estimator.n, n)
                np.testing.assert_array_equal(estimator.successes, successes)


class BinaryDemandEstimator(ScoreEstimator):
    def point_pmf(self, tail="uniform"):
        assert tail == "uniform"
        return np.r_[0.4, 0.6, np.zeros(self.M - 1)]

    def planned_minimax_regret_many(self, quantities, cu, co, planning_samples=64, p=None):
        assert planning_samples >= 1
        return self.expected_minimax_regret_many(quantities, cu, co, p=p)


def test_cost_aware_uses_minimal_informative_probe_and_leaves_budget_unused():
    # Quantities above two give exactly the same information score; their
    # immediate overage cost grows, so a forced full-budget probe is dominated.
    estimator = BinaryDemandEstimator([8, 8, 7, 7, 7, 7])
    orders = allocate_cost_aware(
        np.array([1]), 4, [estimator], 3, 1, np.random.default_rng(1),
        planning_samples=64, remaining_rounds=400,
    )
    np.testing.assert_array_equal(orders, [2])


@pytest.mark.parametrize("planning_samples", [1, 64])
def test_cost_aware_spends_zero_when_all_extra_orders_have_negative_net_value(planning_samples):
    estimator = BinaryDemandEstimator([0] * 6)
    orders = allocate_cost_aware(
        np.array([1]), 4, [estimator], 3, 1, np.random.default_rng(1),
        planning_samples=planning_samples, remaining_rounds=400,
    )
    np.testing.assert_array_equal(orders, [1])


def test_cost_aware_respects_budget_capacity_and_does_not_mutate_history():
    from fresh_demand.active.statistics import RiskSetEstimator

    estimators = [RiskSetEstimator(6, horizon=100, n_contexts=2) for _ in range(2)]
    for estimator in estimators:
        for sales in [0, 1, 2, 3, 3, 3]:
            estimator.observe(3, sales, sales >= 3)
    before = [(e.n.copy(), e.successes.copy()) for e in estimators]
    base = np.array([2, 5])
    for budget in [0, 1, 2, 20]:
        orders = allocate_cost_aware(base, budget, estimators, 3, 1, np.random.default_rng(1))
        assert (orders >= base).all() and (orders <= 6).all()
        assert int((orders - base).sum()) <= min(budget, 5)
        for estimator, (n, successes) in zip(estimators, before):
            np.testing.assert_array_equal(estimator.n, n)
            np.testing.assert_array_equal(estimator.successes, successes)
