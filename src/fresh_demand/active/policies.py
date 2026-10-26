"""Allocate integer extra stock using observed-history models and a budget.

The decision score is a one-step, model-based value-of-information heuristic:
expected reduction in next-step minimax regret relative to ordering the common
baseline quantity. The predictive ablation instead reduces total CDF confidence
band width. Both use the estimator's uniform-tail point PMF to average possible
censored observations. Neither score is an oracle or a guarantee of globally
optimal exploration. The knapsack solver only optimizes the supplied score
curves exactly, including when a fixed budget forces zero/negative-value stock.

With planning_samples > 1, lookahead instead uses fractional expected sufficient
statistics for repeated observations at each candidate quantity. This mean-field
planning approximation can see past the initially flat Hoeffding band widths;
it is not an expectation over realized sample paths or a confidence guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _integer_vector(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional integer array")
    if (array < 0).any() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain nonnegative integers")
    if (array > np.iinfo(np.int64).max // 2).any():
        raise ValueError(f"{name} is too large")
    return array.astype(np.int64)


def _spend(budget: int, capacities: np.ndarray) -> int:
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)):
        raise ValueError("budget must be a nonnegative integer")
    if budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    return min(int(budget), int(capacities.sum()))


def _validate_orders(base_orders, max_demand) -> tuple[np.ndarray, np.ndarray]:
    base = _integer_vector(base_orders, "base_orders")
    limits = np.asarray(max_demand)
    if limits.ndim == 0:
        limits = np.repeat(limits, len(base))
    limits = _integer_vector(limits, "max_demand")
    if base.shape != limits.shape:
        raise ValueError("max_demand must be scalar or match base_orders")
    if (base > limits).any():
        raise ValueError("base_orders must not exceed max_demand")
    return base, limits


def _estimator_inputs(base_orders, estimators: Sequence[Any], budget: int, cu, co):
    if not np.isfinite([cu, co]).all() or min(cu, co) < 0 or cu + co <= 0:
        raise ValueError("cu and co must be nonnegative, finite, and not both zero")
    base, limits = _validate_orders(base_orders, [estimator.M for estimator in estimators])
    if len(estimators) != len(base):
        raise ValueError("One estimator is required per context")
    return base, limits, _spend(budget, limits - base)


def _point_pmf(estimator) -> np.ndarray:
    # This is the fitted estimate, never the environment's latent demand PMF.
    p = np.asarray(estimator.point_pmf(tail="uniform"), dtype=np.float64)
    if p.shape != (int(estimator.M) + 1,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Estimator point_pmf must be a finite nonnegative PMF on 0..M")
    if not np.isclose(p.sum(), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("Estimator point_pmf must sum to one")
    return p / p.sum()


def _uniform_extra(capacities: np.ndarray, spending: int, rng: np.random.Generator) -> np.ndarray:
    allocation = np.zeros(len(capacities), dtype=np.int64)
    for _ in range(spending):
        eligible = np.flatnonzero(allocation < capacities)
        allocation[int(rng.choice(eligible))] += 1
    return allocation


def _optimal_allocation(
    value_curves: Sequence[np.ndarray], budget: int, rng: np.random.Generator,
    force_spend: bool = True,
) -> np.ndarray:
    """Exact multiple-choice knapsack with exchangeable, seeded tie breaking.

    ``value_curves[c][b]`` is the score for spending b units on context c.
    Nonmonotone and negative curves are supported. Objective ties prefer the
    allocation closest to a uniform-unit allocation drawn with the same kernel
    as allocate_uniform. Thus flat scores cannot gain an accidental exploration
    advantage from a more concentrated DP-backtracking tie distribution. A
    random context permutation breaks any remaining context-index symmetry.
    With force_spend=False, spending is at most budget and objective ties first
    prefer the smallest total expenditure, then the usual fixed-total rule.
    """
    curves = [np.asarray(curve, dtype=float) for curve in value_curves]
    if any(curve.ndim != 1 or not len(curve) or not np.isfinite(curve).all() for curve in curves):
        raise ValueError("Value curves must be nonempty finite one-dimensional arrays")
    capacities = np.array([len(curve) - 1 for curve in curves], dtype=np.int64)
    spending = _spend(budget, capacities)
    allocation = np.zeros(len(curves), dtype=np.int64)
    if spending == 0:
        return allocation
    if not force_spend:
        # First find the best total without random tie breaking. The ascending
        # argmax chooses the least spend when several totals achieve the same
        # value. Then sample a tie target for that actual total, rather than
        # comparing lower-spend plans to a target that spends the entire cap.
        previous = np.full(spending + 1, -np.inf)
        previous[0] = 0.0
        for curve in curves:
            current = np.full_like(previous, -np.inf)
            for amount in range(min(len(curve) - 1, spending) + 1):
                current[amount:] = np.maximum(
                    current[amount:], previous[:spending + 1 - amount] + curve[amount],
                )
            previous = current
        selected_total = int(np.argmax(previous))
        return _optimal_allocation(curves, selected_total, rng, force_spend=True)
    tie_target = _uniform_extra(capacities, spending, rng)
    permutation = rng.permutation(len(curves))
    dp = np.full((len(curves) + 1, spending + 1), -np.inf)
    distance = np.full_like(dp, np.inf)
    choice = np.full((len(curves) + 1, spending + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    distance[0, 0] = 0.0
    for row, context in enumerate(permutation, start=1):
        curve = curves[context]
        for total in range(spending + 1):
            amounts = np.arange(min(len(curve) - 1, total) + 1)
            scores = dp[row - 1, total - amounts] + curve[amounts]
            maximum = scores.max()
            if not np.isfinite(maximum):
                continue
            ties = amounts[scores == maximum]
            distances = (
                distance[row - 1, total - ties] + (ties - tie_target[context]) ** 2
            )
            best_distance = distances.min()
            ties = ties[distances == best_distance]
            selected = int(ties[0] if len(ties) == 1 else rng.choice(ties))
            dp[row, total] = maximum
            distance[row, total] = best_distance
            choice[row, total] = selected
    remaining = spending
    for row in range(len(curves), 0, -1):
        selected = int(choice[row, remaining])
        if selected < 0:
            raise RuntimeError("Knapsack backtracking encountered an infeasible state")
        allocation[permutation[row - 1]] = selected
        remaining -= selected
    return allocation


def allocate_uniform(
    base_orders: np.ndarray,
    budget: int,
    max_demand: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Distribute each extra unit uniformly among contexts with spare capacity."""
    base, limits = _validate_orders(base_orders, max_demand)
    spending = _spend(budget, limits - base)
    return base + _uniform_extra(limits - base, spending, rng)


def _expected_width_many(estimator, quantities: np.ndarray, p: np.ndarray) -> np.ndarray:
    batch = getattr(estimator, "expected_cdf_width_many", None)
    if batch is not None:
        return np.asarray(batch(quantities, p=p), dtype=float)
    # Compatibility path for small external estimators. Aggregate all D>=Q
    # into one censored observation; D==Q is a sell-out, not an exact demand.
    values = []
    for q in quantities:
        expected = 0.0
        for sales in range(int(q) + 1):
            mass = float(p[sales] if sales < q else p[int(q):].sum())
            if mass == 0:
                continue
            copied = estimator.copy()
            copied.observe(int(q), sales, sales == q)
            lower, upper = copied.cdf_bounds()
            expected += mass * float(np.sum(np.asarray(upper) - np.asarray(lower)))
        values.append(expected)
    return np.asarray(values)


def _allocate_scored(
    base_orders, budget, estimators, cu, co, rng, criterion, planning_samples=1
):
    if (isinstance(planning_samples, (bool, np.bool_))
            or not isinstance(planning_samples, (int, np.integer)) or planning_samples < 1):
        raise ValueError("planning_samples must be a positive integer")
    base, limits, spending = _estimator_inputs(base_orders, estimators, budget, cu, co)
    if spending == 0:
        return base.copy()
    curves = []
    for start, limit, estimator in zip(base, limits, estimators, strict=True):
        quantities = np.arange(start, start + min(spending, limit - start) + 1)
        p = _point_pmf(estimator)
        if planning_samples > 1:
            if criterion == "width":
                expected = np.asarray(estimator.planned_cdf_width_many(
                    quantities, planning_samples=planning_samples, p=p,
                ), dtype=float)
            else:
                expected = np.asarray(estimator.planned_minimax_regret_many(
                    quantities, cu, co, planning_samples=planning_samples, p=p,
                ), dtype=float)
        elif criterion == "width":
            expected = _expected_width_many(estimator, quantities, p)
        else:
            batch = getattr(estimator, "expected_minimax_regret_many", None)
            expected = (
                np.asarray(batch(quantities, cu, co, p=p), dtype=float)
                if batch is not None
                else np.array([
                    estimator.expected_minimax_regret(int(q), cu, co, p=p)
                    for q in quantities
                ])
            )
        if expected.shape != quantities.shape or not np.isfinite(expected).all():
            raise ValueError("Expected information scores must be finite and match quantities")
        values = expected[0] - expected
        values[0] = 0.0
        curves.append(values)
    return base + _optimal_allocation(curves, spending, rng)


def allocate_predictive(
    base_orders: np.ndarray,
    budget: int,
    estimators: Sequence[Any],
    cu: float,
    co: float,
    rng: np.random.Generator,
    planning_samples: int = 1,
) -> np.ndarray:
    """Maximize expected reduction in total CDF-band width, irrespective of tau.

    This uncertainty-only ablation uses the same PMF and exact budget solver as
    the decision method. cu/co are validated for a shared caller interface but
    do not influence its uncertainty objective. planning_samples=1 uses exact
    one-step expectation; larger values use mean-field repeated-observation
    planning, with the same daily spending constraint.
    """
    return _allocate_scored(
        base_orders, budget, estimators, cu, co, rng, "width", planning_samples,
    )


def allocate_decision(
    base_orders: np.ndarray,
    budget: int,
    estimators: Sequence[Any],
    cu: float,
    co: float,
    rng: np.random.Generator,
    planning_samples: int = 1,
) -> np.ndarray:
    """Maximize model-based minimax-regret reduction under fixed daily spending.

    planning_samples=1 averages exact one-step hypothetical censored outcomes;
    larger values use fractional expected updates over repeated observations.
    """
    return _allocate_scored(
        base_orders, budget, estimators, cu, co, rng, "regret", planning_samples,
    )


def allocate_cost_aware(
    base_orders: np.ndarray,
    budget: int,
    estimators: Sequence[Any],
    cu: float,
    co: float,
    rng: np.random.Generator,
    planning_samples: int = 64,
    remaining_rounds: int = 400,
) -> np.ndarray:
    """Spend at most the daily cap when a planning surrogate justifies the cost.

    For each extra-stock quantity, the net score is
    (R_after_base - R_after_quantity) * remaining_rounds / planning_samples,
    minus the fitted-PMF incremental immediate newsvendor cost. The score uses
    an exact one-step expectation at planning_samples=1 and fractional expected
    sufficient statistics otherwise. Multiplying future ambiguity reduction by
    remaining decisions is a heuristic valuation, not an optimality guarantee,
    regret bound, or measurement of the actual value of future information.
    Zero extra spending is feasible; objective ties prefer spending less.
    """
    from .statistics import cost_curve

    if (isinstance(planning_samples, (bool, np.bool_))
            or not isinstance(planning_samples, (int, np.integer)) or planning_samples < 1):
        raise ValueError("planning_samples must be a positive integer")
    if (isinstance(remaining_rounds, (bool, np.bool_))
            or not isinstance(remaining_rounds, (int, np.integer)) or remaining_rounds < 0):
        raise ValueError("remaining_rounds must be a nonnegative integer")
    base, limits, spending = _estimator_inputs(base_orders, estimators, budget, cu, co)
    if spending == 0:
        return base.copy()
    curves = []
    for start, limit, estimator in zip(base, limits, estimators, strict=True):
        quantities = np.arange(start, start + min(spending, limit - start) + 1)
        p = _point_pmf(estimator)
        if planning_samples > 1:
            expected = np.asarray(estimator.planned_minimax_regret_many(
                quantities, cu, co, planning_samples=planning_samples, p=p,
            ), dtype=float)
        else:
            batch = getattr(estimator, "expected_minimax_regret_many", None)
            expected = (
                np.asarray(batch(quantities, cu, co, p=p), dtype=float)
                if batch is not None
                else np.array([
                    estimator.expected_minimax_regret(int(q), cu, co, p=p)
                    for q in quantities
                ])
            )
        if expected.shape != quantities.shape or not np.isfinite(expected).all():
            raise ValueError("Expected information scores must be finite and match quantities")
        immediate_costs = cost_curve(p, cu, co)
        values = (
            (expected[0] - expected) * remaining_rounds / planning_samples
            - (immediate_costs[quantities] - immediate_costs[start])
        )
        values[0] = 0.0
        curves.append(values)
    return base + _optimal_allocation(curves, spending, rng, force_spend=False)
