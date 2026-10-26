"""Post-confirmation mechanism controls, separate from the frozen policies.

The myopic allocator isolates the value of avoiding immediately expensive
extra stock. It uses the same uniform-tail working PMF and at-most-budget
solver as the cost-aware method, but no confidence bands or information value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .policies import _estimator_inputs, _optimal_allocation, _point_pmf
from .statistics import cost_curve


def allocate_myopic(
    base: np.ndarray,
    budget: int,
    estimators: Sequence[Any],
    cu: float,
    co: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Minimize immediate fitted expected cost using at most ``budget`` extras.

    Each context's score for an order q is C_working(base) - C_working(q).
    Zero extra stock is feasible. The shared exact knapsack implementation
    prefers less total expenditure when scores tie. Returned orders never
    reduce the common baseline and never exceed the estimator's support.
    Estimators need only M and point_pmf(tail="uniform"); their observation
    histories and uncertainty calculations are neither accessed nor changed.
    """
    baseline, limits, spending = _estimator_inputs(base, estimators, budget, cu, co)
    if spending == 0:
        return baseline.copy()
    curves = []
    for start, limit, estimator in zip(baseline, limits, estimators, strict=True):
        quantities = np.arange(start, start + min(spending, limit - start) + 1)
        costs = cost_curve(_point_pmf(estimator), cu, co)
        values = costs[start] - costs[quantities]
        values[0] = 0.0
        curves.append(values)
    return baseline + _optimal_allocation(curves, spending, rng, force_spend=False)
