"""Risk-set CDF estimation and exact regret over a CDF-band ambiguity set.

This module does not implement Kaplan--Meier. Confidence statements require
stationary independent future demand conditional on the estimator's context,
and orders chosen before observing that round's demand.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _integer(value, name: str, low: int, high: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if number != value or number < low or (high is not None and number > high):
        raise ValueError(f"{name} must be an integer in the supported range.")
    return number


def _costs(cu: float, co: float) -> tuple[float, float]:
    if not np.isfinite([cu, co]).all() or min(cu, co) < 0 or cu + co <= 0:
        raise ValueError("Costs must be finite, nonnegative, and not both zero.")
    return float(cu), float(co)


def _probabilities(p) -> np.ndarray:
    probability = np.asarray(p, dtype=np.float64)
    if probability.ndim != 1 or probability.size < 2:
        raise ValueError("p must cover integer demand support 0..M with M >= 1.")
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("p must contain finite nonnegative probabilities.")
    if not np.isclose(probability.sum(), 1.0, atol=1e-8, rtol=1e-8):
        raise ValueError("p must sum to one.")
    return probability / probability.sum()


def _bands(L, U) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = np.asarray(L, dtype=float), np.asarray(U, dtype=float)
    if lower.ndim != 1 or lower.shape != upper.shape or lower.size < 2:
        raise ValueError("CDF bounds must be equal-length one-dimensional arrays.")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("CDF bounds must be finite.")
    tolerance = 1e-12
    if ((lower < -tolerance).any() or (upper > 1 + tolerance).any()
            or (lower > upper + tolerance).any()
            or (np.diff(lower) < -tolerance).any()
            or (np.diff(upper) < -tolerance).any()
            or abs(lower[-1] - 1) > tolerance or abs(upper[-1] - 1) > tolerance):
        raise ValueError("Bounds must describe a nonempty monotone CDF band with F(M)=1.")
    return np.clip(lower, 0, 1), np.clip(upper, 0, 1)


def cost_curve(p, cu: float, co: float) -> np.ndarray:
    """Expected newsvendor cost for every integer order on support 0..M."""
    probability = _probabilities(p)
    cu, co = _costs(cu, co)
    demand = np.arange(probability.size)
    orders = demand[:, None]
    losses = cu * np.maximum(demand - orders, 0) + co * np.maximum(orders - demand, 0)
    return losses @ probability


def quantile_bounds(L, U, tau: float) -> tuple[int, int]:
    """Extremal left quantiles among all CDFs inside the supplied band."""
    lower, upper = _bands(L, U)
    if not np.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("tau must lie in [0, 1].")
    return int(np.searchsorted(upper, tau)), int(np.searchsorted(lower, tau))


def _relative_cost(cdf: np.ndarray, cu: float, co: float) -> np.ndarray:
    # C(q+1)-C(q)=(cu+co)F(q)-cu. The unknown additive C(0) cancels in regret.
    increments = (cu + co) * cdf[..., :-1] - cu
    return np.concatenate([np.zeros(cdf.shape[:-1] + (1,)), np.cumsum(increments, axis=-1)], axis=-1)


def _regret_curves(lower: np.ndarray, upper: np.ndarray, cu: float, co: float) -> np.ndarray:
    cost_upper = _relative_cost(upper, cu, co)
    cost_lower = _relative_cost(lower, cu, co)
    left = cost_upper - np.minimum.accumulate(cost_upper, axis=-1)
    right = cost_lower - np.minimum.accumulate(cost_lower[..., ::-1], axis=-1)[..., ::-1]
    return np.maximum(left, right)


def minimax_regret(L, U, cu: float, co: float) -> tuple[int, np.ndarray]:
    """Exact worst-case regret for each action over a pointwise CDF polytope.

    The comparator knows F but uses the same integer order set 0..M. Ties in
    the minimax action are broken toward the smallest quantity. This is not
    the full finite-sample distribution-of-training-data theorem of Kumar and
    Mouchtaki; it solves the supplied deterministic CDF ambiguity problem.
    """
    lower, upper = _bands(L, U)
    cu, co = _costs(cu, co)
    regret = _regret_curves(lower, upper, cu, co)
    return int(np.argmin(regret)), regret


class RiskSetEstimator:
    """CDF estimates from predictable, known inventory thresholds.

    sold_out means D >= Q, including equality. Therefore threshold event
    D <= d is observed only when Q > d. Each estimator represents one fixed
    context; n_contexts supplies the union-bound factor across all contexts.
    """

    def __init__(self, M: int, delta: float = 0.1, n_contexts: int = 4, horizon: int = 500):
        self.M = _integer(M, "M", 1)
        self.n_contexts = _integer(n_contexts, "n_contexts", 1)
        self.horizon = _integer(horizon, "horizon", 1)
        if not np.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("delta must lie strictly between zero and one.")
        self.delta = float(delta)
        self.n = np.zeros(self.M, dtype=np.int64)
        self.successes = np.zeros(self.M, dtype=np.int64)
        self.rounds = 0
        self.history: list[tuple[int, int, bool]] = []
        self._log_term = np.log(2 * self.n_contexts * self.M * self.horizon / self.delta)

    def copy(self) -> RiskSetEstimator:
        """Independent clone for hypothetical observations; the original is unchanged."""
        return copy.deepcopy(self)

    def observe(self, q: int, sales: int, sold_out: bool) -> None:
        q = _integer(q, "q", 0, self.M)
        sales = _integer(sales, "sales", 0, q)
        if not isinstance(sold_out, (bool, np.bool_)):
            raise ValueError("sold_out must be a boolean.")
        if bool(sold_out) != (sales == q):
            raise ValueError("sold_out must equal (sales == q) under the D>=Q convention.")
        if self.rounds >= self.horizon:
            raise ValueError("The declared confidence horizon has been exhausted.")
        self.n[:q] += 1
        self.successes[sales:q] += 1
        self.rounds += 1
        self.history.append((q, sales, bool(sold_out)))

    def _bounds(self, counts: np.ndarray, successes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positive = counts > 0
        estimates = np.divide(successes, counts, out=np.zeros_like(successes, dtype=float), where=positive)
        radius = np.sqrt(self._log_term / (2 * np.maximum(counts, 1)))
        lower = np.where(positive, np.maximum(estimates - radius, 0), 0)
        upper = np.where(positive, np.minimum(estimates + radius, 1), 1)
        lower = np.maximum.accumulate(lower, axis=-1)
        upper = np.minimum.accumulate(upper[..., ::-1], axis=-1)[..., ::-1]
        if (lower > upper + 1e-12).any():
            raise ValueError("Confidence bands have an empty intersection; the coverage event may have failed.")
        shape = lower.shape[:-1] + (1,)
        return np.concatenate([lower, np.ones(shape)], axis=-1), np.concatenate([upper, np.ones(shape)], axis=-1)

    def cdf_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Anytime Hoeffding bands through the declared finite horizon."""
        return self._bounds(self.n, self.successes)

    def point_pmf(self, tail: str = "uniform") -> np.ndarray:
        """Weighted-isotonic risk-set CDF plus an explicitly assumed unknown tail.

        Remaining mass is spread uniformly over unknown integer demands by
        default, or placed at their lower/upper endpoint. This is a working
        predictive distribution, not identification of the unobserved tail.
        """
        if tail not in {"uniform", "lower", "upper"}:
            raise ValueError("tail must be uniform, lower, or upper.")
        known = np.flatnonzero(self.n > 0)
        probability = np.zeros(self.M + 1, dtype=float)
        first_unknown = 0
        if known.size:
            rates = self.successes[known] / self.n[known]
            fitted = IsotonicRegression(y_min=0, y_max=1, increasing=True).fit_transform(
                known, rates, sample_weight=self.n[known],
            )
            probability[:known[-1] + 1] = np.diff(np.r_[0, fitted])
            first_unknown = int(known[-1] + 1)
        mass = max(0.0, 1 - probability.sum())
        if tail == "uniform":
            probability[first_unknown:] += mass / (self.M + 1 - first_unknown)
        else:
            probability[first_unknown if tail == "lower" else self.M] += mass
        return probability / probability.sum()

    def quantile(self, tau: float, tail: str = "uniform") -> int:
        if not np.isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError("tau must lie in [0, 1].")
        return min(self.M, int(np.searchsorted(np.cumsum(self.point_pmf(tail)), tau)))

    def _quantities(self, qvalues) -> np.ndarray:
        quantities = np.asarray(qvalues)
        if (quantities.ndim != 1 or not quantities.size or quantities.dtype.kind not in "iuf"
                or not np.isfinite(quantities).all()):
            raise ValueError("qvalues must be a nonempty vector of supported integers.")
        if (quantities != np.floor(quantities)).any() or (quantities < 0).any() or (quantities > self.M).any():
            raise ValueError("qvalues must contain integers in 0..M.")
        return quantities.astype(np.int64)

    def _hypothetical_bounds(self, qvalues) -> tuple[np.ndarray, np.ndarray]:
        if self.rounds >= self.horizon:
            raise ValueError("The declared confidence horizon has been exhausted.")
        quantities = self._quantities(qvalues)
        thresholds = np.arange(self.M)
        exposure = thresholds[None, :] < quantities[:, None]
        counts = np.broadcast_to(self.n[None, None, :] + exposure[:, None, :],
                                 (len(quantities), self.M + 1, self.M))
        demand = np.arange(self.M + 1)
        events = exposure[:, None, :] & (demand[None, :, None] <= thresholds[None, None, :])
        successes = self.successes[None, None, :] + events
        return self._bounds(counts, successes)

    def _working_pmf(self, p) -> np.ndarray:
        probability = self.point_pmf() if p is None else _probabilities(p)
        if len(probability) != self.M + 1:
            raise ValueError("p must match this estimator's demand support.")
        return probability

    def expected_minimax_regret_many(self, qvalues, cu: float, co: float, p=None) -> np.ndarray:
        """Expected next-state minimax regret for candidate information actions.

        The expectation uses a supplied working PMF, or point_pmf(uniform),
        rather than unknown true demand. All demand outcomes and candidate
        orders are processed in one NumPy batch. State is never mutated.
        """
        cu, co = _costs(cu, co)
        probability = self._working_pmf(p)
        lower, upper = self._hypothetical_bounds(qvalues)
        regret = _regret_curves(lower, upper, cu, co).min(axis=-1)
        return regret @ probability

    def expected_minimax_regret(self, q: int, cu: float, co: float, p=None) -> float:
        return float(self.expected_minimax_regret_many([q], cu, co, p=p)[0])

    def expected_cdf_width_many(self, qvalues, p=None) -> np.ndarray:
        """Expected sum of next-state CDF-band widths, independent of decision costs.

        This diagnostic uses the same working-PMF outcome convention as the
        decision-aware lookahead, allowing a controlled allocation ablation.
        """
        probability = self._working_pmf(p)
        lower, upper = self._hypothetical_bounds(qvalues)
        return (upper - lower).sum(axis=-1) @ probability

    def _planned_bounds(self, qvalues, planning_samples: int, p) -> tuple[np.ndarray, np.ndarray]:
        samples = _integer(planning_samples, "planning_samples", 1)
        quantities = self._quantities(qvalues)
        probability = self._working_pmf(p)
        exposure = np.arange(self.M)[None, :] < quantities[:, None]
        counts = self.n[None, :] + samples * exposure
        expected_events = np.cumsum(probability)[:-1]
        successes = self.successes[None, :] + samples * exposure * expected_events[None, :]
        return self._bounds(counts, successes)

    def planned_minimax_regret_many(
        self, qvalues, cu: float, co: float, planning_samples: int = 16, p=None,
    ) -> np.ndarray:
        """Mean-field planning score after fractional expected risk-set updates.

        For each candidate order, hypothetically repeat it planning_samples
        times and replace its future counts by their working-PMF expectations.
        Regret is evaluated at that expected state. It is an approximation to
        expected future regret, not an outcome-enumerated expectation; even
        planning_samples=1 generally differs from expected_minimax_regret_many.
        Virtual counts may exceed the remaining real horizon. They mutate no
        state and have no new anytime-coverage interpretation.
        """
        cu, co = _costs(cu, co)
        lower, upper = self._planned_bounds(qvalues, planning_samples, p)
        return _regret_curves(lower, upper, cu, co).min(axis=-1)

    def planned_cdf_width_many(self, qvalues, planning_samples: int = 16, p=None) -> np.ndarray:
        """CDF-width analogue of the approximate fractional-count planner.

        This is width at expected counts, not the expected width over possible
        observation sequences. The operational confidence estimator is unchanged.
        """
        lower, upper = self._planned_bounds(qvalues, planning_samples, p)
        return (upper - lower).sum(axis=-1)
