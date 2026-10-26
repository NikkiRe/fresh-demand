"""Canonical product-limit baseline for known discrete censoring thresholds."""

from __future__ import annotations

import numpy as np


def _integer(value, name: str, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if integer != value or integer < 0 or (maximum is not None and integer > maximum):
        raise ValueError(f"{name} is outside its integer support.")
    return integer


def kaplan_meier_pmf(
    history: list[tuple[int, int, bool]], M: int, tail: str = "uniform",
) -> np.ndarray:
    """Estimate a discrete PMF with censoring removed before same-value events.

    The feedback convention is sales=min(D,Q), sold_out=(D>=Q). Thus a sold-out
    observation at Q=d does not establish whether D=d and cannot be included
    in the risk set for deaths at d. Unknown remaining mass on max(Q)..M uses
    an explicitly assumed uniform/lower/upper completion. At max(Q)=M, that
    remaining support is the known singleton {M}, so completion is unambiguous.
    Independent current demand and predictable thresholds within a stationary
    context are required for the usual censoring interpretation. No confidence
    guarantee or tail identification is claimed by this point estimate.
    """
    M = _integer(M, "M")
    if M < 1:
        raise ValueError("M must be at least one.")
    if tail not in {"uniform", "lower", "upper"}:
        raise ValueError("tail must be uniform, lower, or upper.")
    deaths = np.zeros(M + 1, dtype=np.int64)
    censored = np.zeros(M + 1, dtype=np.int64)
    largest_order = 0
    for observation in history:
        if len(observation) != 3:
            raise ValueError("Each history record must be (q, sales, sold_out).")
        q, sales, sold_out = observation
        q = _integer(q, "q", M)
        sales = _integer(sales, "sales", q)
        if not isinstance(sold_out, (bool, np.bool_)):
            raise ValueError("sold_out must be a boolean.")
        if bool(sold_out) != (sales == q):
            raise ValueError("sold_out must equal (sales == q) under the D>=Q convention.")
        largest_order = max(largest_order, q)
        if sold_out:
            censored[q] += 1
        else:
            deaths[sales] += 1
    uncensored_at_or_above = np.cumsum(deaths[::-1])[::-1]
    censored_strictly_above = np.r_[np.cumsum(censored[::-1])[::-1][1:], 0]
    risk = uncensored_at_or_above + censored_strictly_above
    probability = np.zeros(M + 1, dtype=float)
    survival = 1.0
    for d in range(largest_order):
        hazard = deaths[d] / risk[d] if risk[d] else 0.0
        probability[d] = survival * hazard
        survival *= 1 - hazard
    if tail == "uniform":
        probability[largest_order:] = survival / (M + 1 - largest_order)
    else:
        probability[largest_order if tail == "lower" else M] = survival
    return probability / probability.sum()
