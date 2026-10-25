"""Forecast, inventory, and controlled-censoring evaluation.

All relative metrics are ratios, not percentages. Store bootstrap intervals are
conditional on the observed dates and do not quantify temporal generalization.
"""

from __future__ import annotations

import numpy as np
import polars as pl


OPERATIONAL_HOURS = tuple(range(6, 22))
FIXED_PEAK_HOURS = (9, 10, 16, 17)


def _paired_arrays(y, prediction) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 1 or estimate.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")
    if truth.shape != estimate.shape or not truth.size:
        raise ValueError("Inputs must have the same nonzero length.")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("Inputs must contain finite numeric values.")
    return truth, estimate


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def regression_metrics(y, pred) -> dict[str, float]:
    """Return MAE, WAPE, and signed bias; positive bias is overprediction.

    WAPE and bias use sum(abs(y)) as denominator. They are undefined (NaN)
    when every target is zero.
    """
    truth, estimate = _paired_arrays(y, pred)
    error = estimate - truth
    denominator = float(np.abs(truth).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "wape": _ratio(float(np.abs(error).sum()), denominator),
        "bias": _ratio(float(error.sum()), denominator),
    }


def pinball(y, pred, q: float) -> float:
    """Mean pinball loss for one quantile, with 0 < q < 1."""
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be strictly between zero and one.")
    truth, estimate = _paired_arrays(y, pred)
    residual = truth - estimate
    return float(np.maximum(q * residual, (q - 1) * residual).mean())


def _inventory_components(y, order, underage: float, overage: float):
    truth, quantity = _paired_arrays(y, order)
    if (truth < 0).any() or (quantity < 0).any():
        raise ValueError("Demand and order quantities must be nonnegative.")
    if not np.isfinite([underage, overage]).all() or min(underage, overage) < 0:
        raise ValueError("Underage and overage costs must be finite and nonnegative.")
    shortage = np.maximum(truth - quantity, 0)
    excess = np.maximum(quantity - truth, 0)
    cost = underage * shortage + overage * excess
    return truth, quantity, shortage, excess, cost


def inventory_metrics(y, order, underage: float = 3, overage: float = 1) -> dict[str, float]:
    """Evaluate a single-period order under explicitly assumed cost weights.

    normalized_cost is total asymmetric cost divided by total demand. No
    monetary or multi-period shelf-life interpretation is implied.
    """
    truth, quantity, shortage, excess, cost = _inventory_components(y, order, underage, overage)
    demand = float(truth.sum())
    return {
        "normalized_cost": _ratio(float(cost.sum()), demand),
        "mean_shortage": float(shortage.mean()),
        "mean_excess": float(excess.mean()),
        "fill_rate": _ratio(float(np.minimum(truth, quantity).sum()), demand),
        "cycle_service": float((quantity >= truth).mean()),
    }


def paired_store_bootstrap(
    frame: pl.DataFrame,
    truth_col: str,
    candidate_col: str,
    baseline_col: str,
    underage: float = 3,
    overage: float = 1,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | bool | str]:
    """Paired store-cluster bootstrap of normalized cost, candidate minus baseline.

    Each draw samples stores with replacement and retains every row of each
    sampled store. Ratios are recomputed from the sampled sums, rather than
    averaging store-specific ratios. Lower differences favor the candidate.
    The interval conditions on the observed dates; common temporal shocks are
    not resampled and this is not a guarantee about unseen dates or cities.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer.")
    if "store_id" not in frame.columns or frame["store_id"].null_count():
        raise ValueError("frame requires a non-null store_id column.")
    truth = frame[truth_col].to_numpy()
    _, _, _, _, candidate_cost = _inventory_components(truth, frame[candidate_col].to_numpy(), underage, overage)
    _, _, _, _, baseline_cost = _inventory_components(truth, frame[baseline_col].to_numpy(), underage, overage)
    clusters = (
        frame.select("store_id")
        .with_columns(
            pl.Series("demand", truth),
            pl.Series("cost_difference", candidate_cost - baseline_cost),
        )
        .group_by("store_id", maintain_order=True)
        .agg(pl.col("demand").sum(), pl.col("cost_difference").sum())
    )
    demand = clusters["demand"].to_numpy()
    difference = clusters["cost_difference"].to_numpy()
    n_stores = clusters.height
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    # Bounded temporary allocation even for large numbers of stores.
    for offset in range(0, n_boot, 256):
        batch_size = min(256, n_boot - offset)
        indices = rng.integers(0, n_stores, size=(batch_size, n_stores))
        denominators = demand[indices].sum(axis=1)
        numerators = difference[indices].sum(axis=1)
        np.divide(numerators, denominators, out=draws[offset:offset + batch_size], where=denominators > 0)
    finite_draws = draws[np.isfinite(draws)]
    interval = np.quantile(finite_draws, [0.025, 0.975]) if finite_draws.size else [np.nan, np.nan]
    return {
        "estimate": _ratio(float(difference.sum()), float(demand.sum())),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "n_boot": int(n_boot),
        "n_valid_boot": int(finite_draws.size),
        "n_stores": int(n_stores),
        "n_rows": int(frame.height),
        "metric": "normalized_cost_difference",
        "direction": "candidate_minus_baseline",
        "conditional_on_dates": True,
        "uncertainty_scope": "Store-cluster bootstrap conditional on observed dates; excludes temporal generalization.",
    }


def _hour_matrix(frame: pl.DataFrame, column: str) -> np.ndarray:
    if frame[column].null_count():
        raise ValueError(f"{column} must not contain null lists.")
    try:
        values = frame[column].cast(pl.Array(pl.Float64, 24)).to_numpy().copy()
    except Exception as exc:
        raise ValueError(f"{column} must contain 24 numeric values per row.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{column} must contain finite values.")
    return values


def make_censoring(
    frame: pl.DataFrame,
    training_masks: np.ndarray | None = None,
    regime: str = "empirical",
    seed: int = 42,
) -> pl.DataFrame:
    """Hide operational-hour sales only on originally fully available days.

    ``empirical`` samples nonempty operational masks from the supplied TRAIN
    matrix, shaped (n, 24) or (n, 16); the caller must supply training-only
    stockout masks. ``suffix`` hides hours 18..21. ``peak`` hides the fixed,
    predeclared hours 9, 10, 16, 17; no target-dependent peak selection occurs.
    Sales, stock flags, sale_amount, and stock_hour6_22_cnt are updated. Added
    columns record eligibility and the exact synthetic mask. Preserve original
    truth columns before calling if they are needed for scoring.
    """
    if regime not in {"empirical", "suffix", "peak"}:
        raise ValueError("regime must be empirical, suffix, or peak.")
    sales = _hour_matrix(frame, "hours_sale")
    status = _hour_matrix(frame, "hours_stock_status")
    if (sales < 0).any() or not np.isin(status, [0, 1]).all():
        raise ValueError("Hourly sales must be nonnegative and stock flags binary.")
    eligible = (status[:, 6:22] == 0).all(axis=1)
    indices = np.flatnonzero(eligible)
    hidden = np.zeros(status.shape, dtype=np.int8)
    if regime == "empirical":
        if training_masks is None:
            raise ValueError("empirical requires a training-only mask matrix.")
        masks = np.asarray(training_masks)
        if masks.ndim != 2 or masks.shape[1] not in {16, 24}:
            raise ValueError("training_masks must have shape (n, 16) or (n, 24).")
        if not np.isin(masks, [0, 1]).all():
            raise ValueError("training_masks must be binary.")
        operational_masks = masks[:, 6:22] if masks.shape[1] == 24 else masks
        operational_masks = operational_masks[operational_masks.any(axis=1)]
        if not len(operational_masks):
            raise ValueError("training_masks contains no nonempty operational masks.")
        rng = np.random.default_rng(seed)
        sampled = operational_masks[rng.integers(0, len(operational_masks), size=len(indices))]
        hidden[indices, 6:22] = sampled
    else:
        hours = (18, 19, 20, 21) if regime == "suffix" else FIXED_PEAK_HOURS
        hidden[np.ix_(indices, hours)] = 1
    sales[hidden == 1] = 0
    status[hidden == 1] = 1
    censored = hidden.any(axis=1)
    daily_sales = sales.sum(axis=1)
    stock_counts = status[:, 6:22].sum(axis=1).astype(np.int32)
    if "sale_amount" in frame.columns:
        daily_sales = np.where(censored, daily_sales, frame["sale_amount"].to_numpy())
    if "stock_hour6_22_cnt" in frame.columns:
        stock_counts = np.where(censored, stock_counts, frame["stock_hour6_22_cnt"].to_numpy())
    return frame.with_columns(
        pl.Series("hours_sale", sales).arr.to_list(),
        pl.Series("hours_stock_status", status.astype(np.int32)).arr.to_list(),
        pl.Series("sale_amount", daily_sales),
        pl.Series("stock_hour6_22_cnt", stock_counts),
        pl.Series("controlled_eligible", eligible),
        pl.Series("controlled_censored", censored),
        pl.Series("controlled_mask", hidden).arr.to_list(),
    )
