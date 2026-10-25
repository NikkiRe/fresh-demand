"""Frozen, training-only heuristics for recovering censored hourly demand.

The operational target is hours 06:00 through 21:00, inclusive. A stock-status
value of one denotes absence. Recovered values are estimates, never labels of
observed counterfactual demand. The seven pseudo-days of profile shrinkage,
0.2 minimum exposure share and five-prior cap are predeclared arbitrary safety
choices, rather than parameters identified from the data or tuned on test days.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


class HourlyRecovery:
    """Estimate 16-hour demand using a frozen intraday availability profile.

    Fit profiles on positive, completely available warmup days only. Product
    profiles shrink toward third-category profiles, which shrink toward the
    global profile. Priors use mean daily demand at store/product level; a group
    with no completely available warmup day uses its observed-sales mean.
    Unseen pairs fall back to product, then global priors. Calling ``transform``
    neither updates these estimates nor uses other rows from its input.
    """

    def __init__(self, smoothing_days: float = 7.0) -> None:
        if not np.isfinite(smoothing_days) or smoothing_days <= 0:
            raise ValueError("smoothing_days must be positive and finite")
        self.smoothing_days = float(smoothing_days)
        self.min_share = 0.2
        self.prior_cap_multiplier = 5.0
        self._fitted = False

    @staticmethod
    def _matrix(frame: pl.DataFrame, column: str) -> np.ndarray:
        if column not in frame.columns:
            raise ValueError(f"Missing required column: {column}")
        if frame.height == 0:
            return np.empty((0, 16), dtype=np.float64)
        series = frame.get_column(column)
        # The List -> fixed Array conversion lets Polars construct a dense NumPy
        # matrix without first creating one Python list for every row.
        if isinstance(series.dtype, pl.Array):
            series = series.cast(pl.List(pl.Float64))
        try:
            matrix = (
                series.list.slice(6, 16)
                .list.to_array(16)
                .to_numpy()
                .astype(np.float64, copy=False)
            )
        except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{column} must contain numeric hourly lists with at least 22 entries"
            ) from exc
        if matrix.shape != (frame.height, 16) or not np.isfinite(matrix).all():
            raise ValueError(f"{column} contains missing or non-finite operational values")
        return matrix

    @classmethod
    def _sales(cls, frame: pl.DataFrame) -> np.ndarray:
        sales = cls._matrix(frame, "hours_sale")
        if (sales < 0).any():
            raise ValueError("hours_sale must be nonnegative")
        return sales

    @classmethod
    def _availability(cls, frame: pl.DataFrame) -> np.ndarray:
        status = cls._matrix(frame, "hours_stock_status")
        if not np.isin(status, [0.0, 1.0]).all():
            raise ValueError("hours_stock_status must contain only 0 or 1")
        return status == 0

    @staticmethod
    def _category_column(frame: pl.DataFrame) -> str:
        if "third_category_id" in frame.columns:
            return "third_category_id"
        if "third_category" in frame.columns:
            return "third_category"
        raise ValueError("Missing required column: third_category_id")

    @staticmethod
    def _mean_lookup(
        frame: pl.DataFrame, keys: list[str], mask: np.ndarray | None = None
    ) -> dict[Any, float]:
        selected = frame if mask is None else frame.filter(pl.Series(mask))
        grouped = selected.group_by(keys).agg(pl.col("_raw").mean())
        return {
            (tuple(row[:-1]) if len(keys) > 1 else row[0]): float(row[-1])
            for row in grouped.iter_rows()
        }

    def fit(self, warmup: pl.DataFrame) -> HourlyRecovery:
        """Learn profiles and priors from an explicitly supplied past-only frame."""
        if warmup.height == 0:
            raise ValueError("warmup must contain at least one row")
        for key in ["store_id", "product_id"]:
            if key not in warmup.columns:
                raise ValueError(f"Missing required column: {key}")
            if warmup.get_column(key).null_count():
                raise ValueError(f"{key} must not contain nulls")
        category_column = self._category_column(warmup)
        sales = self._sales(warmup)
        available = self._availability(warmup)
        raw = sales.sum(axis=1)
        complete = available.all(axis=1)
        positive_complete = complete & (raw > 0)

        priors = warmup.select("store_id", "product_id").with_columns(
            pl.Series("_raw", raw)
        )
        pair_observed = self._mean_lookup(priors, ["store_id", "product_id"])
        product_observed = self._mean_lookup(priors, ["product_id"])
        pair_complete = self._mean_lookup(priors, ["store_id", "product_id"], complete)
        product_complete = self._mean_lookup(priors, ["product_id"], complete)
        self._pair_priors = pair_observed | pair_complete
        self._product_priors = product_observed | product_complete
        self._global_prior = float(raw[complete].mean() if complete.any() else raw.mean())

        shape = sales[positive_complete] / raw[positive_complete, None]
        self._global_profile = (
            shape.mean(axis=0) if shape.shape[0] else np.full(16, 1.0 / 16)
        )
        shape_columns = [f"_h{i}" for i in range(16)]
        shapes = warmup.select("product_id", category_column).filter(
            pl.Series(positive_complete)
        ).with_columns(
            [pl.Series(name, shape[:, i]) for i, name in enumerate(shape_columns)]
        )
        aggregations = [pl.len().alias("_n")] + [
            pl.col(name).sum() for name in shape_columns
        ]
        self._category_profiles: dict[Any, np.ndarray] = {}
        for row in shapes.group_by(category_column).agg(aggregations).iter_rows(named=True):
            count = row["_n"]
            total_shape = np.array([row[name] for name in shape_columns], dtype=float)
            self._category_profiles[row[category_column]] = (
                total_shape + self.smoothing_days * self._global_profile
            ) / (count + self.smoothing_days)

        # Product IDs have a stable taxonomy in FreshRetailNet. Taking the first
        # warmup category also keeps all fitted state independent of future rows.
        product_categories = dict(
            warmup.select("product_id", category_column)
            .unique(subset=["product_id"], keep="first", maintain_order=True)
            .iter_rows()
        )
        self._product_profiles: dict[Any, np.ndarray] = {}
        for row in shapes.group_by("product_id").agg(aggregations).iter_rows(named=True):
            product = row["product_id"]
            parent = self._category_profiles.get(
                product_categories[product], self._global_profile
            )
            total_shape = np.array([row[name] for name in shape_columns], dtype=float)
            self._product_profiles[product] = (
                total_shape + self.smoothing_days * parent
            ) / (row["_n"] + self.smoothing_days)

        self.diagnostics_ = {
            "warmup_rows": int(warmup.height),
            "complete_days": int(complete.sum()),
            "positive_complete_days": int(positive_complete.sum()),
            "product_profiles": len(self._product_profiles),
            "category_profiles": len(self._category_profiles),
            "pair_priors": len(self._pair_priors),
            "pairs_using_observed_prior": len(pair_observed.keys() - pair_complete.keys()),
            "global_prior": self._global_prior,
            "operational_hours": list(range(6, 22)),
            "smoothing_pseudo_days": self.smoothing_days,
            "minimum_share": self.min_share,
            "prior_cap_multiplier": self.prior_cap_multiplier,
            "safety_parameters": "predeclared arbitrary sensitivity assumptions",
        }
        self._fitted = True
        return self

    def transform(self, frame: pl.DataFrame, method: str = "profile") -> np.ndarray:
        """Return raw or recovered 06:00--21:00 demand in input row order."""
        if method not in {"raw", "uniform", "profile"}:
            raise ValueError("method must be raw, uniform, or profile")
        sales = self._sales(frame)
        raw = sales.sum(axis=1)
        if method == "raw":
            return raw
        if not self._fitted:
            raise RuntimeError("Call fit(warmup) before recovering demand")
        if frame.height == 0:
            return raw
        available = self._availability(frame)
        products = frame.get_column("product_id").to_list()
        stores = frame.get_column("store_id").to_list()
        prior = np.fromiter(
            (
                self._pair_priors.get(
                    (store, product), self._product_priors.get(product, self._global_prior)
                )
                for store, product in zip(stores, products, strict=True)
            ),
            dtype=float,
            count=frame.height,
        )
        if method == "uniform":
            share = available.mean(axis=1)
        else:
            categories = frame.get_column(self._category_column(frame)).to_list()
            profiles = np.stack(
                [
                    self._product_profiles.get(
                        product,
                        self._category_profiles.get(category, self._global_profile),
                    )
                    for product, category in zip(products, categories, strict=True)
                ]
            )
            share = np.einsum("ij,ij->i", profiles, available)
        observed_available = np.einsum("ij,ij->i", sales, available)
        recovered = observed_available / np.maximum(share, self.min_share)
        recovered = np.where(share > 0, recovered, prior)
        recovered = np.minimum(recovered, np.maximum(raw, self.prior_cap_multiplier * prior))
        recovered = np.maximum(raw, recovered)
        # Avoid tiny floating-point increases when all hours are observed.
        return np.where(available.all(axis=1), raw, recovered)
