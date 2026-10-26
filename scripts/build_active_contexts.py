#!/usr/bin/env python3
"""Fit explicitly semi-synthetic inventory contexts from TRAIN-only retail data.

Features come from the first 28 train days; empirical PMFs use only fully
available operating-hour sales within the first 69 train days. The official
eval file is never opened. This estimates a distribution of selected observed
sales, not latent demand on natural stockout days.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["store_id", "product_id"]
sys.path.insert(0, str(PROJECT / "src"))
from fresh_demand.active.environments import make_environment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_context_model(
    train_path: Path,
    series_file: Path | None = None,
    max_series: int = 10000,
    feature_days: int = 28,
    fit_days: int = 69,
    n_contexts: int = 4,
    max_demand: int = 40,
    scale_target: float = 10.0,
    seed: int = 42,
    threads: int = 4,
    scale_mode: str = "tail_guard",
) -> dict:
    """Return unsmoothed empirical context PMFs and their complete provenance."""
    if not (1 <= feature_days <= fit_days) or min(n_contexts, max_demand, threads) < 1 or scale_target <= 0:
        raise ValueError("Invalid window, context, support, scale, or thread configuration")
    if scale_mode not in {"tail_guard", "median"}:
        raise ValueError("scale_mode must be tail_guard or median")
    train_path = Path(train_path)
    if train_path.name.startswith("eval"):
        raise ValueError("The official evaluation split must not be used to fit contexts")
    source = pl.scan_parquet(train_path)
    schema = source.collect_schema()
    required = set(KEYS + ["dt", "hours_sale", "hours_stock_status"])
    if required.difference(schema.names()):
        raise ValueError(f"Missing train columns: {sorted(required.difference(schema.names()))}")
    if series_file is None:
        pairs = source.select(KEYS).unique().sort(KEYS).collect()
    else:
        pairs = pl.read_parquet(series_file).select(KEYS).unique().sort(KEYS)
    if max_series > 0 and pairs.height > max_series:
        indices = np.sort(np.random.default_rng(seed).choice(pairs.height, max_series, replace=False))
        pairs = pairs[indices]
    if pairs.height < n_contexts:
        raise ValueError("Fewer sampled series than requested contexts")
    date_expression = pl.col("dt").str.to_date() if schema["dt"] == pl.String else pl.col("dt").cast(pl.Date)
    limits = source.select(date_expression.min().alias("first"), date_expression.max().alias("last")).collect().row(0, named=True)
    first = limits["first"]
    fit_end = first + timedelta(days=fit_days - 1)
    feature_end = first + timedelta(days=feature_days - 1)
    if limits["last"] < fit_end:
        raise ValueError("The training file does not cover the requested fit window")
    frame = (
        source.join(pairs.lazy(), on=KEYS, how="semi")
        .with_columns(date_expression.alias("__date"))
        .filter(pl.col("__date") <= fit_end)
        .select(KEYS + ["__date", "hours_sale", "hours_stock_status"])
        .collect()
        .with_columns(
            pl.col("hours_sale").list.slice(6, 16).list.sum().cast(pl.Float64).alias("raw"),
            (pl.col("hours_stock_status").list.slice(6, 16).list.sum() / 16).alias("stockout_hour_share"),
        )
        .with_columns((pl.col("stockout_hour_share") == 0).alias("fully_stocked"))
    )
    if not frame.height:
        raise ValueError("No series rows found in the train fit window")
    for column in ["hours_sale", "hours_stock_status"]:
        if frame.select((pl.col(column).is_null() | (pl.col(column).list.len() != 24)).any()).item():
            raise ValueError(f"{column} must have 24 values per row")
    if frame.select(pl.struct(KEYS + ["__date"]).is_duplicated().any()).item():
        raise ValueError("Duplicate store-product-day rows")
    features = (
        frame.filter(pl.col("__date") <= feature_end).group_by(KEYS).agg(
            pl.col("raw").filter(pl.col("fully_stocked")).mean().alias("complete_mean"),
            pl.col("raw").filter(pl.col("fully_stocked")).std(ddof=0).alias("complete_std"),
            pl.col("raw").mean().alias("raw_mean"),
            pl.col("raw").std(ddof=0).alias("raw_std"),
            pl.col("fully_stocked").sum().alias("complete_feature_days"),
            pl.col("stockout_hour_share").mean().alias("stockout_hour_share"),
        )
        .with_columns(
            pl.coalesce("complete_mean", "raw_mean").alias("feature_mean"),
            pl.coalesce("complete_std", "raw_std").fill_null(0).alias("feature_std"),
        ).sort(KEYS)
    )
    if features.height != pairs.height:
        raise ValueError("Some sampled series have no observations in the feature window")
    matrix = np.column_stack([
        np.log1p(features["feature_mean"].to_numpy()),
        np.log1p(features["feature_std"].to_numpy()),
        features["stockout_hour_share"].to_numpy(),
    ])
    if not np.isfinite(matrix).all() or np.unique(matrix, axis=0).shape[0] < n_contexts:
        raise ValueError("Series features must be finite with enough distinct points for clustering")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    with threadpool_limits(limits=threads):
        kmeans = KMeans(n_clusters=n_contexts, random_state=seed, n_init=10, algorithm="lloyd").fit(scaled)
    order = np.argsort(kmeans.cluster_centers_[:, 0], kind="stable")
    remap = np.empty(n_contexts, dtype=int)
    remap[order] = np.arange(n_contexts)
    assignments = features.select(KEYS).with_columns(pl.Series("context", remap[kmeans.labels_]))
    observed = frame.filter(pl.col("fully_stocked")).join(assignments, on=KEYS, how="inner")
    positive = observed.filter(pl.col("raw") > 0)["raw"].to_numpy()
    if not len(positive):
        raise ValueError("No positive fully-stocked sales for declaring the demand scale")
    median = float(np.median(positive))
    raw_sales = observed["raw"].to_numpy()
    labels = observed["context"].to_numpy()
    raw_q99 = []
    for context in range(n_contexts):
        values = raw_sales[labels == context]
        if not len(values):
            raise ValueError(f"Context {context} has no fully-stocked training outcomes")
        raw_q99.append(float(np.quantile(values, 0.99)))
    tail_reference = max(raw_q99)
    if scale_mode == "tail_guard" and tail_reference <= 0:
        raise ValueError("Tail guard requires a positive 99th percentile in at least one context")
    factor = max_demand / tail_reference if scale_mode == "tail_guard" else scale_target / median
    scaled_sales = raw_sales * factor
    rounded = np.rint(scaled_sales).astype(np.int64)
    clipped = np.clip(rounded, 0, max_demand)
    pmfs, cluster_records = [], []
    inverse_centers = scaler.inverse_transform(kmeans.cluster_centers_[order])
    for context in range(n_contexts):
        mask = labels == context
        if not mask.any():
            raise ValueError(f"Context {context} has no fully-stocked training outcomes")
        counts = np.bincount(clipped[mask], minlength=max_demand + 1)
        mass = counts / counts.sum()
        pmfs.append(mass.tolist())
        cluster_records.append({
            "context": context, "series": int((remap[kmeans.labels_] == context).sum()),
            "fully_stocked_rows": int(mask.sum()), "clipped_row_fraction": float((rounded[mask] > max_demand).mean()),
            "mean_integer_demand": float(mass @ np.arange(max_demand + 1)),
            "q75_integer_demand": int(np.argmax(np.cumsum(mass) >= 0.75)),
            "feature_center_standardized": kmeans.cluster_centers_[order[context]].tolist(),
            "feature_center_unstandardized": inverse_centers[context].tolist(),
            "empirical_counts": counts.tolist(),
        })
    metadata = {
        "source_dataset": "Dingdong-Inc/FreshRetailNet-50K",
        "source_revision": "c374e85975c82816c495acfa904cc55ab14a47d3",
        "source_file": train_path.name,
        "source_sha256": sha256(train_path),
        "source_license": "CC-BY-4.0",
        "source_reference": "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K",
        "source_fields_used": KEYS + ["dt", "hours_sale", "hours_stock_status"],
        "source_original_date_range": [str(first), str(limits["last"])],
        "fit_window": [str(first), str(fit_end)], "fit_days": fit_days,
        "feature_window": [str(first), str(feature_end)], "feature_days": feature_days,
        "official_eval_file_read": False,
        "train_only_series_manifest_sha256": sha256(series_file) if series_file is not None else None,
        "sampling_seed": seed, "sampled_series": pairs.height,
        "sampling_rule": "IDs from a supplied train-only manifest, or uniformly sampled train IDs; no eval outcomes used.",
        "fit_rows": frame.height, "fully_stocked_fit_rows": observed.height,
        "fully_stocked_fit_row_fraction": observed.height / frame.height,
        "series_with_fully_stocked_fit_rows": observed.select(pl.struct(KEYS).n_unique()).item(),
        "feature_series_fallback_to_raw_mean": int((features["complete_feature_days"] == 0).sum()),
        "feature_columns": ["log1p_fully_stocked_mean", "log1p_fully_stocked_std", "stockout_hour_share"],
        "feature_fallback": "If no fully-stocked feature-window day exists, use observed raw mean/std for that series.",
        "clustering": {"method": "StandardScaler then KMeans", "random_state": seed, "n_init": 10,
                       "context_order": "Ascending center of log1p sales mean", "threads": threads,
                       "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()},
        "scale": {"mode": scale_mode, "positive_fully_stocked_sales_median": median,
                  "median_target_units": scale_target if scale_mode == "median" else None,
                  "resulting_scaled_positive_sales_median": median * factor,
                  "raw_q99_by_context": raw_q99,
                  "tail_guard_rule": "One global multiplier = max_demand / max(context raw-sales q99).",
                  "tail_guard_quantile": 0.99,
                  "multiplier": factor, "quantization": "numpy.rint (nearest integer, ties to even)",
                  "clipping_support": [0, max_demand], "clipped_row_fraction": float((rounded > max_demand).mean()),
                  "maximum_context_clipped_row_fraction": max(value["clipped_row_fraction"] for value in cluster_records),
                  "mean_absolute_quantization_error_before_clipping": float(np.abs(scaled_sales - rounded).mean()),
                  "positive_sales_rounded_to_zero_fraction": float(((scaled_sales > 0) & (rounded == 0)).sum() / len(scaled_sales)),
                  "positive_sales_rounded_to_zero_denominator": "All fully-stocked fit rows, including originally zero-sale rows.",
                  "interpretation": "Dimensionless scaled, coarsened normalized operating-hour sales, not physical SKU units or currency."},
        "probability_estimator": "Unsmoothed empirical counts; zero-probability support points remain zero.",
        "clusters": cluster_records,
        "selection_bias": "PMFs condition on observed fully-stocked days; they do not identify demand during naturally censored days.",
        "environment_use": "Semi-synthetic iid resampling within contexts, not chronological replay.",
        "real_policy_counterfactual_identified": False,
        "oracle_pmf_known_only_in_simulator": True,
    }
    return {"pmfs": pmfs, "max_demand": max_demand, "n_contexts": n_contexts, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("work/data/train.parquet"))
    parser.add_argument("--series-file", "--series", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/active/context_model.json")
    parser.add_argument("--max-series", type=int, default=10000)
    parser.add_argument("--feature-days", type=int, default=28)
    parser.add_argument("--fit-days", type=int, default=69)
    parser.add_argument("--n-contexts", type=int, default=4)
    parser.add_argument("--max-demand", type=int, default=40)
    parser.add_argument("--scale-target", type=float, default=10.0)
    parser.add_argument("--scale-mode", choices=["tail_guard", "median"], default="tail_guard",
                        help="tail_guard uses a global multiplier set by the largest context-specific train q99; median reproduces the former scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    model = build_context_model(
        args.train_file, args.series_file, args.max_series, args.feature_days, args.fit_days,
        args.n_contexts, args.max_demand, args.scale_target, args.seed, args.threads, args.scale_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    # Validate the exact serialization consumed by the experiment runner.
    make_environment("fresh_empirical", seed=args.seed, context_path=args.output,
                     n_contexts=args.n_contexts, max_demand=args.max_demand)
    print(json.dumps({"output": str(args.output), "sampled_series": model["metadata"]["sampled_series"],
                      "fully_stocked_fit_rows": model["metadata"]["fully_stocked_fit_rows"],
                      "clipped_row_fraction": model["metadata"]["scale"]["clipped_row_fraction"]}), flush=True)


if __name__ == "__main__":
    main()
