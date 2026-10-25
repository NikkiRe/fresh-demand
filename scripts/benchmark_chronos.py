#!/usr/bin/env python3
"""Optional, CPU-only, fit-free Chronos-2 benchmark on a frozen test cohort.

Install separately from the classic-ML environment:
    uv pip install --python work/chronos-venv/bin/python \
        'chronos-forecasting==2.3.2' 'polars==2.1.0' pyarrow

Official API and model card:
    https://huggingface.co/amazon/chronos-2
    https://github.com/amazon-science/chronos-forecasting

The model uses raw sales during 06:00--21:00 and no covariates. Each forecast
origin sees only earlier observations, including realized sales from earlier
test days. This is daily-updated H=1 evaluation, not a seven-day open-loop
forecast. No model fitting, tuning, or recovery-profile fitting occurs here.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import polars as pl


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from fresh_demand.evaluation import inventory_metrics, pinball, regression_metrics


MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
KEYS = ["store_id", "product_id"]
QUANTILES = [0.5, 0.75, 0.9]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dated(frame: pl.DataFrame) -> pl.DataFrame:
    expression = (
        pl.col("dt").str.to_date()
        if frame.schema["dt"] == pl.String
        else pl.col("dt").cast(pl.Date)
    )
    return frame.with_columns(expression.alias("__date"))


def select_cohort(series_path: Path, max_series: int, seed: int) -> pl.DataFrame:
    """Sample IDs only from an already frozen train-only series manifest."""
    if max_series < 1:
        raise ValueError("max-series must be positive")
    pairs = pl.read_parquet(series_path).select(KEYS).unique().sort(KEYS)
    if pairs.height > max_series:
        indices = np.sort(np.random.default_rng(seed).choice(
            pairs.height, max_series, replace=False,
        ))
        pairs = pairs[indices]
    if not pairs.height:
        raise ValueError("Series manifest is empty")
    return pairs


def load_panel(data_dir: Path, pairs: pl.DataFrame) -> pl.DataFrame:
    panels = []
    for name in ["train.parquet", "eval.parquet"]:
        panel = (
            pl.scan_parquet(data_dir / name)
            .join(pairs.lazy(), on=KEYS, how="semi")
            .select(KEYS + ["dt", "hours_sale"])
            .collect()
        )
        panels.append(panel)
    panel = dated(pl.concat(panels)).with_columns(
        pl.col("hours_sale").list.slice(6, 16).list.sum()
        .cast(pl.Float32).alias("raw_target"),
    ).sort(KEYS + ["__date"])
    if panel.select(pl.struct(KEYS + ["__date"]).is_duplicated().any()).item():
        raise ValueError("Duplicate store-product-date rows in raw input")
    return panel


def history_inputs(
    panel: pl.DataFrame, origin, pairs: pl.DataFrame, lookback: int = 90,
) -> list[np.ndarray]:
    """Build historical contexts; never read the origin's outcome as input."""
    history = panel.filter(
        (pl.col("__date") < origin)
        & (pl.col("__date") >= origin - timedelta(days=lookback))
    )
    grouped = {
        (row["store_id"], row["product_id"]): row
        for row in history.group_by(KEYS).agg(
            pl.col("__date").sort(),
            pl.col("raw_target").sort_by("__date"),
        ).iter_rows(named=True)
    }
    contexts = []
    expected_dates = [origin - timedelta(days=days) for days in range(lookback, 0, -1)]
    for store_id, product_id in pairs.iter_rows():
        row = grouped.get((store_id, product_id))
        if row is None or row["__date"] != expected_dates:
            raise ValueError(
                f"Expected {lookback} consecutive past days for "
                f"store={store_id}, product={product_id}, origin={origin}"
            )
        values = np.asarray(row["raw_target"], dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Historical operating-hour sales must be finite and nonnegative")
        contexts.append(values)
    return contexts


def forecast_one_day(pipeline, inputs: list[np.ndarray], batch_size: int) -> np.ndarray:
    """Use the official tensor/list API; returned point is the q50 median."""
    quantiles, _ = pipeline.predict_quantiles(
        inputs,
        prediction_length=1,
        quantile_levels=QUANTILES,
        context_length=90,
        batch_size=batch_size,
        cross_learning=False,
    )
    result = np.vstack([
        prediction.detach().cpu().numpy()[0, 0, :] for prediction in quantiles
    ])
    if result.shape != (len(inputs), len(QUANTILES)) or not np.isfinite(result).all():
        raise ValueError("Unexpected Chronos-2 forecast output")
    # Demand is nonnegative; apply the same lower bound as the tree forecasts.
    return np.maximum(result, 0)


def metrics_on_same_cohort(predictions: pl.DataFrame) -> dict:
    observed = predictions.filter(pl.col("is_observed"))
    if not observed.height:
        raise ValueError("Selected cohort has no fully available test days")
    y = observed["raw_target"].to_numpy()
    point_columns = [
        name for name in observed.columns
        if name.startswith("pred_")
        and (name.endswith("_q50") or name in {"pred_seasonal_lag7", "pred_mean7"})
    ]
    points = {
        name: regression_metrics(y, observed[name].to_numpy())
        for name in sorted(point_columns)
    }
    quantile_metrics = {}
    for name in sorted(observed.columns):
        match = re.fullmatch(r"pred_(.+)_q(50|75|90)", name)
        if match:
            quantile = int(match.group(2)) / 100
            order = observed[name].to_numpy()
            underage = quantile / (1 - quantile)
            quantile_metrics[name] = {
                "quantile": quantile,
                "pinball_loss": pinball(y, order, quantile),
                "assumed_underage_cost": underage,
                "assumed_overage_cost": 1,
                **inventory_metrics(y, order, underage=underage, overage=1),
            }
    return {
        "rows_all": predictions.height,
        "rows_fully_available": observed.height,
        "series_all": predictions.select(pl.struct(KEYS).n_unique()).item(),
        "stores_fully_available": observed["store_id"].n_unique(),
        "test_dates": sorted(str(value) for value in observed["__date"].unique()),
        "point_metrics": points,
        "quantile_metrics": quantile_metrics,
        "evaluation_scope": (
            "All models use the identical sampled series and fully available "
            "operating-hour test days; this selected cohort does not identify "
            "latent demand on natural stockout days."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--series-file", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="JSON metrics output")
    parser.add_argument("--max-series", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("work/hf-cache"))
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Synthetic sequence only; no test data read")
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch-size must be positive")
    if not args.smoke and any(
        value is None for value in [args.data_dir, args.predictions, args.series_file]
    ):
        parser.error("data-dir, predictions, and series-file are required without --smoke")

    started = time.perf_counter()
    os.environ["HF_HOME"] = str(args.cache_dir.resolve())
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch
    from chronos import Chronos2Pipeline
    from huggingface_hub import snapshot_download

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    load_started = time.perf_counter()
    # Resolve exactly the pinned snapshot first. Loading its local directory
    # also prevents the library's adapter probe from querying an unpinned main.
    snapshot = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=args.model_revision,
        allow_patterns=["config.json", "model.safetensors"],
        cache_dir=str(args.cache_dir.resolve() / "hub"),
        token=False,
        local_files_only=args.local_files_only,
    ))
    pipeline = Chronos2Pipeline.from_pretrained(
        snapshot,
        device_map="cpu",
        dtype=torch.float32,
        token=False,
        local_files_only=True,
    )
    pipeline.model.eval()
    load_seconds = time.perf_counter() - load_started
    metadata = {
        "model": MODEL_ID,
        "model_revision": args.model_revision,
        "model_weights_sha256": sha256(snapshot / "model.safetensors"),
        "model_weights_bytes": (snapshot / "model.safetensors").stat().st_size,
        "model_license": "Apache-2.0",
        "model_parameters": int(sum(value.numel() for value in pipeline.model.parameters())),
        "model_card": "https://huggingface.co/amazon/chronos-2",
        "device": "cpu",
        "dtype": "float32",
        "threads": args.threads,
        "batch_size": args.batch_size,
        "context_days": 90,
        "horizon_days": 1,
        "cross_learning": False,
        "future_covariates": False,
        "fine_tuning": False,
        "used_for_deployment_selection": False,
        "pretraining_overlap_independently_audited": False,
        "quantiles": QUANTILES,
        "model_load_seconds": load_seconds,
        "versions": {
            name: version(name) for name in ["chronos-forecasting", "torch", "transformers", "numpy", "polars"]
        },
        "pretraining_overlap_caveat": (
            "No task-specific fitting is performed. FreshRetailNet is not "
            "named in the published Appendix A training list, but exact "
            "pretraining overlap has not been independently audited. This "
            "benchmark does not establish strictly unseen-dataset performance."
        ),
        "training_data_reference": "https://arxiv.org/html/2510.15821v1#A1",
    }
    if args.smoke:
        synthetic = 2 + np.sin(np.arange(90, dtype=np.float32) * 2 * np.pi / 7)
        infer_started = time.perf_counter()
        forecast = forecast_one_day(pipeline, [synthetic], args.batch_size)
        result = {
            **metadata,
            "mode": "synthetic_api_smoke",
            "test_data_read": False,
            "synthetic_forecast": forecast.tolist(),
            "inference_seconds": time.perf_counter() - infer_started,
        }
    else:
        pairs = select_cohort(args.series_file, args.max_series, args.seed)
        panel = load_panel(args.data_dir, pairs)
        baseline = dated(pl.read_parquet(args.predictions)).join(pairs, on=KEYS, how="semi")
        required = set(KEYS + ["__date", "raw_target", "is_observed", "pred_raw_q50", "pred_profile_q50"])
        if required.difference(baseline.columns):
            raise ValueError(f"Missing test prediction columns: {sorted(required.difference(baseline.columns))}")
        dates = sorted(baseline["__date"].unique())
        if len(dates) != 7:
            raise ValueError(f"Expected seven frozen test dates, found {len(dates)}")
        if baseline.height != pairs.height * len(dates):
            raise ValueError("Frozen predictions do not cover every sampled series on each test date")
        forecast_frames = []
        inference_seconds = 0.0
        per_origin_seconds = {}
        for origin in dates:
            inputs = history_inputs(panel, origin, pairs)
            infer_started = time.perf_counter()
            forecast = forecast_one_day(pipeline, inputs, args.batch_size)
            duration = time.perf_counter() - infer_started
            inference_seconds += duration
            per_origin_seconds[str(origin)] = duration
            forecast_frames.append(pairs.with_columns(
                pl.lit(origin).alias("__date"),
                *[pl.Series(f"pred_chronos_q{int(q * 100)}", forecast[:, index])
                  for index, q in enumerate(QUANTILES)],
            ))
            print(json.dumps({"forecast_origin": str(origin), "series": pairs.height, "inference_seconds": duration}), flush=True)
        scored = baseline.join(pl.concat(forecast_frames), on=KEYS + ["__date"], how="inner", validate="1:1")
        result = {
            **metadata,
            "mode": "frozen_test_daily_updated_h1",
            "sampling_seed": args.seed,
            "sampling_rule": "Uniform fixed subset of the saved train-only sampled_series.parquet IDs",
            "source_predictions_sha256": sha256(args.predictions),
            "source_series_sha256": sha256(args.series_file),
            "inference_seconds": inference_seconds,
            "per_origin_inference_seconds": per_origin_seconds,
            **metrics_on_same_cohort(scored),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored.write_parquet(args.output.with_name(args.output.stem + "_predictions.parquet"))
        pairs.write_parquet(args.output.with_name(args.output.stem + "_series.parquet"))
    result["total_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
