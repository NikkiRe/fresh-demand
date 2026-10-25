"""Expanding-window validation, frozen selection, and a single held-out test."""

from __future__ import annotations

import importlib.metadata
import json
import pickle
import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .data import KEYS, REVISION, audit, file_hash, read_subset, select_series
from .evaluation import (
    inventory_metrics, make_censoring, paired_store_bootstrap,
    pinball, regression_metrics,
)
from .features import ID_FEATURES, build_features
from .recovery import HourlyRecovery

METHODS = ("raw", "uniform", "profile")


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def write_json(path: Path, value) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n")


def fit_model(x: np.ndarray, y: np.ndarray, q: float, config: dict):
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=q,
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        num_leaves=config["num_leaves"],
        min_child_samples=config["min_child_samples"],
        n_jobs=config["threads"], random_state=config["seed"],
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    model.fit(x, y, categorical_feature=list(range(len(ID_FEATURES))))
    return model


def score_predictions(predictions: pl.DataFrame) -> dict:
    observed = predictions.filter(pl.col("is_observed"))
    y = observed["raw_target"].to_numpy()
    result = {
        "n_rows": observed.height,
        "n_total_rows": predictions.height,
        "n_stores": observed["store_id"].n_unique(),
        "n_days": observed["dt"].n_unique(),
        "demand_sum": float(y.sum()),
        "models": {},
    }
    for column in predictions.columns:
        if column.startswith("pred_"):
            estimate = observed[column].to_numpy()
            metrics = regression_metrics(y, estimate)
            metrics.update(inventory_metrics(y, estimate, underage=3, overage=1))
            if "_q" in column:
                q = int(column.rsplit("_q", 1)[1]) / 100
                metrics["pinball"] = pinball(y, estimate, q)
                metrics["quantile"] = q
                metrics["empirical_coverage"] = float((y <= estimate).mean())
            result["models"][column] = metrics
    return result


def prepare_features(frame: pl.DataFrame, recovery: HourlyRecovery, method: str, config: dict):
    target = recovery.transform(frame, method=method)
    return build_features(frame, target, warmup_days=config["warmup_days"])


def predict_window(
    context: pl.DataFrame,
    start: str,
    end: str,
    recovery: HourlyRecovery,
    config: dict,
    quantiles: list[float],
    save_models: bool = False,
):
    result = None
    models = {}
    feature_names = None
    for method in METHODS:
        data, feature_names = prepare_features(context, recovery, method, config)
        fitting = data.filter(pl.col("__warmup_complete") & (pl.col("dt") < start))
        future = data.filter((pl.col("dt") >= start) & (pl.col("dt") <= end))
        if fitting.is_empty() or future.is_empty():
            raise ValueError("Empty fitting or evaluation window")
        if result is None:
            result = future.select(KEYS + ["dt", "raw_target", "is_observed"])
            naive = data.with_columns(
                pl.col("raw_target").shift(7).over(KEYS).alias("pred_seasonal_lag7"),
                pl.col("raw_target").shift(1).rolling_mean(7, min_samples=1).over(KEYS).alias("pred_mean7"),
            ).filter((pl.col("dt") >= start) & (pl.col("dt") <= end))
            result = result.with_columns(
                naive["pred_seasonal_lag7"].clip(lower_bound=0),
                naive["pred_mean7"].clip(lower_bound=0),
            )
        x_train = fitting.select(feature_names).to_numpy().astype(np.float32)
        x_future = future.select(feature_names).to_numpy().astype(np.float32)
        y_train = fitting["target"].to_numpy()
        log(f"window {start}..{end}, {method}, train_rows={len(y_train)}")
        method_predictions = []
        for q in quantiles:
            started = time.perf_counter()
            model = fit_model(x_train, y_train, q, config)
            prediction = np.maximum(model.predict(x_future), 0)
            method_predictions.append(prediction)
            models[f"{method}_q{round(q*100)}"] = model
            log(f"  q={q:.2f}: fit+predict {time.perf_counter()-started:.1f}s")
        # Sorting predictions is a predeclared quantile-crossing correction.
        # Validation and test use the same quantiles to keep q75 comparable.
        ordered = np.sort(np.stack(method_predictions, axis=1), axis=1)
        for i, q in enumerate(quantiles):
            result = result.with_columns(pl.Series(f"pred_{method}_q{round(q*100)}", ordered[:, i]))
    return result, models if save_models else {}, feature_names


def recovery_experiment(frame: pl.DataFrame, train: pl.DataFrame, recovery: HourlyRecovery, seed: int) -> dict:
    observed = frame.filter(pl.col("hours_stock_status").list.slice(6, 16).list.sum() == 0)
    truth = recovery.transform(observed, "raw")
    masks = train["hours_stock_status"].list.to_array(24).to_numpy()
    output = {"n_observed_days": observed.height, "regimes": {}}
    for regime in ("empirical", "suffix", "peak"):
        hidden = make_censoring(observed, masks, regime=regime, seed=seed)
        output["regimes"][regime] = {
            method: regression_metrics(truth, recovery.transform(hidden, method))
            for method in METHODS
        }
    return output


def run(data_dir: Path, config_path: Path, output: Path, final_test: bool = False) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    if config["quantiles"] != sorted(config["quantiles"]) or 0.75 not in config["quantiles"]:
        raise ValueError("Quantiles must be sorted and include primary quantile 0.75")
    if config["primary_quantile"] != 0.75:
        raise ValueError("Primary cost ratio is predeclared as 3:1 (q75)")
    existing = output / "test_summary.json"
    if final_test and existing.exists():
        raise FileExistsError("Held-out test already exists in this run directory; preserve it and use a new explicit experiment")
    pairs = select_series(data_dir / "train.parquet", config["max_series"], config["seed"])
    pairs.write_parquet(output / "sampled_series.parquet")
    train = read_subset(data_dir / "train.parquet", pairs)
    dates = sorted(train["dt"].unique().to_list())
    if len(dates) <= config["warmup_days"] + config["validation_windows"]*config["validation_days"]:
        raise ValueError("Not enough history for warmup and validation")
    warmup_end = dates[config["warmup_days"]]
    warmup = train.filter(pl.col("dt") < warmup_end)
    recovery = HourlyRecovery().fit(warmup)
    metadata = {
        "dataset_revision": REVISION,
        "config": config,
        "train_sha256": file_hash(data_dir / "train.parquet"),
        "config_sha256": file_hash(config_path),
        "n_series": pairs.height,
        "n_train_rows": train.height,
        "sampled_stores": train["store_id"].n_unique(),
        "recovery_warmup_end_exclusive": warmup_end,
        "recovery": recovery.diagnostics_,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "polars", "lightgbm", "scikit-learn", "pyarrow"]},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_protocol": "One day ahead; model refit at each weekly origin, realized history updated daily",
        "target": "Normalized sales sum in hours [6,22); evaluation only on fully available days",
    }
    folds = []
    validation_predictions = []
    for fold in range(config["validation_windows"]):
        offset = len(dates) - (config["validation_windows"]-fold)*config["validation_days"]
        first = dates[offset]
        last = dates[offset+config["validation_days"]-1]
        context = train.filter(pl.col("dt") <= last)
        predictions, _, _ = predict_window(context, first, last, recovery, config, config["quantiles"])
        metrics = score_predictions(predictions)
        folds.append({"start": first, "end": last, **metrics})
        validation_predictions.append(predictions.with_columns(pl.lit(fold).alias("fold")))
    costs = {
        method: float(np.mean([fold["models"][f"pred_{method}_q75"]["normalized_cost"] for fold in folds]))
        for method in METHODS
    }
    selected = min(METHODS, key=lambda method: costs[method])
    frozen = {
        "selected_method": selected,
        "selection_metric": "Mean validation-window normalized cost, Cu=3, Co=1, q=0.75",
        "validation_costs": costs,
        "config_sha256": metadata["config_sha256"],
        "frozen_before_test_read_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "frozen_selection.json", frozen)
    validation = {"folds": folds, "selection": frozen, "pooled": score_predictions(pl.concat(validation_predictions))}
    write_json(output / "validation_summary.json", validation)
    pl.concat(validation_predictions).write_parquet(output / "validation_predictions.parquet")
    # Controlled recovery assessment also uses held-out validation days, never
    # their labels for fitting profiles or selecting a synthetic mask.
    validation_start = folds[0]["start"]
    recovery_results = recovery_experiment(
        train.filter(pl.col("dt") >= validation_start),
        train.filter(pl.col("dt") < validation_start), recovery, config["seed"],
    )
    write_json(output / "recovery_validation.json", recovery_results)
    summary = {"metadata": metadata, "validation": validation, "recovery_validation": recovery_results}
    log(f"Frozen selection: {selected}, validation costs={costs}")
    if final_test:
        # This is the first read of evaluation data in the training workflow.
        test = read_subset(data_dir / "eval.parquet", pairs)
        if min(test["dt"]) <= max(train["dt"]):
            raise ValueError("Evaluation dates overlap training")
        context = pl.concat([train, test]).sort(KEYS + ["dt"])
        predictions, models, features = predict_window(
            context, min(test["dt"]), max(test["dt"]), recovery, config,
            config["quantiles"], save_models=True,
        )
        test_metrics = score_predictions(predictions)
        observed = predictions.filter(pl.col("is_observed"))
        comparisons = {}
        for method in ("uniform", "profile"):
            comparisons[f"{method}_vs_raw"] = paired_store_bootstrap(
                observed, "raw_target", f"pred_{method}_q75", "pred_raw_q75",
                n_boot=config["bootstrap_replicates"], seed=config["seed"],
            )
        sensitivity = {}
        for q, underage in [(0.5, 1), (0.75, 3), (0.9, 9)]:
            sensitivity[str(underage)] = {
                method: inventory_metrics(
                    observed["raw_target"].to_numpy(), observed[f"pred_{method}_q{round(q*100)}"].to_numpy(),
                    underage=underage, overage=1,
                ) for method in METHODS
            }
        test_summary = {
            **test_metrics, "start": min(test["dt"]), "end": max(test["dt"]),
            "bootstrap": comparisons, "cost_sensitivity": sensitivity,
            "test_sha256": file_hash(data_dir / "eval.parquet"),
            "selection": frozen,
        }
        policy_comparison = paired_store_bootstrap(
            observed, "raw_target", "pred_raw_q75", "pred_raw_q50",
            n_boot=config["bootstrap_replicates"], seed=config["seed"],
        )
        test_summary["policy_comparison"] = policy_comparison
        write_json(output / "policy_comparison.json", policy_comparison)
        write_json(output / "test_summary.json", test_summary)
        predictions.write_parquet(output / "test_predictions.parquet")
        write_json(output / "recovery_test.json", recovery_experiment(test, train, recovery, config["seed"]))
        artifacts = output / "artifacts"
        artifacts.mkdir(exist_ok=True)
        bundle = {
            "models": {f"q{round(q*100)}": models[f"{selected}_q{round(q*100)}"] for q in config["quantiles"]},
            "method": selected, "recovery": recovery, "features": features,
            "config": config, "train_end": max(train["dt"]),
            "training_start": min(train["dt"]), "series": pairs,
        }
        with (artifacts / "model.pkl").open("wb") as stream:
            pickle.dump(bundle, stream)
        importance_model = models[f"{selected}_q75"]
        pl.DataFrame({"feature": features, "gain": importance_model.booster_.feature_importance(importance_type="gain")}).sort("gain", descending=True).write_csv(output / "feature_importance.csv")
        summary["test"] = test_summary
        log(f"Held-out test completed: {test_metrics['n_rows']} fully available rows")
    metadata["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "run_summary.json", summary)
    log(f"Run complete in {metadata['elapsed_seconds']:.1f}s")
    return summary


def forecast(model_path: Path, history_path: Path, destination: Path) -> pl.DataFrame:
    # Only load model files generated locally or obtained from a trusted source.
    with model_path.open("rb") as stream:
        bundle = pickle.load(stream)
    history = pl.read_parquet(history_path).sort(KEYS + ["dt"])
    if history.is_empty() or history.group_by(KEYS).len()["len"].min() < 28:
        raise ValueError("Predict requires at least 28 consecutive observed days per series")
    last_date = history["dt"].max()
    if last_date < bundle["train_end"]:
        raise ValueError("History ends before model training end")
    next_date = (datetime.fromisoformat(last_date).date() + timedelta(days=1)).isoformat()
    latest = history.filter(pl.col("dt") == last_date)
    if latest.height != history.select(pl.struct(KEYS).n_unique()).item():
        raise ValueError("Every series must have observations on the latest history date")
    future = latest.with_columns(
        pl.lit(next_date).alias("dt"),
        pl.lit([0.0]*24).alias("hours_sale"),
        pl.lit([0]*24).cast(pl.List(history.schema["hours_stock_status"].inner)).alias("hours_stock_status"),
        pl.lit(0.0).alias("sale_amount"),
        pl.lit(0).cast(history.schema["stock_hour6_22_cnt"]).alias("stock_hour6_22_cnt"),
    )
    context = pl.concat([history, future])
    data, features = prepare_features(context, bundle["recovery"], bundle["method"], bundle["config"])
    if features != bundle["features"]:
        raise ValueError("Feature schema differs from saved model")
    # day_index must use the original training origin even when predict receives
    # a shorter recent-history file.
    data = data.with_columns((pl.col("__date") - pl.lit(datetime.fromisoformat(bundle["training_start"]).date())).dt.total_days().cast(pl.Int32).alias("day_index"))
    prediction_rows = data.filter(pl.col("dt") == next_date)
    x = prediction_rows.select(features).to_numpy().astype(np.float32)
    quantile_predictions = np.sort(np.stack([np.maximum(model.predict(x), 0) for model in bundle["models"].values()], axis=1), axis=1)
    result = prediction_rows.select(KEYS + ["dt"]).with_columns([
        pl.Series(f"demand_{name}", quantile_predictions[:, i]) for i, name in enumerate(bundle["models"])
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".csv":
        result.write_csv(destination)
    else:
        result.write_parquet(destination)
    return result
