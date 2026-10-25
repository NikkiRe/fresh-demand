"""Train an inference bundle using past history and an already frozen selection.

This command consumes the supplied history only. It does not open the official
evaluation file, select recovery methods, or tune model hyperparameters.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .data import KEYS, file_hash
from .features import build_features
from .pipeline import fit_model, log
from .recovery import HourlyRecovery


def retrain(
    history_path: Path,
    selection_path: Path,
    config_path: Path,
    model_path: Path,
) -> dict:
    """Refit the preselected method and save a ``forecast``-compatible bundle.

    Every supplied series needs at least 35 consecutive days ending on the
    common latest date. The frozen selection must refer to the exact config
    bytes supplied here, so retraining cannot silently change model settings.
    A JSON sidecar records input hashes, fitted dates, and sample sizes.
    """
    config = json.loads(config_path.read_text())
    selection = json.loads(selection_path.read_text())
    config_sha256 = file_hash(config_path)
    if selection.get("config_sha256") != config_sha256:
        raise ValueError("Config hash must match the frozen selection config_sha256")
    method = selection.get("selected_method")
    if method not in {"raw", "uniform", "profile"}:
        raise ValueError("Frozen selected_method must be raw, uniform, or profile")
    if config.get("quantiles") != [0.5, 0.75, 0.9]:
        raise ValueError("Retrain requires the predeclared quantiles [0.5, 0.75, 0.9]")
    warmup_days = config.get("warmup_days")
    if isinstance(warmup_days, bool) or not isinstance(warmup_days, int) or warmup_days < 1:
        raise ValueError("warmup_days must be a positive integer")

    history = pl.read_parquet(history_path).sort(KEYS + ["dt"])
    if history.is_empty():
        raise ValueError("Retrain history must not be empty")
    if history["dt"].null_count():
        raise ValueError("History dates must not contain nulls")
    # forecast() uses ISO date strings in its bundle and placeholder rows.
    if history.schema["dt"] == pl.Date:
        history = history.with_columns(pl.col("dt").cast(pl.String))
    minimum_history = max(35, warmup_days + 1)
    series_counts = history.group_by(KEYS).len()
    if series_counts["len"].min() < minimum_history:
        raise ValueError(f"Retrain requires at least {minimum_history} days per series")
    train_end = history["dt"].max()
    if history.filter(pl.col("dt") == train_end).height != series_counts.height:
        raise ValueError("Every retraining series must end on the latest history date")
    dates = history["dt"].unique().sort().to_list()
    warmup_end = dates[warmup_days]
    recovery = HourlyRecovery().fit(history.filter(pl.col("dt") < warmup_end))
    target = recovery.transform(history, method)
    data, features = build_features(history, target, warmup_days=warmup_days)
    # build_features validates duplicate keys and daily calendar continuity.
    fitting = data.filter(pl.col("__warmup_complete"))
    if fitting.is_empty():
        raise ValueError("No fitting rows remain after the recovery warmup")
    x = fitting.select(features).to_numpy().astype(np.float32)
    y = fitting["target"].to_numpy()
    models = {}
    for q in config["quantiles"]:
        log(f"retrain {method}, q={q:.2f}, train_rows={len(y)}")
        models[f"q{round(q * 100)}"] = fit_model(x, y, q, config)
    bundle = {
        "models": models,
        "method": method,
        "recovery": recovery,
        "features": features,
        "config": config,
        "train_end": train_end,
        "training_start": history["dt"].min(),
        "series": history.select(KEYS).unique().sort(KEYS),
    }
    metadata = {
        "operation": "retrain_frozen_selection",
        "selected_method": method,
        "history_sha256": file_hash(history_path),
        "selection_sha256": file_hash(selection_path),
        "config_sha256": config_sha256,
        "history_rows": history.height,
        "fitting_rows": fitting.height,
        "n_series": series_counts.height,
        "training_start": bundle["training_start"],
        "train_end": train_end,
        "fitting_start": fitting["dt"].min(),
        "recovery_warmup_end_exclusive": warmup_end,
        "recovery": recovery.diagnostics_,
        "quantiles": config["quantiles"],
        "official_evaluation_read": False,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        pickle.dump(bundle, stream)
    model_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return metadata
