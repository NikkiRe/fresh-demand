"""Sensitivity to censoring threshold, exploration allowance and critical fractile."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import polars as pl

from .experiment import (
    _job, _resolve_context_path, _source_hashes, log, paired_comparison, save_json, sha256,
)


def run_grid(config_path: Path, selection_path: Path, output: Path):
    config = json.loads(config_path.read_text())
    selected = int(json.loads(selection_path.read_text())["planning_samples"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Completed sensitivity exists; use a new output directory")
    jobs = []
    for cu in config["underage_costs"]:
        for initial in config["initial_orders"]:
            for budget in config["budgets"]:
                cell = {**config, "underage": cu, "initial_order": initial, "extra_budget": budget}
                for seed in config["seeds"]:
                    jobs.append((cell, "heterogeneous", seed, selected))
    save_json(output / "protocol.json", {"config": config, "config_sha256": sha256(config_path),
        "selection_sha256": sha256(selection_path), "planning_samples": selected,
        "context_sha256": sha256(_resolve_context_path(config["context_path"])),
        "source_sha256": _source_hashes()})
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
        futures = [executor.submit(_job, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            summaries, _, _, _, _ = future.result()
            records.extend(summaries)
            if count % 12 == 0 or count == len(jobs):
                log(f"Sensitivity {count}/{len(jobs)} paired environments")
    frame = pl.DataFrame(records).sort(["underage", "initial_order", "budget", "seed", "policy"])
    frame.write_csv(output / "outcomes.csv")
    cells = []
    for key, data in frame.group_by(["underage", "initial_order", "budget"], maintain_order=True):
        cu, initial, budget = key
        comparisons = {baseline: paired_comparison(data, "heterogeneous", "decision", baseline,
            n_boot=config["bootstrap_replicates"]) for baseline in ["uniform", "uncertainty"]}
        adaptive = paired_comparison(data, "heterogeneous", "adaptive_cost", "uniform",
                                    n_boot=config["bootstrap_replicates"])
        cells.append({"underage": cu, "initial_order": initial, "budget": budget,
            "difference": comparisons["uniform"]["difference"],
            "relative_reduction": comparisons["uniform"]["relative_reduction"],
            "comparisons": comparisons, "adaptive_cost_vs_uniform": adaptive,
            "means": data.group_by("policy").agg(pl.col("cumulative_expected_regret").mean(),
                pl.col("normalized_cost").mean(), pl.col("extra_units").mean()).sort("policy").to_dicts()})
    result = {"config": config, "planning_samples": selected, "cells": cells,
              "elapsed_seconds": time.perf_counter()-started}
    save_json(output / "summary.json", result)
    log(f"Sensitivity complete in {result['elapsed_seconds']:.1f}s")
    return result
