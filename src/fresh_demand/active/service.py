"""Reconstruct observed state and propose the next inventory vector."""

from pathlib import Path
import json

import numpy as np
import polars as pl

from .experiment import save_json, sha256
from .policies import allocate_cost_aware, allocate_decision, allocate_uniform
from .statistics import RiskSetEstimator, minimax_regret, quantile_bounds


def plan_next(history_path: Path, config_path: Path, selection_path: Path, output: Path,
              policy: str = "adaptive_cost", seed: int = 42) -> dict:
    """Use known-Q censored logs only; source stock flags cannot replace Q."""
    if policy not in {"adaptive_cost", "decision", "uniform"}:
        raise ValueError("Serving policy must be adaptive_cost, decision, or uniform")
    config = json.loads(config_path.read_text())
    selection = json.loads(selection_path.read_text())
    planning = int(selection["planning_samples"])
    maximum, contexts = config["max_demand"], config["n_contexts"]
    history = pl.read_csv(history_path)
    needed = {"round", "context", "order", "sales", "sold_out"}
    if not needed.issubset(history.columns) or history.is_empty():
        raise ValueError(f"History needs nonempty columns: {sorted(needed)}")
    if history.select(pl.any_horizontal([pl.col(name).is_null() for name in needed]).any()).item():
        raise ValueError("History contains missing observation fields")
    if sorted(history["context"].unique().to_list()) != list(range(contexts)):
        raise ValueError("History contexts must match every configured ID in 0..K-1")
    if history.select(pl.struct(["round", "context"]).is_duplicated().any()).item():
        raise ValueError("Duplicate context-round observations")
    next_round = max(0, int(history["round"].max()) + 1)
    if next_round >= config["horizon"]:
        raise ValueError("Declared online horizon exhausted; start a new explicit protocol")
    expected_rounds = list(range(-config["warmup"], next_round))
    estimators = []
    for context in range(contexts):
        rows = history.filter(pl.col("context") == context).sort("round")
        if rows["round"].to_list() != expected_rounds:
            raise ValueError("Each context needs the same complete warmup and online history")
        estimator = RiskSetEstimator(maximum, config["delta"], contexts,
                                     config["horizon"] + config["warmup"])
        for row in rows.iter_rows(named=True):
            if row["sold_out"] not in (True, False, 0, 1):
                raise ValueError("sold_out must be boolean or 0/1")
            estimator.observe(row["order"], row["sales"], bool(row["sold_out"]))
        estimators.append(estimator)
    cu, co = config["underage"], config["overage"]
    tau = cu / (cu + co)
    base = np.array([estimator.quantile(tau, "lower") for estimator in estimators])
    rng = np.random.default_rng(np.random.SeedSequence([seed, next_round, 991]))
    if policy == "uniform":
        orders = allocate_uniform(base, config["extra_budget"], maximum, rng)
    elif policy == "decision":
        orders = allocate_decision(base, config["extra_budget"], estimators, cu, co, rng,
                                  planning_samples=planning)
    else:
        orders = allocate_cost_aware(base, config["extra_budget"], estimators, cu, co, rng,
            planning_samples=planning, remaining_rounds=config["horizon"]-next_round)
    diagnostics = []
    for context, estimator in enumerate(estimators):
        lower, upper = estimator.cdf_bounds()
        lo, hi = quantile_bounds(lower, upper, tau)
        robust_order, curve = minimax_regret(lower, upper, cu, co)
        diagnostics.append({"context": context, "base_order": int(base[context]),
            "order": int(orders[context]), "extra_units": int(orders[context]-base[context]),
            "quantile_lower": lo, "quantile_upper": hi, "minimax_order": robust_order,
            "minimax_regret_bound": float(curve.min()), "observations": estimator.rounds})
        diagnostics[-1]["chosen_order_regret_bound"] = float(curve[int(orders[context])])
    result = {"next_round": next_round, "policy": policy, "planning_samples": planning,
        "budget_limit": config["extra_budget"], "extra_units": int((orders-base).sum()),
        "underage": cu, "overage": co, "contexts": diagnostics,
        "history_sha256": sha256(history_path), "config_sha256": sha256(config_path),
        "selection_sha256": sha256(selection_path), "oracle_data_used": False,
        "assumptions": "Known bounded support; known inventory Q; predictable orders; stationary independent demand conditional on context",
        "planning": "Mean-field virtual batches are heuristic scores, not guaranteed future confidence sets"}
    save_json(output, result)
    return result
