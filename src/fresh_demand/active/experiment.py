"""Predeclared sequential experiments with common random demand trajectories.

The policy interface receives past censored observations only. Hidden demands
and PMFs are used after actions for scoring; the explicitly privileged
full-information reference receives past demands, never future realizations.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import zlib

import numpy as np
import polars as pl

from .environments import make_environment, sample_demands
from .policies import allocate_cost_aware, allocate_decision, allocate_predictive, allocate_uniform
from .baselines import kaplan_meier_pmf
from .statistics import RiskSetEstimator, cost_curve, minimax_regret, quantile_bounds

PROJECT = Path(__file__).resolve().parents[3]
ALLOCATION_POLICIES = {"uniform", "uncertainty", "decision", "adaptive_decision", "adaptive_cost"}
KNOWN_POLICIES = ALLOCATION_POLICIES | {"sales", "lightgbm_sales", "risk_lower", "risk_uniform", "risk_upper", "km_uniform", "km_upper", "minimax", "geometric_mean", "geometric_ts", "full_information"}


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def empirical_quantile(values, tau, initial):
    return int(np.quantile(values, tau, method="inverted_cdf")) if len(values) else initial


def geometric_quantile(probability, tau, maximum):
    probability = np.clip(probability, 1e-10, 1 - 1e-10)
    return np.minimum(maximum, np.maximum(0, np.ceil(np.log1p(-tau) / np.log1p(-probability)) - 1)).astype(int)


def _empty_intersection(error):
    return "empty intersection" in str(error)


def simulate_policy(pmfs, demands, warmup_demands, policy, config, seed, planning_samples=1, keep_observations=False):
    """Execute one policy; PMFs affect scoring, never ordinary action selection."""
    horizon, contexts, support = pmfs.shape
    if policy not in KNOWN_POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    maximum = support - 1
    cu, co = config["underage"], config["overage"]
    tau = cu / (cu + co)
    budget = config["extra_budget"]
    initial = config["initial_order"]
    estimators = [RiskSetEstimator(maximum, config["delta"], contexts, horizon + len(warmup_demands)) for _ in range(contexts)]
    sales_history = [[] for _ in range(contexts)]
    full_history = [[] for _ in range(contexts)]
    a, b = np.ones(contexts), np.full(contexts, 8.0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(policy.encode()), 512]))
    observations = []

    def update(orders, latent, t, phase, base_orders=None):
        sales = np.minimum(orders, latent)
        sold_out = latent >= orders
        for c in range(contexts):
            estimators[c].observe(int(orders[c]), int(sales[c]), bool(sold_out[c]))
            sales_history[c].append(int(sales[c]))
            if policy == "full_information":
                full_history[c].append(int(latent[c]))
            a[c] += not sold_out[c]
            b[c] += sales[c]
            if keep_observations:
                observations.append({"round": t, "phase": phase, "context": c, "order": int(orders[c]), "base_order": int(orders[c] if base_orders is None else base_orders[c]), "sales": int(sales[c]), "sold_out": bool(sold_out[c])})
        return sales

    for t, latent in enumerate(warmup_demands):
        update(np.full(contexts, initial), latent, t - len(warmup_demands), "warmup")
    records = []
    fitted_lgbm = None
    context_x = np.arange(contexts, dtype=np.float32)[:, None]
    for t in range(horizon):
        fallback = False
        # Only history-based state is read until the order vector is fixed.
        if policy in {"sales", "full_information"}:
            histories = full_history if policy == "full_information" else sales_history
            base = np.array([empirical_quantile(h, tau, initial) for h in histories])
        elif policy == "lightgbm_sales":
            if fitted_lgbm is None or t % 25 == 0:
                import lightgbm as lgb
                x = np.concatenate([np.full((len(h), 1), c, dtype=np.float32) for c, h in enumerate(sales_history)])
                y = np.concatenate(sales_history)
                fitted_lgbm = lgb.LGBMRegressor(objective="quantile", alpha=tau, n_estimators=80,
                    learning_rate=.1, num_leaves=8, min_child_samples=10, n_jobs=1,
                    verbosity=-1, deterministic=True, force_col_wise=True, random_state=42)
                fitted_lgbm.fit(x, y, categorical_feature=[0])
            base = np.rint(fitted_lgbm.predict(context_x)).astype(int)
        elif policy in {"geometric_mean", "geometric_ts"}:
            probability = rng.beta(a, b) if policy == "geometric_ts" else a / (a + b)
            base = geometric_quantile(probability, tau, maximum)
        elif policy in {"km_uniform", "km_upper"}:
            tail = "uniform" if policy == "km_uniform" else "upper"
            base = np.array([min(maximum, int(np.searchsorted(np.cumsum(kaplan_meier_pmf(e.history, maximum, tail)), tau))) for e in estimators])
        elif policy == "minimax":
            try:
                base = np.array([minimax_regret(*e.cdf_bounds(), cu, co)[0] for e in estimators])
            except ValueError as error:
                if not _empty_intersection(error):
                    raise
                base = np.array([e.quantile(tau, "uniform") for e in estimators])
                fallback = True
        else:
            tail = {"risk_lower": "lower", "risk_uniform": "uniform", "risk_upper": "upper"}.get(policy, "lower")
            base = np.array([e.quantile(tau, tail) for e in estimators])
        base = np.clip(base, 0, maximum)
        orders = base.copy()
        allocation_rng = np.random.default_rng(np.random.SeedSequence([seed, t, 991]))
        if policy in ALLOCATION_POLICIES and budget:
            try:
                if policy == "uniform":
                    orders = allocate_uniform(base, budget, maximum, allocation_rng)
                elif policy == "uncertainty":
                    orders = allocate_predictive(base, budget, estimators, cu, co, allocation_rng, planning_samples=planning_samples)
                elif policy == "adaptive_decision":
                    active = np.array([quantile_bounds(*e.cdf_bounds(), tau)[0] != quantile_bounds(*e.cdf_bounds(), tau)[1] for e in estimators])
                    if active.any():
                        orders[active] = allocate_decision(base[active], budget, [e for e, yes in zip(estimators, active) if yes], cu, co, allocation_rng, planning_samples=planning_samples)
                elif policy == "adaptive_cost":
                    orders = allocate_cost_aware(base, budget, estimators, cu, co, allocation_rng,
                        planning_samples=planning_samples, remaining_rounds=horizon-t)
                else:
                    orders = allocate_decision(base, budget, estimators, cu, co, allocation_rng, planning_samples=planning_samples)
            except ValueError as error:
                if not _empty_intersection(error):
                    raise
                orders = allocate_uniform(base, budget, maximum, allocation_rng)
                fallback = True
        extra = int((orders - base).sum())
        if policy in {"uniform", "uncertainty", "decision"}:
            assert extra == min(budget, int((maximum - base).sum()))
        assert (orders >= base).all() and (orders <= maximum).all()

        # Hidden outcomes become available to the evaluator only after ordering.
        latent = demands[t]
        shortage, excess = np.maximum(latent - orders, 0), np.maximum(orders - latent, 0)
        costs = cu * shortage + co * excess
        expected = np.stack([cost_curve(p, cu, co) for p in pmfs[t]])
        expected_cost = expected[np.arange(contexts), orders]
        oracle_cost = expected.min(axis=1)
        covered, identified = True, 0
        for c, estimator in enumerate(estimators):
            try:
                lower, upper = estimator.cdf_bounds()
                true_cdf = pmfs[t, c].cumsum()
                covered &= bool(((true_cdf >= lower - 1e-10) & (true_cdf <= upper + 1e-10)).all())
                lo, hi = quantile_bounds(lower, upper, tau)
                identified += lo == hi
            except ValueError as error:
                if not _empty_intersection(error):
                    raise
                covered = False
        sales = update(orders, latent, t, "online", base)
        records.append({"round": t, "cost": float(costs.sum()), "expected_cost": float(expected_cost.sum()),
            "oracle_cost": float(oracle_cost.sum()), "regret": float((expected_cost - oracle_cost).sum()),
            "shortage": float(shortage.sum()), "excess": float(excess.sum()), "demand": int(latent.sum()),
            "sales": int(sales.sum()), "ordered": int(orders.sum()), "extra": extra,
            "available_extra_capacity": int((maximum - base).sum()),
            "identified_fraction": identified / contexts, "cdf_covered": covered, "fallback": fallback})
    frame = pl.DataFrame(records)
    total_demand = float(frame["demand"].sum())
    summary = {"policy": policy, "planning_samples": planning_samples,
        "cost": float(frame["cost"].sum()), "normalized_cost": float(frame["cost"].sum() / max(total_demand, 1)),
        "cumulative_expected_regret": float(frame["regret"].sum()),
        "expected_cost": float(frame["expected_cost"].sum()), "oracle_expected_cost": float(frame["oracle_cost"].sum()),
        "demand": total_demand, "fill_rate": float(frame["sales"].sum() / max(total_demand, 1)),
        "excess": float(frame["excess"].sum()), "extra_units": int(frame["extra"].sum()),
        "band_familywise_coverage": bool(frame["cdf_covered"].all()),
        "final_identified_fraction": float(frame["identified_fraction"][-1]), "fallback_rounds": int(frame["fallback"].sum())}
    return summary, frame, pl.DataFrame(observations) if observations else None


def _job(arguments):
    config, scenario, seed, planning = arguments
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    environment = make_environment(scenario, seed, config["horizon"], config["max_demand"], config["n_contexts"], PROJECT / config["context_path"])
    pmfs = environment["pmfs"]
    warmup_pmf = np.repeat(pmfs[:1], config["warmup"], axis=0)
    all_demands = sample_demands(np.concatenate([warmup_pmf, pmfs]), seed)
    warmup, demand = all_demands[:config["warmup"]], all_demands[config["warmup"]:]
    summaries, rounds, example = [], [], None
    started = time.perf_counter()
    for policy in config["policies"]:
        keep = seed == config["seeds"][0] and scenario == config["scenarios"][0] and policy == "decision"
        summary, frame, observations = simulate_policy(pmfs, demand, warmup, policy, config, seed, planning, keep)
        prefix = {"scenario": scenario, "seed": seed, "planning_samples": planning,
                  "budget": config["extra_budget"], "underage": config["underage"], "initial_order": config["initial_order"]}
        summaries.append({**prefix, **summary})
        rounds.append(frame.with_columns([pl.lit(value).alias(key) for key, value in {**prefix, "policy": policy}.items()]))
        if observations is not None:
            example = observations
    return summaries, pl.concat(rounds), example, environment["metadata"], time.perf_counter()-started


def paired_comparison(frame, scenario, candidate, baseline, metric="cumulative_expected_regret", n_boot=5000, seed=2026):
    relevant = frame.filter(pl.col("scenario") == scenario)
    left = relevant.filter(pl.col("policy") == candidate).select("seed", pl.col(metric).alias("candidate"))
    right = relevant.filter(pl.col("policy") == baseline).select("seed", pl.col(metric).alias("baseline"))
    pair = left.join(right, on="seed", how="inner").sort("seed")
    if pair.height != relevant["seed"].n_unique():
        raise ValueError("Incomplete or duplicated paired seed comparison")
    difference = pair["candidate"].to_numpy() - pair["baseline"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = difference[rng.integers(0, pair.height, size=(n_boot, pair.height))].mean(axis=1)
    baseline_mean = float(pair["baseline"].mean())
    return {"candidate": candidate, "baseline": baseline, "metric": metric,
        "difference": float(difference.mean()), "ci_low": float(np.quantile(draws, .025)),
        "ci_high": float(np.quantile(draws, .975)), "relative_reduction": float(-difference.mean()/baseline_mean) if baseline_mean else None,
        "paired_seeds": pair.height, "bootstrap_replicates": n_boot,
        "scope": "Independent demand trajectories paired across policies; conditional on this fixed simulator and protocol"}


def freeze_selection(development: Path, config_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("Frozen selection already exists; preserve it")
    config = json.loads(config_path.read_text())
    data = pl.read_csv(development / "outcomes.csv")
    if sorted(data["seed"].unique().to_list()) != sorted(config["seeds"]) or max(config["seeds"]) >= 1000:
        raise ValueError("Selection must use only the explicitly configured development seeds")
    candidates = data.filter((pl.col("scenario") == "heterogeneous") & (pl.col("policy") == "decision")).group_by("planning_samples").agg(pl.col("cumulative_expected_regret").mean()).sort(["cumulative_expected_regret", "planning_samples"])
    result = {"planning_samples": int(candidates["planning_samples"][0]),
        "criterion": "Mean decision-policy cumulative expected regret on heterogeneous development seeds only",
        "development_seeds": config["seeds"], "confirmation_seeds_seen": False,
        "candidates": candidates.to_dicts(), "development_config_sha256": sha256(config_path),
        "frozen_utc": datetime.now(timezone.utc).isoformat()}
    save_json(output, result)
    return result


def run(config_path: Path, output: Path, selection_path: Path | None = None):
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Completed run exists; use a new explicitly named output directory")
    if config["planning_samples"] == "frozen_development_selection":
        if selection_path is None:
            raise ValueError("Confirmation requires the frozen development selection")
        selection = json.loads(selection_path.read_text())
        plans = [int(selection["planning_samples"])]
        selection_hash = sha256(selection_path)
    else:
        plans = config["planning_samples"] if isinstance(config["planning_samples"], list) else [config["planning_samples"]]
        selection_hash = None
    protocol = {"config": config, "config_sha256": sha256(config_path),
        "context_sha256": sha256(PROJECT / config["context_path"]),
        "selection_sha256": selection_hash, "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "lightgbm", "polars"]},
        "python": platform.python_version(), "platform": platform.platform(),
        "hypothesis": "At identical extra-unit budgets, decision-directed information allocation reduces expected regret relative to uniform allocation",
        "censoring": "sales=min(D,Q), sold_out=(D>=Q); latent D never supplied to ordinary learners",
        "warmup": "Common censored history at fixed initial order; warmup costs excluded equally from all policies"}
    protocol["source_sha256"] = {str(path.relative_to(PROJECT)): sha256(path) for path in sorted((PROJECT / "src/fresh_demand/active").glob("*.py"))}
    save_json(output / "protocol.json", protocol)
    jobs = [(config, scenario, seed, plan) for plan in plans for scenario in config["scenarios"] for seed in config["seeds"]]
    all_summaries, all_rounds, environments = [], [], {}
    with ProcessPoolExecutor(max_workers=config.get("workers", 1)) as executor:
        futures = {executor.submit(_job, job): job[1:] for job in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            summaries, frame, example, metadata, seconds = future.result()
            all_summaries.extend(summaries)
            all_rounds.append(frame)
            scenario, seed, plan = futures[future]
            environments[scenario] = metadata
            if example is not None:
                example.write_csv(output / f"example_observations_m{plan}.csv")
            log(f"{completed}/{len(jobs)} {scenario}, seed={seed}, planning={plan}, {seconds:.1f}s")
    outcomes = pl.DataFrame(all_summaries).sort(["planning_samples", "scenario", "seed", "policy"])
    curves = pl.concat(all_rounds).sort(["planning_samples", "scenario", "seed", "policy", "round"])
    outcomes.write_csv(output / "outcomes.csv")
    curves.write_parquet(output / "learning_curves.parquet")
    comparison = {}
    if len(plans) == 1:
        for scenario in config["scenarios"]:
            comparison[scenario] = {baseline: paired_comparison(outcomes, scenario, "decision", baseline,
                n_boot=config.get("bootstrap_replicates", 2000)) for baseline in ("uniform", "uncertainty") if baseline in config["policies"]}
    summary = {"protocol": protocol, "environment_metadata": environments,
               "comparisons": comparison, "elapsed_seconds": time.perf_counter()-started}
    save_json(output / "summary.json", summary)
    log(f"Finished {len(jobs)} paired environments in {summary['elapsed_seconds']:.1f}s")
    return summary
