"""A transparent identification example and a fixed-design passive-data study.

This is a conceptual reproduction of an indistinguishability mechanism, not
a numerical replication of a paper's tables or experimental checkpoints.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .statistics import RiskSetEstimator, cost_curve, minimax_regret, quantile_bounds


MAX_DEMAND = 40
HISTORICAL_CAP = 8
SAMPLE_SIZES = (20, 80, 320, 1280)
AUDIT_COUNTS = (0, 1, 4, 16)


def distributions() -> dict[str, np.ndarray]:
    """Two distinct demands with exactly the same cap-eight observations."""
    result = {}
    for name, tail_demand in [("A", 12), ("B", 35)]:
        probability = np.zeros(MAX_DEMAND + 1)
        probability[5] = 0.5
        probability[tail_demand] = 0.5
        result[name] = probability
    return result


def observation_law(probability: np.ndarray, order: int) -> list[dict]:
    law: dict[tuple[int, bool], float] = defaultdict(float)
    for demand, mass in enumerate(probability):
        if mass > 0:
            law[(min(demand, order), demand >= order)] += float(mass)
    return [{"sales": sales, "sold_out": bool(sold_out), "probability": probability}
            for (sales, sold_out), probability in sorted(law.items())]


def population_band() -> tuple[np.ndarray, np.ndarray]:
    """The sharp CDF band after infinitely many cap-eight observations."""
    lower = np.zeros(MAX_DEMAND + 1)
    upper = np.zeros(MAX_DEMAND + 1)
    lower[5:] = 0.5
    upper[5:HISTORICAL_CAP] = 0.5
    upper[HISTORICAL_CAP:] = 1
    lower[-1] = 1
    return lower, upper


def analytic_identification() -> dict:
    worlds = distributions()
    lower, upper = population_band()
    costs = []
    for cu, co in [(1, 3), (3, 1)]:
        tau = cu / (cu + co)
        curves = {world: cost_curve(p, cu, co) for world, p in worlds.items()}
        action, robust_regret = minimax_regret(lower, upper, cu, co)
        restricted_regret = np.max(np.asarray([
            curve - curve.min() for curve in curves.values()
        ]), axis=0)
        restricted_actions = np.flatnonzero(np.isclose(restricted_regret, restricted_regret.min())).tolist()
        costs.append({
            "underage_cost": cu, "overage_cost": co, "critical_quantile": tau,
            "oracle_orders": {world: int(np.argmin(curve)) for world, curve in curves.items()},
            "oracle_expected_costs": {world: float(curve.min()) for world, curve in curves.items()},
            "identified_quantile_bounds": list(quantile_bounds(lower, upper, tau)),
            "full_cdf_band_minimax_order": action,
            "full_cdf_band_minimax_regret": float(robust_regret[action]),
            "full_cdf_band_regret_by_order": robust_regret.tolist(),
            "two_witness_worlds_minimax_orders": restricted_actions,
            "two_witness_worlds_minimax_regret": float(restricted_regret.min()),
            "two_witness_worlds_regret_by_order": restricted_regret.tolist(),
        })
    return {
        "support": [0, MAX_DEMAND], "historical_order": HISTORICAL_CAP,
        "sold_out_convention": "Demand >= order, including equality",
        "world_pmfs": {name: p.tolist() for name, p in worlds.items()},
        "observation_laws": {name: observation_law(p, HISTORICAL_CAP) for name, p in worlds.items()},
        "population_cdf_lower": lower.tolist(), "population_cdf_upper": upper.tolist(),
        "population_band_is_finite_sample_confidence_band": False,
        "population_band_scope": "All distributions with mass 0.5 at demand five and remaining mass anywhere on eight through forty.",
        "cost_settings": costs,
        "analytic_verification": {
            "high_quantile_regret_formula_for_orders_8_to_40": "max(q - 8, 40 - q)",
            "balance_equation": "q - 8 = 40 - q",
            "balanced_order": 24, "balanced_regret": 16,
            "two_world_restriction_is_different": "Restricting uncertainty to only A and B gives orders 23 or 24 and regret 12, not 16.",
            "low_quantile": "At costs 1:3, order five has zero regret over the entire population CDF band.",
        },
    }


def _fit_record_batch(demand: np.ndarray, audits: int, horizon: int) -> RiskSetEstimator:
    """Exact batch sufficient statistics for the same public risk-set estimator.

    Audit orders replace the last k historical orders, so each alternative
    uses the identical N demand draws. No outcome-dependent audit selection.
    """
    demand = np.asarray(demand, dtype=int)
    if demand.ndim != 1 or not len(demand) or not 0 <= audits <= len(demand):
        raise ValueError("Require a nonempty demand vector and 0 <= audits <= N")
    orders = np.full(len(demand), HISTORICAL_CAP, dtype=int)
    if audits:
        orders[-audits:] = MAX_DEMAND
    sales = np.minimum(demand, orders)
    exposure = orders[:, None] > np.arange(MAX_DEMAND)[None, :]
    events = sales[:, None] <= np.arange(MAX_DEMAND)[None, :]
    estimator = RiskSetEstimator(MAX_DEMAND, delta=0.1, n_contexts=1, horizon=horizon)
    estimator.n[:] = exposure.sum(axis=0)
    estimator.successes[:] = (exposure & events).sum(axis=0)
    estimator.rounds = len(demand)
    estimator.history = list(zip(orders.tolist(), sales.tolist(), (demand >= orders).tolist()))
    return estimator


def passive_plateau(
    seed_count: int = 50,
    sample_sizes: tuple[int, ...] = SAMPLE_SIZES,
    audit_counts: tuple[int, ...] = AUDIT_COUNTS,
) -> dict:
    """Fixed-budget study; report expected true cost, including audit history.

    Increasing passive N cannot identify the hidden tail. A fixed number of
    audits may improve or worsen an estimator in finite samples; no outcome is
    required to improve monotonically. These are post-history decisions, not
    an adaptive learning-policy regret experiment.
    """
    if seed_count < 1 or not sample_sizes or min(sample_sizes) < 1 or not audit_counts or min(audit_counts) < 0:
        raise ValueError("Invalid Monte Carlo settings")
    if max(audit_counts) > min(sample_sizes):
        raise ValueError("Every audit count must fit inside every sample size")
    worlds = distributions()
    horizon = max(sample_sizes)
    curves = {(world, cu, co): cost_curve(p, cu, co)
              for world, p in worlds.items() for cu, co in [(1, 3), (3, 1)]}
    records = []
    for seed in range(seed_count):
        uniforms = np.random.default_rng(seed).random(horizon)
        demands = {world: np.where(uniforms < 0.5, 5, tail)
                   for world, tail in [("A", 12), ("B", 35)]}
        for n in sample_sizes:
            for audits in audit_counts:
                for world in worlds:
                    estimator = _fit_record_batch(demands[world][:n], audits, horizon)
                    for tail_assumption in ["uniform", "lower"]:
                        working_pmf = estimator.point_pmf(tail=tail_assumption)
                        for cu, co in [(1, 3), (3, 1)]:
                            tau = cu / (cu + co)
                            quantity = min(MAX_DEMAND, int(np.searchsorted(np.cumsum(working_pmf), tau)))
                            curve = curves[world, cu, co]
                            history_cost = (n - audits) * curve[HISTORICAL_CAP] + audits * curve[MAX_DEMAND]
                            records.append({
                                "n": int(n), "audits": int(audits), "seed": seed,
                                "world": world, "tail_assumption": tail_assumption,
                                "underage_cost": cu, "overage_cost": co, "critical_quantile": tau,
                                "estimated_order": quantity,
                                "terminal_expected_cost": float(curve[quantity]),
                                "terminal_expected_regret": float(curve[quantity] - curve.min()),
                                "history_expected_cost": float(history_cost),
                                "audit_history_cost_difference_vs_no_audits": float(audits * (curve[MAX_DEMAND] - curve[HISTORICAL_CAP])),
                                "history_plus_one_terminal_decision_expected_cost": float(history_cost + curve[quantity]),
                            })
    grouped = defaultdict(list)
    for record in records:
        key = tuple(record[name] for name in ["n", "audits", "world", "tail_assumption", "underage_cost", "overage_cost"])
        grouped[key].append(record)
    summary = []
    for key, rows in sorted(grouped.items()):
        item = dict(zip(["n", "audits", "world", "tail_assumption", "underage_cost", "overage_cost"], key))
        values = np.asarray([row["terminal_expected_regret"] for row in rows])
        stderr = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        item.update({
            "seeds": len(rows),
            "mean_terminal_expected_regret": float(values.mean()),
            "terminal_expected_regret_standard_error": stderr,
            "terminal_expected_regret_normal_95_interval": [float(values.mean() - 1.96 * stderr), float(values.mean() + 1.96 * stderr)],
            "mean_estimated_order": float(np.mean([row["estimated_order"] for row in rows])),
            "mean_terminal_expected_cost": float(np.mean([row["terminal_expected_cost"] for row in rows])),
            "mean_audit_history_cost_difference_vs_no_audits": float(np.mean([row["audit_history_cost_difference_vs_no_audits"] for row in rows])),
            "mean_history_plus_one_terminal_decision_expected_cost": float(np.mean([row["history_plus_one_terminal_decision_expected_cost"] for row in rows])),
        })
        summary.append(item)
    return {
        "settings": {"seed_count": seed_count, "seeds": list(range(seed_count)),
                     "sample_sizes": list(sample_sizes), "audit_counts": list(audit_counts),
                     "passive_order": HISTORICAL_CAP, "audit_order": MAX_DEMAND,
                     "tail_assumptions": ["uniform", "lower"], "declared_confidence_horizon": horizon},
        "design": "The last k cap-eight records are replaced by cap-forty records within the same fixed N demands; audit placement is independent of outcomes.",
        "common_random_numbers": "The same uniforms are used across worlds, N prefixes, audit allocations, tail completions, and cost settings.",
        "interval_interpretation": "Approximate normal interval for the mean over independent simulation seeds, not an identification or CDF confidence bound.",
        "audit_cost_interpretation": "Historical cost differences include both underage and overage; they can be negative when larger stock also avoids expensive shortages.",
        "claim_scope": "Terminal decisions after fixed observation designs; no monotonic improvement, active-policy regret bound, or business effect is asserted.",
        "summary": summary, "raw_records": records,
    }


def run_reproduction(
    output: Path,
    *,
    seed_count: int = 50,
    sample_sizes: tuple[int, ...] = SAMPLE_SIZES,
    audit_counts: tuple[int, ...] = AUDIT_COUNTS,
) -> dict:
    """Save identification.json with exact calculations and every MC record."""
    started = time.perf_counter()
    result = {
        "label": "Conceptual reproduction of an indistinguishability mechanism, not numeric replication of a paper table.",
        "source_reference": "https://arxiv.org/abs/2602.16842",
        "source_scope": "The two-point distributions, finite support, costs, and fixed audit experiment are constructed here; source theorem and table values are not claimed.",
        "analytic": analytic_identification(),
        "monte_carlo": passive_plateau(seed_count=seed_count, sample_sizes=sample_sizes, audit_counts=audit_counts),
        "runtime_seconds": time.perf_counter() - started,
    }
    output = Path(output)
    path = output if output.suffix == ".json" else output / "identification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/active/reproduction"))
    args = parser.parse_args()
    result = run_reproduction(args.output)
    print(json.dumps({"output": str(args.output), "runtime_seconds": result["runtime_seconds"],
                      "raw_records": len(result["monte_carlo"]["raw_records"]),
                      "summary_rows": len(result["monte_carlo"]["summary"])}), flush=True)


if __name__ == "__main__":
    main()
