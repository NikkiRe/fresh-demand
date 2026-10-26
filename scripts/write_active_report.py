#!/usr/bin/env python3
"""Render the inventory learning report from saved experiment results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import polars as pl


PROJECT = Path(__file__).resolve().parents[1]
POLICIES = ["sales", "lightgbm_sales", "risk_lower", "risk_uniform", "risk_upper",
            "km_uniform", "km_upper", "minimax", "geometric_mean", "geometric_ts",
            "uniform", "uncertainty", "decision", "adaptive_decision", "adaptive_cost", "full_information"]
LABELS = {"sales": "Sales quantile", "lightgbm_sales": "LightGBM sales", "risk_lower": "Risk set: lower",
          "risk_uniform": "Risk set: uniform", "risk_upper": "Risk set: upper", "km_uniform": "KM: uniform",
          "km_upper": "KM: upper", "minimax": "CDF-band minimax", "geometric_mean": "Geometric posterior mean",
          "geometric_ts": "Geometric Thompson", "uniform": "Uniform allocation", "uncertainty": "CDF uncertainty",
          "decision": "Decision allocation", "adaptive_decision": "Identifiability gate",
          "adaptive_cost": "Cost-aware gate", "full_information": "Past full information"}
SCENARIOS = {"heterogeneous": "Mixtures", "geometric": "Geometric", "heavy_tail": "Bounded long tail",
             "fresh_empirical": "Retail iid", "identified": "Identified", "shift": "Constructed shift",
             "fresh_shift": "Retail shift"}
COLORS = {"uniform": "#65758a", "uncertainty": "#c68c36", "decision": "#167d8d", "adaptive_cost": "#8d428f",
          "km_uniform": "#428e48", "geometric_ts": "#d05d4e", "sales": "#9b794e", "full_information": "#1b263b"}


def load(path: Path) -> dict:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"JSON object required: {path}")
    return result


def num(value, digits=2, signed=False):
    return "undefined" if value is None or not np.isfinite(value) else format(value, f"{'+' if signed else ''},.{digits}f")


def pct(value):
    return num(100 * value, 1) + "%"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def link(path: Path, anchor: Path = PROJECT):
    return Path(os.path.relpath(path, anchor)).as_posix()


def interval(comparison):
    return f"{num(comparison['difference'], signed=True)} [{num(comparison['ci_low'])}, {num(comparison['ci_high'])}]"


def conclusion(comparison):
    if comparison["ci_high"] < 0:
        return "Decision allocation has lower expected regret in this simulator."
    if comparison["ci_low"] > 0:
        return "Decision allocation has higher expected regret than uniform allocation."
    return "The interval includes zero; there is no clear advantage over uniform allocation."


def aggregate(outcomes):
    columns = ["cumulative_expected_regret", "cost", "normalized_cost", "fill_rate", "excess", "extra_units",
               "band_familywise_coverage", "final_identified_fraction", "fallback_rounds"]
    return outcomes.group_by("scenario", "policy").agg([
        pl.col(column).cast(pl.Float64).mean().alias(column) for column in columns
    ] + [pl.len().alias("seeds")]).sort("scenario", "policy")


def row_lookup(averages):
    return {(row["scenario"], row["policy"]): row for row in averages.iter_rows(named=True)}


def paired_delta(outcomes, scenario, candidate, baseline, metric, n_boot):
    relevant = outcomes.filter(pl.col("scenario") == scenario)
    left = relevant.filter(pl.col("policy") == candidate).select("seed", pl.col(metric).alias("candidate"))
    right = relevant.filter(pl.col("policy") == baseline).select("seed", pl.col(metric).alias("baseline"))
    pair = left.join(right, on="seed", validate="1:1").sort("seed")
    if pair.height != relevant["seed"].n_unique():
        raise ValueError("Incomplete secondary paired comparison.")
    difference = pair["candidate"].to_numpy() - pair["baseline"].to_numpy()
    rng = np.random.default_rng(2026)
    draws = difference[rng.integers(0, len(difference), size=(n_boot, len(difference)))].mean(axis=1)
    mean_baseline = pair["baseline"].mean()
    return {"difference": difference.mean(), "ci_low": np.quantile(draws, .025), "ci_high": np.quantile(draws, .975),
            "relative_reduction": -difference.mean() / mean_baseline if mean_baseline else None}


def mechanism_sections(path: Path, confirmation_seeds, development_seeds):
    if not path.exists():
        return "", ""
    summary = load(path)
    config = summary["protocol"]["config"]
    if config.get("stage") != "post_confirmation_mechanism_check":
        raise ValueError("Mechanism control must retain its post-confirmation label.")
    if set(config["seeds"]) & (set(confirmation_seeds) | set(development_seeds)):
        raise ValueError("Explanatory mechanism seeds overlap earlier stages.")
    outcomes = pl.read_csv(path.parent / "outcomes.csv")
    if outcomes.height != len(config["scenarios"]) * len(config["policies"]) * len(config["seeds"]):
        raise ValueError("Incomplete explanatory-control outcomes.")
    contrasts, rows = {}, []
    for scenario in config["scenarios"]:
        regret = paired_delta(outcomes, scenario, "adaptive_cost", "myopic_cost", "cumulative_expected_regret", config["bootstrap_replicates"])
        cost = paired_delta(outcomes, scenario, "adaptive_cost", "myopic_cost", "cost", config["bootstrap_replicates"])
        contrasts[scenario] = (regret, cost)
        extras = [outcomes.filter((pl.col("scenario") == scenario) & (pl.col("policy") == policy))["extra_units"].mean()
                  for policy in ["adaptive_cost", "myopic_cost"]]
        rows.append([scenario, interval(regret), pct(regret["relative_reduction"]) if regret["relative_reduction"] is not None else "undefined",
                     interval(cost), pct(cost["relative_reduction"]) if cost["relative_reduction"] is not None else "undefined",
                     num(extras[0], 1), num(extras[1], 1)])
    regret, cost = contrasts["heterogeneous"]
    if regret["ci_low"] > 0:
        interpretation = "The myopic policy has lower expected regret in this scenario."
    elif regret["ci_high"] < 0:
        interpretation = "The cost-aware policy has lower expected regret in this scenario."
    else:
        interpretation = "The interval includes zero; neither policy has a clear expected-regret advantage."
    worse = [SCENARIOS.get(scenario, scenario) for scenario, (contrast, _) in contrasts.items()
             if scenario != "heterogeneous" and contrast["ci_low"] > 0]
    counterevidence = (" It has higher expected regret than the myopic policy in " + ", ".join(worse) + ".") if worse else ""
    text = f"""
## Comparison with a myopic policy

This comparison was added after confirmation, using new seeds
{min(config['seeds'])}–{max(config['seeds'])} and the unchanged cost-aware policy.
The myopic policy uses the same estimated demand distribution and budget cap,
but optimizes only immediate cost. On heterogeneous demand, cost-aware minus myopic
regret is **{interval(regret)}** and realized cost is **{interval(cost)}**.
{interpretation}{counterevidence} This is a secondary analysis.
"""
    detail = f"""
## Myopic policy comparison added after confirmation

The candidate algorithm, selected planning parameter, and primary comparison
were unchanged. The added `myopic_cost` policy uses the same estimated demand
distribution, uniform tail completion, and budget cap B. It minimizes immediate
estimated cost without assigning value to future information. This checks
whether reduced spending alone explains the cost-aware policy's results.

There are {len(config['seeds'])} new paired seeds, {min(config['seeds'])} through
{max(config['seeds'])}, with {config['horizon']} rounds. Both metrics are computed
from [outcomes.csv](../{link(path.parent / 'outcomes.csv')}) with a paired bootstrap
over whole trajectories. Positive relative reduction favors the cost-aware
policy. Intervals are descriptive and are not adjusted for multiple comparisons.

{md_table(['Scenario', 'Regret Δ, 95% CI', 'Regret reduction', 'Realized-cost Δ, 95% CI', 'Cost reduction', 'Cost-aware extra', 'Myopic extra'], rows)}

Heterogeneous result: {interpretation} These results apply to the stated simulator.
Source: [summary.json](../{link(path)}).
"""
    return text, detail


def plot_identification(reproduction, output, plt, source_path):
    analytic = reproduction["analytic"]
    worlds = analytic["world_pmfs"]
    support = np.arange(len(next(iter(worlds.values()))))
    high = max(analytic["cost_settings"], key=lambda row: row["critical_quantile"])
    low = min(analytic["cost_settings"], key=lambda row: row["critical_quantile"])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    for name, probability in worlds.items():
        axes[0].plot(support, probability, marker="o", markersize=3, label=f"World {name}")
    axes[0].axvline(analytic["historical_order"], color="#728099", linestyle="--", label="Historical order")
    axes[0].set(xlabel="Demand (declared integer units)", ylabel="Probability", title="Different latent demand")
    axes[0].legend(fontsize=8, frameon=False)
    laws = analytic["observation_laws"]
    names = list(laws)
    if laws[names[0]] != laws[names[1]]:
        raise ValueError("Identification witness laws differ.")
    values = laws[names[0]]
    positions = np.arange(len(values))
    for index, name in enumerate(names):
        axes[1].bar(positions + (index - .5) * .3, [row["probability"] for row in laws[name]], width=.28, label=name)
    axes[1].set_xticks(positions, [f"sales={row['sales']}\nsold out={row['sold_out']}" for row in values])
    axes[1].set(ylabel="Observable probability", title="Identical censored observations", ylim=(0, 1))
    axes[1].legend(title="World", frameon=False, fontsize=8)
    axes[2].plot(support, high["full_cdf_band_regret_by_order"], color="#167d8d", label="Entire CDF band")
    axes[2].plot(support, high["two_witness_worlds_regret_by_order"], linestyle="--", color="#c68c36", label="Only worlds A/B")
    q, regret = high["full_cdf_band_minimax_order"], high["full_cdf_band_minimax_regret"]
    axes[2].scatter([q], [regret], color="#167d8d", zorder=4)
    axes[2].annotate(f"q={q}, R={num(regret, 0)}", (q, regret), xytext=(6, 8), textcoords="offset points", fontsize=9)
    axes[2].set(xlabel="Order", ylabel="Worst-case expected regret (cost units)", title=f"Critical quantile {num(high['critical_quantile'], 2)}")
    axes[2].legend(fontsize=8, frameon=False)
    figure.suptitle("Censored observations and optimal inventory orders", x=.06, ha="left", weight="bold", fontsize=14)
    figure.text(.06, .025, f"Constructed population example; no sampling interval. Lower critical quantile {num(low['critical_quantile'], 2)} has zero-regret order {low['full_cdf_band_minimax_order']}.\nSource: {link(source_path)}. Full-band regret is distinct from the smaller two-witness problem.", fontsize=8)
    figure.subplots_adjust(left=.06, right=.98, bottom=.24, top=.80, wspace=.33)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_learning(curves, config, output, plt):
    selected = ["sales", "uniform", "uncertainty", "decision", "adaptive_cost", "km_uniform", "geometric_ts", "full_information"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    common = curves.filter(pl.col("scenario") == "heterogeneous")
    for policy in selected:
        data = common.filter(pl.col("policy") == policy).sort("seed", "round")
        if data.is_empty():
            continue
        matrix = np.array(data.group_by("seed", maintain_order=True).agg(pl.col("regret").sort_by("round"))["regret"].to_list())
        cumulative = matrix.cumsum(axis=1)
        mean = cumulative.mean(axis=0)
        rng = np.random.default_rng(2026)
        weights = rng.multinomial(len(matrix), np.full(len(matrix), 1 / len(matrix)), size=1000) / len(matrix)
        low, high = np.quantile(weights @ cumulative, [.025, .975], axis=0)
        rounds = np.arange(1, len(mean) + 1)
        axes[0].plot(rounds, mean, label=LABELS[policy], color=COLORS[policy], linewidth=1.8)
        axes[0].fill_between(rounds, low, high, color=COLORS[policy], alpha=.10)
        if policy in {"uniform", "uncertainty", "decision", "adaptive_cost", "full_information"}:
            axes[1].plot(rounds, mean, color=COLORS[policy], linewidth=1.8)
            axes[1].fill_between(rounds, low, high, color=COLORS[policy], alpha=.10)
    for axis in axes:
        axis.set(xlabel="Online round (shared warmup excluded)", ylabel="Cumulative conditional expected regret (cost units)")
    axes[0].set_title("Full scale with independent baselines", fontsize=11)
    axes[1].set_title("Allocation policies and full-information reference", fontsize=11)
    figure.suptitle(f"Heterogeneous confirmation · {len(config['seeds'])} paired seeds · daily cap B={config['extra_budget']}", x=.07,
                   ha="left", fontsize=14, weight="bold")
    figure.legend(*axes[0].get_legend_handles_labels(), ncol=4, frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(.07, .10))
    figure.text(.07, .025, "Shading: pointwise 95% bootstrap intervals for the mean over complete independent seed trajectories; not simultaneous bands.\nThe oracle knows the generating PMF, never the coming realized demand. Cost-aware allocation may spend less than the same cap.", fontsize=8)
    figure.subplots_adjust(left=.07, right=.98, bottom=.29, top=.81, wspace=.28)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_scenarios(averages, config, output, plt):
    lookup = row_lookup(averages)
    policies = [policy for policy in POLICIES if policy in config["policies"]]
    scenarios = config["scenarios"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=True)
    for axis, metric, title in zip(axes, ["cumulative_expected_regret", "normalized_cost", "extra_units"],
                                   ["Mean expected regret", "Mean realized cost / demand", "Mean extra units actually spent"]):
        values = np.array([[lookup[(scenario, policy)][metric] for scenario in scenarios] for policy in policies])
        shown = axis.imshow(values, aspect="auto", cmap="YlGnBu")
        for y in range(len(policies)):
            for x in range(len(scenarios)):
                color = "white" if values[y, x] > shown.norm.vmax * .65 else "#183249"
                axis.text(x, y, num(values[y, x], 2 if metric == "normalized_cost" else 0), ha="center", va="center", fontsize=7, color=color)
        axis.set_xticks(np.arange(len(scenarios)), [SCENARIOS.get(name, name) for name in scenarios], rotation=45, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(policies)), [LABELS[name] for name in policies], fontsize=9)
        axis.set_title(title, fontsize=11, weight="bold")
        figure.colorbar(shown, ax=axis, orientation="horizontal", pad=.15, fraction=.03)
    figure.suptitle("Every confirmation scenario and baseline", x=.15, ha="left", weight="bold", fontsize=15)
    figure.text(.15, .025, f"{len(config['seeds'])} trajectories per cell · horizon {config['horizon']} · {config['n_contexts']} contexts · shortage:excess {config['underage']}:{config['overage']}.\nSame budget cap does not mean same actual spending. Full-information reference is privileged. All values condition on the declared simulator.", fontsize=8)
    figure.subplots_adjust(left=.15, right=.98, bottom=.17, top=.92, wspace=.13)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def phase_plot(path, output, plt):
    if not path.exists():
        return False
    data = load(path)
    rows = data.get("cells", data.get("rows", data.get("results", [])))
    if not isinstance(rows, list) or not rows:
        warnings.warn("Optional phase-grid schema not recognized; core report is complete.", stacklevel=2)
        return False
    try:
        xkey = "budget" if "budget" in rows[0] else "extra_budget"
        xs = sorted({row[xkey] for row in rows})
        ys = sorted({row["initial_order"] for row in rows})
        cost_ratios = sorted({row["underage"] for row in rows})
        panels = []
        seen = set()
        for underage in cost_ratios:
            matrix = np.full((len(ys), len(xs)), np.nan)
            for row in rows:
                if row["underage"] != underage:
                    continue
                identity = (underage, row["initial_order"], row[xkey])
                if identity in seen:
                    raise ValueError("Duplicate phase-grid cost/cap/budget cell.")
                seen.add(identity)
                comparison = row.get("comparisons", {}).get("uniform", row.get("comparison", {}))
                value = comparison.get("relative_reduction", row.get("relative_reduction"))
                if value is not None:
                    matrix[ys.index(row["initial_order"]), xs.index(row[xkey])] = 100 * value
            panels.append(matrix)
        finite = np.concatenate([panel[np.isfinite(panel)] for panel in panels])
        if not finite.size:
            return False
        figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.7), squeeze=False, sharey=True)
        limit = max(float(np.max(np.abs(finite))), 1e-9)
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#edf0f3")
        phase_config = data.get("config", {})
        overage = phase_config.get("overage", 1)
        for axis, underage, matrix in zip(axes[0], cost_ratios, panels):
            shown = axis.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
            for i in range(len(ys)):
                for j in range(len(xs)):
                    if np.isfinite(matrix[i, j]):
                        axis.text(j, i, num(matrix[i, j], 1, signed=True) + "%", ha="center", va="center", fontsize=9,
                                  color="white" if abs(matrix[i, j]) > .65 * limit else "#183249")
                    else:
                        axis.text(j, i, "undefined", ha="center", va="center", fontsize=8, color="#65758a")
            axis.set_xticks(range(len(xs)), xs)
            axis.set_yticks(range(len(ys)), ys)
            axis.set(xlabel="Daily extra-stock budget cap", title=f"Cost ratio {underage}:{overage} · τ={underage / (underage + overage):.2f}")
        axes[0, 0].set_ylabel("Historical warmup order")
        figure.suptitle("Phase grid: relative regret reduction of decision versus uniform", x=.07, ha="left", weight="bold", fontsize=14)
        figure.subplots_adjust(left=.07, right=.88, bottom=.25, top=.78, wspace=.15)
        color_axis = figure.add_axes([.90, .25, .015, .53])
        figure.colorbar(shown, cax=color_axis, label="Relative reduction (%) · positive favors decision")
        seed_list = phase_config.get("seeds", [])
        phase_scope = f"Horizon {phase_config.get('horizon', 'unspecified')} rounds · {len(seed_list)} paired seeds ({min(seed_list)}–{max(seed_list)}) per cell" if seed_list else "Paired-seed sensitivity analysis"
        figure.text(.07, .025, phase_scope + f" · descriptive, unadjusted sensitivity analysis.\nUndefined relative reductions are omitted from the colour scale. Source: {link(path)}; not the primary confirmation.", fontsize=8)
        figure.savefig(output, dpi=180)
        plt.close(figure)
        return True
    except (KeyError, TypeError, ValueError) as error:
        warnings.warn(f"Skipping optional phase grid: {error}", stacklevel=2)
        return False


def confirmation_table(summary):
    rows = []
    for scenario in summary["protocol"]["config"]["scenarios"]:
        comparisons = summary["comparisons"].get(scenario, {})
        rows.append([scenario, interval(comparisons["uniform"]) if "uniform" in comparisons else "unavailable",
                     interval(comparisons["uncertainty"]) if "uncertainty" in comparisons else "unavailable"])
    return md_table(["Scenario", "Decision − uniform, 95% CI", "Decision − uncertainty, 95% CI"], rows)


def selected_table(averages, scenario, complete=False):
    chosen = POLICIES if complete else ["sales", "lightgbm_sales", "km_uniform", "km_upper", "minimax", "geometric_ts", "uniform", "uncertainty", "decision", "adaptive_cost", "full_information"]
    lookup = row_lookup(averages)
    rows = []
    for policy in chosen:
        if (scenario, policy) not in lookup:
            continue
        row = lookup[(scenario, policy)]
        rows.append([LABELS[policy], num(row["cumulative_expected_regret"]), num(row["cost"], 0),
                     pct(row["fill_rate"]), num(row["extra_units"], 0)])
    return md_table(["Policy", "Expected regret", "Realized cost", "Fill rate", "Extra units"], rows)


def write_documents(summary, averages, outcomes, reproduction, context, selection, report_dir, figures, has_phase,
                    mechanism_path, phase_path, reproduction_path, context_path, selection_path):
    protocol, config = summary["protocol"], summary["protocol"]["config"]
    primary = summary["comparisons"]["heterogeneous"]["uniform"]
    mechanism = summary["comparisons"]["heterogeneous"]["uncertainty"]
    analytic = reproduction["analytic"]
    high = max(analytic["cost_settings"], key=lambda row: row["critical_quantile"])
    low = min(analytic["cost_settings"], key=lambda row: row["critical_quantile"])
    meta = context["metadata"]
    comparison_table = confirmation_table(summary)
    main_table = selected_table(averages, "heterogeneous")
    lookup = row_lookup(averages)
    stationary = [name for name, env in summary["environment_metadata"].items() if env.get("stationary", True)]
    adaptive = lookup[("heterogeneous", "adaptive_cost")]
    uniform = lookup[("heterogeneous", "uniform")]
    mechanism_text, mechanism_detail = mechanism_sections(mechanism_path, config["seeds"], selection["development_seeds"])
    readme = f"""# Fresh Demand: inventory learning from censored sales

Inventory policies learn demand from sales limited by stock. Experiments test
whether allocating extra stock to informative contexts reduces ordering cost.
They use known synthetic distributions and distributions fitted to selected
retail records; they do not estimate real lost demand or policy effects.

## Confirmation results

The predeclared primary comparison is fixed-budget **decision versus uniform**
allocation on heterogeneous demand. Cumulative expected-regret difference:
**{interval(primary)}**, candidate minus baseline, with a paired 95% interval.
{conclusion(primary)} The comparison with allocation based on CDF uncertainty
is **{interval(mechanism)}**. Both comparisons were fixed before confirmation.

{comparison_table}

Negative differences favor decision allocation. Intervals resample complete
paired trajectories. Secondary intervals are descriptive and are not adjusted
for multiple comparisons. All results depend on the stated demand distributions
and protocol.

![Confirmation learning curves]({link(figures / 'learning_curves.png')})

## Methods

- A risk-set estimator counts `D <= d` only when order `Q > d` reveals that
  event. Its anytime Hoeffding bands require stationary independent demand
  within fixed contexts and orders chosen before observing current demand.
- An exact minimax-regret calculation over all CDFs between the supplied bounds,
  computed in linear time and checked against independent linear programs.
- A decision score prioritizes information that reduces regret. An exact integer
  allocator distributes the budget. Uniform and CDF-width allocation use the
  same base orders and constraint. Mean-field planning uses expected counts,
  without a Bayesian value function or regret bound.
- The cost-aware policy can leave budget unused when estimated immediate cost
  exceeds the information benefit. Baselines: Kaplan–Meier, geometric Bayes and
  Thompson sampling with censored feedback, LightGBM, and alternative unknown-tail
  completions.

## Identifiability example

Two constructed demand laws yield identical sales and sold-out observations at
historical order {analytic['historical_order']}. At critical quantile
{num(low['critical_quantile'], 2)}, order {low['full_cdf_band_minimax_order']} has
zero regret across the entire population CDF band. At quantile
{num(high['critical_quantile'], 2)}, the full-band minimax order is
{high['full_cdf_band_minimax_order']} with regret {num(high['full_cdf_band_minimax_regret'], 0)};
restricting uncertainty to the two witness worlds gives regret
{num(high['two_witness_worlds_minimax_regret'], 0)}. These minimize regret over different
ambiguity sets. The distributions and costs illustrate a published identification argument.

![Identification mechanism]({link(figures / 'identification.png')})

## Cost and stock allocation

Confirmation uses {len(config['seeds'])} independent seeds, {config['horizon']}
online rounds, {config['n_contexts']} contexts per round, known support
0..{config['max_demand']}, and shortage:excess costs
{config['underage']}:{config['overage']}. Shared warmup: {config['warmup']} rounds
at order {config['initial_order']}; its costs are excluded equally. Every
online shortage and excess cost, including exploration stock, is charged.

{main_table}

Values are seed means in the heterogeneous scenario. The PMF oracle uses the
same action set without seeing next demand; the full-information learner sees
past demand. Fixed policies spend B={config['extra_budget']} extra units when feasible.
The cost-aware policy uses the same cap but spends {num(adaptive['extra_units'], 0)}
mean extra units versus {num(uniform['extra_units'], 0)} for uniform. This secondary
comparison allows different spending.

{mechanism_text}

![All baselines and scenarios]({link(figures / 'scenario_results.png')})

## Data and limitations

Retail contexts use {meta['sampled_series']:,} training-only store–product pairs.
Clustering features use {meta['feature_days']} days; empirical demand PMFs use
{meta['fully_stocked_fit_rows']:,} fully stocked rows from {meta['fit_days']} days
({meta['fit_window'][0]}–{meta['fit_window'][1]}). A single scale factor based on
context-specific 99th percentiles, followed by rounding and clipping, sets the
integer support. The clipped-row fraction is
{num(100 * meta['scale']['clipped_row_fraction'], 4)}%. Zero empirical probabilities
remain zero. Demand is resampled independently within each context. Stock
availability depends on demand, so fully stocked records are a selected subset;
they do not identify demand during natural stockouts.

Real data lack historical inventory thresholds and assignment propensities.
Known order thresholds, bounded support, costs, context frequencies, and shifts
are simulator assumptions. CDF guarantees apply only under their stationary
conditions; constructed shift is an empirical stress test. Empty confidence
sets cause the experiment runner to log a fallback to uniform allocation.
The CLI refuses a recommendation when the confidence set is empty. Costs use
scenario units; the model does not represent currency or product shelf life.

## Reproduce

Run these commands from the repository root.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-lock.txt
uv pip install ".[dev]"
fresh-demand download --data-dir data/raw
python scripts/build_active_contexts.py --train-file data/raw/train.parquet --series-file reports/main/sampled_series.parquet --output reports/active/context_model.json --scale-mode tail_guard
fresh-demand active-reproduce --output reports/reproduction_active/reproduction
fresh-demand active-run --config configs/active/development.json --output reports/reproduction_active/development
fresh-demand active-select --development reports/reproduction_active/development --config configs/active/development.json --output reports/reproduction_active/frozen_selection.json
fresh-demand active-run --config configs/active/confirmation.json --selection reports/reproduction_active/frozen_selection.json --output reports/reproduction_active/confirmation
fresh-demand active-grid --config configs/active/phase_grid.json --selection reports/reproduction_active/frozen_selection.json --output reports/reproduction_active/phase_grid
fresh-demand active-run --config configs/active/mechanism_check.json --selection reports/reproduction_active/frozen_selection.json --output reports/reproduction_active/mechanism_check
python scripts/write_active_report.py --report-dir reports/reproduction_active/confirmation --selection reports/reproduction_active/frozen_selection.json --reproduction reports/reproduction_active/reproduction/identification.json --phase-grid reports/reproduction_active/phase_grid/summary.json --mechanism-check reports/reproduction_active/mechanism_check/summary.json
fresh-demand active-plan --history reports/active/service_demo/history.csv --config configs/active/confirmation.json --selection reports/reproduction_active/frozen_selection.json --output reports/reproduction_active/next_orders.json --policy adaptive_cost --seed 42
pytest -q
```

Existing completed runs are preserved; use fresh output directories when
rerunning. The serving demo replays a complete known-order censored log; raw
FreshRetail stock flags cannot substitute for historical inventory thresholds.
Development seeds {selection['development_seeds']} and confirmation
seeds {min(config['seeds'])}..{max(config['seeds'])} are disjoint. Planning batch
{selection['planning_samples']} was frozen on development outcomes. The
[protocol](docs/experimental_protocol.md) records the cost-aware policy added during development.
The [research report](docs/research_report.md) contains every policy,
coverage diagnostics, spending, and evidence paths; [identification](docs/identification.md)
gives the exact scope and proof. The [forecasting and Chronos baseline](docs/baseline_forecasting.md)
is documented separately. [Russian notes](docs/interview_notes_ru.md) cover the
decision problem, implementation, and interview discussion.

Related work: [Kumar–Mouchtaki (2026)](https://arxiv.org/abs/2602.16842)
and [Mukherjee et al. (2026)](https://arxiv.org/abs/2605.14840).
The project implements and evaluates methods without new theoretical claims.
[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
is CC BY 4.0. Independently written software is [MIT](LICENSE).
"""
    detail = f"""# Confirmation research report

Source files: `{link(report_dir / 'summary.json')}` and
`{link(report_dir / 'outcomes.csv')}`. The report script reads completed results.
Configuration hash: `{protocol['config_sha256']}`; context hash:
`{protocol['context_sha256']}`; frozen-selection hash:
`{protocol.get('selection_sha256')}`. Runtime: {num(summary['elapsed_seconds'], 1)} seconds.

## Predeclared comparisons

{comparison_table}

{conclusion(primary)} Differences are candidate minus baseline in cumulative
conditional expected regret. The reference order minimizes expected cost under
the true PMF using the same finite action set, without access to the next
realized demand. Paired intervals resample whole independent trajectories,
preserving all rounds and contexts. Secondary comparisons are descriptive.

## Full policy results

Tables include every configured policy. Cost is the total realized scenario
cost; the second table gives cost divided by demand and coverage diagnostics.
Means describe repeated runs of the stated environment. Every policy also
maintains a risk-set CDF band from its own observations. Coverage refers to
that auxiliary band, rather than the Kaplan–Meier or Bayesian point estimate.

"""
    for scenario in config["scenarios"]:
        detail += f"\n## {scenario}\n\n{selected_table(averages, scenario, complete=True)}\n"
        diagnostic = []
        for policy in config["policies"]:
            row = lookup[(scenario, policy)]
            diagnostic.append([LABELS.get(policy, policy), num(row["normalized_cost"], 3),
                               pct(row["band_familywise_coverage"]), pct(row["final_identified_fraction"]),
                               num(row["fallback_rounds"], 2)])
        detail += "\n" + md_table(["Policy", "Mean cost/demand", "Seeds with path coverage", "Final identified fraction", "Fallback rounds"], diagnostic) + "\n"
        stationary_flag = summary["environment_metadata"][scenario].get("stationary", True)
        detail += ("\nThis environment satisfies the stationary confidence assumptions. " if stationary_flag else "\nThis environment shifts, so the stationary coverage guarantee does not apply. ")
        detail += "Coverage is checked before each action. The final identified fraction uses observations available before the last action.\n"
    secondary_rows = []
    for scenario in config["scenarios"]:
        for baseline in ["uniform", "km_uniform", "geometric_ts"]:
            regret = paired_delta(outcomes, scenario, "adaptive_cost", baseline, "cumulative_expected_regret", config.get("bootstrap_replicates", 5000))
            cost = paired_delta(outcomes, scenario, "adaptive_cost", baseline, "cost", config.get("bootstrap_replicates", 5000))
            secondary_rows.append([scenario, LABELS[baseline], interval(regret),
                                   pct(regret["relative_reduction"]) if regret["relative_reduction"] is not None else "undefined",
                                   interval(cost), pct(cost["relative_reduction"]) if cost["relative_reduction"] is not None else "undefined"])
    detail += "\n## Secondary cost-aware policy comparisons\n\n"
    detail += "These secondary comparisons use complete seed pairs and 95% bootstrap intervals computed from outcomes.csv. Positive relative reduction favors the cost-aware policy. Intervals are not adjusted for multiple comparisons. The cost-aware policy may leave budget unused; this differs from the fixed-budget primary test.\n\n"
    detail += md_table(["Scenario", "Baseline", "Regret Δ, 95% CI", "Relative regret reduction", "Realized-cost Δ, 95% CI", "Relative cost reduction"], secondary_rows) + "\n"
    detail += mechanism_detail
    detail += f"""
## Assumptions and limits

The full population ambiguity band gives order {high['full_cdf_band_minimax_order']}
and regret {num(high['full_cdf_band_minimax_regret'], 0)}; only the two witness worlds
give regret {num(high['two_witness_worlds_minimax_regret'], 0)}. Neither is a
finite-sample confidence interval. Exact CDF-band optimization is verified
independently by LP; this does not reproduce the full finite-sample theorem in
Kumar–Mouchtaki. See [identification](identification.md) and [reproduction](reproduction.md).

Mean-field planning evaluates nonlinear regret at expected fractional counts;
it is not expected regret over future datasets. Unknown-tail completions are
assumptions. The geometric baseline is correctly specified only for
the geometric control. Kaplan–Meier uses censored-before-event tie handling
under the declared D>=Q observation convention.

All policies observe one outcome from each context every round. Context
frequency and the extra-stock cap are experimental design choices. Fixed
allocation and allocation with an upper budget limit use different spending
rules. Both extra units and total cost are reported.

Stationary scenarios: {', '.join(stationary)}. Under constructed shift, old
observations need not represent current demand and confidence coverage is not
asserted. Empty-polytope action fallbacks are counted. Even stationary
guarantees condition on the known finite support, fixed contexts, predictable
orders, independence, and the declared finite horizon.
The experiment runner logs a fallback to uniform allocation when a confidence
set is empty. The serving CLI returns an error without a recommendation.

Retail-derived PMFs describe selected fully stocked historical sales, scaled
and coarsened. They are not observed natural-stockout counterfactuals. Context
model provenance, clipping, and cluster sizes are in
[`context_model.json`](../{link(context_path)}). Official retail
evaluation outcomes are excluded from context construction.

## Result files

- [Confirmation summary](../{link(report_dir / 'summary.json')}) and
  [complete per-seed results](../{link(report_dir / 'outcomes.csv')}).
- [Round-level trajectories](../{link(report_dir / 'learning_curves.parquet')}).
- [Frozen development selection](../{link(selection_path)}).
- [Constructed identification and fixed-audit study](../{link(reproduction_path)}).
- [Experimental protocol and development amendment](experimental_protocol.md).
- [Former forecasting baseline](baseline_forecasting.md).

Software versions: {json.dumps(protocol['versions'], sort_keys=True)}.
Results apply to the stated simulators. They do not estimate operational ROI,
financial savings, or latent demand in the original retail data. The project
does not introduce a new statistical theory.
"""
    verification = report_dir / "source_verification.json"
    if verification.exists():
        checked = load(verification)
        detail += (
            "\n## Source verification\n\n"
            f"The recorded simulation-core hash match is `{checked.get('all_core_match')}` "
            f"against commit `{checked.get('confirmation_core_commit')}`. "
            "The check covers the simulation core. Serving changes made after the run began "
            "are recorded separately, as are the post-confirmation mechanism extensions. "
            f"See [source_verification.json](../{link(verification)}).\n"
        )
    tests_path = PROJECT / "reports/active/test_summary.json"
    if tests_path.exists():
        tests = load(tests_path)
        passed = int(tests["tests"]) - int(tests["failures"]) - int(tests["errors"]) - int(tests["skipped"])
        test_text = (
            f"Tests: **{passed} passed**, {tests['failures']} failures, "
            f"{tests['errors']} errors, and {tests['skipped']} skips; "
            "[test_summary.json](reports/active/test_summary.json)."
        )
        readme += "\n" + test_text + "\n"
        detail += "\n## Tests\n\n" + test_text.replace(
            "](reports/active/test_summary.json)", "](../reports/active/test_summary.json)"
        ) + "\n"
    serving = PROJECT / "reports/active/service_demo/verification.json"
    if serving.exists():
        verified = load(serving)
        detail += (
            "\n## Serving replay\n\n"
            f"The saved known-order history reproduces the recorded order at round {verified['round']}: "
            f"exact match `{verified['exact_match']}`, oracle inputs used `{verified['oracle_inputs_used']}`. "
            "The check verifies log replay through the censored-feedback interface, rather than a real retail policy effect. "
            "See [verification.json](../reports/active/service_demo/verification.json), "
            "[history.csv](../reports/active/service_demo/history.csv), "
            "[next_orders.json](../reports/active/service_demo/next_orders.json), and "
            "[decision_replay.json](../reports/active/service_demo/decision_replay.json).\n"
        )
    if has_phase:
        detail += f"\n## Exploratory phase grid\n\n![Sensitivity grid](../reports/active/figures/phase_grid.png)\n\nThis grid is a sensitivity study; it does not replace the predeclared primary comparison. Source: [summary.json](../{link(phase_path)}).\n"
    return readme, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=PROJECT / "reports/active/confirmation")
    parser.add_argument("--reproduction", type=Path, default=PROJECT / "reports/active/reproduction/identification.json")
    parser.add_argument("--context", type=Path, default=PROJECT / "reports/active/context_model.json")
    parser.add_argument("--selection", type=Path, default=PROJECT / "reports/active/frozen_selection.json")
    parser.add_argument("--phase-grid", type=Path, help="Phase summary; defaults to the selected run's sibling phase_grid directory")
    parser.add_argument("--mechanism-check", type=Path, help="Post-confirmation summary; defaults to the selected run's sibling mechanism_check directory")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    phase_path = args.phase_grid.resolve() if args.phase_grid is not None else report_dir.parent / "phase_grid/summary.json"
    mechanism_path = args.mechanism_check.resolve() if args.mechanism_check is not None else report_dir.parent / "mechanism_check/summary.json"
    for explicit, path in [(args.phase_grid, phase_path), (args.mechanism_check, mechanism_path)]:
        if explicit is not None and not path.is_file():
            raise FileNotFoundError(f"Explicit report evidence is missing: {path}")
    summary = load(report_dir / "summary.json")
    config = summary["protocol"]["config"]
    if config.get("stage") != "confirmation":
        raise ValueError("A completed confirmation run is required; do not report development as confirmation.")
    reproduction, context, selection = load(args.reproduction), load(args.context), load(args.selection)
    if set(config["seeds"]) & set(selection["development_seeds"]):
        raise ValueError("Development and confirmation seeds overlap.")
    outcomes = pl.read_csv(report_dir / "outcomes.csv")
    curves = pl.read_parquet(report_dir / "learning_curves.parquet")
    if outcomes.select(pl.struct("scenario", "seed", "policy").is_duplicated().any()).item():
        raise ValueError("Duplicate confirmation outcomes.")
    if outcomes.height != len(config["seeds"]) * len(config["scenarios"]) * len(config["policies"]):
        raise ValueError("Incomplete policy/seed/scenario confirmation coverage.")
    averages = aggregate(outcomes)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fresh-demand-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    figures = PROJECT / "reports/active/figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_identification(reproduction, figures / "identification.png", plt, args.reproduction.resolve())
    plot_learning(curves, config, figures / "learning_curves.png", plt)
    plot_scenarios(averages, config, figures / "scenario_results.png", plt)
    has_phase = phase_plot(phase_path, figures / "phase_grid.png", plt)
    readme, detail = write_documents(summary, averages, outcomes, reproduction, context, selection, report_dir, figures, has_phase,
                                   mechanism_path, phase_path, args.reproduction.resolve(), args.context.resolve(), args.selection.resolve())
    (PROJECT / "README.md").write_text(readme)
    (PROJECT / "docs/research_report.md").write_text(detail)
    print(json.dumps({"readme_words": len(readme.split()), "outcomes": outcomes.height,
                      "scenarios": config["scenarios"], "figures": str(figures), "phase_grid_included": has_phase}))


if __name__ == "__main__":
    main()
