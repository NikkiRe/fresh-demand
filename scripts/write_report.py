#!/usr/bin/env python3
"""Render the forecasting report and figure from saved experiment results."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import warnings


PROJECT = Path(__file__).resolve().parents[1]
DOCUMENT_ROOT = PROJECT
METHODS = ("raw", "uniform", "profile")
COLORS = {"raw": "#62738a", "uniform": "#d58c39", "profile": "#23877a"}


def load_json(path: Path) -> dict:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def number(value, digits: int = 4, signed: bool = False) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "undefined"
    return format(value, f"{'+' if signed else ''},.{digits}f")


def percent(value, signed: bool = False) -> str:
    if value is None:
        return "undefined"
    return number(100 * value, digits=2, signed=signed) + "%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def relative(path: Path, anchor: Path | None = None) -> str:
    return Path(os.path.relpath(path, DOCUMENT_ROOT if anchor is None else anchor)).as_posix()


def optional_chronos(report_dir: Path) -> str:
    path = report_dir / "chronos.json"
    if not path.exists():
        return ""
    try:
        result = load_json(path)
        if result.get("mode") != "frozen_test_daily_updated_h1":
            return ""
        points = result.get("point_metrics")
        orders = result.get("quantile_metrics")
        if not isinstance(points, dict) or not isinstance(orders, dict) or "pred_chronos_q50" not in points:
            return ""
        rows = []
        comparisons = [
            ("Seasonal lag 7", "pred_seasonal_lag7", None),
            ("Trailing mean 7", "pred_mean7", None),
            ("LightGBM raw", "pred_raw_q50", "pred_raw_q75"),
            ("LightGBM profile", "pred_profile_q50", "pred_profile_q75"),
            ("Chronos-2", "pred_chronos_q50", "pred_chronos_q75"),
        ]
        for label, point_key, order_key in comparisons:
            if point_key not in points:
                continue
            metrics = points[point_key]
            cost = orders.get(order_key, {}).get("normalized_cost") if order_key else None
            rows.append([label, percent(metrics["wape"]), percent(metrics["bias"], signed=True),
                         number(cost) if cost is not None else "—"])
        count = result["series_all"]
        scored = result["rows_fully_available"]
        date_list = result.get("test_dates", [])
        dates = f"{date_list[0]}–{date_list[-1]}" if date_list else "the frozen test dates"
        measured = table(["Same-cohort model", "Point WAPE", "Bias", "q75 cost"], rows)
        return f"""
## Chronos-2 comparison

Chronos-2 and the baselines are evaluated on the same {count:,} sampled series
and {scored:,} fully available rows on {dates}. Inference runs on CPU without
fine-tuning. Each H=1 forecast uses the updated history of raw sales, without
covariates or cross-series learning. Chronos-2 is excluded from deployment selection.

{measured}

Inference time: {number(result.get('inference_seconds'), 1)} seconds. Overlap with
pretraining data was not independently checked, so this is not verified as an
unseen-dataset evaluation. Source: [`chronos.json`]({relative(path)}).
"""
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        warnings.warn(f"Skipping optional Chronos report: {exc}", stacklevel=2)
        return ""


def chart(summary: dict, recovery: dict, output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fresh-demand-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    test = summary["test"]
    primary = test["bootstrap"]["profile_vs_raw"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=180)
    figure.patch.set_facecolor("#fbfcfe")
    for axis in axes:
        axis.set_facecolor("#fbfcfe")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#dde3eb", linewidth=0.6)

    costs = [test["models"][f"pred_{method}_q75"]["normalized_cost"] for method in METHODS]
    bars = axes[0].bar(METHODS, costs, color=[COLORS[method] for method in METHODS], width=0.62)
    axes[0].bar_label(bars, labels=[number(value) for value in costs], padding=5, fontsize=9)
    axes[0].set_ylim(0, max(costs) * 1.22 if max(costs) > 0 else 1)
    axes[0].set_title("Day-ahead replenishment: q75", loc="left", weight="bold", fontsize=12)
    axes[0].set_ylabel("Asymmetric cost / total normalized demand")
    axes[0].set_xlabel("Recovery variant · Cu = 3, Co = 1")

    regimes = list(recovery["regimes"])
    locations = np.arange(len(regimes))
    for index, method in enumerate(METHODS):
        values = [100 * recovery["regimes"][regime][method]["wape"] for regime in regimes]
        rectangles = axes[1].bar(locations + (index - 1) * 0.25, values, width=0.23,
                                 color=COLORS[method], label=method)
        axes[1].bar_label(rectangles, labels=[number(value, 1) for value in values], padding=3, fontsize=8)
    axes[1].set_xticks(locations, regimes)
    axes[1].set_ylim(bottom=0, top=axes[1].get_ylim()[1] * 1.16)
    axes[1].set_title("Controlled reconstruction of hidden sales", loc="left", weight="bold", fontsize=12)
    axes[1].set_ylabel("WAPE (%)")
    axes[1].set_xlabel("Artificial stockout regime")
    axes[1].legend(ncol=3, frameon=False, fontsize=9, loc="upper center")

    figure.suptitle(f"Fresh Demand · {test['start']} to {test['end']}", x=0.075,
                    ha="left", fontsize=16, weight="bold", color="#233247")
    caption = (
        f"Fully available test days only · primary Δ(profile − raw) = {number(primary['estimate'], signed=True)}; "
        f"95% store-cluster CI [{number(primary['ci_low'])}, {number(primary['ci_high'])}].\n"
        "CI is conditional on the observed dates. Costs are scenario units; reconstruction uses synthetic masks, not natural-stockout truth."
    )
    figure.text(0.075, 0.025, caption, fontsize=8.2, color="#4d5c70", va="bottom")
    figure.subplots_adjust(left=0.075, right=0.98, top=0.81, bottom=0.23, wspace=0.31)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)


def readme(summary: dict, recovery: dict, report_dir: Path, figure_path: Path) -> str:
    metadata, validation, test = summary["metadata"], summary["validation"], summary["test"]
    config = metadata["config"]
    models = test["models"]
    selection = test["selection"]
    primary = test["bootstrap"]["profile_vs_raw"]
    raw_point = models["pred_raw_q50"]
    raw_order = models["pred_raw_q75"]
    cost_reduction = 1 - raw_order["normalized_cost"] / raw_point["normalized_cost"]
    policy_direction = "reduces" if cost_reduction >= 0 else "increases"
    profile_cost_change = models["pred_profile_q75"]["normalized_cost"] / raw_order["normalized_cost"] - 1
    profile_direction = "higher" if profile_cost_change >= 0 else "lower"
    policy = test.get("policy_comparison")
    policy_file = report_dir / "policy_comparison.json"
    if policy is None and policy_file.exists():
        policy = load_json(policy_file)
    policy_interval = ""
    if isinstance(policy, dict) and all(key in policy for key in ("estimate", "ci_low", "ci_high")):
        policy_interval = (
            f"The secondary comparison of q75 with q50 orders gives cost difference "
            f"{number(policy['estimate'], signed=True)}, with 95% store-cluster interval "
            f"[{number(policy['ci_low'])}, {number(policy['ci_high'])}], conditional on the same dates."
        )
        if policy_file.exists():
            policy_interval += f" Source: [`policy_comparison.json`]({relative(policy_file)})."
    rows = []
    for label, point_key, order_key in [
        ("Seasonal lag 7", "pred_seasonal_lag7", "pred_seasonal_lag7"),
        ("Trailing mean 7", "pred_mean7", "pred_mean7"),
        ("LightGBM raw", "pred_raw_q50", "pred_raw_q75"),
        ("LightGBM uniform", "pred_uniform_q50", "pred_uniform_q75"),
        ("LightGBM profile", "pred_profile_q50", "pred_profile_q75"),
    ]:
        point, order = models[point_key], models[order_key]
        rows.append([label, percent(point["wape"]), percent(point["bias"], signed=True),
                     number(order["normalized_cost"]), percent(order["fill_rate"])])
    results_table = table(["Model", "Point WAPE", "Point bias", "Order cost", "Fill rate"], rows)
    recovery_table = table(
        ["Mask regime", "Raw WAPE", "Uniform WAPE", "Profile WAPE"],
        [[regime, *[percent(values[method]["wape"]) for method in METHODS]]
         for regime, values in recovery["regimes"].items()],
    )
    validation_rows = [[
        f"{fold['start']}–{fold['end']}",
        *[number(fold["models"][f"pred_{method}_q75"]["normalized_cost"]) for method in METHODS],
    ] for fold in validation["folds"]]
    validation_rows.append(["Arithmetic fold mean", *[number(selection["validation_costs"][method]) for method in METHODS]])
    validation_table = table(["Validation dates", "Raw cost", "Uniform cost", "Profile cost"], validation_rows)
    sensitivity_table = table(
        ["Shortage:excess cost", "Raw cost", "Uniform cost", "Profile cost"],
        [[f"{ratio}:1", *[number(values[method]["normalized_cost"]) for method in METHODS]]
         for ratio, values in sorted(test["cost_sensitivity"].items(), key=lambda item: float(item[0]))],
    )
    versions = ", ".join(f"{name} {version}" for name, version in metadata["versions"].items())
    return f"""# Demand forecasting baseline

Day-ahead forecasting of fresh-retail demand with corrections for historical
stockouts. The experiment compares raw sales, correction by available hours,
and correction by the hourly demand profile. LightGBM quantiles determine
single-period order quantities.

During stockouts, observed sales are a lower bound on demand. Corrected demand
is an estimate and is evaluated separately from observed sales.

On fully available test days, raw LightGBM q50 has
**{percent(raw_point['wape'])} WAPE**, versus {percent(models['pred_mean7']['wape'])}
for the trailing mean and {percent(models['pred_seasonal_lag7']['wape'])} for seasonal lag 7.
Using q75 instead of q50 for the raw model {policy_direction} scenario cost by
**{percent(abs(cost_reduction))}** at a 3:1 shortage:excess cost ratio; fill rate
changes from **{percent(raw_point['fill_rate'])} to {percent(raw_order['fill_rate'])}**.
Validation selected **{selection['selected_method']}**. Profile recovery has
{percent(abs(profile_cost_change))} {profile_direction} test order cost than raw.

![Forecast and controlled recovery results]({relative(figure_path)})

## Held-out results

The run samples **{metadata['n_series']:,} store–product pairs** from
official training identifiers (seed {config['seed']}); it contains
{metadata['n_train_rows']:,} training rows across {metadata['sampled_stores']:,}
stores. Test dates: **{test['start']}–{test['end']}**, {test['n_days']} days.
Scoring uses {test['n_rows']:,} fully available rows out of
{test['n_total_rows']:,} test rows, across {test['n_stores']:,} scored stores.

{results_table}

LightGBM point metrics use q50; its order metrics use q75. The two simple
baselines use the same point forecast as their order quantity. WAPE and bias
are percentages; positive bias means overprediction. Cost is total
`3 × shortage + 1 × excess` divided by total scored demand. Fill rate is total
fulfilled demand divided by total demand. These are normalized scenario units.

**Predeclared comparison, profile versus raw:** q75 cost difference
**{number(primary['estimate'], signed=True)}**, with 95% paired store-cluster
interval **[{number(primary['ci_low'])}, {number(primary['ci_high'])}]**;
negative favors profile. This primary comparison was fixed before model
selection. The interval is conditional on the observed dates,
using {primary['n_boot']:,} bootstrap draws; it does not quantify future-date
uncertainty.

{policy_interval}

## Validation and deployment selection

{validation_table}

The lowest arithmetic mean validation cost selected **{selection['selected_method']}**
before training read the official evaluation file. Three expanding
validation windows precede the final test; the configuration fixes
{config['n_estimators']} trees, with no test-tuned early stopping. Quantile
crossings are corrected by sorting each row's quantile predictions.

Recovery profiles and priors use only the first {config['warmup_days']} days and
remain frozen. Forecast models refit at each weekly origin; actual preceding
days update lag features within each scoring window. Each day receives a new
**H=1 forecast**; the model does not forecast the entire week from one origin. Same-day realized
weather, discounts, stock flags, and sales are excluded from forecast features.

## Recovery check and cost assumptions

Known sales on {recovery['n_observed_days']:,} fully available store–product days are
artificially hidden. Empirical masks come from training, suffix masks hide the
last four operational hours, and the fixed peak scenario hides predeclared
hours. Evaluation sales never determine mask placement.

{recovery_table}

These tests measure recovery under artificial masks. True demand during natural
stockouts remains unknown. Cost-ratio sensitivity uses the corresponding
quantile for each ratio; the deployed variant remains the validation selection:

{sensitivity_table}

## Reproduce

Python {metadata['python']}; package versions: {versions}.
Elapsed main-run time: {number(metadata.get('elapsed_seconds'), 1)} seconds.
On macOS, LightGBM also requires OpenMP (`brew install libomp`).

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-lock.txt
uv pip install -e ".[dev]"
fresh-demand download --data-dir data/raw
fresh-demand audit data/raw/train.parquet --output reports/train_audit.json
pytest -q

# Validation only: no held-out evaluation data is read.
fresh-demand train --data-dir data/raw --config configs/default.json --output reports/validation

# A fresh output directory preserves the existing frozen results.
fresh-demand train --data-dir data/raw --config configs/default.json --output reports/reproduction --final-test
python scripts/write_report.py --report-dir reports/reproduction

fresh-demand subset --input data/raw/train.parquet --series reports/reproduction/sampled_series.parquet --output data/history_cohort.parquet
fresh-demand predict --model reports/reproduction/artifacts/model.pkl --history data/history_cohort.parquet --output forecasts/next_day.parquet
fresh-demand retrain --history data/history_cohort.parquet --selection reports/reproduction/frozen_selection.json --config configs/default.json --model reports/reproduction/artifacts/retrained.pkl
```

`--final-test` creates the deployment bundle; validation-only training does not.
Prediction history follows the source schema and contains consecutive completed
days. For the original cohort, filter history using `sampled_series.parquet`.
Retraining uses the saved selection. Model bundles are excluded from Git and
can be regenerated locally or loaded from the supplied artifact. Optional Chronos
inference uses a separate locked environment and the same sampled cohort:

```bash
uv venv --python 3.12 .venv-chronos
uv pip install --python .venv-chronos/bin/python -r requirements-chronos-lock.txt
.venv-chronos/bin/python scripts/benchmark_chronos.py --data-dir data/raw --predictions reports/reproduction/test_predictions.parquet --series-file reports/reproduction/sampled_series.parquet --output reports/reproduction/chronos.json --max-series 256 --threads 4 --cache-dir .cache/hf
python scripts/write_report.py --report-dir reports/reproduction
```

## Evidence and limitations

Results are generated from [`run_summary.json`]({relative(report_dir / 'run_summary.json')})
and [`recovery_test.json`]({relative(report_dir / 'recovery_test.json')}). These
record data/config hashes, selection, package versions, counts, and scores.
Dataset revision: `{metadata['dataset_revision']}`. The downloader pins public
files and verifies their hashes. Raw data and binary model artifacts are ignored.

Evaluation uses fully stocked days, which may have a different demand distribution
from stockout days. The seven-day test does not establish performance stability
over longer periods. Positive sales can coexist
with stock flags, so treating flagged hours as entirely unavailable is an
approximation. Fixed warmup profiles can drift; shrinkage, exposure floors,
and recovery caps are heuristic assumptions. Costs use scenario units. The model
does not represent monetary profit, actual spoilage, shelf life, multi-period
inventory, or causal inventory effects. Details are in the
[methodology]({relative(PROJECT / 'docs/methodology.md')}).
{optional_chronos(report_dir)}
## Data, related work, and license

[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
is released by Dingdong-Inc under **CC BY 4.0**; see the
[dataset paper](https://arxiv.org/abs/2505.16319) and
[official baseline](https://github.com/Dingdong-Inc/frn-50k-baseline).
[Azar's pipeline](https://www.mdpi.com/2411-5134/11/5/89) is closely related prior
work on recovery, forecasting, and replenishment. This project compares
implementations and evaluation procedures; it does not propose a new method.
Code is [MIT]({relative(PROJECT / 'LICENSE')}); dataset licensing is
separate. Software citation metadata is in [CITATION.cff]({relative(PROJECT / 'CITATION.cff')}).
"""


def main() -> None:
    global DOCUMENT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=PROJECT / "reports" / "main")
    parser.add_argument("--readme", type=Path, default=PROJECT / "docs/baseline_forecasting.md")
    args = parser.parse_args()
    DOCUMENT_ROOT = args.readme.resolve().parent
    report_dir = args.report_dir.resolve()
    summary = load_json(report_dir / "run_summary.json")
    if "test" not in summary:
        raise ValueError("A completed --final-test run is required to generate the report.")
    recovery = load_json(report_dir / "recovery_test.json")
    figure_path = report_dir / "overview.png"
    content = readme(summary, recovery, report_dir, figure_path)
    chart(summary, recovery, figure_path)
    args.readme.parent.mkdir(parents=True, exist_ok=True)
    args.readme.write_text(content)
    print(f"Generated {args.readme} and {figure_path} from {report_dir}")


if __name__ == "__main__":
    main()
