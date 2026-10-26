# Fresh Demand — learning when lost demand matters

**When should an inventory policy pay to learn demand it cannot see?** This
classic ML and operations-research project studies censored feedback, decision
identification, and budgeted exploration. It uses known synthetic demand laws
and a separately labelled retail-derived simulator. It estimates no real
retailer's counterfactual profit or lost demand.

## Main result, including the failed cases

The predeclared primary comparison is fixed-budget **decision versus uniform**
allocation on heterogeneous demand. Cumulative expected-regret difference:
**+0.91 [-58.26, 59.81]**, candidate minus baseline, with a paired 95% interval.
The interval includes zero; the primary comparison does not establish a clear advantage. The mechanism contrast against CDF-uncertainty allocation
is **-6.79 [-83.54, 80.13]**. Neither comparison is replaced by a more favourable
scenario or policy after confirmation.

| Scenario | Decision − uniform, 95% CI | Decision − uncertainty, 95% CI |
| --- | --- | --- |
| heterogeneous | +0.91 [-58.26, 59.81] | -6.79 [-83.54, 80.13] |
| geometric | +146.76 [92.45, 199.51] | -320.31 [-409.36, -238.35] |
| heavy_tail | -70.07 [-127.39, -11.59] | +43.48 [-11.00, 96.00] |
| fresh_empirical | -48.28 [-91.55, -6.99] | -473.13 [-547.54, -395.83] |
| identified | -11.46 [-16.82, -4.19] | +5.29 [4.13, 6.44] |
| shift | +763.95 [380.66, 1,171.62] | +186.36 [-335.15, 717.62] |

All differences use complete paired simulation seeds. Negative favors decision
allocation. Secondary intervals are descriptive, without multiplicity correction.
They condition on the fixed demand laws and protocol, not the real economy.

![Confirmation learning curves](reports/active/figures/learning_curves.png)

## What was built

- An independently written risk-set estimator observes a threshold event only
  when the known order exposes it. Anytime Hoeffding bands require stationary
  independent demand within fixed contexts and predictable orders.
- An exact, linear-time minimax-regret calculation over a pointwise CDF-band
  polytope, independently checked against probability-simplex LPs.
- A decision-directed information score and exact integer budget allocator,
  compared with uncertainty-directed and uniform allocation using the same
  base rule and spending constraint. Mean-field lookahead is a planning
  approximation, not a Bayesian value function or new regret theorem.
- A secondary cost-aware gate can decline exploration when its working-model
  benefit does not justify its immediate cost. Canonical Kaplan–Meier,
  censored geometric Bayes/Thompson, LightGBM, and explicit tail assumptions
  provide independent classical baselines.

## Economic decision versus demand identification

Two constructed demand laws yield identical sales and sold-out observations at
historical order 8. At critical quantile
0.25, order 5 has
zero regret across the entire population CDF band. At quantile
0.75, the full-band minimax order is
24 with regret 16;
restricting uncertainty to the two witness worlds gives regret
12. These are different ambiguity
sets. This is a conceptual reproduction, not replication of a published table.

![Identification mechanism](reports/active/figures/identification.png)

## Policies, actual cost, and exploration spending

Confirmation uses 30 independent seeds, 400
online rounds, 4 contexts per round, known support
0..40, and shortage:excess costs
3:1. Shared warmup: 40 rounds
at order 8; its costs are excluded equally. Every
online shortage and excess cost, including exploration stock, is charged.

| Policy | Expected regret | Realized cost | Fill rate | Extra units |
| --- | --- | --- | --- | --- |
| Sales quantile | 21,093.09 | 40,340 | 44.4% | 0 |
| LightGBM sales | 21,082.93 | 40,318 | 45.2% | 0 |
| KM: uniform | 3,113.50 | 22,387 | 77.1% | 0 |
| KM: upper | 3,156.83 | 22,434 | 77.1% | 0 |
| CDF-band minimax | 2,639.38 | 21,827 | 92.0% | 0 |
| Geometric Thompson | 3,508.66 | 22,740 | 89.2% | 0 |
| Uniform allocation | 1,004.81 | 20,230 | 87.3% | 3,200 |
| CDF uncertainty | 1,012.51 | 20,252 | 87.1% | 3,200 |
| Decision allocation | 1,005.72 | 20,232 | 87.2% | 3,200 |
| Cost-aware gate | 858.96 | 20,088 | 86.3% | 2,466 |
| Past full information | 160.55 | 19,345 | 85.6% | 0 |

Values are seed means on the primary heterogeneous scenario. The oracle knows
the true PMF, uses the same order set, and never observes next demand. Past
full information is a privileged learning reference. Fixed allocation policies
spend B=8 extra units when capacity permits. The cost-aware
policy has the **same cap**, spending 2,466 mean extra
units versus 3,200 for uniform; actual spending differs.
It is secondary and cannot rescue a failed fixed-budget primary claim.


## Post-confirmation mechanism control

New seeds 4000–4019 compare the unchanged cost-aware
candidate with a myopic policy using the same working PMF and at-most budget,
without information value. On heterogeneous demand, candidate-minus-myopic
regret is **-9,643.05 [-11,722.22, -7,540.05]** and realized cost is **-9,776.30 [-11,915.21, -7,646.54]**.
Expected regret is lower than the myopic control on these new simulator trajectories; this is explanatory evidence, not a new primary test. The cost-aware policy has higher expected regret than myopic control in Geometric, Retail iid, Identified. This follow-up does not alter the original primary comparison.


![All baselines and scenarios](reports/active/figures/scenario_results.png)

## Data and limitations

Retail contexts use 10,000 training-only store–product pairs.
Clustering features use 28 days; empirical demand PMFs use
379,138 fully stocked rows from 69 days
(2024-03-28–2024-06-04). A global tail-guard multiplier,
integer rounding, and clipping define dimensionless support. The clipped-row
fraction is 0.0633%; no probability floor is
added. Within-context iid resampling is constructed, rather than chronological replay. Fully
stocked selection is endogenous, so this does not identify natural-stockout demand.

Real data lack historical inventory thresholds and assignment propensities.
Known order thresholds, bounded support, costs, context frequencies, and shifts
are simulator assumptions. CDF guarantees apply only under their stationary
conditions; constructed shift is an empirical stress test. Empty confidence
sets trigger a logged uniform-allocation fallback. Costs are scenario units,
not currency or a shelf-life simulator. No scientific novelty is claimed.
The CLI serving interface instead fails closed on an empty confidence set.

## Reproduce and inspect

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
Development seeds [0, 1, 2, 3] and confirmation
seeds 1000..1029 are disjoint. Planning batch
64 was frozen on development outcomes. The
[protocol](docs/experimental_protocol.md) records the development-only cost-gate
amendment. The [research report](docs/research_report.md) contains every policy,
coverage diagnostics, spending, and evidence paths; [identification](docs/identification.md)
gives the exact scope and proof. [Earlier forecasting experiments](docs/baseline_forecasting.md)
and their Chronos comparison remain separate.
A [Russian technical walkthrough](docs/interview_notes_ru.md) explains the
decision problem, implementation, and findings for an interview discussion.

The research builds on [Kumar–Mouchtaki (2026)](https://arxiv.org/abs/2602.16842)
and the online completion perspective of [Mukherjee et al. (2026)](https://arxiv.org/abs/2605.14840).
[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
is CC BY 4.0. Independently written software is [MIT](LICENSE).

Measured test evidence: **109 passed**, 0 failures, 0 errors, and 0 skips; [test_summary.json](reports/active/test_summary.json).
