# Fresh Demand

Day-ahead demand forecasting and order quantities for fresh products. When a product
sells out, sales only show a lower bound of demand, so the project also studies how
stockouts bias ordering. Data: [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
(Dingdong, CC BY 4.0), daily and hourly sales of fresh products with stockout flags.

## Forecasting and ordering on held-out data

10,000 store–product pairs sampled from the official training identifiers (seed 42):
900,000 training rows across 890 stores. Test week: 2024-06-26–2024-07-02, scored on
41,220 fully available store–product days out of 70,000. Each day gets a new
one-day-ahead forecast; models refit at weekly origins. Three expanding validation
windows precede the test, and the method was selected before the test file was read.

| Model | WAPE | Bias | Order cost | Fill rate |
| --- | --- | --- | --- | --- |
| Seasonal lag 7 | 39.84% | +0.09% | 0.7959 | 80.13% |
| Trailing mean 7 | 32.14% | -1.00% | 0.6529 | 83.43% |
| LightGBM, raw sales | **29.98%** | -3.47% | **0.5006** | 93.18% |
| LightGBM, uniform stockout recovery | 32.36% | +10.63% | 0.5679 | 96.06% |
| LightGBM, profile stockout recovery | 31.12% | +6.95% | 0.5439 | 95.48% |

LightGBM WAPE and bias use the median forecast (q50); its orders use q75. The baselines
order their point forecast. Order cost is `(3 × shortage + 1 × excess) / total demand`.

- **Forecast.** Quantile LightGBM lowers WAPE to 29.98% from 32.14% for the trailing
  mean and 39.84% for seasonal lag 7.
- **Order quantity.** Ordering at q75 instead of q50 lowers cost by 21.08% at a 3:1
  shortage:excess ratio; fill rate rises from 83.27% to 93.18%. Paired store-cluster
  bootstrap for the cost difference: −0.1337 [−0.1410, −0.1262].
- **Stockout recovery did not pay off.** Correcting sales on stockout days by the hourly
  demand profile costs 8.65% more than raw sales (predeclared comparison:
  +0.0433 [0.0392, 0.0478]). Validation selected raw sales.
- **Foundation model.** On the same 256 sampled series, LightGBM has 31.47% WAPE versus
  33.44% for zero-shot Chronos-2 without covariates.

Intervals are conditional on the observed test dates. Details, validation folds and the
Chronos-2 setup: [docs/baseline_forecasting.md](docs/baseline_forecasting.md).

![Forecast and controlled recovery results](reports/main/overview.png)

## Simulation: ordering when sales hide demand

A simulator draws demand from known distributions (synthetic and fitted to FreshRetailNet
records) and shows the policy only `sales = min(demand, order)` and a sold-out flag.
It compares 16 ordering policies in 6 scenarios, 30 paired seeds each, 400 rounds.

- In the synthetic heterogeneous scenario, ordering at a quantile of observed sales
  serves 44.4% of demand. Censoring-aware estimators serve 77.1% (Kaplan–Meier),
  89.2% (geometric Thompson sampling) and 92.0% (CDF-band minimax), with expected
  regret 2,639.38–3,508.66 instead of 21,093.09.
- Spending extra stock to learn demand: the predeclared comparison of decision-based
  and uniform allocation shows no clear difference, +0.91 [−58.26, 59.81].
- The extra-stock budget is allocated exactly with a multiple-choice knapsack (dynamic
  programming); minimax-regret orders are checked against linear programs.

Full results, methods and limitations: [simulation study](docs/simulation_study.md),
[research report](docs/research_report.md), [identification](docs/identification.md),
[experimental protocol](docs/experimental_protocol.md).

## Usage

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-lock.txt
uv pip install -e ".[dev]"
fresh-demand download --data-dir data/raw
fresh-demand train --data-dir data/raw --config configs/default.json --output reports/reproduction --final-test
fresh-demand subset --input data/raw/train.parquet --series reports/reproduction/sampled_series.parquet --output data/history_cohort.parquet
fresh-demand predict --model reports/reproduction/artifacts/model.pkl --history data/history_cohort.parquet --output forecasts/next_day.parquet
fresh-demand retrain --history data/history_cohort.parquet --selection reports/reproduction/frozen_selection.json --config configs/default.json --model reports/reproduction/artifacts/retrained.pkl
pytest -q
```

On macOS LightGBM needs OpenMP (`brew install libomp`). Simulation commands are listed in
the [simulation study](docs/simulation_study.md#reproduce).

## Data and license

FreshRetailNet-50K is released by Dingdong-Inc under CC BY 4.0
([dataset paper](https://arxiv.org/abs/2505.16319)). Related work on censored inventory:
[Kumar–Mouchtaki (2026)](https://arxiv.org/abs/2602.16842),
[Mukherjee et al. (2026)](https://arxiv.org/abs/2605.14840). The project implements and
evaluates existing methods; it does not propose new theory. Code is [MIT](LICENSE).
