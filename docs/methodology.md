# Methodology

## Question and evidence

Does accounting for the timing of historical stockouts improve next-day demand
forecasts and a simple replenishment decision? The project uses the public
[FreshRetailNet-50K release](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
and its [original paper](https://arxiv.org/abs/2505.16319). Sales are globally
normalized quantities. They are not prices, revenue, or an observation of unmet
demand. A stockout flag identifies a potential censoring event; it does not
reveal the counterfactual quantity customers would have bought.

The operational target is the sum of hourly quantities over **[06:00, 22:00)**,
namely hours 6 through 21. Overnight quantities are excluded consistently from
recovery, forecasting, and decision evaluation. The default experiment uniformly
samples 10,000 store–product pairs from sorted identifiers in the official
training file, using seed 42. Sampling uses neither targets nor evaluation rows.
The sample supports a laptop experiment rather than a claim about all pairs.

## Information available at prediction time

The task is day-ahead forecasting at the end of the preceding day. Features
include identifiers, forecast-day calendar variables, lagged demand, shifted
rolling summaries, historical stockout rates, and lagged discounts and weather.
No same-day realized weather, discounts, stock availability, or sales enter the
forecast. No future promotion plan or archived weather forecast is assumed.

The forecasting model is refitted at each fold origin. Inside a seven-day scoring
window, its parameters remain fixed, while lag features receive actual sales and
stock observations from each completed preceding day. Recovery uses its frozen
parameters to process that history. This is a sequence of **H=1 forecasts**, not
a fixed-origin seven-step forecast. Deployments requiring an entire week's
orders in advance would need a different experiment.

## Three recovery variants

One recovery model is fitted on the first 28 training days and then frozen for
every validation and final-test window. Those warmup days are excluded from
forecast-model fitting. No evaluation observation updates a recovery profile,
prior, or safety parameter.

- **Raw:** observed operational sales, without correction.
- **Uniform:** estimate total demand from sales in available hours divided by
  the fraction of operational hours marked available.
- **Profile:** replace that fraction with the expected demand share in the
  exact available hours. Positive, fully available warmup days supply normalized
  intraday profiles. Product profiles shrink toward their third-category
  profile; category profiles shrink toward the global profile, using seven
  pseudo-days at each level.

Both corrected variants use a minimum exposure share of 0.2. Fully unavailable
days use a warmup demand prior. Pair priors prefer fully available warmup days;
pairs without such days fall back to observed means, with product/global
fallbacks for unseen identifiers. Estimates are capped at five times their
prior, unless observed sales already exceed that cap, and never fall below
observed sales. These are predeclared heuristic assumptions, not identified
properties of customer demand.

Positive sales sometimes coexist with a stockout flag in the same hour. An
hour may contain partial availability; flags do not establish how long it was
available. The exposure heuristic treats a flagged hour as unavailable when
estimating the share, while retaining total observed sales as a lower bound.
This approximation can overcorrect or undercorrect. Recovery does not establish
a causal effect of inventory availability.

## Forecast comparison and selection

Each variant trains LightGBM quantile regressors at 0.5, 0.75, and 0.9, with a
fixed budget of 250 trees and configuration recorded in `configs/default.json`.
The same feature construction and hyperparameters apply to all variants.
Seven-day seasonal naive provides a simple reference. There is no test-driven
hyperparameter search or early stopping.

Three consecutive seven-day validation windows occupy the final 21 days of the
90-day training period, with an expanding training prefix before each window.
The official following seven-day evaluation split is reserved for final scoring.
The primary decision uses quantile 0.75, corresponding to shortage cost 3 and
excess cost 1. Select the recovery variant with the smallest arithmetic mean of
validation normalized costs across the three windows, then refit its forecast
model on eligible official training days. Other quantiles are diagnostics.

## Metrics, uncertainty, and limits

Forecast metrics are MAE, WAPE, signed bias, and pinball loss. Positive bias means
overprediction. Decision loss is `3 * max(demand - order, 0) + max(order - demand, 0)`.
Normalized cost divides total loss by total scored demand. Fill rate is total
fulfilled demand divided by total demand; cycle service is the fraction of rows
without a shortage. These are single-period scenario metrics, not measured
financial savings or a shelf-life inventory simulator.

Fully available scoring days provide observable operational-demand targets.
However, selecting them using realized stock status creates selection bias:
their demand distribution need not represent naturally censored days. Metrics
against observed sales on all days are censored-sales diagnostics. Neither
evaluation identifies real unmet demand, waste reduction, or production profit.

Paired bootstrap draws resample stores with replacement, retaining all their
products and dates together and recomputing normalized-cost differences. The
95% interval is conditional on observed dates; it does not capture common
temporal shocks, future-date performance, or transportability across cities.

Controlled recovery tests hide known sales on fully available evaluation days.
Masks come from nonempty official-training stockout masks, fixed suffix hours
18–21, or fixed hours 9, 10, 16, 17. These last hours are predeclared rather than
selected from evaluation demand. Original targets are preserved before masking.
This retrospective reconstruction experiment tests specified artificial
mechanisms, not natural-stockout counterfactual accuracy or forecast-time access
to partial same-day sales.

## Attribution and scope

The [official baseline](https://github.com/Dingdong-Inc/frn-50k-baseline) and
[Azar's related pipeline](https://www.mdpi.com/2411-5134/11/5/89) establish relevant
prior work. This implementation was written independently; baseline code was
not cloned. Its modest engineering contribution is a reproducible comparison of
exact-hour and uniform recovery, explicit information boundaries, controlled
masking, and decision evaluation. No scientific novelty is claimed. Code uses
the MIT license; the external dataset remains CC BY 4.0 and requires attribution.
