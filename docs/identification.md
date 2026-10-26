# Censored observations, confidence sets, and minimax regret

## What is observed

Demand has known discrete support `0, ..., M`. The agent selects an integer
order `Q` before observing current demand `D`, and receives only
`sales = min(D, Q)` and `sold_out = (D >= Q)`. Equality is treated
as censored: a sold-out observation with sales `Q` means `D >= Q`, rather than
`D > Q`. At `Q = 0`, every observation is sold out and provides no demand
information. This observation rule is specified by the simulator; the original
retail dataset does not establish it.

For threshold `d < M`, the event `D <= d` is observable for every possible outcome
when `Q > d`. Then `sales <= d` equals `D <= d`, including
when the order sells out. If `Q <= d`, a censored observation cannot determine
the event, so it must not be counted as a negative example.

Each fixed-context estimator accumulates

```
n[d] = sum_t 1{Q_t > d}
s[d] = sum_t 1{Q_t > d and sales_t <= d}.
```

This risk-set estimator counts observations separately for each threshold. Its
unadjusted CDF estimates can be nonmonotone because their risk sets differ.
The working point estimator uses weighted isotonic regression with weights
`n[d]`. Remaining unobserved tail mass is explicitly placed uniformly over
unknown support points by default. Lower- and upper-endpoint completions are
available for sensitivity analysis. Each completion is an assumption about the
unknown tail. The estimated PMF also assigns probabilities to hypothetical
observations when scoring exploration. That score depends on the fitted model;
it is not the exact information value under the unknown demand distribution.

A separate Kaplan–Meier baseline uses a product-limit estimate. Under the simulator's
`D >= Q` censoring convention, censored observations at a value are removed
**before** uncensored events at the same value: the hazard denominator at `d`
counts uncensored outcomes at or above `d` and censors strictly above `d`.
The estimated hazard identifies mass only below the largest historical order.
Remaining mass on that order through `M` receives an explicitly reported tail
completion. If the largest order equals the known support bound `M`, the
remaining mass must lie at `M` because it is the known support bound.

## Conditions behind the confidence statement

Within each context, future demand must follow a stationary distribution and
be independent of the past conditional on that context. Orders must be
predictable: they may depend on past observations and current context, but
not on current unobserved demand. Context groups and their maximum number `K`
are fixed in advance. An order can adapt to past outcomes, but exposure to a
threshold must be decided before the current demand is observed.

For a declared finite horizon `T` and error probability `delta`, use

```
r(n) = sqrt(log(2*K*M*T/delta) / (2*n)).
```

Hoeffding's inequality at each positive risk-set sample size, followed by a
union bound over contexts, thresholds, and sample sizes through `T`, gives
simultaneous coverage at least `1-delta`. Unknown thresholds receive `[0,1]`.
Monotonicity tightens lower bounds by a forward cumulative maximum and upper
bounds by a reverse cumulative minimum. Both CDFs end at `F(M)=1`.

These monotone envelopes preserve simultaneous coverage. An empty intersection
raises an error. The implementation
rejects observations beyond the declared horizon. The statement does not
cover drifting demand, hidden context confounding, demand-dependent order
selection, estimated support bounds, or arbitrary endogenous retail data.

## Exact minimax regret for a CDF-band polytope

Fix nonnegative underage and overage costs `cu, co`, with positive sum. For a
CDF `F`, let `C_F(q)` be expected newsvendor cost. Integer support gives

```
C_F(q+1) - C_F(q) = (cu+co)*F(q) - cu.
```

Define the ambiguity set as **all** probability distributions on `0..M` whose
CDF lies pointwise between monotone bounds `L` and `U`. Both endpoint CDFs
themselves belong to this set. The comparator knows the distribution but uses
the same action set `0..M`.

For a candidate order `q`, its worst-case regret is

```
R(q) = sup_F [C_F(q) - min_b C_F(b)].
```

The finite maximum over comparator actions commutes with the supremum. If
`b <= q`, the cost difference is a sum of positive multiples of `F(d)` for
`b <= d < q`, so its maximum is attained at `U`. If `b >= q`, the coefficients
are negative, so its maximum is attained at `L`. Therefore

```
R(q) = max(C_U(q) - min_{b<=q} C_U(b),
           C_L(q) - min_{b>=q} C_L(b)).
```

All costs can be computed by cumulative sums starting at an arbitrary zero
constant: the unknown distribution-specific `C_F(0)` cancels. Prefix and suffix
minima give every `R(q)` in linear time. Choose the smallest minimizing order.
The implementation batches all hypothetical orders and demand outcomes in
NumPy when computing expected next-state minimax regret.

The optional mean-field planner uses a different approximation: for a candidate
order repeated `m` virtual times, it adds `m * 1{Q>d}` to risk-set counts and
`m * F_working(d) * 1{Q>d}` to successes, then evaluates regret or CDF width at
these fractional expected counts. Because envelopes, clipping, and action
minimization are nonlinear, this is not the expectation over future datasets.
Even `m=1` generally differs from exact one-step outcome enumeration. Virtual
counts can exceed the remaining real horizon; this changes no observed counts
and creates no confidence guarantee for those virtual states. It is a planning
heuristic for overcoming a flat one-step information score during cold start.

The formula solves the **deterministic pointwise-CDF ambiguity polytope**
defined above. The finite-sample policy analysis in
[Kumar and Mouchtaki (2026), updated 11 October](https://arxiv.org/abs/2602.16842)
studies a different problem: regret averaged over training datasets and a
broader distributional optimization. Neither this calculation nor the
confidence band establishes sublinear regret for the exploration policy.

## An observational equivalence example

Consider two distributions: both put probability `0.6` at demand `2`; one puts
the remaining `0.4` at `10`, the other at `20`. With every historical order
capped at `10`, both produce exactly the same observable law: `(sales=2,
sold_out=False)` with probability `0.6`, and `(sales=10, sold_out=True)` with
probability `0.4`. No amount of passive data distinguishes their tails.

At `cu=3, co=1`, their optimal orders are respectively `10` and `20`. At equal
costs, both optimal orders are `2`: the unidentified tail does not affect the
required median. The tail is unidentified in both cases, while the optimal
median order is identified. Higher orders can reveal additional thresholds.
Their shortage and excess costs are included in evaluation, and competing
policies have the same exploration budget.

## Independent checks and empirical boundaries

Tests maximize every candidate-versus-comparator cost difference using
`scipy.optimize.linprog` over PMF variables, explicit CDF constraints, and the
probability simplex. This independently constructed LP verifies the closed
form on small random grids. Other tests verify censored equality, isotonic
weights, observational equivalence, and vectorized hypothetical updates
against explicit cloned-state enumeration.

FreshRetail does not provide historical order thresholds, assignment
probabilities, or latent demand. It therefore cannot identify the counterfactual
policy effects studied here. Retail-derived contexts define a semisynthetic
environment; its results depend on the specified generator, support, costs,
and observation rule.
