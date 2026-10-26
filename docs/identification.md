# Identification, confidence sets, and an exact regret calculation

## What is observed

Demand has known discrete support `0, ..., M`. The agent selects an integer
order `Q` before observing current demand `D`, and receives only
`sales = min(D, Q)` and `sold_out = (D >= Q)`. Equality is deliberately treated
as censored: a sold-out observation with sales `Q` means `D >= Q`, rather than
`D > Q`. At `Q = 0`, every observation is sold out and provides no demand
information. These conventions are part of the simulator, not inferred facts
about the original retail dataset.

For threshold `d < M`, the Bernoulli event `D <= d` is fully observable exactly
when `Q > d`. If this condition holds, `sales <= d` equals `D <= d`, including
when the order sells out. If `Q <= d`, a censored observation cannot determine
the event, so it must not be counted as a negative example.

Each fixed-context estimator accumulates

```
n[d] = sum_t 1{Q_t > d}
s[d] = sum_t 1{Q_t > d and sales_t <= d}.
```

This is a threshold-specific risk-set estimator, **not Kaplan–Meier**. Its
unadjusted CDF estimates can be nonmonotone because their risk sets differ.
The working point estimator uses weighted isotonic regression with weights
`n[d]`. Remaining unobserved tail mass is explicitly placed uniformly over
unknown support points by default; lower- and upper-endpoint alternatives
are exposed for sensitivity analysis. Those are assumptions, not recovered
counterfactual demand. The working PMF also supplies hypothetical-observation
probabilities for exploration; a resulting information-value score is a
model-based heuristic, not an exact value under unknown true demand.

An independent Kaplan–Meier product-limit baseline supplies a stronger canonical
comparison than naming the risk-set estimator Kaplan–Meier. Under the simulator's
`D >= Q` censoring convention, censored observations at a value are removed
**before** uncensored events at the same value: the hazard denominator at `d`
counts uncensored outcomes at or above `d` and censors strictly above `d`.
The estimated hazard identifies mass only below the largest historical order.
Remaining mass on that order through `M` receives an explicitly reported tail
completion. If the largest order equals the known support bound `M`, the
remaining support is the singleton `M`, not a freely estimated tail.

## Conditions behind the confidence statement

Within each context, future demand must follow a stationary distribution and
be independent of the past conditional on that context. Orders must be
predictable: they may depend on past observations and current context, but
not on current unobserved demand. Context groups and their maximum number `K`
are fixed in advance. Under these conditions, threshold exposure cannot select
current Bernoulli outcomes. Adaptive orders are allowed.

For a declared finite horizon `T` and error probability `delta`, use

```
r(n) = sqrt(log(2*K*M*T/delta) / (2*n)).
```

Hoeffding's inequality at each positive risk-set sample size, followed by a
union bound over contexts, thresholds, and sample sizes through `T`, gives
simultaneous coverage at least `1-delta`. Unknown thresholds receive `[0,1]`.
Monotonicity tightens lower bounds by a forward cumulative maximum and upper
bounds by a reverse cumulative minimum. Both CDFs end at `F(M)=1`.

This envelope preserves the coverage event. An empty intersection raises an
error rather than silently claiming a valid ambiguity set. The implementation
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

This formula is exact for this **deterministic pointwise-CDF ambiguity
polytope**. It is not the full finite-sample worst-case policy analysis in
[Kumar and Mouchtaki (2026), updated 11 October](https://arxiv.org/abs/2602.16842).
That paper studies regret averaged over training datasets and a broader
distributional optimization problem. A confidence band is also not a proof
that any proposed exploration policy has sublinear regret.

## An observational equivalence example

Consider two distributions: both put probability `0.6` at demand `2`; one puts
the remaining `0.4` at `10`, the other at `20`. With every historical order
capped at `10`, both produce exactly the same observable law: `(sales=2,
sold_out=False)` with probability `0.6`, and `(sales=10, sold_out=True)` with
probability `0.4`. No amount of passive data distinguishes their tails.

At `cu=3, co=1`, their optimal orders are respectively `10` and `20`. At equal
costs, both optimal orders are `2`: the unidentified tail does not affect the
required median. Thus uncertainty about demand and uncertainty about the
economically relevant decision are different objects. Higher exploratory
orders can reveal otherwise inaccessible thresholds, but incur real simulated
excess-stock costs and must share the same budget across competing policies.

## Independent checks and empirical boundaries

Tests maximize every candidate-versus-comparator cost difference using
`scipy.optimize.linprog` over PMF variables, explicit CDF constraints, and the
probability simplex. This independently constructed LP verifies the closed
form on small random grids. Other tests verify censored equality, isotonic
weights, observational equivalence, and vectorized hypothetical updates
against explicit cloned-state enumeration.

FreshRetail does not provide the known action thresholds, assignment
probabilities, or latent demand needed to treat these simulated guarantees as
an identified counterfactual evaluation. Its real contexts can motivate a
semisynthetic environment, but simulated policy improvements remain conditional
on its declared generator, support, costs, and information structure.
