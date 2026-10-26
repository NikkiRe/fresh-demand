# Reproducing the identification mechanism

This experiment is a conceptual reproduction of the indistinguishability
mechanism discussed by [Kumar and Mouchtaki](https://arxiv.org/abs/2602.16842).
It is **not** a numerical replication of their tables, theorem, data,
implementation, or policy. The finite support, two distributions, cost ratios,
and Monte Carlo design below are constructed for this project.

## Two demands that produce the same observations

Demand takes integer values from zero through forty. World A has probability
one half at five and one half at twelve. World B has probability one half at
five and one half at thirty-five. Historical orders are always eight. The
observable pair is `(sales = min(D,Q), sold_out = D >= Q)`, including equality.
In either world it is `(5, false)` with probability one half and `(8, true)`
with probability one half. More passive cap-eight observations distinguish
neither world, even with an unlimited sample size.

The population CDF is known below eight: it is zero below five and one half
at five through seven. At eight through thirty-nine, only a lower bound of
one half and an upper bound of one are identified. Both bounds equal one at
forty because the support is known. This population identification band is
different from a finite-sample confidence band.

## Identification depends on the cost ratio

The critical quantile is `cu / (cu + co)`. With underage cost one and overage
cost three, it is one quarter. Ordering five is optimal in both worlds and
throughout the full CDF ambiguity band. Its regret is exactly zero, although
its absolute expected cost differs between the worlds.

With costs three and one, the critical quantile is three quarters. A knows to
order twelve; B knows to order thirty-five. The full CDF ambiguity band permits
every optimal quantile from eight through forty.

For an order `q` between eight and forty, the full-band worst-case regret is

```
R(q) = max(q - 8, 40 - q).
```

The first term is attained when the unknown half of demand mass sits at eight;
the second is attained when it sits at forty. Equating the two terms gives
`q = 24` and worst-case regret `16`. The code evaluates every integer order
using the CDF-band minimax solver. An independent test enumerates all tail
vertices from eight through forty and verifies the entire regret curve.

Restricting ambiguity to the **two witness worlds A and B alone** is a smaller
problem: its minimax orders are twenty-three or twenty-four and its regret is
twelve. The experiment reports both problems separately. The value sixteen
must not be described as minimax regret over only A and B.

## Fixed-design passive and audited observations

The simulation uses fifty independent seeds, historical sample sizes
`20, 80, 320, 1280`, and audit counts `0, 1, 4, 16`. For each fixed N, the last
k cap-eight orders are replaced by cap-forty orders. Audit selection is
independent of outcomes; it adds no extra demand records. All alternatives use
the same underlying uniform draws, including across A and B.

The threshold risk-set estimator is evaluated with two explicit completions
of its unidentified tail: uniform mass over unknown support points and mass
at their lower endpoint. Those completions are working assumptions. The
Monte Carlo study records the post-history order and its exact expected cost
and regret under each known true PMF. It does not force improvement with N
or audit count, and it does not run an adaptive inventory policy.

Audits are not free. The output includes the exact expected historical cost,
the difference from cap-eight history, and the expected cost of history plus
one terminal decision. The historical cost difference can be negative when
higher stock also avoids expensive shortages. Terminal regret improvement
alone therefore establishes neither audit return on investment nor a claim
about cumulative online-policy performance.

Approximate normal intervals in the summary describe variation of the seed
mean. They are neither identification bounds nor CDF confidence statements.
Every per-seed record is retained in the JSON, including unfavourable results.

## Run and inspect

From the project root after installing the package:

```bash
python -m fresh_demand.active.reproduction --output reports/active/reproduction
```

The output is `reports/active/reproduction/identification.json`. Its `analytic`
section contains PMFs, observation laws, CDF bounds, and full regret curves.
Its `monte_carlo` section contains the fixed settings, summaries, historical
audit costs, and all raw records. The experiment uses no FreshRetail test
outcomes and makes no counterfactual claim about real replenishment policies.
