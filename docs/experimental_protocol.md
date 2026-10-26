# Experimental protocol

The primary comparison tests allocation of extra stock by expected reduction
in decision regret against uniform allocation. Allocation by CDF uncertainty
is a secondary comparison. Demand has known integer support 0..40. These
simulations do not estimate a real retailer's policy effect.

Each round has four contexts. Policies order before demand is drawn and observe
only sales=min(D,Q) and the sell-out indicator D>=Q. Each receives the same
40-round historical data collected at Q=8. Latent demand and PMFs are accessible
only to the evaluator and to an explicitly privileged past-full-information
reference. The oracle knows the generating PMF, uses the same order set, and
does not know the coming realized demand.

The main allocation comparison uses the same risk-set quantile base rule and
exactly eight extra inventory units per round whenever capacity allows. All
shortage and excess costs, including exploration quantities, are charged.
An adaptive gating variant can leave budget unused and is reported separately.

Development seeds 0..3 are disjoint from confirmation seeds 1000..1029. The
only development choice is the planning approximation horizon in {1,16,64},
selected by mean expected regret of decision allocation on the heterogeneous
development scenario. m=1 is exact one-observation model-based lookahead;
m>1 uses expected fractional counts for a virtual batch and is a planning
heuristic, not an actual future confidence statement. Selection is written and
hashed before confirmation starts. All main parameters and scenarios are in
the versioned JSON configs.

The primary confirmation comparison is decision versus uniform allocation on
the heterogeneous scenario. Decision versus global CDF-uncertainty allocation
tests the proposed mechanism. Other stationary, misspecified, empirical, and
nonstationary scenarios are stress tests and are reported regardless of sign.
The empirical environment bootstraps training-only fully stocked observations;
its context frequencies and inventory mechanism are constructed, not observed
real policy counterfactuals. Costs have scenario units, not monetary units.

The primary metric is cumulative conditional expected regret relative to the
PMF-optimal order. Realized cost, service, excess, extra-stock spending and
stationary confidence diagnostics are also reported. Paired bootstrap resamples
complete independent demand seeds, preserving all dates and contexts within a
trajectory. These intervals condition on the fixed simulator and do not imply
robustness across unspecified real-world environments. The primary contrast is
predeclared; secondary intervals are descriptive and not familywise adjusted.

Confidence bands require stationary independent demand conditional on context
and predictable inventory. They have no claimed coverage after the constructed
shift. If a confidence polytope becomes empty, the allocation falls back to
uniform and logs the event. The serving CLI returns an error without a
recommendation when its confidence set is empty.

The exact CDF-band optimization and its LP verification are described in
[identification.md](identification.md). They do not reproduce the complete
finite-sample worst-case-regret theorem of Kumar and Mouchtaki.

## Development revision before confirmation

The first development run showed that forced spending can dominate ordering
cost when little additional information is needed. A cost-aware policy was
added before reading confirmation seeds. Its score is the planned regret reduction times remaining rounds
divided by the planning batch size, minus the working-model expected immediate
incremental inventory cost. It optimizes an at-most-B budget and may leave
inventory unallocated. This is an approximate planning score, not an optimal
multi-step Bayesian value function. It is secondary; the original fixed-budget
primary comparison is retained. Kaplan-Meier tail-completion variants and
LightGBM are also retained as independent baselines.
