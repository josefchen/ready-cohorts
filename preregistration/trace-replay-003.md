# Trace replay 003 preregistration: exact full-trace deadline packing

Status: frozen at **2026-08-11T20:46:38Z**, before computing any exact-packing
outcome on the empirical replay. The earlier fixed-window and local-bound
results are allowed design data and are already public within this artifact.

## Question

For the frozen equal-relative-deadline trace model, what share of compatible
agent-control events can an offline scheduler actually batch above a measured
GPU crossover threshold? This experiment places the exact achievable optimum
`P*` between the already measured fixed-window policy `F` and local-overlap
upper bound `U`:

```text
F <= P* <= U.
```

The result is a zero-service, unlimited-capacity workload feasibility bound.
It is not an online-runtime achievement or an end-to-end throughput claim.

## Frozen implementation and clock

- solver: the per-route contiguous-block dynamic program in
  `src/gpu_agent_crossover/ready_cohort.py`;
- complexity claim: `O(N log N)` including stable per-route sorting and linear
  time after sorting;
- solver source SHA-256:
  `c49a840cca619f61fb4470d544914abc8cd7fbeb5df214b33ccead4846a692ab`;
- all floating release times are rounded to the nearest integer nanosecond;
- relative deadlines are rounded to integer nanoseconds;
- interval endpoints are inclusive and all feasibility comparisons are exact
  on that integer clock;
- service time is zero, GPU capacity is unlimited, routes cannot mix, and any
  batch of size at least `K` is assumed beneficial;
- the recovered schedule is a maximum-cardinality witness. Batch count and
  waiting time are not separate optimization objectives.

Before this freeze, the recurrence was checked against brute-force assignment
on tiny random cases, two published greedy counterexamples, route-specific
threshold cases, and a floating-boundary adversarial case. Those are method
tests, not trace outcomes.

## Frozen data and replay

- span features:
  `data/processed/exgentic-tau2-span-features.parquet`, SHA-256
  `da57b75db69916a4c1036909e453e41e470b7daef304e61bf2f8b2d8720d4dc2`;
- session summary:
  `data/processed/exgentic-tau2-session-summary.csv`, SHA-256
  `389c4e700aad48f322f7e82560594f402b62f56123c6d39a27b3f18ffe409fc6`;
- same stationary Poisson template replay and seed construction as trace
  replays 001 and 002;
- target active sessions: `1,000`, `10,000`, and `100,000`;
- three replay seeds per concurrency, root seed `20260811`;
- retained horizon: 60 seconds;
- deadlines: 10, 25, 50, 100, and 250 ms;
- groupings: pooled, event class, and exact route;
- crossover thresholds: 32, 64, 128, and 256;
- every generated event is retained.

The script must deterministically reproduce the matched `F` and `U` values
from replay 002 before its exact results are accepted.

## Estimands and frozen hypotheses

For every matched cell, report `F`, `P*`, `U`, batch count, accelerated event
count, and, when `U > F`, the alignment-gap closure

```text
G = (P* - F) / (U - F).
```

The following invariants are validity gates, not statistical hypotheses:

- **E1:** `F <= P* <= U` in every repetition and cell;
- **E2:** coarsening compatibility from exact route to event class to pooled
  cannot decrease `P*` for a common threshold;
- **E3:** increasing the deadline cannot decrease `P*`;
- **E4:** increasing `K` cannot increase `P*`;
- **E5:** cells where the previously measured `F == U` must also have
  `P* == F == U`, up to exact integer-count equality.

The one directional pilot hypothesis is:

- **E6:** at 100,000 target active sessions, exact-route grouping, `K=256`,
  and a 50 ms deadline, mean `P*` is strictly above mean fixed-window `F`.

E6 is descriptive conditional evidence over three simulation seeds on one
fixed 851-session panel. It carries no population p-value. The primary effect
size is `P* - F`; `G` explains how much of the boundary-alignment opportunity
is actually jointly packable.

## Failure and reporting rules

- Any E1--E5 violation invalidates the implementation or matched-input claim
  and halts interpretation.
- No failed or inconvenient cell is removed. A crash is recorded with the
  last completed cell and exact configuration.
- No threshold, deadline, grouping, concurrency, or seed is added or removed
  after outcomes are viewed; later workload models receive new experiment IDs.
- Repetitions are Monte Carlo seeds conditional on the fixed trace panel, not
  independent production workloads.
- Report the full grid and negative regimes. Do not select only cells where
  `P* > F`.

## Non-generalization boundary

This experiment does not establish feasibility under finite kernel service,
one-at-a-time GPU launches, upper batch limits, heterogeneous per-event
deadlines, cross-route fusion, queueing, fairness, correlated non-Poisson
arrivals, or task-utility constraints. Those belong to the runtime and
sensitivity stages.
