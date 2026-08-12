# Ready-cohort formalism

Evidence/status cutoff: 2026-08-11. This note distinguishes the quantity
already measured by trace replay from the stronger deadline-constrained bound
the paper should ultimately report.

## Events, compatibility, and hardware thresholds

Each control event `i` has:

- release time `t_i`;
- latest admissible launch time `d_i >= t_i`;
- semantics-preserving execution class or route `r_i`;
- a measured safe-suffix crossover `K(r_i, h, v, H)` over a declared cohort
  range.

The trace artifact does not observe a semantics-preserving execution class. It
uses an outcome-derived route-key proxy that omits state-machine, schema,
arguments, policy context, and multi-tool identities. Every trace result is
therefore conditional on that declared grouping and does not prove fusion is
executable.

A feasible GPU batch `(tau, B)` for a fixed hardware/runtime configuration
contains events from one route, has `|B| >= K`, and satisfies
`t_i <= tau <= d_i` for every `i in B`. A schedule is a collection of feasible
batches in which each event appears at most once. This definition deliberately
ignores accelerator capacity and execution duration; adding either can only
reduce the accelerable share.

The threshold abstraction assumes the profitable region is a monotone suffix
and has no upper batch limit. A measured `K` must therefore be the start of a
safe suffix through a declared maximum, not merely the first isolated winning
cohort size. Non-monotone benefit and upper profitable limits are outside this
model.

## Quantity measured by trace replay 001

Trace replay 001 freezes a non-overlapping, origin-aligned window partition
`pi_delta`. Each event belongs to exactly one `(time window, route)` bucket
`b`, with count `n_b`. Its measured fixed-window eligible share is

```text
F(pi_delta, K, g) = sum_b n_b 1[n_b >= K] / sum_b n_b.
```

### Proposition 1: fixed-partition exactness

`F` is achievable and is the exact maximum event share among schedulers that
may batch only events from the same frozen bucket and have unlimited GPU
capacity.

**Proof sketch.** If `n_b < K`, no legal batch can be formed inside bucket
`b`. If `n_b >= K`, one batch containing all `n_b` events is legal. Buckets
are disjoint, so optimizing each independently gives the expression above.

### Important scope correction

`F` is **not** a universal deadline-scheduler ceiling. If events use sliding
per-event deadlines, a legal cohort can straddle the boundary of two fixed
windows even when neither window contains `K` events. Therefore:

- call existing outputs **fixed-window eligibility** or the
  **fixed-partition ceiling**;
- do not claim that no scheduler can exceed them under the same maximum wait;
- evaluate a sliding-deadline optimum or upper bound before using the shorter
  phrase “ready-cohort ceiling” in the title/abstract.

This clarification does not change any replay value; it changes its domain of
interpretation.

## General deadline bounds

For one route, define the active interval of event `i` as `[t_i, d_i]`. Let

```text
A_r(tau) = |{j : r_j = r and t_j <= tau <= d_j}|.
```

Event `i` is *locally eligible* if there exists a launch time inside its own
interval at which at least `K(r_i)` compatible events are active:

```text
u_i = 1[max_{tau in [t_i,d_i]} A_{r_i}(tau) >= K(r_i)].
U = sum_i u_i / |E|.
```

### Proposition 2: local-eligibility upper bound

For any deadline-respecting schedule, its accelerated event share `P*`
satisfies `P* <= U`.

**Proof sketch.** Every accelerated event belongs to a feasible batch of at
least `K` intervals sharing its launch time. It must therefore be locally
eligible. Counting all locally eligible events can double-count mutually
incompatible batching opportunities, so `U` is an upper bound rather than an
achievability claim.

The exact offline packing optimum can be written as a binary program over
candidate launch times (unique deadlines are sufficient):

```text
maximize   sum_i sum_tau x[i,tau]
subject to sum_tau x[i,tau] <= 1                           for every event i
           x[i,tau] = 0 unless t_i <= tau <= d_i
           K_r y[r,tau] <= sum_{i:r_i=r} x[i,tau]
           sum_{i:r_i=r} x[i,tau] <= M[r,tau] y[r,tau]
           x[i,tau], y[r,tau] in {0,1}.
```

`P*` is the optimum divided by event count. Under unlimited compute:

```text
F(pi_delta, K, g) <= P* <= U.
```

The left inequality holds when each fixed-window batch is also legal under
the event deadlines. Define a runtime's achieved share `A` over the identical
event set `E` and retained horizon used by `P*`, counting every unaccelerated,
late, missing, or failed event in the denominator. Then `A <= P*`; finite
service time, queueing, fairness, fallback, and GPU contention can make the
inequality strict.

### Proposition 3: unique deadlines suffice

There is an optimal offline packing whose launch times are drawn only from
event deadlines. Consider any non-empty feasible batch launched at `tau`, and
let `d_min` be the smallest deadline among its assigned events. Feasibility
gives `tau <= d_min`; every assigned release is at most `tau`, and every
assigned deadline is at least `d_min`. Moving that batch from `tau` to
`d_min` therefore preserves every assignment. Applying this independently to
all batches proves the claim.

If two same-route batches move to the same deadline, they can be merged. This
argument assumes zero service time, unlimited simultaneous capacity, and a
deadline on launch rather than completion.

### Proposition 4: exact equal-deadline packing admits an `O(N log N)` evaluator

For the experiment's equal-relative-deadline case, sort one route's releases
as `t_1 <= ... <= t_n`. There is an optimum whose batches are disjoint
contiguous index blocks. To see this, order batches by launch time. Whenever
an earlier event is assigned to a later batch and a later event to an earlier
batch, exchange them. Equal interval lengths preserve both assignments:

```text
t_i <= t_j <= tau_early <= tau_late <= t_i + delta <= t_j + delta.
```

Repeated exchanges remove crossings. Any unassigned event between a batch's
first and last releases can then be added to that batch without changing its
release span. Define `D[j]` as the maximum number assigned among the first `j`
events. Then

```text
D[0] = 0
D[j] = max(
    D[j-1],
    max_i D[i-1] + j-i+1
)
subject to i <= j-K+1 and t_j-t_i <= delta.
```

The inner objective is a sliding range maximum of `D[i-1]-i+1`. A grouped sort,
route slicing, a moving left feasibility pointer, and a monotone deque give an
`O(N log N)` evaluator, including sorting. The frozen implementation used for
the reported outputs does not realize that global bound: it scans the full
group array once per route and uses a binary search for each event. Its bound is
`O(NR + sum_r n_r log n_r)` for `R` routes and `n_r` events per route, which is
quadratic in the worst case. This affects runtime complexity, not the returned
optimum or the reported shares. Routes remain separable and may use distinct
`K_r`. The implementation also normalizes releases and deadlines to integer
nanoseconds before applying inclusive comparisons, so “exact” refers to that
explicit clock rather than a floating-point epsilon.

This evaluator is a fixed-diameter, minimum-cardinality clustering problem
with outliers on the line, equivalently a lower-bounded clique-packing problem
on a unit interval graph. Its structure is closely related to prior
one-dimensional `r`-gathering and microaggregation results. The paper therefore
uses the recurrence as a specialized reference evaluator and makes no
algorithm-priority claim.

This result does not extend to arbitrary deadlines, finite service/capacity,
upper batch limits, cross-route fusion, or non-monotone benefit beyond `K_r`;
the binary program remains the general reference formulation for those cases.

## Regularity and residency

If grouping `g_fine` refines `g_coarse`, every fine bucket is contained in one
coarse bucket. For a fixed partition and threshold,

```text
F(pi, K, g_fine) <= F(pi, K, g_coarse).
```

The measured difference is a fixed-window regularity tax. It is useful, but it
does not prove that route fusion is legal: merging routes may change actions,
memory access, compilation shape, or numerical trajectory. Residency and
visibility enter through the measured threshold `K(r,h,v,H)`; changing them
can move both the hardware boundary and the set of events that clear it.

## Required next validation

1. Compare the full-trace fixed-window `F`, exact offline `P*`, local bound
   `U`, and online
   scheduler achievement under equal per-event deadlines.
2. Repeat under bursty and correlated arrivals, route-specific `K`, finite
   GPU service time, and CPU fallback.
3. Reserve the paper-level word **ceiling** for `P*` or a proven upper bound;
   label current plots as fixed-window eligibility until then.
