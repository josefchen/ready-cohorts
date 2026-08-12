# Trace replay 001 preregistration: the ready-cohort ceiling

Status: frozen before cohort-replay computation on 2026-08-11 at 16:49 UTC.

Post-analysis scope clarification (2026-08-11): the frozen expression is an
exact ceiling for schedulers constrained to the same origin-aligned,
non-overlapping window partition. It is not a universal ceiling for arbitrary
sliding per-event deadlines, because a feasible cohort can straddle a fixed
window boundary. The frozen design and outputs are unchanged; the paper and
processed-data documentation use “fixed-window eligibility” or
“fixed-partition ceiling.” `paper/formalism.md` defines the stronger interval-
packing optimum and a valid general upper bound.

The source panel and content-free feature extractor had already been selected,
implemented, and quality-audited when this document was frozen. Descriptive
counts such as the number of sessions, spans, route labels, and timestamp
anomalies were inspected. No microbatch cohort sizes or threshold-eligibility
outcomes had been computed.

## Question

For an asynchronous population of tool-using agents, what fraction of control
events can ever be presented to a GPU in same-operator cohorts large enough to
clear a hardware crossover threshold, under a fixed latency budget?

This study estimates a fixed-partition scheduling ceiling rather than the
performance of a particular scheduler. An event is *eligible* at threshold
`K` when at least `K` events assigned to the same execution class become ready
within the same fixed microbatch window. The fixed-window eligible share is
the event-weighted share satisfying that condition. No implementation
restricted to the same frozen partition can exceed it without fusing execution
classes, changing the workload, or reducing the hardware crossover threshold.

## Source panel

The source is the complete public tau2 panel from
`Exgentic/agent-llm-traces`, pinned to revision
`70036b93a04e61b0ea2706a68b962f4f26774587`:

- benchmarks: `tau2_airline`, `tau2_retail`, and `tau2_telecom`;
- 851 unique sessions and 9,031 recorded LLM spans;
- all four represented harnesses retained;
- source Parquet files are SHA-256 fingerprinted in the extraction manifest;
- derived features exclude prompt text, tool arguments, and tool results.

Each recorded span completion is one candidate control event. Its class is
derived from the recorded outcome: final/text, error, one named tool, or a
multi-tool outcome. The replay uses relative event order and timing only. It
does not interpret recorded span duration as model-service latency because
duration semantics vary materially by harness.

## Stationary swarm replay

Independent session arrivals follow a homogeneous Poisson process. Session
templates are sampled uniformly from the 851-session empirical panel. For a
target mean number of active sessions `C`, the arrival rate is
`C / mean(session_duration)`, which gives mean active population `C` under an
infinite-server replay.

Arrivals are generated from `-max(session_duration)` through the end of the
measurement interval so the retained interval is in steady state. Every event
from those sessions whose recorded completion time falls inside the
measurement interval is retained. The main factorial is:

- target active sessions `C`: 100, 1,000, 10,000, and 100,000;
- fixed microbatch window: 1, 5, 10, 25, 50, 100, 250, 500, and 1,000 ms;
- grouping rule: pooled universal transition, coarse event class, and exact
  route key;
- candidate crossover threshold `K`: 32, 64, 128, 256, 512, 1,024, and 4,096;
- five independently seeded replay repetitions;
- 60-second retained measurement interval;
- root seed: 20260811.

Fixed windows are aligned to the start of the retained interval. No event is
dropped as an outlier. The primary result aggregates all benchmarks and
harnesses; benchmark and harness strata are secondary diagnostics rather than
separate hypothesis tests.

## Outcomes

Primary:

- event-weighted eligible share at every `(C, window, grouping, K)` cell;
- event-weighted median, p90, and p99 ready-cohort size;
- smallest target concurrency reaching 50% and 90% eligibility, when observed.

Secondary:

- mean and p95 fixed-window waiting tax;
- route-heterogeneity penalty, defined as pooled eligibility minus exact-route
  eligibility at the same `(C, window, K)`;
- coarse-state recovery, defined as event-class eligibility minus exact-route
  eligibility;
- between-replay standard deviation and minimum/maximum eligible share;
- benchmark, harness, and route-frequency diagnostics.

## Hypotheses

- **R1 (readiness):** eligibility increases monotonically with active-session
  concurrency and batching-window duration, up to Monte Carlo variation.
- **R2 (regularity tax):** exact-route eligibility is no greater than coarse
  event-class eligibility, which is no greater than pooled eligibility in every
  cell by construction.
- **R3 (latency tension):** at `K >= 256`, exact-route eligibility remains below
  50% for at least one of the two largest tested populations when the batching
  window is at most 25 ms.
- **R4 (threshold leverage):** reducing `K` changes eligibility more strongly
  than an equal-factor increase in the batching window in at least one
  high-concurrency exact-route stratum.

R1 is evaluated after averaging repetitions; isolated Monte Carlo inversions
are reported rather than repaired. R2 is an implementation invariant and a
failed check invalidates the replay code. R3 and R4 are empirical pilot
hypotheses, not confirmatory significance tests.

## Quality checks and interpretation limits

The replay must reproduce the configured target mean active population within
Monte Carlo error and must retain nonzero events in every repetition. Results
are invalid if a finer grouping produces a larger eligible share than its
coarser parent at the same event times, window, and threshold.

The public traces contain failed and nonpositive-duration spans. They remain in
the event stream as error outcomes because the scheduler must still process
them; duration is clipped only implicitly by using the recorded completion
timestamp. Reported timestamps describe recorded harness behavior and do not
establish production arrival processes. A stationary Poisson swarm is a
controlled load model, not a claim that real deployments are Poisson.

The ceiling is necessary but not sufficient for a GPU win. It does not include
kernel runtime, transfer, launch, compilation, queueing outside the fixed
window, tool-service capacity, or end-to-end task utility. Hardware crossover
measurements from pilots 003/004 are joined only after this replay is computed.
