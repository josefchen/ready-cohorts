# Resident policy 002: measurement-hardened nuisance-pilot design

Status: **design draft only — not preregistered, not frozen, and not an
authorization to launch compute**.

Post-draft decision (2026-08-12): **blocked in its present form; do not
implement or scale this design**. Independent statistical, systems, and thesis
reviews found outcome-dependent sampling, an undefined intention-to-run value
for failures, provider/carryover confounding, infeasible artifact volume,
unstable placement-level power planning, and a mismatch between the trace
launch deadline and this draft's completion-SLO language. The corrected
research sequence is recorded in
[`resident-policy-001-report-2026-08-12.md`](resident-policy-001-report-2026-08-12.md).
This file is retained as an auditable design record, not as the next runbook.

This document uses the disclosed `resident-policy-001` results to design the
next source and experiment. No `resident-policy-002` timing result exists yet.
The source, runners, provider mix, schedules, workload seeds, artifact schema,
and analysis must receive content hashes in a separate preregistration before
the first performance-bearing placement.

## Decision this experiment supports

`resident-policy-001` established a useful mechanism result on four named
placements (local GTX 1660 Ti, Modal L4, RunPod L4, and Lambda H100): keeping a
GPU-computed global route decision on device was faster than copying that
decision to the host before launching the same route work. It did not provide:

- individual-invocation latency tails;
- CPU core-seconds or context-switch measurements;
- placement-balanced method order;
- a durable partial ledger after a hard remote failure; or
- complete, normalized provider provenance.

`resident-policy-002` is the **measurement-hardened six-placement nuisance
pilot** for that same matched mechanism. Its decision is whether the mechanism
is sufficiently correct, measurable, and practically useful to justify a
separately powered confirmation. It is not itself a confirmatory study.

The primary comparison remains:

```text
H = host_roundtrip
D = device_resident
```

Both methods use the same initialized state, predicate, route bodies, GPU,
compiler flags, and observation boundary. `H` returns the 4-byte decision to
the host and dispatches from there; `D` selects the uploaded route body on the
device. The oracle `O = no_decision_lower_bound` remains a diagnostic floor,
not a legal online baseline.

For the nuisance pilot, `H` must be the fastest trajectory-exact host-visible
GPU implementation selected using disclosed development-only data before the
new preregistration. At minimum it contains the current matched host-roundtrip
path; tuning may improve graph reuse, memory layout, affinity, and polling
without changing the required host observation. If another legal host baseline
wins that tournament, it becomes `H` before schedules are frozen. A baseline
change after a `002` outcome is observed is prohibited. The optimized CPU
engine remains necessary for the eventual system paper but is not a substitute
for this causal, same-GPU mechanism control.

## Why this is still part of the exact ready-cohort thesis

The paper's thesis joins two independently necessary quantities:

1. the hardware/runtime threshold `K(r,h,v,H)` at which compatible resident
   work is worthwhile; and
2. the trace's ability to assemble that compatible work before its deadline.

The frozen exact replay's primary condition is:

```text
active sessions C = 100,000
compatibility      = exact route
cohort threshold K = 256
relative deadline  = 50 ms
F                   = 0.301902
P*                  = 0.430007
U                   = 0.458487
```

The primary `resident-policy-002` shape remains `N=K=256`, `H=32`, with a
50 ms deadline/SLO label. This experiment asks whether one already-compatible
cohort can cross multiple deterministic control epochs with lower true tail
latency and lower CPU consumption when the route decision stays resident.
That is the hardware/runtime half of the joined boundary.

This experiment deliberately does **not** claim to realize the offline schedule.
It does not implement deadline queues, finite-capacity admission, route
compaction, or CPU fallback, and therefore does not estimate online accelerated
share `A` or gap closure `(A-F)/(P*-F)`. A later online-runtime experiment must
show `A > F`, preserve exact trajectories, and recover a frozen fraction of
`P*-F`. The manifest for every `002` placement should nevertheless record the
hash of the frozen trace-exact manifest so that `K`, the 50 ms deadline, and the
opportunity values cannot drift after performance is observed.

## Audit gap to design response

| Audit gap | Required response in `002` |
|---|---|
| P95/P99 were quantiles of 30 batch averages | Persist every invocation's wall duration and calculate an empirical nearest-rank P99 from at least 20,000 raw invocations per method, placement, and workload seed. |
| CPU displacement was unmeasured | Measure process CPU nanoseconds over the exact control interval, plus user/system CPU, voluntary and involuntary context switches, and equivalent occupied cores. |
| Method positions were shuffled but not balanced across placements | Assign the six permutations of `H`, `D`, and `O` exactly once in each six-placement hardware stratum. |
| A missing final manifest could destroy a remote crash record | Journal lifecycle events and upload immutable, checksummed sample chunks to durable storage throughout the run; collection must work without a final manifest. |
| Cloud provenance omitted or blurred important host/provider fields | Require a versioned provider receipt with requested and realized location, GPU, CPU/cgroup, image, software, pricing, lifecycle, and failure fields; unavailable values carry an explicit reason. |
| `O` reported an “observed” decision trace copied from the oracle | Mark decision validation for `O` as `not_applicable_oracle_replay`; validate its final state, but never count its decision trace as independently observed. |
| The prior R2 phrase “advantage increases” was ambiguous | `002` has one primary P99 estimand. Horizon surfaces and saved time per epoch are named exploratory diagnostics, not pass/fail hypotheses. |

## Minimal source changes

The route functions, predicate, state initialization, graph topologies, and
timed control boundaries should remain byte-for-byte or mechanically equivalent
to `resident-policy-001`. Any semantic or algorithmic change beyond the items
below requires a distinct treatment label and cannot be hidden as measurement
instrumentation.

### 1. Retain raw invocation measurements

`run_invocation` already produces one wall duration and one CUDA-event duration.
Instead of summing and discarding them, append one fixed-schema record to a
preallocated in-memory chunk:

```text
attempt_id, placement_id, hardware_stratum, provider_block
schedule_slot, period_id, workload_seed, microblock_id, invocation_id
method, N, H, status, failure_stage
start_monotonic_ns, end_monotonic_ns, wall_ns, device_ns
process_cpu_ns, thread_cpu_ns
voluntary_context_switches, involuntary_context_switches
decision_d2h_bytes, decision_syncs, terminal_syncs, host_graph_launches
deadline_ns, deadline_miss
exact_state_match, decision_validation_status
source_sha256, binary_sha256
```

`wall_ns` is the monotonic host interval from dispatch start through the common
terminal completion synchronization for one complete cohort-horizon invocation.
It must not be reconstructed from a batch average. Preserve the unsummarized
integer nanoseconds in the released artifact. This is the tail of a complete
cohort invocation, not a separately timestamped latency for each agent or each
internal epoch; all paper labels must retain that distinction.

The primary empirical quantile uses the nearest-rank definition:

```text
Q99(x_1,...,x_n) = sorted(x)[ceil(0.99 n) - 1].
```

Do not use interpolation and do not pool invocations across placements. At
`n=20,000`, the tail is supported by 200 observations at or above the empirical
99th-percentile rank.

### 2. Add CPU and scheduler counters at the matched boundary

Immediately around the existing control interval, collect:

- `CLOCK_PROCESS_CPUTIME_ID` for the primary process CPU time;
- `CLOCK_THREAD_CPUTIME_ID` for the calling-thread diagnostic;
- `getrusage(RUSAGE_SELF).ru_utime`, `.ru_stime`, `ru_nvcsw`, and `ru_nivcsw`
  immediately outside the matched interval, with an additional microblock-level
  reconciliation snapshot; and
- exact software counters for decision-induced D2H copies/bytes, decision
  synchronizations, the common terminal synchronization, and host graph
  launches.

The primary CPU numerator is the sum of process CPU nanoseconds inside the same
control boundary used for `wall_ns`. It excludes compilation, graph creation,
state reset, correctness copies, artifact writes, and the fixed rewarm between
microblocks. Those costs are reported separately as cold/setup and audit costs.

For method `m`, placement `p`, and workload seed `s`:

```text
Cinv_mps = sum(process_cpu_ns) / count(valid scheduled invocations)
C_mps = 1e-9 * sum(process_cpu_ns)
        / (count(valid scheduled invocations) * N * H)
equivalent_cores_mps = sum(process_cpu_ns) / sum(wall_ns)
CSv_mps = sum(voluntary switches) / count(valid scheduled invocations)
CSi_mps = sum(involuntary switches) / count(valid scheduled invocations)
```

`C_mps` is CPU core-seconds per valid synthetic agent-transition;
`Cinv_mps` is the per-cohort-invocation diagnostic. Their method ratios are the
same in the fixed primary cell, but the event denominator keeps the resource
unit aligned with the later runtime study.

No baseline subtraction is applied to CPU time or context switches. An
instrumentation-only calibration reports clock/counter overhead; before source
freeze, its P99 must be below 1% of the `N=256,H=32` host median. If not, the
counters must move to microblock boundaries and the timed boundary must be
revalidated before any pilot run.

Optional `perf_event_open`, cgroup `cpu.stat`, CPU migrations, page faults, and
GPU telemetry are diagnostics. Permission denial is recorded with errno and is
not silently converted to zero. The complete clock-plus-`getrusage`
instrumentation sequence is included in the overhead calibration.
`CLOCK_PROCESS_CPUTIME_ID` and `getrusage` support are mandatory for a placement
to contribute to the CPU endpoint.

### 3. Make the method order an input, not an internal reshuffle

Add required run-plan inputs for `attempt_id`, `placement_slot`, exact method
sequence, workload-seed sequence, scheduled invocation count, warm-up count,
deadline, chunk size, and journal destination. The binary must echo the parsed
plan and its SHA-256 before touching CUDA, then include that hash in every
record.

The binary must reject a duplicated method, an unknown method, an unsealed
seed, a schedule that disagrees with its placement slot, or a source/config
hash mismatch. It must not choose a replacement method after observing a
failure.

### 4. Replace the single terminal artifact with committed chunks

Use immutable chunks of 1,024 measured invocations. For each chunk:

1. collect samples in preallocated memory;
2. write a uniquely named temporary file outside the next timed period;
3. flush and `fsync` it;
4. atomically rename it to its final name and `fsync` the directory;
5. calculate SHA-256 and append a `CHUNK_COMMITTED` journal event;
6. copy it to a durable, versioned object prefix using create-if-absent
   semantics and verify the remote checksum;
7. append `CHUNK_DURABLE` only after that acknowledgement; and
8. execute the frozen untimed rewarm before collecting the next chunk.

The local and durable object names include `attempt_id`, period, method, seed,
and monotonically increasing chunk number. Nothing overwrites or resumes an
existing attempt. A retry receives a new attempt ID and retains the prior
prefix.

The append-only JSONL journal uses at least:

```text
LAUNCH_REQUESTED, PROVISIONED, PROCESS_STARTED
PERIOD_STARTED, CHUNK_COMMITTED, CHUNK_DURABLE, PERIOD_COMPLETED
FAILURE_OBSERVED, PROCESS_EXITED, COLLECTION_COMPLETED, TERMINATION_REQUESTED
TERMINATED
```

Each event carries UTC wall time, monotonic time where available, previous-event
hash, payload hash, and attempt ID. The final manifest is a convenient index,
not a prerequisite for collection. If the process or VM disappears, the
controller synthesizes a terminal `PROCESS_LOST` record from the last durable
event and retrieves every durable chunk. A provider runner must never require
“exactly one CSV and one manifest” before collecting partial evidence.

`SIGTERM` may trigger a best-effort terminal event. Correctness after `SIGKILL`,
OOM kill, host loss, and provider timeout comes from the external journal and
durable chunks, not from an unsafe signal handler.

### 5. Correct the validation vocabulary

For `H` and `D`, preserve independent field-by-field final-state comparison and
the complete observed decision trace after every invocation. For `O`, preserve
field comparison but set:

```text
decision_validation_status = not_applicable_oracle_replay
```

The aggregate manifest reports state-exact counts for all methods and
independently observed decision-exact counts only for `H` and `D`.

## Runtime and provider changes

### Provider receipt

Write a controller-side launch record before provisioning. The provider schema
must contain the following fields or an explicit `unavailable` value plus a
reason:

- experiment, attempt, placement-slot, schedule, and workload-plan IDs/hashes;
- request time, provider API/version, requested provider, region/datacenter,
  exact SKU, GPU count, tenancy/spot mode, CPU, memory, disk, and maximum cost;
- sanitized launch response, provider instance/pod/task ID, machine/host ID
  when exposed, provisioned/start/stop/termination times, retry relation, and
  final resource state;
- actual GPU name, UUID, PCI ID, compute capability, memory, MIG state,
  clocks/power limit and whether they were controllable;
- actual CPU model, requested quota, cgroup quota/period, cpuset, process
  affinity, NUMA node, logical CPUs visible, RAM, kernel, OS, and container
  runtime;
- CUDA driver/runtime/toolkit, host compiler, compile command, source,
  Makefile, binary, run-plan, workload, runner, and analysis hashes;
- immutable OCI base-image digest plus provider image ID; a mutable image tag
  alone is insufficient;
- provider price and currency with observation time; and
- artifact-prefix, per-object hashes, collection outcome, termination receipt,
  and any persistent volume or IP left behind.

Secrets, bearer tokens, SSH private material, account numbers, and complete
environment dumps are prohibited. Record an allowlisted environment schema.

### CPU and placement controls

Every method on one placement uses the same four-logical-CPU affinity mask and
the same cgroup quota. Select CPUs nearest the GPU's NUMA node when topology is
available. If a provider cannot enforce four CPUs, record the realized quota
and treat that provider configuration as a separately named block; do not pool
it silently with controlled placements.

GPU clocks and power limits are held fixed when the provider permits it. When
they cannot be controlled, sample available clocks, temperature, power, and
throttle reasons at period boundaries and retain the placement. Never exclude a
slow placement based on telemetry observed after treatment starts.

## Workload, periods, and balanced schedule

### Primary and secondary shapes

- Primary decision-bearing shape: `N=256`, `H=32`, exact-route-compatible
  cohort, deadline/SLO `50 ms`.
- Mechanism diagnostics: `N={128,256,512}` and `H={1,8,32,64}`. These do not
  carry pass/fail inference.
- Five common sealed state/workload seeds per placement. The same initialized
  states and seed order are used by all methods.

The diagnostic atlas may use fewer samples after the primary periods complete,
but its sample count and stopping rule must be frozen before launch. It cannot
replace the primary cell if a more favorable crossover appears.

### One measured period

For one method and one workload seed:

1. reset deterministic state and verify plan/source hashes;
2. instantiate/upload graphs and record cold/setup time separately;
3. warm for at least 1,000 invocations and 60 seconds, whichever is longer;
4. collect at least 20,000 valid individual invocations; and
5. continue to 30 measured minutes if that is longer, without looking at the
   running effect estimate.

Correctness validation, journal commits, and durable upload occur outside the
per-invocation timed boundary. The fixed rewarm after each artifact commit is
identical for every method. No adaptive warm-up, clock stabilization, sample
deletion, or early stopping based on latency is allowed.

### Six-placement schedule per hardware stratum

Let `H=host_roundtrip`, `D=device_resident`, and `O=oracle floor`. Assign these
six sequences exactly once across the six fresh placements:

| Slot | Period 1 | Period 2 | Period 3 |
|---:|---|---|---|
| 1 | H | D | O |
| 2 | D | O | H |
| 3 | O | H | D |
| 4 | D | H | O |
| 5 | H | O | D |
| 6 | O | D | H |

This balances every method twice in every period and every directed first-order
carryover pair twice. For a planned 3/3 provider block, slots 1–3 form one
period-balanced Latin square and slots 4–6 its reverse. Randomize and freeze
which provider receives which half, then randomly map its three placement
labels to slots. The exact assignment seed and resulting table are part of the
future preregistration.

Apply the assigned method sequence within every workload-seed block. Randomize
the common seed-block order independently of method and freeze it per placement.
Do not regenerate an order after seeing provisioning or benchmark outcomes.

The proposed nuisance strata are:

| Hardware stratum | Provider block | Pilot placements |
|---|---|---:|
| Exact NVIDIA L4 24 GB | Modal 3, RunPod 3 | 6 |
| One frozen H100 80 GB form factor/PCI identity | Modal 3, Lambda 3 | 6 |

Do not pool H100 PCIe and SXM variants under one label. If the exact H100 form
factor cannot be matched across providers, re-scope that stratum before any
timing result is opened. Spread each stratum over at least three days for the
nuisance pilot. Confirmation retains the existing minimum of five days.

A provisioning failure before `PROCESS_STARTED` may be replaced by a fresh
attempt inheriting the same schedule slot; both records remain. A repeated GPU
UUID/machine detected before timing is a predeclared non-independence failure
and may be replaced the same way. A failure after any measured sample is an
intention-to-run outcome and is not erased by a replacement. Instrumentation
corruption may invalidate an entire placement only through a treatment-blind
predeclared rule.

## Outcomes and estimands

### Primary performance estimand

For method `m`, placement `p`, and common workload seed `s`, calculate the
nearest-rank P99 from the raw exact-completion invocation durations:

```text
Q_mps = Q99({wall_ns for method m, placement p, seed s}).
```

The placement summary averages paired seed-level log ratios:

```text
z99_p = mean_s log(Q_Dps / Q_Hps).
```

The target estimand within a frozen hardware/provider population is:

```text
theta99 = E_placement[z99_p]
R99     = exp(theta99).
```

`R99 < 1` favors device residency. Report `1/R99` as tail speedup only as a
human-readable transform. Do not pool raw invocations across placements, and do
not treat the five seeds or millions of invocations as new deployment samples.

Completed/exact P99 is interpretable only when the placement passes the frozen
completion and correctness gates. Deadline misses, CUDA failures, crashes, and
scheduled-but-unobserved invocations are reported as intention-to-run outcomes,
not silently dropped. Before preregistration, freeze whether the confirmatory
composite latency endpoint assigns noncompletion the 50 ms SLO cap; `002` must
report both completed-invocation P99 and the unconditional completion/SLO rate.

### CPU resource estimand

For the same cell, define:

```text
C_mps    = 1e-9 * sum(process_cpu_ns)
           / (valid_invocations * N * H)
zcpu_p   = mean_s log(C_Dps / C_Hps)
thetaCPU = E_placement[zcpu_p]
RCPU     = exp(thetaCPU).
```

`RCPU < 1` favors device residency. CPU core-seconds per valid synthetic
agent-transition is a prespecified co-primary engineering outcome and a
required component of any later deployment-level claim. CPU core-seconds per
cohort invocation, equivalent occupied cores, user/system split,
voluntary/involuntary switches per 10,000 invocations, host synchronizations,
D2H bytes, and graph launches are mechanism diagnostics.

### Guardrails and secondary outcomes

- zero unexplained state mismatch for all methods;
- zero unexplained observed-decision mismatch for `H` and `D`;
- completion rate, CUDA/runtime failure rate, crash rate, and 50 ms deadline-miss
  share;
- P50/P95 and median wall time from raw invocations;
- CUDA-event latency, throughput, cold/setup time, amortized lifetime cost, and
  artifact/audit overhead;
- `D` overhead above `O`, with `O` clearly labeled an illegal oracle floor;
- horizon and population surfaces; and
- provider-block, day, CPU-quota, clock, and temperature sensitivity.

R2-style horizon monotonicity is exploratory. If reported, separately show
P99 ratio, absolute saved time per invocation, and saved time per epoch; never
collapse them into an undefined word such as “advantage.”

## Independent unit and power plan

The inferential unit is one fresh GPU/server placement. A workload seed is a
crossed repeated factor shared by methods; an invocation is a nested technical
observation used to estimate the placement/seed tail. Therefore:

```text
six placements in one hardware stratum means n=6, not n=600,000.
```

The six-placement stage estimates nuisance quantities only: placement-level
variance of `z99_p` and `zcpu_p`, seed heterogeneity, within-period dependence,
P99 density, context-switch dispersion, completion/failure rates, and provider
sensitivity. It reports no efficacy p-value.

After the six placements, use the upper 80% confidence limit of the paired
placement variance in 10,000 simulations of the exact balanced design. Choose
the smallest confirmatory placement count with at least 90% power at one-sided
`alpha=0.025`, familywise error at most 5% across the two hardware strata, and
the frozen smallest worthwhile effect. Add a 10% provisioning reserve. The
current 30-per-stratum plan is a floor, not a guaranteed final sample size; a
blinded variance re-estimate may increase but never decrease it.

The eventual confirmatory analysis uses placement-paired log ratios and a
two-way cluster bootstrap over placement and workload seed, with provider as a
fixed block. Report a 95% interval and next-placement prediction interval.
Within-period block bootstrap intervals are diagnostics for quantile precision,
not substitutes for placement replication.

## Pre-scale qualification gates

All must pass before provisioning the six-placement nuisance pilot:

1. **Frozen equivalence:** `H`, `D`, and `O` reproduce the `001` oracle states
   and legal decision traces on adversarial small grids and both target compute
   capabilities.
2. **Quantile oracle:** fixture distributions, including ties and a single
   extreme tail, reproduce the nearest-rank Q99 exactly.
3. **Counter validity:** process CPU time is monotone; aggregate process CPU
   reconciles with user plus system CPU within a frozen tolerance; context-switch
   deltas are nonnegative; instrumentation overhead passes the 1% gate.
4. **Schedule validity:** all six sequences balance period and directed
   carryover; the binary rejects a schedule/hash mismatch.
5. **Crash recovery:** forced exit, `SIGTERM`, `SIGKILL`, simulated OOM, network
   loss, and missing-final-manifest tests recover every previously durable chunk
   and produce an explicit incomplete attempt without overwrite.
6. **Provenance completeness:** every required field is populated or carries an
   explicit unavailable reason; source, binary, plan, image, chunk, and runner
   hashes reconcile from a clean collector.
7. **Provider canary:** compile/help and a correctness-only smoke succeed on the
   exact L4 and H100 forms without exposing performance outcomes. The binary
   therefore needs an untimed `--correctness-only` mode that does not emit
   duration fields. A performance canary is not permitted outside the frozen
   pilot.

Failure of a qualification gate blocks cloud scaling; fix it under a new source
hash and rerun qualification. Passing these tests is measurement readiness, not
evidence of performance.

## Six-placement scale, kill, and scope gates

Evaluate gates independently in each hardware stratum after all six scheduled
placements or their retained intention-to-run failures are accounted for.

### Validity gate

- Any unexplained state, decision, order, timestamp, or artifact-integrity
  mismatch kills that source version for performance interpretation.
- A systematic inability to collect mandatory process CPU or context-switch
  counters blocks the CPU claim and must be fixed before confirmation.
- Provider provisioning failures remain in the launch ledger but do not count
  as semantic failures when they occur before the benchmark begins.

### Engineering scale gate

A stratum may advance to confirmation planning only if correctness passes and
one of these pilot point-estimate paths holds:

1. **tail-performance path:** `R99 <= 0.90`; or
2. **CPU-displacement path:** `RCPU <= 0.90` and `R99 <= 1.02`.

Additionally, at least five of six placement summaries must favor `D` on the
metric that opens the path. This direction-consistency rule is an engineering
gate, not a significance test. It prevents one unusually favorable host from
opening a costly confirmatory program.

The device-minus-host 50 ms deadline-miss share may not exceed `0.01`
percentage point in the primary pilot panel. Report the paired difference by
placement as well as the aggregate counts; any larger adverse difference blocks
scaling pending a separately designed reliability study.

### Kill or re-scope rules

- If neither L4 nor H100 clears either path, kill the general conditional-graph
  scaling program and publish the measured mechanism boundary.
- If exactly one stratum clears, design and power a card-specific confirmation;
  do not retain a two-card claim.
- If CPU improves but P99 exceeds the `+2%` no-harm limit, treat the method as a
  CPU/latency tradeoff and do not enter the current confirmation.
- Do not change `N=256`, `H=32`, the 50 ms label, provider mixture, baseline,
  quantile definition, or method after viewing pilot outcomes. A change is a new
  experiment and preregistration.
- A wide placement distribution is inconclusive, not equivalence. More timing
  invocations cannot replace more placements.

### Later confirmation thresholds

The six-placement gate only decides whether to invest. A deployment-level
positive result remains stricter:

- performance point estimate at least 15% better than the strongest frozen
  legal baseline, with adjusted confidence bound favoring `D`;
- CPU core-seconds point estimate at least 25% lower, with adjusted upper ratio
  below 1.0;
- task completion noninferior within `+2%` and utility difference above the
  frozen `-1 percentage point` margin; and
- trajectory exactness and intention-to-run reliability gates all pass.

`resident-policy-002` alone cannot satisfy the task-completion or utility gates.

## Required artifacts

Before running, define versioned schemas for:

```text
plans/resident-policy-002/<placement-slot>.json
launch-ledger/resident-policy-002.jsonl
attempts/<attempt-id>/lifecycle.jsonl
attempts/<attempt-id>/samples/<period>-<seed>-<chunk>.csv
attempts/<attempt-id>/block-summary.jsonl
attempts/<attempt-id>/final-manifest.json        # optional on failure
processed/resident-policy-002-placement-summary.csv
processed/resident-policy-002-pilot-manifest.json
```

The processed manifest lists all requested, provisioned, started, completed,
failed, invalidated, retried, and analyzed attempts; every source and output
hash; all missing schedule slots; and the exact denominator used by every rate.
No raw timing observation is edited in place.

## Claims permitted if the nuisance pilot passes

Permitted:

> Across the six observed placements in the named hardware/provider stratum,
> the frozen device-resident decision mechanism passed trajectory checks and
> showed a prespecified pilot-level reduction in true invocation P99 or CPU
> core-seconds relative to the matched host-roundtrip path.

Not permitted:

- GPUs are cheap or general CPU cores;
- the method improves all L4, H100, providers, or agent workloads;
- millions of invocations provide millions of independent samples;
- `002` realizes the exact offline ready-cohort optimum or measures `A`;
- the oracle floor is an online baseline;
- the current synthetic global binary policy is a complete agent orchestrator;
  or
- the nuisance pilot is a powered confirmatory result.

The publication-worthy chain remains:

```text
exact trace opportunity P*  ->  measured resident mechanism boundary
                             ->  finite online runtime achievement A
                             ->  CPU/end-to-end leverage without semantic harm.
```

`resident-policy-002` hardens the second link. It neither replaces nor skips
the third.

## Items that must be frozen in the future preregistration

- exact source, Makefile, runner, collector, schema, image, workload, and
  analysis hashes;
- H100 form factor and provider mixture;
- all twelve placement schedule assignments and workload-seed orders;
- CPU quota/affinity and permitted provider-specific deviations;
- invocation count, warm-up, chunk size, deadline, timeout, and noncompletion
  rule;
- empirical quantile definition and all primary formulas;
- instrumentation-overhead and data-validity tolerances;
- provisioning retry, repeated-machine, and instrumentation-invalidation rules;
- pilot scale/kill gates and later SESOIs;
- variance-estimation and 10,000-study power procedure; and
- a content hash linking the primary native cell to the frozen trace-exact
  opportunity manifest.

Until those items are frozen, this file is an engineering design review, not a
preregistration.
