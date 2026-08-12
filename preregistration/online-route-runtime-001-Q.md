# Online route runtime 001-Q: finite-capacity qualification

Status: **draft, not frozen, no outcome-bearing run authorized**.

This experiment replaces the blocked `resident-policy-002` design. It receives
a new identifier because the treatment has changed from a resident-decision
microbenchmark to a finite-capacity online route scheduler. No Modal, RunPod,
Lambda, or other performance-bearing launch may use this ID until the source,
workload bundle, attempt allocation, analysis, image, binary contract, and this
document have final hashes and a freeze timestamp.

## Question

For the same runtime source and frozen event lists, can a finite online
device-resident route compactor:

1. measure its own route/body-specific profitable cohort threshold `K`;
2. accelerate a larger share `A` than an origin-aligned fixed-window policy
   while respecting a 50 ms latest-launch deadline; and
3. close at least half of the exact offline opportunity `P* - F` without
   harming raw tail latency, correctness, ordered commits, or CPU use?

This qualification joins the missing chain:

```text
measured K -> recomputed F and P* -> finite-online observed A
```

It is descriptive development evidence. It cannot support a provider,
hardware-population, or datacenter-impact claim.

## Explicit corrections from earlier designs

- `K=256` is not inherited from the trace study. Every placement measures `K`
  for the exact runtime binary before replay.
- Primary temporal horizon is `H=1`. The trace does not establish that 32
  observation-free decisions are legal.
- 50 ms is release-to-device-start launch slack, not a completion SLO.
- Event lists, attempt IDs, and method periods are fixed. There is no
  completed-only target and no optional continuation.
- Missing, late, crashed, and mismatched events remain in the intention-to-run
  denominator.
- L4 is the only qualification stratum. Lambda H100 belongs to a later,
  separately designed stratum.

## Fixed qualification layout

Eight exact-L4 attempt IDs are scheduled, with no post-outcome replacement:

```text
online-route-runtime-001Q-modal-l4-a1
online-route-runtime-001Q-modal-l4-a2
online-route-runtime-001Q-modal-l4-a3
online-route-runtime-001Q-modal-l4-a4
online-route-runtime-001Q-runpod-l4-a1
online-route-runtime-001Q-runpod-l4-a2
online-route-runtime-001Q-runpod-l4-a3
online-route-runtime-001Q-runpod-l4-a4
```

Attempts span at least two calendar days. One attempt requests one fresh L4
placement. Repeated physical GPU UUIDs are reported rather than replaced.

### Methods

- `C`: optimized CPU-only immediate execution on a pinned pool of four logical
  CPUs;
- `W`: origin-aligned 50 ms fixed-window host-GPU batching with CPU fallback;
- `H`: online earliest-deadline host-controlled executable-route compactor;
- `R`: the same online policy with the route decision and queue service kept on
  device.

`P*` is an illegal future-aware, zero-service, unlimited-capacity reference. It
is never described as a baseline.

### Period and carryover control

Each provider block executes one complete four-method Williams square:

| Slot | Method order |
| --- | --- |
| 1 | `C, W, R, H` |
| 2 | `W, H, C, R` |
| 3 | `R, C, H, W` |
| 4 | `H, R, W, C` |

The final attempt-to-slot mapping and within-method replay-seed order are
generated from root seed `20260812`, written to an immutable allocation file,
and hashed before source freeze. Method-effect summaries remain sealed until
all fixed attempts finish or a prespecified validity stop fires.

## Frozen workload target

The intended source is the pinned Exgentic tau2-derived panel and stationary
replay already used by trace-replay-003:

- source revision:
  `70036b93a04e61b0ea2706a68b962f4f26774587`;
- target active sessions: `100000`;
- outcome-derived route keys from the trace, used only as input metadata;
- three existing replay seeds from root seed `20260811`;
- release horizon: exactly 60 seconds;
- launch deadline: `release_ns + 50_000_000`;
- no event deletion.

The runtime bundle materializes content-free fixed event lists with:

```text
event_id
session_id
sequence_id
route_id
executable_route_id
body_family
state_schema_version
policy_version
release_ns
launch_deadline_ns
state_input
oracle_output
```

The qualification uses synthetic typed deterministic bodies mapped before the
freeze:

- tool-call routes: a frozen integer tool-bookkeeping transition;
- final/text routes: a frozen integer terminal-bookkeeping transition;
- error routes: CPU-only;
- `executable_route_id` is a hash of the frozen body binary identity, state
  schema version, policy version, and required effect schema. It determines the
  queue. The outcome-derived trace `route_id` is retained for diagnostics but
  cannot establish compatibility.

State uses integer or fixed-point fields. CPU and GPU outputs must match
field-for-field. Each session preserves sequence order and produces an ordered
commit log. This qualifies scheduler mechanics, not real tau2 tool execution.

## Runtime semantics

- one GPU and one non-preemptive GPU execution stream;
- maximum GPU batch `B=1024`;
- stable route-ID tie breaking;
- no access to unreleased events;
- no cross-route fusion;
- no temporal fusion;
- no retry under another method;
- CPU fallback uses the same pinned four-logical-CPU pool;
- all graph construction, queue allocation, and cold initialization are
  measured separately rather than hidden.

`R` and `H` share this policy:

1. enqueue only released events into executable-route queues;
2. when the GPU is idle, consider routes with at least their measured `K`;
3. choose the route with the smallest earliest deadline, breaking ties by
   route ID;
4. take at most `B` events, ordered by deadline and then event ID;
5. send underfilled or capacity-blocked events to CPU at the frozen fallback
   guard;
6. count a GPU event as accelerated only when its exact output is produced and
   observed device start is no later than its launch deadline.

`W` uses the same finite GPU and CPU fallback but restricts cohorts to one
origin-aligned 50 ms partition. `C` executes every event immediately on the CPU
subject to the same sequence and commit rules.

## Per-placement crossover measurement

Before replay periods, every started attempt runs this fixed calibration on its
exact binary:

```text
body families: tool_bookkeeping, terminal_bookkeeping
N: 8, 16, 32, 64, 128, 256, 512, 1024
H: 1
warm-up: 256 cohort invocations per cell
measurement: four fixed microblocks of 512 cohort invocations per cell
```

Resident GPU and optimized CPU execute in a frozen paired block order. Report:

- `K0`: smallest safe-suffix `N` whose median resident-GPU wall time beats CPU;
- `K15`: smallest safe-suffix `N` for which the GPU/CPU median ratio is at most
  `1/1.15` and the empirical P99 ratio is at most `1.02`.

Safe suffix means that `N` and every larger tested population satisfy the
rule. If no value passes, record `K=infinity`. Do not add a larger population
or skip the replay. Primary online qualification uses `K15` and recomputes `F`
and `P*` for each placement, body family, and replay seed.

A development-only tournament selects the strongest legal baseline `B*` among
`C`, `W`, and `H` before qualification. All three remain in the final table.
Qualification outcomes cannot redefine `B*`.

## Fixed method periods

For every method and replay seed:

- fixed 10-second warm-up;
- exactly the 60-second frozen event release list;
- fixed 500 ms drain interval;
- 75-second hard watchdog;
- no continuation to obtain a target success count;
- no removal of overload or tail events.

Counters are partitioned into 60 fixed one-second logical-time microblocks.
Raw per-event latency is retained. Process, thread, cgroup CPU, context switch,
queue, and GPU counters are sampled at microblock boundaries. Event records are
chunked by fixed ranges of 8192 scheduled event IDs. A final manifest is useful
but not required for earlier chunks to remain durable.

## Primary quantities

For placement `p`, replay seed `s`, and method `m`, every scheduled event is in
the denominator:

```text
A_mps = count(events executed on GPU in a batch >= K,
              with exact output and observed device start <= deadline)
        / count(all scheduled events)
```

Missing, late, crashed, mismatched, duplicated, and unobserved events do not
contribute to the numerator.

Using the same list and the placement-specific `K_R`:

```text
F_ps  = fixed-partition eligible share
P*_ps = exact zero-service offline packing share
D_ps  = A_Rps - F_ps
G_ps  = (A_Rps - F_ps) / (P*_ps - F_ps), when P*_ps > F_ps
```

`A_Rps <= P*_ps` is a validity invariant. A violation indicates an accounting,
clock, threshold, or future-visibility defect.

Co-primary engineering outcomes are:

- empirical release-to-ordered-commit P99;
- process/cgroup CPU core-seconds per scheduled event;
- valid event throughput;
- launch-deadline miss share;
- exact state, route, sequence, and ordered-commit status.

For an intention-to-run latency view, late, missing, and crash-lost events
receive a frozen 500 ms failure value. The paper must keep that analysis value
separate from the 50 ms launch deadline. Completed-event latency is reported as
a secondary descriptive distribution.

Other recorded outcomes include queue wait, batch-size distribution, CPU
fallback share, GPU busy share, service duration, host-device bytes, host
synchronizations, graph launches, compilation, cold setup, power where
available, and artifact overhead. Seeds are aggregated within placement before
any across-placement summary. Raw events are never pooled as independent
placement samples.

## Intention-to-run rules

- Each of the eight IDs receives at most one launch request.
- A provisioning failure before `PROCESS_STARTED` is retained and not replaced.
- Wrong GPU, unavailable exact L4, or price above the frozen ceiling is a
  retained pre-start outcome.
- After `PROCESS_STARTED`, compilation failure, OOM, crash, timeout, missing
  chunks, and missing final manifest are system outcomes.
- Durable exact events keep their observed statuses after a post-start failure;
  every remaining scheduled event receives `A=0` and the frozen failure
  latency.
- A method period that never begins after process start has all scheduled
  events encoded as failures.
- Instrumentation invalidation requires a treatment-blind mechanical rule. The
  attempt remains in the launch ledger.
- Skipped later attempts after a validity kill are reported as stopped by that
  rule rather than silently omitted.

Operators may inspect correctness, lifecycle, and spend state. They may not
inspect unsealed method effects while the fixed attempt sequence is running.

## Pre-cloud gates

All gates must pass before the first performance-bearing request:

1. exact CPU/GPU field and ordered-commit equality on adversarial small cases;
2. brute-force agreement of event accounting and offline `P*` on tiny traces;
3. event conservation: each event reaches exactly one terminal state;
4. Williams-square and seed-order validation;
5. mechanical rejection of future-event access, duplicate IDs, sub-`K`
   accelerated batches, and late launches counted as valid;
6. device-start clock calibration error below 0.5 ms, with events in the
   unresolved band counted as misses;
7. counter overhead below 1% of a development primary-cell median;
8. forced exit, `SIGKILL`, OOM, network loss, and missing-manifest recovery;
9. hashes for source, binary, image, allocation, plan, workload, runner, and
   analysis;
10. correctness-only provider canaries with duration fields suppressed;
11. development-only baseline tournament complete and `B*` frozen;
12. launch, collect, terminate, and absence-verification dry runs pass without
    secrets in receipts.

## Validity and kill rules

The source version is killed after any:

- unexplained state, route, sequence, or commit mismatch;
- use of an unreleased event;
- batch below `K` counted in `A`;
- late or unobserved device start counted as deadline-valid;
- `A > P*`;
- event-conservation or artifact hash-chain failure.

This is a validity stop, not an efficacy stop. It may prevent later launches
for safety, but the skipped fixed IDs remain in the report.

The qualification is operationally interpretable only when at least six of
eight requests reach `PROCESS_STARTED`, with both providers represented.
Otherwise the result is inconclusive and any retry needs a new preregistration.

## Advancement gates

Advance to nuisance estimation only when equal placement weighting and
intention-to-run post-start failures give all of the following:

- every started attempt is trajectory-exact;
- `P* - F >= 0.05` in the primary opportunity;
- mean `G >= 0.50`;
- at least six of eight attempt summaries have `A > F`;
- either `R/B*` P99 ratio is at most `0.90`, or `R/B*` CPU ratio is at most
  `0.90` while its P99 ratio is at most `1.02`;
- treatment-minus-baseline launch-miss share is no more than 0.01 percentage
  point.

These are engineering gates, not significance tests. If `A` does not beat `F`
or closes less than half of `P* - F`, retain the negative result and pivot to
an offline/online boundary analysis. Do not retune `K`, deadline, body mapping,
event panel, or outcome definitions under this experiment ID.

## Provider lifecycle and spend

Before provisioning, append `LAUNCH_REQUESTED` with the attempt ID, provider,
exact GPU, price ceiling, hard timeout, and all plan/bundle hashes. Lifecycle
states are:

```text
LAUNCH_REQUESTED
PROVISIONED
PROCESS_STARTED
PERIOD_STARTED
CHUNK_COMMITTED
CHUNK_DURABLE
PERIOD_COMPLETED
FAILURE_OBSERVED
PROCESS_EXITED
COLLECTION_COMPLETED
TERMINATION_REQUESTED
TERMINATED
ABSENCE_VERIFIED
```

Qualification limits:

- maximum accepted L4 price: USD 1.25 per GPU-hour;
- maximum provisioned lifetime: 45 minutes per attempt;
- maximum GPU commitment: USD 7.50;
- storage and egress allowance: USD 2.50;
- hard all-in cap: USD 10.00.

If the next request could exceed the cap, record
`NOT_REQUESTED_COST_CAP`. Do not substitute hardware. Use provider-side
timeouts plus controller-side termination in `finally`. Delete RunPod pods and
volumes after validated collection; Modal jobs are ephemeral. Verify zero live
inventory after every attempt and after the experiment. Public receipts use an
allowlist and exclude credentials, complete environments, SSH material, and
bearer-bearing artifact URLs.

## Later confirmation, not part of 001-Q

The eight qualification attempts are development data and remain outside
confirmation. Each retained hardware stratum first receives at least 12 fresh,
blinded nuisance placements over at least five days and, where possible, two
zones. L4 and one exact H100 form are separate strata. A simulation of at least
100000 full studies uses the exact provider, day, order, failure, workload, and
multiplicity design. Confirmation has a floor of 30 analyzed fresh placements
per stratum and increases until the lower Monte Carlo power bound reaches 90%
at one-sided alpha 0.025 with familywise error at most 5%.

The future claim is intersection-union: online gap recovery, at least 15%
performance improvement, at least 25% CPU reduction, end-to-end completion
noninferiority within 2%, task-utility loss below one percentage point, exact
trajectories, and intention-to-run reliability must all pass their adjusted
criteria. A broad two-card claim requires both strata to pass.

## Fields required before freeze

- [ ] implementation source path and SHA-256;
- [ ] runner and lifecycle controller SHA-256;
- [ ] analysis source SHA-256;
- [ ] workload bundle and event-allocation SHA-256;
- [ ] container digest and compile command;
- [ ] exact CPU compiler flags, affinity, and NUMA contract;
- [ ] exact attempt-to-slot and replay-seed allocation;
- [ ] frozen `B*` selection packet;
- [ ] all pre-cloud gate receipts;
- [ ] freeze timestamp and signed decision record.

Until every box is complete, this document is a design record only and no
outcome from `online-route-runtime-001-Q` is admissible.
