# Research decision memo: from a feasibility boundary to an exact resident runtime

Decision date: **2026-08-11**
Evidence cutoff: **2026-08-11, after trace replay 003 and native dispatch 001**
Audience: systems researchers deciding what to build, preregister, and scale over
the next twelve weeks.

Supersession note (2026-08-12): the later four-placement
[`resident-policy-001` report](resident-policy-001-report-2026-08-12.md)
qualifies the device-resident route-decision mechanism and records an
adversarial review of this memo's proposed follow-up. The six-placement
nuisance plan, upper-80%-variance rule, completed-only `20,000 valid or 30
minutes` rule, and five-seed bootstrap below are retained as historical design
text but are **not executable guidance**. The corrected sampling and assurance
rules are in [`statistical-design.md`](statistical-design.md); placement-scale
spend is reserved for the first finite-capacity online runtime that measures
`A` against matching `F` and `P*`.

## Technical decision

The flagship should be **The Ready-Cohort Boundary: When LLM-Agent Control Has
Enough Work for a GPU**. The paper should not claim that GPUs are cheap CPU
cores, that a GPU state machine is new, or that the current pilot accelerates
agent orchestration. Its narrow thesis is:

> GPU acceleration of deterministic agent control is governed by the
> intersection of a route-specific hardware crossover and the workload's
> ability to assemble semantically compatible ready events before their
> deadlines. A runtime earns a systems claim only if it preserves exact
> trajectories, converts a material share of that offline opportunity into
> online work, reduces CPU consumption, and does not harm end-to-end task
> behavior.

The first half of that thesis now has unusually strong pilot evidence. Under
the frozen equal-relative-deadline model, an exact `O(N log N)` dynamic program
places the achievable offline share `P*` between fixed-window eligibility `F`
and a local-eligibility upper bound `U`. In the preregistered primary cell,
`F=30.19%`, `P*=43.00%`, and `U=45.85%`; the exact schedule closes `81.83%` of
the alignment gap between `F` and `U`. This is evidence that deadline-aware
cohort formation is a real opportunity rather than a plotting artifact.

The second half is not established. A five-placement native pilot proved that
the reviewed CUDA mechanism runs on a local GTX 1660 Ti, Modal L4, RunPod L4,
and Lambda H100 while matching every checked final-state field. It also showed
that a fixed nested device graph is the wrong intervention: across all 60
placement-by-shape cells, its median wall time was `1.075x` to `1.994x` the
matched host-graph median. The next experiment must introduce a genuine
device-side route decision and remove a matched host
synchronize/copy/dispatch epoch. More repetitions of the fixed nested graph
would add precision to a mechanism we already know has no structural advantage.

## The paper's named object and exact scope

For route `r`, hardware/runtime configuration `h`, visibility policy `v`, and
fusion horizon `H`, let `K(r,h,v,H)` be the smallest compatible resident cohort
for which the GPU clears the frozen tuned baseline. For released control events
with launch deadlines, distinguish four quantities:

```text
F  = share eligible in a frozen, origin-aligned window partition
P* = maximum share assigned by an offline deadline-respecting scheduler
U  = share that is locally eligible somewhere inside each event's interval
A  = share actually accelerated by a finite online runtime
```

Under the current zero-service, unlimited-capacity, equal-relative-deadline
model,

```text
F <= P* <= U       and       A <= P*.
```

`F` is exact only for its frozen partition. `P*` is the exact offline optimum
only for the current model and integer-nanosecond clock. `U` is an upper bound,
not necessarily a jointly achievable schedule. A production runtime introduces
service time, bounded capacity, queueing, fairness, route-specific kernels, and
external observation barriers, so its achieved share can be strictly below
`P*`. These definitions and proofs are recorded in
[`paper/formalism.md`](../paper/formalism.md); the frozen empirical test is
[`trace-replay-003.md`](../preregistration/trace-replay-003.md).

This framing is the residual gap that survives the literature collision audit.
Agent state machines already run on GPUs in
[FLAME GPU 2](https://doi.org/10.1002/spe.3207). Conditional and device-launched
graphs are documented CUDA mechanisms
([CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)),
and resident queues, task graphs, and dynamic operator dispatch are occupied by
systems such as [GPUOS](https://arxiv.org/abs/2604.17861). Workflow-aware
scheduling is also occupied by [Agentix](https://www.usenix.org/conference/nsdi26/presentation/luo),
[SAGA](https://arxiv.org/abs/2605.00528), and other compound-agent systems.
[Agentic CPU--GPU Scheduling](https://arxiv.org/abs/2607.22242) is the closest
title-level collision, but its unit is a complete AI tool; ours is an internal,
deterministic transition between model/tool effects. The defensible novelty is
the **joined hardware-and-workload boundary plus a trajectory-exact runtime
evaluated against it**, not any one GPU mechanism. The full collision record is
in [`literature-audit-2026-08.md`](literature-audit-2026-08.md).

## What the current evidence supports

| Evidence layer | Correct independent unit | Observed result | Permitted inference |
|---|---|---|---|
| Compiler-matched crossover pilots | one host/process launch | Resident GPU paths cross the observed compiled CPU reference in regular, fused regimes; host visibility can erase the win. | Descriptive hardware-boundary evidence on the observed launches, pending an optimized C++/SIMD/OpenMP baseline. |
| Hardware replication atlas | fresh placement; 18 launches total, three per requested GPU class | 19,440 timing rows, but only 18 independent placements; all observed UUIDs were distinct. | Preliminary placement variability and cell selection, not a population claim. Three wins in three placements have one-sided sign-test `p=0.125`. |
| Exact trace packing | Monte Carlo replay seed conditional on one fixed 851-session trace panel | All 540 preregistered cells passed `F <= P* <= U`, compatibility, deadline, threshold, and closed-bound invariants. `P*>F` in 102 cells; `P*<U` in 90; all three were equal in 438. | Exact algorithmic validity on the tested grid and conditional simulation evidence. The three seeds are not independent production traces. |
| Primary exact-trace cell | three replay seeds conditional on the same panel | At 100,000 target active sessions, exact-route grouping, `K=256`, and 50 ms: mean `F=0.301902`, `P*=0.430007`, `U=0.458487`, 1,046.3 exact batches, and alignment-gap closure `0.818332`. | A material schedulable alignment opportunity under the frozen Poisson-template, zero-service model; no population `p`-value and no online-runtime claim. |
| Native dispatch 001 | fresh GPU placement | Five distinct GPU UUIDs and 12,000 measured rows; every row was status `ok` and field-exact. | Mechanism compatibility and checked exactness on the observed systems, not an empirical reliability rate or broad portability claim. |
| Fixed nested graph contrast | placement-by-`(N,H)` cell; 60 cells | Every cell's nested-device-graph median exceeded its matched host-graph median; ratios ranged `1.075--1.994`. | A replicated negative calibration across the observed hardware/provider cells. It is not evidence against device-resident orchestration in general. |
| Provider coverage | placement within named provider and GPU cell | Modal L4: 2; RunPod L4: 1; Lambda H100: 1; local GTX 1660 Ti: 1. | Initial execution portability only. Provider, GPU, host CPU, driver, and CUDA version are confounded at this sample size. |

The trace outputs are immutable through
[`trace-exact-packing-manifest.json`](../data/processed/trace-exact-packing-manifest.json),
which records 540 repetition rows, 180 summary rows, input/source hashes, and
all six passed preregistered gates. The native calculation and provenance checks
are recorded in
[`native-dispatch-pilot-manifest.json`](../data/processed/native-dispatch-pilot-manifest.json)
and the frozen [`native-dispatch-001.md`](../preregistration/native-dispatch-001.md);
the five raw CSV/manifest pairs remain under
[`data/raw`](../data/raw/README.md). The statistical audit and all recomputed
denominators are in
[`statistical-design.md`](statistical-design.md).

## Why the fixed nested graph is a negative calibration, not the treatment

The host-graph path replays a pre-instantiated chain from the host. The nested
path performs that same host graph launch, executes a launcher kernel, and then
starts the same uploaded child graph from the device. Nothing in the benchmark
requires a host decision between transitions, so the nested path removes no
work and adds a launch layer. Its observed slowdown is therefore consistent
with the mechanism, not a surprising failure of GPU control.

This negative result is useful in three ways:

1. It demonstrates device-graph support and exact state evolution across three
   provider environments and two cloud GPU classes.
2. It supplies a direct warning against attributing novelty or performance to
   device launch by itself.
3. It defines the matched requirement for the next intervention: the device
   must make a decision that otherwise forces host synchronization, result
   delivery, and dispatch.

The fixed nested graph is therefore **killed as a candidate paper treatment**.
It remains in the evaluation as an overhead ablation. The pilot CPU path is
also not a headline baseline: it is single-threaded and lacks `-march=native`,
SoA/SIMD, a persistent thread pool, affinity, and NUMA control. Any GPU/CPU
speedup quoted from native dispatch 001 would be technically premature.

## The next treatment: a device-resident conditional route epoch

The next source version should receive a new experiment ID and preregistration.
Its smallest publishable mechanism is a **typed, trajectory-exact route epoch**:

1. Agent state remains in a structure-of-arrays layout on the GPU.
2. A transition kernel computes the next route/state predicate for each ready
   event.
3. A device-resident bucket/queue stage compacts compatible events and selects
   among pre-uploaded route bodies, using a conditional graph or a persistent
   dispatcher.
4. Multiple decision epochs can execute on device until an explicit model,
   tool, host-observation, deadline, or commit boundary is reached.
5. The CPU receives only typed external-effect mailboxes and ordered commit
   records; sequence numbers make the observation contract explicit.

The matched host baseline must use the **same state layout, classifier, route
bodies, compilation flags, and GPU work**. After each decision epoch it must
synchronize, copy only the information required to make the route decision,
and launch the same uploaded route body from the host. This comparison isolates
the eliminated control epoch. Ordinary kernel launch, host CUDA Graph, a
GPUOS-like persistent command ring, and an optimized CPU implementation remain
legal baselines; the strongest is frozen per stratum using pilot-only data.

### Workload and measurement contract

- Engineering shapes should bracket the measured route threshold at
  `K/2`, `K`, and `2K`, with decision horizons `1`, `8`, and `64`.
- The primary replay condition should remain the frozen exact-route
  `C=100,000`, `K=256`, `delta=50 ms` cell until pilot data justify a new
  preregistration. The runtime must additionally test recorded/bursty and
  trace-resampled arrivals; the stationary Poisson template cannot carry a
  production claim.
- Each implementation receives identical initialized state, ready-event order,
  and route seeds. Exactness is checked field by field and at sampled per-step
  observation boundaries; checksums alone are insufficient.
- Measure P50/P95/P99 event latency, valid throughput, CPU core-seconds per
  event, host synchronizations and copies, achieved accelerated share `A`,
  `(A-F)/(P*-F)` when defined, queueing/deadline misses, cold setup, steady
  state, amortized lifetime, and energy after idle subtraction.
- A useful runtime result must exceed fixed-window execution online and close
  at least half of the `P*-F` alignment opportunity in the primary engineering
  cell. This is an engineering exit gate, not yet a confirmatory population
  claim; it must be frozen before the new outcomes are exposed.
- The live-workload replication must report task-completion time, task utility,
  errors, and the fraction of end-to-end CPU/task time attributable to control.
  A fast microkernel with negligible system-level leverage is a boundary
  measurement, not a top systems result.

## Confirmatory design and statistical threshold

### Units, population, and blocking

The population unit is a **fresh GPU/server placement**, not a timing row.
Within-placement repetitions and requests estimate that placement's summary;
they do not increase deployment-level sample size. Every placement runs every
method in a balanced randomized order with common workload seeds, reset state,
frozen warm-up, and a recorded cache policy.

Use L4 and H100 as separate primary hardware strata. Freeze a provider mixture
before confirmation rather than treating provider labels as interchangeable.
The two-stratum layout below is used only if **both** strata clear the pilot
gate; otherwise the confirmation must be re-scoped and re-powered as a
card-specific study before outcomes are collected:

| Hardware stratum | Proposed fixed provider blocks | Analyzed placements | Interpretation |
|---|---|---:|---|
| L4 | Modal and RunPod, balanced 15/15 when capacity permits | 30 | Effect over the frozen 50/50 provider mixture, not all L4 deployments. |
| H100 | Modal and Lambda, balanced 15/15 when capacity permits | 30 | Effect over the frozen 50/50 provider mixture, not all H100 deployments. |

Provider is a fixed blocking factor, with a prespecified
treatment-by-provider sensitivity term. If stock makes the balanced mixture
impossible, re-freeze the target population and power **before** viewing
outcomes; do not silently substitute a provider or pool A10/L4/H100 results.
Spread placements over at least five calendar days and, where exposed, two
zones; analyze no more than six placements of one hardware class from one day.
Capture provider instance/host ID, GPU UUID and PCI ID, actual SKU, region,
driver/runtime, clocks/power limit, CPU allocation/affinity, image digest,
source revision, price timestamp, and every provisioning failure.

Current provider counts are too small for a provider effect. Even a successful
balanced 30-placement hardware analysis estimates a prespecified provider mix,
not a random population of cloud providers. A literal cross-provider
generalization claim requires separately powered named provider cells or a
larger provider sample; it should not be obtained by adding provider as a
random effect with only two levels.

### Power, endpoints, and decision rule

Run six fresh pilot placements in each intended hardware stratum to estimate
only nuisance variance. Use the upper 80% confidence limit of the paired
placement variance in 10,000 simulations of the exact randomized design. Freeze
the smallest sample with at least 90% power at one-sided `alpha=0.025`, and add
a 10% provisioning reserve. The current planning distribution has p90
placement-level log-speedup SD `0.219`; at SD `0.22`, approximately 29
placements are needed for 90% power to detect a 15% effect. This is the basis
for 30 analyzed placements per L4 and H100 stratum, not a guarantee that 30
will suffice for the new contrast.

The native claim is co-primary and succeeds only if all gates pass:

1. **Correctness:** no unexplained state, action, order, or trajectory mismatch.
2. **Performance:** adjusted lower speedup bound exceeds `1.0`, and the point
   estimate is at least `1.15` over the strongest frozen legal baseline.
3. **CPU displacement:** the adjusted upper CPU-core-seconds ratio is below
   `1.0`, and the point estimate shows at least a 25% reduction.
4. **End-to-end no harm:** task-completion-time ratio is noninferior within
   `+2%`.
5. **Task utility no harm:** the lower bound on the paired success difference
   is above `-1.0` percentage point.

Compute placement-paired log ratios, then a two-way cluster bootstrap over
placement and trace/workload seed. Report the geometric ratio, 95% confidence
interval, next-placement prediction interval, full placement distribution, and
provider-block sensitivity. Use a max-statistic bootstrap or Holm adjustment
for the two hardware-specific primary contrasts. The broad two-card statement
requires both strata to pass; if only one passes, report that card-specific
boundary rather than averaging a win and a loss. Keep the route/horizon/load
atlas exploratory and show simultaneous effect surfaces instead of hundreds of
cell-wise `p`-values.

Each measured period should contain at least 20,000 valid events or 30 minutes,
whichever is longer. P99 trials need at least 10,000, preferably 20,000,
eligible requests. Compilation failure, OOM, timeout, crash, and deadline miss
are intention-to-run outcomes, not removable timing rows. The launch ledger
must distinguish requested, provisioned, started, completed, invalidated,
retried, and analyzed placements.

## Twelve-week staged program and decision gates

| Stage | Work | Exit gate | Kill or pivot decision |
|---|---|---|---|
| **Weeks 1--2: treatment and legal baselines** | Implement the conditional route epoch, optimized C++17/SoA/SIMD/OpenMP CPU engine, matched host graph, and persistent command-ring baseline. Add per-step exactness, host-sync/copy counters, cold-start accounting, and immutable build artifacts. | All implementations compile on L4/H100, pass field and trajectory equality, and reproduce state from a clean artifact. | Any unexplained mismatch kills that source version. If the only difference is an extra nested launch, retain it solely as the negative ablation. |
| **Weeks 3--4: six-placement nuisance pilots** | Run six fresh placements per intended L4/H100 stratum in balanced method order. Profile with Nsight; estimate placement variance, failure rate, and P99 density without performing confirmatory tests. | Every stratum retained for confirmation improves either P99 or CPU core-seconds by 10% over the tuned host graph, with zero unexplained mismatch. Freeze the treatment, baseline, primary cell, and power simulation. | If neither stratum clears 10%, stop building a general graph compiler. If only one clears, proceed only with a newly preregistered card-specific study; do not retain a broad two-card claim. |
| **Weeks 5--6: finite online ready-cohort runtime** | Add deadline queues, route-specific `K`, CPU fallback, bounded service, deterministic commits, and replay against exact `P*`. Replace stationary-only arrivals with trace-resampled, bursty, and correlated sensitivities. | In the primary engineering cell, online `A>F`, at least 50% of `P*-F` is recovered, deadlines/correctness pass, and failure regimes are mapped. | If `A` cannot beat `F` or closes less than half the opportunity after profiled tuning, the runtime contribution is weak; pivot to a measurement/benchmark paper instead of hiding the gap. |
| **Weeks 7--8: live workload and shared-GPU pilot** | Integrate a tool-using agent replay and a coding-agent workload. Co-run one frozen vLLM configuration with low-priority control work; compare CPU control, separate L4, and shared H100. | Control work has measurable system leverage; live utility passes; a pilot rate does not regress P99 TTFT or TPOT by more than 10%. | If control is negligible end to end, drop the end-to-end speedup claim. Stop any shared rate after OOM or sustained >10% P99 regression; keep a negative interference boundary. |
| **Weeks 9--10: confirmatory placements** | Run the frozen 30-per-hardware design, balanced across provider blocks/days, with common seeds and full launch ledger. No efficacy peeking or post hoc cell substitution. | Both hardware strata pass the intersection of correctness, performance, CPU, completion-time, and utility gates, or the evidence yields a clearly bounded card-specific/negative result. | A wide interval is inconclusive, not equivalence. Practical equivalence requires the entire performance interval to lie within the frozen equivalence band. Do not increase repetitions to compensate for too few placements. |
| **Week 11: robustness and artifact audit** | Run frozen burstiness, route-fragmentation, host-observation, branch-entropy, shape-churn, provider, energy, and shared-LLM sensitivity panels. Repeat the novelty search and have an independent reader audit claims. | Every headline regenerates from raw ledgers; primary and exploratory results are visually and textually separated; negative cells remain. | If a new paper occupies the joined boundary-plus-runtime claim, reposition around the open benchmark or measured residual gap; never preserve priority language by narrowing definitions after the fact. |
| **Week 12: paper and release** | Freeze figures/tables, publish source, configs, raw/processed ledgers, derived content-free traces, schemas, preregistrations, bounded-cost reproduction, and a clean-environment artifact check. | The paper can be audited from placement counts through exact schedules and end-to-end results without private prompts or hidden exclusions. | If the runtime has not cleared the gates, release a rigorous measurement preprint; do not write the unsupported system into the abstract. |

The Modal ten-GPU limit is a concurrency constraint, not a reason to reduce the
independent placement count. Run fresh placements in bounded waves and spread
them across days. Lambda and RunPod are scientifically valuable as fixed
provider blocks and external replications; they must not be used merely to
inflate a pooled `n`.

## Frozen kill criteria

The following decisions should be treated as portfolio discipline, not as
failures to produce a positive result:

- **Semantic validity:** stop and version any runtime with an unexplained
  trajectory, action, ordering, or commit mismatch. Never exclude its failed
  launch.
- **Fixed nested graph:** killed now as the treatment; retain only as an
  overhead ablation.
- **Conditional runtime futility:** after six placements per intended primary
  stratum, stop the graph-compiler path if no stratum improves P99 or CPU
  core-seconds by at least 10% over the strongest tuned host graph. A stratum
  that misses the gate cannot enter confirmation under the same treatment ID.
- **Scheduling futility:** if finite online execution cannot beat `F` and close
  at least half of `P*-F` in the frozen primary engineering cell, publish the
  offline/online gap instead of calling the scheduler near-optimal.
- **System-level futility:** if the runtime speeds a microkernel but cannot
  materially reduce CPU use or preserve end-to-end completion/utility, the
  result is a boundary/benchmark paper, not an agent-runtime performance claim.
- **Shared-GPU safety:** stop a pilot rate after OOM or sustained P99 regression
  above 10%. A confirmatory shared-GPU claim later requires one-sided upper
  bounds below `1.02` for both TTFT and TPOT, SLO-hit loss above `-0.5` pp, and
  a frozen useful-work benefit.
- **Provider availability:** capacity failure is recorded. It does not license
  an outcome-dependent replacement GPU, provider, region, or time window.
- **Novelty collision:** repeat the primary-source search before submission.
  If the joined ready-cohort boundary and exact-runtime evaluation is occupied,
  remove priority language and pivot to the open measurement artifact or the
  independently gated diversity--regularity lane.
- **GPU-created VMs/DPU lane:** exclude it from this flagship unless bare-metal
  ConnectX/BlueField access and a capability-enforcing CPU/DPU authority path
  are verified. A CUDA kernel does not directly create a secure VM.

## Citation-safe claim ladder

### Safe now, with the qualifier kept in the same sentence

- “Under a frozen equal-relative-deadline, zero-service, unlimited-capacity
  replay model, the exact offline compatible-event share lies between
  fixed-window eligibility and a local upper bound.”
- “In the preregistered primary replay cell, mean fixed-window, exact-optimal,
  and local-upper shares were 30.19%, 43.00%, and 45.85% across three Monte
  Carlo seeds conditional on one fixed 851-session panel.”
- “On five observed placements spanning local, Modal, RunPod, and Lambda
  environments, all 12,000 native-pilot rows matched every checked final-state
  field.”
- “For the fixed nested-launch calibration, the nested path was slower than the
  host graph in all 60 observed placement-by-shape cells.”
- “The evidence identifies regimes where resident regular GPU execution wins
  and regimes where host visibility, sparse cohorts, route fragmentation, or
  launch structure erase that advantage.”

### Target claims that require the new powered experiment

- “A device-resident route runtime reduces deployment-level P99 control latency
  and CPU core-seconds over the strongest tuned legal baseline.”
- “The online runtime converts a material fraction of the trace-conditioned
  offline opportunity into deadline-valid work.”
- “The runtime improves or preserves end-to-end agent task completion and task
  utility.”
- “The result holds in the prespecified L4 and H100 provider mixtures.”
- “Low-priority control work can harvest same-GPU inference slack under frozen
  TTFT/TPOT/SLO noninferiority margins.”

### Prohibited without materially new evidence

- “GPUs are cheaper CPU cores for agent swarms.”
- “This is the first GPU runtime, GPU agent state machine, persistent GPU task
  queue, device-launched graph, or agent-aware scheduler.”
- “The current nested device graph accelerates orchestration.”
- “The current CPU reference is hardware-optimal.”
- “The Poisson-template replay represents production arrivals.”
- “The measured crossover or `K=256` is universal.”
- “Zero observed mismatches establishes a field failure rate of zero.”
- “The result generalizes across GPUs or cloud providers.”
- “A GPU kernel directly spawns secure VMs.”

The abstract should prefer “we characterize,” “under the evaluated model,” and
“on the prespecified hardware/provider population” over priority or universal
language. If the powered runtime misses its gates, the exact boundary, negative
nested-launch result, open trace artifact, and failure-regime atlas remain
publishable scientific contributions.

## Open questions that can still change the decision

1. Does a true device-side route epoch remove enough host synchronization and
   CPU work to clear the 10% pilot gate once the CPU baseline is optimized?
2. How much of the exact `P*-F` opportunity survives finite service time,
   route-specific kernels, queue contention, and non-Poisson arrivals?
3. Is deterministic control a material fraction of CPU or task time in live
   tool/coding agents, or only a fast but economically negligible microkernel?
4. Can the runtime maintain exact observation/commit semantics across external
   tool and model boundaries, failure recovery, and host fallback?
5. Is useful control work available on an already allocated inference GPU
   while both P99 TTFT and TPOT remain within a 2% noninferiority margin?
6. Are provider interactions small enough for a prespecified mixed-provider
   estimand to be useful, or must conclusions remain provider-card specific?

Those questions define the next twelve weeks. More fixed nested-graph timing
does not.
