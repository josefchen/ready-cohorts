# Twelve-week research program

Objective: produce one coherent, reusable systems artifact rather than a pile
of disconnected GPU benchmarks. Citation potential comes from a named boundary,
an open workload/replay artifact, a useful runtime, and results that others can
reuse—not from claiming a guaranteed citation count.

## Flagship question

When can a GPU execute the non-neural AI-agent control plane more efficiently
than CPU cores, once readiness, route heterogeneity, state residency, launch
overhead, compilation, correctness, queueing, cost, and interference with LLM
serving are included?

## Execution checkpoint: foundation completed

The initial campaign now includes the exact trace opportunity bound and a
[four-placement native mechanism result](resident-policy-001-report-2026-08-12.md).
Under the frozen primary replay,
`F=0.3019`, `P*=0.4300`, and `U=0.4585`; the exact offline schedule closes
81.8% of the fixed-window alignment gap under the stated zero-service model.
In `resident-policy-001`, the device-resident route decision beat the matched
host round trip in all 36 observed placement-cells across local GTX 1660 Ti,
Modal L4, RunPod L4, and Lambda H100, with zero observed correctness failures.
That result qualifies the mechanism but does not measure invocation tails,
CPU displacement, online accelerated share, or a placement population.

Decision: do not scale the mechanism microbenchmark directly. First run a
bounded instrumentation qualification, then spend placement-scale compute on
the first finite-capacity route-compacting runtime that measures achieved
online share `A` against `F` and `P*`.

## Weeks 1–2: measurement foundation (completed)

- Freeze and run the compiler-matched crossover atlas.
- Refine the regular crossover below 256 agents.
- Sweep T4, A10, L4, L40S, A100-80GB, H100, and a consumer GTX.
- Pin public agent traces and release content-free derived event features.
- Measure fixed-window eligibility, then implement and test the general
  sliding-deadline packing optimum and local upper bound.
- Preserve branch-threshold correctness failures as a first-class result.

Exit gate: every figure regenerates from raw ledgers; hardware manifests include
CPU affinity/governor and GPU metadata; claims ledger distinguishes pilot from
confirmatory evidence.

## Weeks 3–4: qualify measurement and eliminate benchmark objections

- Replace the blocked `resident-policy-002` draft with a bounded qualification:
  fixed scheduled attempt IDs, no replacement of failures, fixed microblocks,
  raw empirical tails, block-level process/cgroup CPU counters, an explicit
  CUDA wait policy, and a mechanical intention-to-run reliability endpoint.
- Keep this stage compact; it qualifies instrumentation and baselines rather
  than testing a placement-population efficacy claim.

- Implement a deterministic integer/fixed-point transition and a controlled
  margin-to-branch family.
- Implement optimized CPU baselines: C++17, OpenMP, explicit SoA layout, SIMD,
  thread pinning, NUMA checks, and compiler-vectorization reports.
- Implement custom CUDA, CUDA Graph, and Triton variants; profile with Nsight
  Systems/Compute.
- Repeat selected cells on at least two hosts per GPU class and on Modal,
  Runpod, and Lambda where equivalent cards are available.
- Measure cold/warm compile cache, shape churn, allocation, transfer, launch,
  and host delivery separately.
- Replace a single stationary Poisson replay with recorded/bursty/Hawkes-style
  and trace-resampled arrival sensitivity analyses.

Exit gate: no headline relies on PyTorch-versus-PyTorch alone; crossover effects
replicate at the host/card level; exactness is per-step or mathematically
guaranteed.

## Weeks 5–7: build the online runtime intervention

Build a small “ready-cohort runtime” with:

- structure-of-arrays agent state retained on device;
- per-route/state bucket queues;
- a latency deadline and minimum profitable cohort from measured `K`;
- CPU fallback for sparse, branchy, or near-deadline events;
- CUDA Graph cache for stable routes;
- the verified device-resident route-selection backend;
- a command-ring or persistent-kernel path only as a named ablation if it
  clears the collision audit and beats the simpler backend;
- sequence numbers and deterministic commit order;
- optional multi-step temporal fusion with explicit observation points;
- OpenTelemetry instrumentation and a replayable decision log.

The primary runtime quantities are release-to-launch eligibility and achieved
accelerated share `A`, not an unrelated invocation-completion deadline. Measure
route/hardware/horizon-specific crossover `K`, recompute `P*` for those frozen
thresholds and permissible observation-free horizons, and report `A-F` and
`(A-F)/(P*-F)` wherever defined.

Ablations must isolate bucketing, residency, graph replay, persistent dispatch,
deadline policy, and fusion. Add equal-wait and equal-memory placebos.

Exit gate: the runtime processes real route/state events, preserves exact
semantics, and exposes achieved eligible share versus the offline packing
optimum and local-eligibility upper bound.

## Weeks 8–9: end-to-end and shared-GPU experiments

- Integrate at least two agent runtimes or trace adapters (for example,
  smolagents/tool-calling and a coding-agent trace player).
- Replay tau2, BFCL-style tool use, and a code-agent workload at controlled
  concurrency and burstiness.
- Measure task completion time, control-plane P50/P95/P99, throughput, queueing,
  CPU utilization/cores displaced, GPU utilization, and task utility.
- Co-run the control engine with vLLM on the same GPU using stream priorities;
  sweep model load and enforce TTFT/TPOT/P99 SLOs.
- Compare separate control GPU, shared LLM GPU, and CPU-only control.

Unique high-value angle: **slack harvesting**. Test whether deterministic
control transitions can occupy decode/communication gaps on an already-paid-for
LLM GPU without violating serving SLOs. This is stronger economically than
renting a separate flagship GPU merely to replace cheap CPU cores.

Exit gate: at least one real workload shows a reproducible end-to-end win or a
clear negative boundary; shared-GPU interference is quantified rather than
hand-waved.

## Weeks 10–11: confirmatory freeze

- Choose primary hardware/workload cells using pilot evidence, then freeze a
  confirmatory preregistration.
- Repeat complete matrices with randomized order and independent host/card
  replicates.
- Sample NVML power and host power when available; subtract idle baseline and
  report energy per valid transition/task.
- Freeze provider prices and report cost sensitivity rather than one timeless
  dollar number.
- Test failures: route explosion, tiny populations, high branch entropy,
  frequent host observation, shape churn, tool bursts, priority inversion,
  GPU contention, and correctness margins.
- Have a second person rerun the novelty search and artifact instructions.

Exit gate: all primary statistical families, exclusions, and plots are frozen;
no post-hoc cell selection enters the abstract.

## Week 12: paper and release

- Write from the claims ledger and regenerate every table/figure in a clean
  environment.
- Release raw ledgers, manifests, processed tables, replay generator, runtime,
  configs, and an artifact-evaluation script.
- Add a one-command smoke test and a bounded-cost reproduction path.
- Publish a dataset card and machine-readable schema so other systems papers can
  use the ready-cohort workload without copying prompts/tool payloads.
- Prepare both the systems-paper version and a compact arXiv artifact report;
  do not delay the open artifact for an uncertain venue cycle.

## Secondary paper angles, ranked

1. **Ready-cohort boundary + runtime** — flagship; strongest conceptual and
   systems contribution.
2. **GPU slack harvesting for agent control** — potentially a separate short
   paper if shared-LLM interference results are surprising and robust.
3. **Trajectory amplification across CPU/GPU branch thresholds** — worthwhile
   correctness paper only if it generalizes across kernels/frameworks and leads
   to a usable exact/fixed-point method.
4. **GPU choice inversion for launch-dominated control** — measurement note if
   cheap cards consistently beat flagship cost/performance across providers.
5. **Open orchestration readiness benchmark** — dataset/benchmark paper if the
   trace panel expands beyond one public source and includes real arrival timing.

Do not split early. Run the experiments so these angles remain separable, then
let the evidence determine whether one flagship paper or multiple artifacts are
scientifically justified.

## Parallel frontier lanes

The broader portfolio, including the diversity–regularity frontier,
co-escalation at fleet scale, GPU fork/rollback, device-resident tool-call
front ends, and a hardware-gated GPU/DPU sandbox design, is ranked in
[`parallel-frontiers.md`](parallel-frontiers.md). Only the device-resident graph
engine and shared-GPU slack experiment belong in the flagship by default.
Adjacent lanes must clear their own preregistered kill criteria before consuming
a full implementation month.
