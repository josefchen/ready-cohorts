# Parallel research frontiers

Last reviewed: 2026-08-11.

This document separates ideas that can strengthen the ready-cohort paper from
ideas that deserve an independent paper. The ranking rewards a sharp named
object, a falsifiable boundary, a reusable artifact, and a result that remains
interesting when the GPU loses.

## Executive decision

Run four lanes, but do not build four full systems at once:

1. **Flagship systems lane:** a device-resident agent graph engine that tests
   the measured readiness × regularity × residency boundary.
2. **Joint algorithms/systems lane:** the diversity–regularity frontier: model
   diversity reduces co-failure but fragments GPU residency, batching, and
   cache reuse.
3. **Shared-infrastructure lane:** harvest otherwise idle LLM-GPU capacity for
   deterministic agent control without violating inference SLOs.
4. **High-risk reliability lane:** fleet-scale escalation storms, where many
   agents demand a strong backup policy at the same time.

Keep a fifth, **GPU-signalled sandbox fabric**, behind a strict hardware gate.
A GPU cannot directly create a Linux VM: VM creation and isolation require a
privileged CPU or DPU executor. The researchable design is for the GPU to make
and transmit the decision while an authority processor performs the lifecycle
operation.

## Ranked portfolio

| Rank | Research object | Novelty potential | First real data | Main dependency | Intended outcome |
|---|---|---:|---:|---|---|
| 1 | Device-resident agent graph engine | High | 2–4 days | CUDA C++ and Modal GPUs | Main systems paper |
| 2 | Diversity–regularity frontier | Very high | 3–7 days | Existing co-failure panel plus open-model serving | Independent MLSys/AI-systems paper or flagship section |
| 3 | Same-GPU slack harvesting | High | 3–5 days | vLLM and stream-priority experiments | Strong paper section; separate paper if the regime is broad |
| 4 | Co-escalation ceiling for agent/robot fleets | Very high | 1–2 weeks | AEGIS-style risk traces and parallel simulation | Independent robotics/systems paper |
| 5 | GPU fork/rollback for pure agent continuations | High | 3–6 days | Device-resident state and copy-on-write | Reliability/runtime paper |
| 6 | GPU-signalled CPU/DPU sandbox fabric | Very high | Hardware gate first | Root, ConnectX/BlueField, GPUDirect/DOCA | OSDI/NSDI-style moonshot |
| 7 | GPU virtual-agent bytecode machine | Medium–high | 1–2 weeks | CUDA interpreter/compiler work | PL/systems paper if semantics are compelling |
| 8 | GPU parallel discrete-event simulator | Medium as a paper; very high as an enabler | 1–2 days | CUDA/Triton | Artifact and policy-search engine |

## 1. Device-resident agent graph engine

### Question

Can deterministic regions of an agent workflow execute as one long-lived GPU
program, with the CPU handling only asynchronous external effects and sparse
fallbacks?

The implementable unit is not an entire autonomous agent and not a generic
Python process. It is a typed, side-effect-free region containing state
updates, routing, bookkeeping, policy checks, JSON/schema state, and commit
logic. Tool calls and model calls cross explicit mailbox boundaries.

### Mechanisms to compare

- host-launched CUDA kernels;
- host-launched CUDA Graph replay;
- CUDA Graph conditional `IF`, `SWITCH`, and `WHILE` nodes;
- device-launched graphs;
- a persistent GPU worker with route queues;
- CPU fallback for underfilled, divergent, or deadline-critical cohorts.

CUDA Graphs already support conditional nodes and device-side graph launch, so
the contribution cannot be “graphs can express control flow.” The contribution
must be the agent semantics, the measured feasibility boundary, deadline-aware
route formation, exact commit behavior, and evidence that the runtime approaches
the trace-conditioned bound. See the
[CUDA Graph programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
and [GPUOS](https://arxiv.org/abs/2604.17861).

### First falsification experiment

Implement a three-route finite-state transition with `H={1,8,64}`, population
`N={8,...,4096}`, and resident state. Compare pinned C++ CPU, ordinary CUDA,
host graph replay, conditional graph replay, and persistent dispatch on T4, L4,
H100, and the local GTX 1660 Ti.

Measure dispatch latency, P50/P95/P99 transition latency, CPU time, launch and
synchronization counts, achieved occupancy, energy per valid transition, and
exact trajectory equality. Randomize implementation order and use a fresh
container as the hardware replication unit.

**Kill criterion:** if conditional/device dispatch does not improve P99 or CPU
consumption over a tuned host graph in any preregistered ready-cohort regime,
retain it as a negative ablation and stop building a general graph compiler.

### High-value sub-angle: device-resident tool-call front end

Keep token IDs, a small DFA/schema validator, route selection, and the action
mailbox on device. Measure last-token-to-tool-dispatch latency and host CPU
interrupts. This is narrower and more defensible than claiming that the whole
agent runs on a GPU.

## 2. The diversity–regularity frontier

### Question

The co-failure paper shows why heterogeneous models can be epistemically useful:
they may fail on different examples. Systems work shows why homogeneous calls
are operationally useful: they share model weights, prefixes, KV state, kernels,
and route queues. What is the attainable quality–latency–cost surface when
those forces oppose one another?

For a selected model set `S`, retain the co-failure quantity

```text
beta(S) = P(all models in S are wrong).
```

Any selection-only combiner has accuracy at most `1 - beta(S)`. Independently,
the ready-cohort analysis gives an operational feasible share bounded by the
deadline/grouping quantities `F <= P* <= U`. Do not multiply these bounds
without additional assumptions. Instead, measure a joint Pareto frontier over

```text
(1 - beta(S), task utility, task-completion time, GPU-seconds, dollars,
 model swaps, prefix/KV reuse, and route-ready share).
```

### Core hypothesis

The ensemble with the lowest co-failure is often not the best ensemble under a
latency or GPU-memory budget. Conversely, the most batchable homogeneous swarm
often has a worse all-wrong tail. A joint selector using both error
complementarity and systems compatibility should dominate quality-only routing
and throughput-only scheduling.

### Experiment

1. Reuse the frozen 67-model correctness panel to enumerate or optimize model
   subsets offline.
2. Choose a preregistered 8–16-model open-weight subset spanning families and
   sizes.
3. Serve homogeneous self-samples and heterogeneous ensembles on identical
   GPU budgets.
4. Compare best-single-model, Self-MoA, minimum-`beta`, minimum pairwise
   correlation, model-family diversity, systems-only affinity, and the joint
   optimizer.
5. Repeat on execution-graded code and open-ended mathematics so selection
   utility is objectively scored.

Measure not only aggregate accuracy but per-query oracle gain, all-wrong tail,
model-residency churn, cache reuse, batch size, queue delay, task-completion
time, and cost. [Chimera](https://arxiv.org/abs/2603.22206) already jointly
considers model performance and latency, while
[SAGA](https://arxiv.org/abs/2605.00528) is workflow-aware; our distinguishing
object must be the measured conflict between **joint error complementarity**
and **hardware regularity**, with co-failure certificates rather than a generic
quality predictor.

**Kill criterion:** if minimum-`beta` subsets incur no material scheduling,
residency, or cache penalty at matched hardware and load, publish the negative
result inside the co-failure extension rather than constructing a new runtime.

## 3. Same-GPU slack harvesting

### Question

Can an already-paid-for inference GPU execute agent control during decode,
communication, or tool-wait gaps, avoiding both extra CPU cores and a separate
control GPU?

### Experiment

Co-run vLLM with the resident transition engine. Sweep model size, prefill/decode
mix, sequence length, request concurrency, control arrival burstiness, and
control-kernel horizon. Compare:

- CPU-only control;
- a separate cheap T4/L4 control GPU;
- a shared L4/H100 with ordinary streams;
- shared GPU with low-priority streams;
- MPS where the provider exposes it;
- MIG on supported bare metal.

Primary metrics are inference TTFT, TPOT, P99 latency, SLO attainment, agent
task-completion time, control queue delay, GPU SM/memory utilization, CPU cores
displaced, energy, and marginal cost. The economic headline is incremental
cost on a GPU that was already allocated, not raw kernel throughput.

**Kill criterion:** stop the shared-GPU design if it cannot recover useful
control work while keeping the preregistered inference SLO guardrail. A
separate cheap GPU may still win and is a scientifically useful boundary.

This is adjacent to inference co-scheduling systems such as
[OmniServe](https://arxiv.org/abs/2603.12831), so the paper must center agent
control work, exact trajectories, and the ready-cohort predictor rather than a
generic utilization claim.

## 4. Co-escalation ceiling for fleets

### Question

Selective escalation looks efficient for one embodied agent, but what happens
when a fleet encounters a common shock and many agents request the strong
policy simultaneously?

Average escalation rate is not enough. Let `A_t` be the number of agents whose
risk gate requests backup service before deadline `t`, and let `m` be the
number the strong-policy pool can serve. The operational safety quantity is the
tail

```text
P(A_t > m),
```

not merely `E[A_t]`. Correlated observations, shared weather, identical scene
types, software regressions, or coordinated tasks can make this tail much
heavier than an independence model predicts.

### Experiment

- Extract time-to-failure and risk-score trajectories from AEGIS-style runs.
- Construct common-random-number fleets with controllable trigger dependence.
- Replay 10–10,000 simultaneous robots against a bounded strong-policy GPU
  pool.
- Compare late escalation, early-warning admission, earliest-deadline-first,
  risk-only priority, diversity-aware admission, random admission, and extra
  reserved capacity.
- Confirm selected cells with real strong-policy inference, not simulation
  alone.

Measure missed-rescue probability, recovered-task rate, overload probability,
warning time, queue delay, fairness, GPU reserve, energy, and cost. The intended
named object is a **co-escalation ceiling** or **escalation-storm boundary**.
It combines the early-warning logic of
[AEGIS](https://arxiv.org/abs/2606.06660) with the tail-dependence discipline of
the [co-failure ceiling](https://arxiv.org/abs/2606.27288).

**Kill criterion:** if realistic common shocks do not move overload or rescue
rates beyond an independence-matched placebo, keep this as a capacity-planning
appendix rather than a separate paper.

## 5. GPU fork/rollback for pure continuations

### Question

While an agent waits for a typed tool result, can the GPU precompute all likely
side-effect-free continuations, then commit the matching state when the result
arrives?

This is not speculative tool execution. Never duplicate a side effect. Fork
only pure state-transition branches, assign versioned state pages, and commit
exactly one branch after the real result is validated.

### Experiment

Benchmark branch factors 2–64, state sizes 64 B–1 MB, prediction entropy,
arrival rates, and continuation depths. Compare sequential CPU/GPU execution,
most-likely-only speculation, all-branch GPU speculation, and copy-on-write
branching. Report commit latency, end-to-end tool-return-to-next-action latency,
wasted GPU work, memory amplification, exactness, and break-even probability.

The design must distinguish itself from speculative **tool** execution such as
[PASTE](https://arxiv.org/abs/2603.18897) and from model-serving checkpointing
such as [Concordia](https://arxiv.org/abs/2606.23521).

**Kill criterion:** if host notification and commit synchronization dominate
the saved transition time across realistic states, retain the result as a
negative limit on speculation.

## 6. GPU-signalled CPU/DPU sandbox fabric

### What is and is not possible

A CUDA kernel cannot issue privileged KVM operations or securely create a
microVM by itself. A real design has three roles:

```text
GPU policy/state engine
        |
        | command ring / doorbell
        v
CPU or BlueField authority executor ----> Firecracker/container/tool sandbox
        ^                                         |
        | result descriptor / DMA                 |
        +------------------------------------------+
```

The strongest version removes the host CPU from the common data path, not from
the authority path. NVIDIA's
[DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca-gpunetio/) supports
GPU-controlled network send/receive after CPU-side setup. Lambda now advertises
[bare-metal instances](https://lambda.ai/blog/lambda-bare-metal-instances)
whose lifecycle is implemented with BlueField and BMC infrastructure, which
validates the architectural direction but does not imply that a tenant can
program Lambda's zero-trust DPU.

### Three-stage feasibility ladder

1. **Host executor:** GPU writes sandbox commands into mapped pinned memory;
   a dedicated CPU thread manages a preallocated Firecracker/container pool.
2. **NIC data path:** tool results enter GPU memory through GPUDirect while the
   host retains lifecycle authority.
3. **DPU executor:** a programmable BlueField handles network/isolation and
   privileged lifecycle actions; the host is the slow-path fallback.

Do not pitch generic speculative sandbox prewarming. That space is already
occupied by [SpecBox](https://arxiv.org/abs/2607.23933). The defensible new
combination is GPU-resident readiness/route state, authority-separated sandbox
lifecycle, and CPU-bypass result delivery.

### Hardware gate

Proceed past stage 1 only after verifying root access, IOMMU/Resizable BAR,
ConnectX or BlueField model, GPUDirect RDMA, DOCA compatibility, and permission
to access the relevant devices. Modal is excellent for ordinary CUDA and
multi-GPU experiments but its managed container environment does not expose
host kernel modules. Lambda bare metal or a dedicated BlueField server is the
credible target.

**Kill criterion:** no programmable DPU/NIC access means no three-month
full-system claim. Publish the host-executor result only if GPU signalling and
direct result ingress yield a robust latency or CPU-core benefit over a tuned
CPU event loop.

## 7. GPU virtual-agent bytecode machine

Interpret thousands to millions of restricted agent programs as small virtual
machines on the GPU. Each virtual agent owns a program counter, typed registers,
state pages, deadline, and mailbox. Regroup runnable agents by program counter
before executing the next bytecode block, turning divergent programs into
temporarily regular cohorts.

This is not a Linux VM and should never be described as one. It is closer to a
Wasm-like actor machine. The novelty burden is high because
[FLAME GPU 2](https://doi.org/10.1002/spe.3207) already executes state-machine
agents on GPUs. The defensible additions would be asynchronous external I/O,
per-agent capability isolation, exact effect/commit semantics, deadline-aware
PC bucketing, and evidence from real LLM-agent traces.

**Kill criterion:** if PC regrouping, interpretation, and mailbox traffic lose
to compiled route kernels across the trace-derived route distribution, stop at
the compiler result; do not build a general VM.

## 8. GPU parallel discrete-event simulator

Build this even if it never becomes a paper. It can evaluate millions of
counterfactual scheduling policies across arrival processes, route entropy,
deadlines, hardware crossover `K`, failures, preemptions, and shared-GPU
contention. Use it to discover candidate regimes, then confirm a frozen subset
on real systems.

The simulator must not be the sole evidence for a systems claim. Validate every
timing distribution against raw measurements, preserve common random numbers
across policies, and keep an explicit simulation-to-system gap table.

## 72-hour decision sprint

### Day 1: cheapest falsifiers

- Implement and locally validate the CUDA Graph conditional/device-launch
  microbenchmark.
- Run an offline diversity–regularity analysis using the existing 67-model
  correctness panel plus measured/provider latency and residency constraints.
- Add a GPU fork/rollback synthetic benchmark with exact commit checks.

### Day 2: bounded cloud measurements

- Run the graph microbenchmark on fresh T4, L4, and exact H100 placements.
- Start a single-GPU vLLM interference sweep on L4 and H100 with a strict cost
  cap and an SLO abort condition.
- Measure a local host-executor sandbox command ring without claiming DPU or
  bare-metal results.

### Day 3: go/no-go

- Freeze one preregistration per surviving lane.
- Promote at most two lanes to full implementation.
- Treat the other lanes as short probes until they clear their kill criteria.

The recommended promotion order is: device-resident graph engine first,
diversity–regularity second, slack harvesting third. Run the co-escalation
analysis in parallel only if the AEGIS risk trajectories are immediately
available. Keep the DPU design gated until the hardware is real.

## Publication strategy

The current paper should remain **The Ready-Cohort Boundary**. It can absorb
the graph engine and slack-harvesting result because both directly test its
boundary. Do not overload it with model-ensemble quality or robot-fleet safety.

The two strongest independent-paper candidates are:

1. **The Diversity–Regularity Frontier:** when error complementarity fights
   hardware efficiency in multi-model agents.
2. **When Every Agent Escalates at Once:** a co-escalation ceiling for bounded
   backup inference in embodied fleets.

This structure matches the strongest pattern in the existing Josef Chen work:
a named ceiling or resource boundary, a simple certificate or economic object,
large controlled evidence, explicit placebos and kill criteria, and an artifact
that remains useful beyond the paper's preferred mechanism.
