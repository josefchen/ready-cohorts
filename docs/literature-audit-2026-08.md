# Collision-first literature audit: GPU systems for agent workloads

Audit date: **2026-08-11**
Scope: primary sources published or posted from 2024 through 2026-08-11,
plus older priority-setting anchors that invalidate broad novelty claims. Sources
were limited to arXiv/publisher pages, conference pages, official documentation,
and official repositories. For the closest papers, the abstract and available
mechanism/design material were read; this is not a search-snippet survey.

This audit is deliberately adversarial. Its purpose is to identify claims that
would fail a knowledgeable systems review, not to maximize the apparent size of
the gap. A second author should repeat the negative search and backward/forward
citation chaining before any paper says “first.”

## Bottom line

Two directions survive as credible paper centers:

1. **Ready-cohort feasibility boundary plus an exact resident implementation.**
   The novelty is not a persistent kernel, GPU state machine, device-launched
   graph, task ring, or generic agent scheduler. It is the joined measurement of
   (a) the per-route CPU/GPU crossover and (b) whether interrupted real-agent
   traces form enough semantically compatible cohorts before their deadlines,
   followed by a trajectory-exact runtime that approaches that trace-conditioned
   limit.
2. **The co-failure–regularity frontier for simultaneous heterogeneous
   ensembles.** Existing systems optimize model routing, placement, quality,
   latency, or cost, and existing ensemble work optimizes prediction. I found no
   source that jointly measures an ensemble's all-wrong probability and the
   systems penalty caused by simultaneously resident, error-complementary model
   sets: weight churn, batching fragmentation, KV/prefix reuse, and GPU-seconds.

Three directions remain conditional: GPU/DPU authority-split orchestration is a
hardware-gated moonshot; correlated escalation storms may support a separate
reliability paper only if they outperform an independence-matched explanation;
and same-GPU slack harvesting is useful as a flagship section but is too crowded
to lead a paper on its own.

Generic “GPU agents,” persistent work queues, GPU bytecode interpreters,
speculative tool/sandbox execution, serverless GPU prewarming, and GPU-created
VMs do **not** survive as broad novelty claims.

## Compact collision matrix

| Proposed direction | Nearest primary-source mechanisms | Exact collision | Defensible residual gap | Verdict |
|---|---|---|---|---|
| GPU-resident agent state/control machine | [FLAME GPU 2](https://doi.org/10.1002/spe.3207); [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html); [GPUOS](https://arxiv.org/abs/2604.17861); [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219); [Event Tensor](https://arxiv.org/abs/2604.13327); [Fleet](https://arxiv.org/abs/2604.15379); [Ada-MK](https://arxiv.org/abs/2605.11581); [DVM](https://arxiv.org/abs/2603.24239) | FLAME GPU already maps communicating agent state machines and state partitions to GPUs. CUDA Graphs already provide replay, conditional nodes, and device launch. The 2025–26 megakernel/runtime papers already provide resident queues, in-kernel task graphs, decentralized dispatch, dynamic dependencies, and operator injection/interpreters. | Interrupt-driven LLM agents are only intermittently ready and routes are semantically heterogeneous. Join hardware crossover with trace-conditioned ready-cohort supply; preserve exact trajectories across external model/tool boundaries. | **Survives only in narrow form; strongest flagship.** Never claim the state machine or persistent queue as the invention. |
| Compound-agent/workflow scheduling | [Parrot](https://arxiv.org/abs/2405.19888); [Agentix](https://www.usenix.org/conference/nsdi26/presentation/luo); [SAGA](https://arxiv.org/abs/2605.00528); [MARS](https://arxiv.org/abs/2604.26963); [SwarmX](https://arxiv.org/abs/2606.21401); [OpRAG](https://arxiv.org/abs/2608.08340); [Agentic CPU–GPU Scheduling](https://arxiv.org/abs/2607.22242); [AgentServe](https://arxiv.org/abs/2603.10342) | Program/workflow-aware priority, preemption, KV retention, admission, fairness, session affinity, tool placement, telemetry-driven CPU/GPU placement, and learned scaling are occupied. AgentServe also isolates concurrent agents on consumer GPUs using CUDA Green Contexts. | The internal deterministic transition between LLM/tool events is a different scheduling unit from a whole call, tool, or workflow. It matters only if measured CPU/latency cost and trace-ready supply are material end to end. | **Plane distinction survives; generic “agent scheduler” does not.** These are mandatory system-level baselines/neighbors. |
| Heterogeneous multi-model serving and ensembles | [MuxServe](https://arxiv.org/abs/2404.02015); [Mélange](https://arxiv.org/abs/2404.14527); [MixLLM](https://arxiv.org/abs/2502.18482); [BOute](https://arxiv.org/abs/2602.10729); [Chimera](https://arxiv.org/abs/2603.22206); [Prism](https://www.usenix.org/conference/osdi26/presentation/yu-shan); [DeePEn](https://arxiv.org/abs/2404.12715); [co-failure ceiling](https://arxiv.org/abs/2606.27288); [diversity-metric audit](https://arxiv.org/abs/2607.20768) | Multi-model placement, query routing, load/cost/quality optimization, memory elasticity, and probabilistic ensemble fusion already exist. A generic heterogeneous-model scheduler or diversity heuristic is not new. | Jointly optimize or characterize **simultaneous** ensemble co-failure/all-wrong risk and measured systems regularity. Route-one-model systems do not answer when epistemically complementary models must co-reside or execute together. | **Strongest independent lane**, provided it uses all-wrong certificates plus real serving costs, not another quality predictor. |
| Speculative actions, tools, or continuations | [Speculative Actions](https://arxiv.org/abs/2510.04371); [SPAgent](https://arxiv.org/abs/2511.20048); [PASTE](https://arxiv.org/abs/2603.18897); [Speculate with Memory](https://arxiv.org/abs/2607.12236) | Future action/tool prediction, parallel branches, result isolation, verification/commit, online prediction memory, and load-aware speculation scheduling are all occupied. Generic fork–match–commit is an exact collision. | At most: page-level copy-on-write of a typed, side-effect-free **control continuation** already resident on GPU, evaluated specifically at the measured ready-cohort boundary. It must show a semantic or systems mechanism absent from action/tool speculation. | **Demote.** Useful ablation or follow-on; weak 3-month flagship after these papers. |
| GPU-triggered sandbox prewarm/execution | [SpecBox](https://arxiv.org/abs/2607.23933); [Foundry](https://arxiv.org/abs/2604.06664); [Concordia](https://arxiv.org/abs/2606.23521) | Intent-driven sandbox prewarm, stochastic dependency prefetch, shared-memory delivery, persisted GPU graph/context materialization, and persistent-kernel checkpoint/recovery are occupied. | Only authority separation plus GPU-signalled lifecycle decisions and CPU-bypass result delivery on real hardware; prewarm prediction alone is not enough. | **Do not lead with this.** High collision. |
| GPU-initiated networking, syscalls, and DPU orchestration | NVIDIA [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca-gpunetio/); [NCCL GIN](https://arxiv.org/abs/2511.15076); [GPU System Calls / GENESYS](https://arxiv.org/abs/1705.06965); Linux [KVM API](https://docs.kernel.org/virt/kvm/api.html) | GPUNetIO already gives GPU kernels direct Ethernet/RDMA data-path control after privileged CPU setup. NCCL GIN exposes device-initiated remote-memory operations with direct and proxy transports. GENESYS demonstrated GPU-issued OS calls historically. KVM VM creation remains host-userspace ioctl authority. | An agent-specific split in which the GPU decides and handles the common unprivileged data path while a CPU/DPU validates capabilities and performs privileged lifecycle actions. Novelty would be the agent protocol, authority semantics, and end-to-end benefit—not GPU networking itself. | **High-risk moonshot.** Hardware-gated by bare metal plus ConnectX/BlueField; never claim that a CUDA kernel directly creates a VM. |
| GPU serverless, microVMs, and isolation | [ServerlessLLM](https://arxiv.org/abs/2401.14351); [StreamBox](https://www.usenix.org/conference/atc24/presentation/wu-hao); [KRYPTON](https://www.usenix.org/conference/atc25/presentation/zhang-shulai); [HAS-GPU](https://arxiv.org/abs/2505.01968); [Dilu](https://arxiv.org/abs/2503.05130); [AgileOS](https://arxiv.org/abs/2606.06697); [C2CServe](https://arxiv.org/abs/2605.19481); Firecracker [GPU passthrough discussion](https://github.com/firecracker-microvm/firecracker/discussions/4845); [Behind Bars](https://www.usenix.org/conference/usenixsecurity26/presentation/gu-cheng) | Fast model loading/migration, stream sandboxes, virtual GPUs, spatio-temporal sharing, autoscaling, protected CUDA-service virtualization, MIG model switching, and GPU passthrough prototypes are crowded. The Firecracker prototype reports VFIO passthrough rather than oversubscription/snapshot support. MIG itself has demonstrated cross-instance interference/side channels. | No broad gap. An agent-specific capability protocol may fit the DPU lane, but it does not establish a new GPU microVM substrate. | **Kill as a primary lane.** Use existing isolation; evaluate its security and cold-start constraints. |
| Shared-GPU slack harvesting | [ParvaGPU](https://arxiv.org/abs/2409.14447); [SIRIUS](https://www.usenix.org/conference/atc25/presentation/wang-jiali); [FlexLLM](https://www.usenix.org/conference/nsdi26/presentation/oliaro); [OmniServe](https://arxiv.org/abs/2603.12831); [MPS/MIG evaluation](https://arxiv.org/abs/2604.22430); [AgentServe](https://arxiv.org/abs/2603.10342) | MIG/MPS partitioning, inference/training colocation, token-level inference/PEFT interleave, latency-sensitive/best-effort sharing, CPU attention piggybacking, and Green Context isolation already harvest GPU slack under SLOs. | Treat exact deterministic agent control as a newly characterized best-effort workload; predict eligibility from route-ready cohorts and report inference SLO guardrails. | **Strong integrated section, weak standalone paper** unless a new general sharing boundary emerges. |
| Fleet overload and escalation | [MultiTASC](https://arxiv.org/abs/2306.12830); [AEGIS](https://arxiv.org/abs/2606.06660); [SwarmX](https://arxiv.org/abs/2606.21401) | MultiTASC already adapts local-to-shared-heavy-model cascade thresholds under server load, latency, and accuracy for many devices. AEGIS already uses an early risk probe for selective strong-policy invocation. SwarmX handles production-scale agent load prediction/routing/scaling. Average escalation rate and adaptive thresholding are not new. | The dependence-sensitive tail `Pr(A_delta > capacity)` during common shocks, with missed-rescue safety and an independence-matched placebo. It must use realistic correlated fleet traces and bounded strong-policy capacity. | **Conditional reliability paper.** Survives only as co-escalation tail risk, not generic cascade scheduling. |
| Runtime persistence semantics | [Agents Learn Their Runtime](https://arxiv.org/abs/2603.01209) | Interpreter persistence is not a neutral implementation detail: changing persistent/stateless execution can alter learned behavior and token use. | Make trajectory equivalence, state visibility, failure recovery, and train/runtime match explicit in the GPU-control evaluation. | **Cross-cutting requirement**, not yet a standalone systems claim. |

## What the nearest papers actually occupy

### 1. Resident execution is mechanism prior art, not the paper thesis

- **FLAME GPU 2** represents agent-based simulations as communicating stream
  X-machine state machines, partitions agents by state, infers function
  dependencies, and runs very large populations/ensembles on GPUs. This defeats
  “first GPU agent state machine,” “first state-bucketed agents,” and “first GPU
  swarm runtime.” The distinction is that simulated agents are continuously
  runnable while LLM agents are externally interrupted and asynchronously
  ready.
- **CUDA Graphs** let applications define work once and replay it, update some
  graph parameters, use conditional `IF`/`WHILE`/`SWITCH` nodes, and launch
  graphs from the device under documented constraints. Conditional control flow
  or device launch alone is therefore not novel.
- **GPUOS** keeps a persistent worker kernel, accepts work through a device
  queue, and dynamically injects/operators via runtime compilation and an
  indirection table. **Mirage MPK**, **Event Tensor**, **Fleet**, and **Ada-MK**
  likewise cover decentralized in-kernel task graphs, dynamic dependencies,
  hierarchical task scheduling, and static/DAG megakernel construction.
- **DVM** is an especially close warning for “agent bytecode”: a device virtual
  machine decodes dynamically constructed instructions on an accelerator. A
  CUDA interpreter is implementation engineering unless the agent semantics
  and measured workload boundary are themselves the contribution.

The surviving question is two-sided: even if a resident kernel can make tiny
transitions fast at cohort size `K`, how often do real interrupted agents expose
`K` route-compatible transitions within deadline `delta`? None of the sources
above joins that hardware crossover to real agent arrival geometry and then
tests an exact runtime against the resulting attainable share.

### 2. Workflow awareness is heavily occupied

- **Parrot** exposes semantic variables and program structure to an LLM service.
  **Agentix** intercepts program-level LLM calls and schedules them using program
  state, completed work, preemption, and priority rather than treating requests
  independently.
- **SAGA** represents an agent workflow as an Agent Execution Graph and uses its
  structure for KV retention, session affinity, work stealing, and fair sharing.
  **MARS** uses an external control plane with unified CPU/GPU pressure,
  admission, and KV-aware scheduling for agents. **SwarmX** predicts workload
  structure and latency for large fleet routing/scaling. **OpRAG** uses
  resource-deterministic operators, persistent workers, bounded queues, and
  overlap for RAG workflows.
- **Agentic CPU–GPU Scheduling** profiles complete AI tools and assigns them to
  immediate GPU, queued GPU, or CPU execution under utilization and memory
  pressure. This is the closest title-level collision. The proposed work must
  say in its first paragraph that its unit is an internal deterministic
  transition, not a complete tool or model invocation.

This plane distinction is necessary but not sufficient. A credible paper must
show that orchestration transitions consume enough CPU, tail latency, energy, or
fleet capacity to matter beside the model/tool stages. If the end-to-end gain is
negligible, the microbenchmark boundary remains an artifact rather than a top
systems result.

### 3. The multi-model opening is narrow and promising

**MuxServe** and **Mélange** optimize placement and batching across multiple
models/heterogeneous GPUs. **MixLLM** routes queries using contextual learning.
**BOute** jointly searches routing and heterogeneous deployment under quality
and resource goals. **Chimera** uses per-model semantic confidence and workflow
load to schedule heterogeneous models. **Prism** addresses memory elasticity for
multi-model serving at large production scale. **DeePEn** fuses heterogeneous
LLM predictions; the co-failure papers show that nominal model diversity is not
the same as useful error complementarity.

Therefore the claim cannot be “joint quality/latency/cost scheduling.” A
defensible named object is the **co-failure–regularity frontier**:

```text
(all-wrong probability, task utility, TCT, GPU-seconds, dollars,
 model swaps, weight/KV/prefix reuse, batch fragmentation)
```

The decisive experiment compares simultaneous error-complementary ensembles
with homogeneous self-sampling and route-one-model systems at matched hardware,
load, and task utility. If minimum-co-failure sets incur no material regularity
penalty, that negative result belongs in an ensemble paper rather than a new
scheduler.

### 4. Speculation and serverless are saturated

**Speculative Actions** executes predicted future actions in parallel and
commits matching branches. **SPAgent** specializes speculation and scheduling
for search agents. **PASTE** predicts future tool invocation patterns, isolates
results until confirmation, and jointly schedules tools with returning model
work. **Speculate with Memory** expands action/observation/chained speculation
with online memory while preserving the original trajectory.

On the cold-start side, **SpecBox** predicts sandbox intent and dependencies,
prewarms/prefetches, caches, and returns data over shared memory. **Foundry**
persists GPU execution topology/context for fast materialization. **Concordia**
adds checkpoint/recovery to persistent GPU execution. These make
“prewarm the likely sandbox,” “fork branches,” and “rollback/commit” poor lead
claims.

### 5. GPU I/O is real; GPU privilege is not implied

DOCA GPUNetIO separates setup from the fast path: the CPU creates and configures
privileged objects and launches the kernel; the GPU can then drive Ethernet/RDMA
queues and access data directly with supported NICs/topology. NCCL GIN similarly
supports device-initiated communication through direct GDAKI-capable operations
or GPU-to-CPU proxy queues. GENESYS is historical evidence that GPU system calls
are not a new concept.

Linux KVM creates a VM by opening `/dev/kvm` and issuing host-userspace ioctls.
The Firecracker GPU discussion describes a VFIO passthrough proof of concept,
not GPU-originated VM creation, oversubscription, or snapshot/resume. The honest
architecture is therefore:

```text
GPU resident policy -> capability-checked request -> CPU/DPU authority
                    -> privileged lifecycle action -> GPU/NIC fast data path
```

The paper-worthy question is whether this split removes control-path latency and
CPU load without weakening authority or isolation. It requires real bare-metal
ConnectX/BlueField hardware; a cloud VM without exposed GPUDirect/DOCA support
cannot substantiate the claim.

### 6. Fleet escalation must be about correlated tails

MultiTASC is the critical collision: many edge devices cascade from local light
models to a shared heavy-model server and adapt thresholds to throughput,
latency, and accuracy. AEGIS supplies early risk-triggered strong-policy
escalation, while SwarmX addresses large-scale fleet load.

The residual question is not the mean escalation rate. For arrivals `A_delta`
to a backup pool of capacity `m`, it is:

```text
Pr(A_delta > m)
```

under shared weather, scene, policy, software, or task shocks. A paper survives
only if dependence changes overload and missed-rescue tails relative to a
marginally matched independent trigger process, and if policies improve safety
at a fixed reserve-GPU budget.

## Ranked research verdict

| Rank | Direction | Decision | Required evidence before scaling |
|---:|---|---|---|
| 1 | Ready-cohort feasibility boundary + exact resident runtime | **Primary flagship** | Non-Poisson public agent traces; tuned CPU ceiling; CUDA Graph and GPUOS-like persistent baselines; route/deadline ablations; end-to-end task and CPU/energy impact; trajectory equality. |
| 2 | Co-failure–regularity frontier | **Independent paper candidate** | Frozen correctness panel plus 8–16 open-weight models served concurrently; matched quality/load/hardware; direct weight/KV/prefix/batch/churn measurements; Chimera/BOute/MixLLM-style routing and homogeneous self-sampling baselines. |
| 3 | GPU-signalled CPU/DPU authority fabric | **Moonshot behind hardware gate** | Verified ConnectX/BlueField access; GPUNetIO direct and proxy baselines; explicit threat/capability model; real lifecycle and data-path latency; CPU offload and failure-isolation results. |
| 4 | Correlated co-escalation ceiling | **Conditional reliability lane** | Realistic common-shock traces; independence-matched placebo; MultiTASC adaptive-threshold baseline; bounded-capacity strong-policy inference; overload and missed-rescue confidence intervals. |
| 5 | Same-GPU deterministic-control slack | **Flagship section** | Pre-registered TTFT/TPOT/P99 guardrails; low-priority stream, MPS/MIG/Green Context baselines where supported; useful work per marginal dollar; exact control trajectories. |
| 6 | Persistence semantics | **Cross-cutting study** | 2x2 train/runtime persistence check, state visibility/recovery specification, and evidence that GPU lowering does not change learned or executed behavior. |
| 7 | GPU fork/rollback | **Demote** | Proceed only if a typed COW continuation offers a mechanism and regime not covered by action/tool speculation; otherwise retain one negative ablation. |
| 8 | GPU serverless/microVM substrate | **Stop** | No defensible broad novelty; adopt and benchmark existing mechanisms instead. |
| 9 | Generic GPU OS, state engine, or bytecode VM | **Stop as a claim** | Use as implementation substrate only. FLAME GPU, GPUOS/megakernels, CUDA Graphs, and DVM occupy the mechanism. |

## Claims that are and are not defensible

Potentially defensible after a repeated search:

> We characterize a two-sided boundary for deterministic agent-control
> acceleration: the hardware cohort at which a route becomes GPU-beneficial and
> the trace-conditioned supply of compatible ready events before a latency
> deadline. We then evaluate an exact resident runtime against that boundary.

> We characterize the Pareto conflict between ensemble all-wrong probability
> and the serving regularity of simultaneous heterogeneous model sets.

Not defensible:

- “first GPU runtime for agents”;
- “first agent state machine on a GPU”;
- “first persistent GPU task queue/OS”;
- “first agent/workflow-aware scheduler”;
- “first agent CPU/GPU scheduler”;
- “first GPU-controlled network path”;
- “GPU kernels spawn secure VMs”;
- “first speculative agent/tool execution”;
- “first shared-GPU slack harvester”;
- “first adaptive fleet escalation scheduler.”

## Evaluation implications from the collision audit

1. Treat independent machine/container placement, trace, and random seed as
   replication units; millions of per-request timings from one deployment do
   not create millions of independent samples.
2. Pre-register one primary systems estimand per lane, a practically important
   effect threshold, and the workload cells used for confirmation. Report effect
   sizes and cluster/bootstrap confidence intervals, not only `p` values.
3. Randomize implementation order within paired workload blocks. Use Holm
   correction for the small family of confirmatory pairwise comparisons; mark
   hardware sweeps and mechanism grids as exploratory.
4. For exact decisions on the same tasks, use paired tests such as McNemar plus
   confidence intervals. For overload or all-wrong tails, report binomial or
   bootstrap intervals and the number of independent shock/task clusters.
5. Separate the feasibility quantities: fixed-window eligible share, optimal
   interval packing under the chosen deadline semantics, queueing losses, and
   realized end-to-end acceleration. Do not call a fixed-window share a universal
   upper bound.
6. Preserve negative regimes. “GPU loses below `K`,” “route fragmentation erases
   the gain,” and “shared inference SLOs forbid useful slack work” are boundary
   results, not failed experiments.

## Search record

Search performed: **2026-08-11**. Primary-source search surfaces were arXiv,
USENIX conference proceedings, publisher pages, NVIDIA CUDA/DOCA documentation,
Linux kernel documentation, and the official Firecracker repository.

Representative queries and citation-chain terms:

- `GPU persistent kernel task queue runtime dynamic operator injection 2025 2026`
- `GPU OS persistent megakernel task graph device interpreter`
- `CUDA Graph conditional node device launch persistent control flow`
- `GPU agent state machine agent based simulation FLAME GPU`
- `compound AI system agent program workflow scheduling KV cache 2026`
- `multi agent serving scheduler CPU GPU tool orchestration`
- `heterogeneous multi model LLM serving placement routing ensemble GPU`
- `LLM ensemble diversity co-failure all wrong systems cost`
- `speculative agent action tool execution sandbox prewarm rollback`
- `GPU initiated networking syscall GPUDirect DOCA DPU control plane`
- `GPU microVM Firecracker VFIO GPU passthrough snapshot isolation`
- `serverless GPU cold start CUDA context function isolation MIG security`
- `shared GPU slack harvesting inference best effort MPS MIG Green Context`
- `multi agent fleet escalation overload shared server cascade correlation`

Older anchors were retained when they defeat priority claims: GENESYS (2017)
for GPU system calls, FLAME GPU 2 (2023 publication) for GPU agent state
machines, and MultiTASC (2023) for adaptive many-device escalation. Search
exclusions included secondary surveys, vendor blogs, news articles, benchmark
roundups, and unlinked search-result summaries.

## Submission-time audit queue

Before freezing related work:

1. Repeat exact-title and abstract searches in arXiv, DBLP/OpenAlex/Semantic
   Scholar, and the proceedings for SOSP, OSDI, NSDI, ASPLOS, EuroSys, ATC,
   MLSys, SC, HPCA, MICRO, FAST, SoCC, and Middleware.
2. Backward- and forward-chain the closest ten sources: FLAME GPU 2, GPUOS,
   Agentix, SAGA, Agentic CPU–GPU Scheduling, Chimera, Speculative Actions,
   SpecBox, DOCA GPUNetIO, and MultiTASC.
3. Record screened titles and explicit exclusion reasons. Negative-search
   evidence should be an artifact, not a memory of queries.
4. Re-run the search immediately before submission; the 2026 agent-systems area
   is moving too quickly for this August snapshot to remain sufficient.

## 12 August 2026 close-neighbor addendum

Two papers added after the first audit materially sharpen the positioning:

- [Architectural Implications of Agentic AI Workflows](https://arxiv.org/abs/2608.04458)
  characterizes production Azure and open-source agentic workflows. It reports
  repeated CPU--GPU crossings, host orchestration on the critical path, bursty
  CPU demand, and the Agora server prototype. This occupies any broad claim
  that the present work discovered agentic CPU pressure or the importance of
  orchestration. It does not measure deadline-conditioned supply of
  same-route post-event transitions, compute `F/P*/U`, or compare a matched
  device-resident decision with a host-observed GPU path.
- [ThunderAgent](https://arxiv.org/abs/2602.13692) represents workflows as LLM
  Programs and jointly manages KV cache, system state, and tool environments
  through a program-aware scheduler. It occupies broad program-aware agent
  scheduling and resource-management claims. Its scheduling unit and reported
  interventions are inference and tool resources, not the deterministic
  post-event transition measured here.

After adding these neighbors, the defensible current-paper claim remains
scoped: the audited sources do not combine route-key-conditioned deadline
supply with a matched host-observation placement test for deterministic
post-event agent control. This is a collision audit, not a first-ever or
exhaustive-priority claim. The larger online runtime remains open.
