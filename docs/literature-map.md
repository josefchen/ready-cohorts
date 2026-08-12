# Literature map and novelty boundary

Last reviewed: 2026-08-12. This is a working map, not a claim that the search is
exhaustive. The paper should say “we found no prior system that joins X and Y”
only after a second author repeats the search and citation chaining.

## Closest work

| Line of work | What it establishes | What remains distinct here | Threat to novelty |
|---|---|---|---|
| [Architectural Implications of Agentic AI Workflows / Agora](https://arxiv.org/abs/2608.04458) | Production Azure and open-source characterization finds repeated CPU--GPU crossings, host orchestration on the critical path, bursty CPU demand, and heterogeneous resource roles. Agora pools CPU cores and oversubscribes GPU memory. | We do not claim to discover agentic CPU pressure. Our unit is the deterministic post-event transition; we measure route-key-conditioned deadline supply and a matched host-observation placement choice. | **Very high for motivation, low for the scoped measurement pair.** Cite centrally and prohibit generic CPU-critical-path novelty language. |
| [ThunderAgent](https://arxiv.org/abs/2602.13692) | Models workflows as LLM Programs and jointly schedules inference state, KV cache, and tool environments. | Its unit is the program's inference and tool resource lifecycle. It does not measure `F/P*/U` for deterministic post-event transitions or the resident decision mechanism. | **High systems-neighbor overlap.** Program-aware scheduling is not a novel claim here. |
| [FLAME GPU 2](https://doi.org/10.1002/spe.3207) | General agent-based simulations can be expressed as state machines, partitioned by state to reduce divergence, and executed efficiently on GPUs. It explicitly discusses small-population utilization, ensembles, data movement, and heterogeneous states. | LLM-agent runtimes are externally interrupted and asynchronously ready. We measure the hardware crossover and then derive a trace-conditioned ceiling on how many compatible events can actually use it inside a deadline. | **High.** We must cite it centrally and cannot claim state-bucketed GPU agents as new. |
| [Agentic CPU-GPU Scheduling for Heterogeneous AI Workloads](https://arxiv.org/abs/2607.22242) | Profiles 19 whole AI tools and maps each tool to immediate GPU, queued GPU, or CPU execution under utilization and VRAM contention. | Our scheduling unit is a non-neural agent state transition inside the orchestration runtime, not a model/tool invocation. The proposed contribution is the readiness × regularity × residency boundary and a route-bucketed transition engine. | **Very high.** The title and abstract must make the different scheduling granularity explicit. |
| [Agentix / Autellix](https://www.usenix.org/conference/nsdi26/presentation/luo), [Parrot](https://arxiv.org/abs/2405.19888), [SAGA](https://arxiv.org/abs/2605.00528), [SwarmX](https://arxiv.org/abs/2606.21401), [Kairos](https://arxiv.org/abs/2508.06948) | Treat programs/workflows rather than isolated LLM requests as scheduling units; improve LLM-call routing, fairness, throughput, tail latency, or cache affinity. | They optimize the neural inference plane and workflow placement. We isolate deterministic control transitions between LLM/tool events and ask whether those transitions themselves should execute as resident GPU batches. | High systems-neighbor overlap, but a clean plane/granularity distinction. |
| [MARS](https://arxiv.org/abs/2604.26963) and [OpRAG](https://arxiv.org/abs/2608.08340) | Coordinate heterogeneous GPU inference, CPU tools, and multi-stage RAG operators through an external resource-aware control plane. | Their control plane decides where workflow stages execute; it is not itself lowered into a resident GPU state-transition engine. Their real agent integrations are important end-to-end baselines for our eventual runtime. | High. Reviewers may reasonably ask why optimizing non-neural transitions matters beside whole-workflow co-scheduling. |
| [Chimera](https://arxiv.org/abs/2603.22206) and capability/cost-aware multi-model serving | Jointly schedule heterogeneous models using predicted quality, latency, load, or cost. | A possible adjacent paper studies the conflict between co-failure reduction and hardware regularity: error-complementary model sets can fragment model residency, cache reuse, and batching. The distinguishing evidence would be joint all-wrong certificates plus measured systems costs, not another quality predictor. | **High** for the diversity–regularity lane. A generic multi-model scheduler is not novel. |
| [InferCept](https://arxiv.org/abs/2402.01869), [Continuum](https://arxiv.org/abs/2511.02230) | Tool calls interrupt generation; preserving or predicting KV-cache residency across pauses improves augmented/agentic LLM serving. | Their resident state is model KV cache. Our resident state is agent runtime state and route-specific transition data. | Medium. They strongly motivate interruption-aware residency. |
| [SpecBox](https://arxiv.org/abs/2607.23933), [PASTE](https://arxiv.org/abs/2603.18897), and [Foundry](https://arxiv.org/abs/2604.06664) | Hide agent sandbox/cold-start latency through speculative preallocation or execution, and accelerate GPU context materialization. | A GPU-signalled sandbox design is defensible only if it adds authority separation, ready-cohort-aware lifecycle decisions, and CPU-bypass result delivery. Pure “predict the tool and prewarm its sandbox” is already occupied. Fork/rollback must speculate only side-effect-free state continuations. | **Very high.** Do not lead with generic VM or sandbox prewarming. |
| [TokTier](https://arxiv.org/abs/2607.29678), [GPUTOK](https://arxiv.org/abs/2603.02597) | CPU-side serving stages can move to GPU, but only with exactness checks, state reuse, batching, and a measured crossover. | We target the broader state-transition loop and expose route readiness as the workload-side limit. | Medium-to-high methodological neighbor; useful positive precedent. |
| [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219), [Event Tensor](https://arxiv.org/abs/2604.13327), [Fleet](https://arxiv.org/abs/2604.15379), [Ada-MK](https://arxiv.org/abs/2605.11581), and [GPUOS](https://arxiv.org/abs/2604.17861) | Persistent/mega-kernels remove launch gaps and can dynamically schedule or inject fine-grained GPU work, primarily for tensor programs and LLM inference. | These provide candidate mechanisms for the runtime intervention. We contribute the agent-specific readiness model, workload trace, crossover atlas, and deadline-aware route batching rather than a general tensor compiler or transparent operator-fusion layer. | **High after GPUOS.** A persistent work queue or small-operation fusion alone is not a contribution; the trace-conditioned agent boundary and semantics must carry the novelty. |
| [Static Batching of Irregular Workloads](https://arxiv.org/abs/2501.16103) and [CUDA Graph kernel batching](https://arxiv.org/abs/2501.09398) | Irregular tasks can be mapped into static batches, and repeated fine-grained iterative kernels can be unrolled/captured to amortize launch overhead. NVIDIA's [CUDA Graph programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) explicitly frames graphs as a way to pay submission costs once and supports device-side graph launch for dynamic control flow. | These are mechanism and baseline families. Our open question is whether real agent arrivals produce sufficiently large *semantically compatible* cohorts before their deadlines, and when route fusion changes behavior. | High for mechanism novelty, low for the two-sided feasibility boundary. |
| [Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference](https://arxiv.org/abs/2603.22774) | CPU starvation can delay launches, communication, and tokenization even after CUDA Graph optimization. | We test moving a particular CPU control workload to a resident GPU path and quantify when that is beneficial versus simply provisioning CPU. | Medium and important as a skeptical baseline. |
| NVIDIA [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca-gpunetio/) and [AgileOS](https://arxiv.org/abs/2606.06697) | Demonstrate GPU-controlled network data paths after privileged setup and protected GPU service/virtualization mechanisms. | The moonshot architecture keeps lifecycle authority on a CPU/DPU while GPU-resident agent state drives the common data path. It would need real ConnectX/BlueField hardware and isolation evidence, not a simulated direct-I/O claim. | High mechanism overlap and a strict hardware gate. |
| [MAgent](https://arxiv.org/abs/1712.00600) and GPU RL simulation | Millions of homogeneous agents or environments can run on a GPU. | LLM agents are not continuously runnable simulated particles: readiness is fragmented by model/tool waits and route heterogeneity. | Low-to-medium; prevents broad “first GPU agent swarm” claims. |

## Defensible gap

The project should not claim “the first GPU system for agents,” “the first
agent CPU/GPU scheduler,” or “the first state-bucketed agent runtime.” Those
claims are false or dangerously broad.

The strongest current gap is:

> No identified system jointly (i) measures the CPU/GPU crossover for
> non-neural agent state transitions under residency, branch regularity,
> temporal fusion, and host visibility; (ii) measures from public agent traces
> whether enough same-route transitions become ready inside a latency budget;
> and (iii) implements a deadline-aware resident runtime whose observed
> acceleration approaches that trace-conditioned ceiling.

Items (i) and the trace replay behind (ii) now have pilot evidence. Item (iii),
non-Poisson validation, an optimized CPU ceiling, and end-to-end task results
remain required for a strong systems paper.

The August 2026 literature makes the positioning narrower than “GPU control
plane.” GPUOS already makes transparent small-operation fusion a first-class
runtime abstraction, while MARS and OpRAG optimize agent/RAG orchestration.
The defensible paper is therefore a **feasibility-boundary and semantics
paper**: it must show when agent transitions are accelerable, prove why a
deadline/route distribution limits acceleration, and demonstrate a runtime
that approaches the bound without changing trajectories.

## Core conceptual object

Let `K(h, r, v, H)` be the smallest cohort on hardware `h` at which route or
kernel class `r`, visibility policy `v`, and fused horizon `H` beat the chosen
CPU baseline. For a frozen non-overlapping window partition `pi_delta` and
grouping rule `g`, let `n_b(pi_delta, g)` be the number of compatible ready
events in bucket `b`.

The event-weighted fixed-window eligible share is

```text
F(pi_delta, K, g) = Σ_b n_b · 1[n_b ≥ K] / Σ_b n_b.
```

`F` is achievable and exact among schedulers constrained to that same frozen
partition. It is not a universal ceiling for sliding per-event deadlines,
because a valid cohort may straddle a fixed boundary. `paper/formalism.md`
defines the general interval-packing optimum `P*` and a local-eligibility upper
bound `U`, with `F <= P* <= U`. The fixed-window empirical regularity tax is

```text
R(pi_delta, K) = F(pi_delta, K, pooled) - F(pi_delta, K, exact-route).
```

This boundary model is deliberately simple. The implementation study must then add
queueing, execution time, CPU fallback, fairness, state movement, and shared-
GPU interference.

## Paper-level differentiators to preserve

1. **Scheduling granularity:** internal deterministic transition, not whole
   model/tool/workflow.
2. **Two-sided measurement:** hardware crossover `K` plus workload quantities
   `F`, `P*`, and `U`, rather than a benchmark of either side alone.
3. **Negative regimes:** wide/branchy correctness failures, no GTX one-step
   crossover by 256, host-visible losses, and route fragmentation remain
   headline results rather than appendix debris.
4. **Matched intervention:** route bucketing / persistent residency changes one
   missing term and is compared with equal-budget placebos.
5. **Exactness:** trajectory decisions are verified, not just final floats or
   throughput.
6. **Open artifact:** raw timing ledgers, pinned trace-derived features,
   simulator, manifests, and a reproducible figure pipeline.

## Citation-chaining queue

Before submission, inspect references and citing papers for each of the
following terms: GPU agent-based simulation; discrete-event GPU simulation;
state-machine batching; heterogeneous CPU/GPU task placement; dynamic batching
with deadlines; persistent work queues; megakernel runtimes; compound AI
serving; tool-interrupted KV caching; orchestration traces; serverless GPU cold
starts; and exact GPU tokenization. Record both inclusions and exclusions so a
negative novelty search is auditable.
