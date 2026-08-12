# Working paper outline

## Candidate title

**The Ready-Cohort Boundary: When LLM-Agent Control Has Enough Work for a GPU**

Avoid leading with “GPU as cheap CPU”; it is memorable but technically
misleading and now overlaps a July 2026 whole-tool CPU/GPU scheduling paper.

## One-sentence thesis

GPU acceleration of an agent control plane is governed not by total swarm
population but by the intersection of a hardware crossover for a resident,
regular transition and the workload’s ability to assemble same-route ready
cohorts inside its latency budget.

## Abstract skeleton

1. Agent systems increasingly execute large numbers of deterministic state,
   routing, bookkeeping, filtering, and policy transitions between expensive
   model/tool calls; these stages remain CPU/Python controlled.
2. Offloading them to GPU appears attractive but can lose to launch, transfer,
   divergence, compilation, and route fragmentation.
3. Define a hardware crossover `K`, fixed-window eligibility `F`, a
   sliding-deadline packing optimum `P*`, and a local-eligibility upper bound
   `U`.
4. Characterize `K` across hardware, state width, branching, residency,
   observation frequency, and temporal fusion, preserving correctness failures.
5. Characterize `C` on pinned public orchestration traces and quantify the
   regularity tax between pooled and exact-route batching.
6. Present a deadline-aware resident route-bucket runtime and evaluate it on
   trace replay and live agent workloads.
7. Report the boundary, not just peak speedup: when CPU wins, when GPU wins,
   and how close the runtime comes to the ceiling.

## Contributions in the working manuscript

1. **Ready-cohort boundary.** A framework separating the hardware safe-suffix
   threshold from fixed-partition eligibility, an exact offline opportunity,
   a local upper bound, and unmeasured online achievement.
2. **Conditional trace instrument.** A hash-pinned public panel and a
   route-key-conditioned replay using an exact specialized evaluator that is
   equivalent to established one-dimensional minimum-size clustering. No
   algorithm-priority claim is made.
3. **Matched mechanism test.** A four-named-placement comparison of a bundled
   host observation and redispatch path against a GPU-resident decision path,
   with exact state and decision checking.
4. **Negative calibration.** A fixed nested device launch that is slower in all
   60 measured cells across five named placements.

The route-compacting runtime, tuned CPU comparison, raw invocation P99, CPU
displacement, shared-inference interference, and end-to-end utility remain the
next paper stage rather than contributions of the current draft.

## Main figures

1. Conceptual readiness × regularity × residency cube with measured regimes.
2. Full hardware crossover phase diagram with invalid correctness cells marked.
3. Ready-cohort heatmap by active sessions and batching deadline, separating
   the frozen partition `F`, exact offline optimum `P*`, and local bound `U`.
4. Opportunity decomposition: pooled → event class → exact route → online
   runtime achieved, with the primary `F=0.3019`, `P*=0.4300`, `U=0.4585`
   calibration identified as conditional on the frozen trace/arrival model.
5. Runtime end-to-end latency/throughput frontier against CPU and GPU baselines.
6. Cost/energy and CPU-core displacement across GPU classes.
7. Negative-regime panel: host visibility, shape churn, branch divergence,
   burstiness, and shared-LLM interference.

## Evaluation questions

- RQ1: What determines the steady-state CPU/GPU crossover for agent control
  transitions?
- RQ2: How often do public agent workloads form cohorts above that crossover
  inside realistic latency budgets?
- RQ3: Can a route-bucketed resident runtime approach the offline
  deadline-packing optimum while remaining below the local upper bound?
- RQ4: Does offload improve end-to-end task completion, cost, and energy after
  compilation, queueing, transfers, and CPU fallback?
- RQ5: Can control transitions harvest slack on an LLM GPU without degrading
  model-serving P99 SLOs?
- RQ6: Which regimes remain CPU-preferred, numerically unsafe, or too fragmented?

## Baselines

- optimized C++/OpenMP/SIMD CPU transition engine;
- PyTorch eager CPU/GPU and `torch.compile` CPU/GPU;
- resident versus host-visible GPU;
- CUDA Graph per route;
- fixed nested device launch as a retained negative-overhead ablation;
- matched host-round-trip versus device-resident conditional route epochs;
- route-bucketed microbatcher without persistent kernel;
- persistent-kernel command queue as an optional named ablation;
- pooled illegal oracle (upper-bound diagnostic only);
- deadline-matched random bucketing and equal-wait CPU placebos;
- whole-tool CPU/GPU placement where relevant, clearly separated from our unit.

## Submission positioning

The intended contribution is a systems measurement + runtime paper. A top
systems venue will require an implemented end-to-end system and multiple
independent hosts, not just an arXiv benchmark report. If the runtime is not
ready, release the trace/crossover artifact as a careful measurement preprint
and avoid overclaiming the system contribution.

The deployment-level native claim uses fresh placements, not timing rows. A
placement-scale confirmation is reserved for the online route-compacting
runtime; its default is 30 analyzed placements in each frozen L4 and H100
provider mixture, increased when blinded nuisance-variance simulation requires
it. The smallest worthwhile runtime speedup is 1.15; CPU displacement, task
completion, utility, and exactness are co-primary gates rather than optional
secondary plots.
