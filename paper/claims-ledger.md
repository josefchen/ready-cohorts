# Claims ledger

Evidence cutoff: 2026-08-12. “Supported” means supported by the current local
artifact, not yet peer-reviewed or externally replicated.

| ID | Candidate claim | Status | Evidence | Required before paper headline |
|---|---|---|---|---|
| C1 | Resident compiled GPU execution can beat matched compiled CPU execution for a regular agent transition. | **Supported, qualified.** | Pilots 003–006; L4 `H=1` crosses at tested `N=16`; fused `H=64` crosses at or below `N=8` on L4 and GTX 1660 Ti. | Optimized C++/SIMD/OpenMP CPU baseline and a deterministic custom GPU kernel. |
| C2 | A GPU should be described as a resident SIMD/SIMT transition engine, not as interchangeable cheap CPU cores. | **Interpretation supported.** | Strong sensitivity to state residency, host visibility, branching, and horizon fusion. | Preserve as framing, not a universal hardware claim. |
| C3 | Temporal fusion moves the crossover more than population alone. | **Supported in tested regular kernel.** | `H=64` gives a 2.36–4.25× proportional GPU fusion advantage over tuned CPU across every sub-256 population/mode; broader atlas agrees directionally. | Custom-kernel replication and horizons other than 1/64. |
| C4 | Host observation can erase a GPU win. | **Supported.** | L4 resident `H=1` crosses at 16; host-visible `H=1` never crosses by 256. Median host penalty is 1.55× on L4 and 1.36× on GTX. | Pin transfer semantics and isolate copy, synchronization, and Python delivery. |
| C5 | Tiny CPU/GPU numeric differences can cross a discrete decision boundary and amplify into materially different trajectories. | **Supported.** | Identical 9/32 invalid `A=8` shapes on GTX and L4 in pilots 003/004; budget divergence reaches about 0.107 despite small state error. | Per-step divergence instrumentation; integer/fixed-point control and margin sweep. |
| C6 | Large swarm population does not imply large same-route-key cohorts inside a fixed batching window. | **Supported under a qualified replay model.** | Hash-pinned 851-session tau2 panel; at `C=100k`, `K=256`, 25 ms: pooled fixed-window eligibility 85.6%, route-key grouping 0%. Sliding-local analysis confirms that boundary alignment can materially raise a valid upper bound but does not erase label fragmentation in the tested panel. | Verify executable semantic classes; replay recorded production arrivals, additional corpora, and sensitivity to bursts and correlated arrivals. |
| C7 | For zero-service, unlimited-capacity, equal-relative-deadline events, the sliding-deadline ready-cohort optimum is exactly computable between the frozen-window lower bound and local upper bound. | **Supported under the stated model.** | Exact integer-clock dynamic program in `paper/formalism.md` and `ready_cohort.py`; brute-force agreement tests; 540 prospectively frozen cell-seed rows from nine generated swarms satisfy `F <= P* <= U` and all frozen monotonicity checks. The recurrence admits an `O(N log N)` grouped implementation, but the frozen code is `O(NR + sum_r n_r log n_r)` because it rescans route masks. In the primary route-key cell (`C=100k`, `K=256`, 50 ms), mean eligibility is `F=0.3019`, `P*=0.4300`, and `U=0.4585`, closing 81.8% of the alignment gap. | Add finite service time/capacity, online decisions, non-Poisson arrivals, additional trace corpora, and runtime realization. Do not generalize the exact algorithm beyond equal relative deadlines. |
| C8 | A route-bucketed resident runtime improves end-to-end agent completion time and cost. | **Unsupported / target claim.** | No runtime intervention or end-to-end agent run yet. | Implement runtime; compare CPU, eager GPU, graph batching, persistent kernel, and placebos on live/replayed agents. |
| C9 | GPU offload reduces required CPU cores in agent serving without harming LLM SLOs. | **Unsupported / high-value target.** | Current microbenchmarks do not co-run an LLM server. | Same-GPU and separate-GPU co-scheduling experiments with vLLM, CPU saturation sweeps, P99 SLO and interference metrics. |
| C10 | At the frozen provider prices and primary fused control cell, a cheaper inference GPU can beat H100 on cost even when H100 wins wall time. | **Supported, qualified.** | In the preregistered 18-placement confirmation at `N=256, H=64`, H100 has the fastest median placement time (0.122803 ms), while T4 has the lowest estimated GPU cost per billion valid transitions (0.002429 versus 0.008222 for H100). All four cheaper requested classes beat H100 cost in every placement; price-rank versus wall-time-rank Spearman correlation is -0.486. | Repeat selected cells on Runpod/Lambda; report price timestamp and sensitivity; do not generalize beyond this kernel, allocation, or provider. |
| C11 | The native CUDA mechanism executes correctly across providers. | **Initial portability evidence; no performance generalization.** | Field-exact native pilot on two Modal L4 placements, one RunPod L4 placement, and one Lambda H100 placement, plus a local GTX 1660 Ti: 12,000 measured rows, five distinct GPU UUIDs, and zero correctness failures. Provider-specific receipts and hardware metadata are retained. | Carry the mechanism into the finite-capacity online runtime, qualify its invocation-tail/CPU instrumentation, and reserve powered placement-level confirmation for that deployment-relevant treatment. |
| C12 | The system saves energy. | **Unsupported.** | No power telemetry. | NVML sampling, host power where available, energy per valid transition, and idle-baseline subtraction. |
| C13 | A fixed device-launched child graph is faster than replaying the same graph from the host. | **Falsified for the tested calibration.** | Across all five native pilot placements and every tested `(N, H)` cell, `cuda_device_graph / cuda_host_graph` exceeds 1 (observed placement ranges approximately 1.07--1.99). The device path adds a launcher kernel while eliminating no host decision. | Do not scale this mechanism. Test a device-resident predicate/route decision that removes matched host synchronize/copy/dispatch epochs under a new preregistration. |
| C14 | Keeping a GPU-computed route decision on device can beat the tested host-observation and redispatch bundle. | **Supported as a four-placement mechanism pilot; not a population claim.** | Frozen `resident-policy-001` source on local GTX 1660 Ti, Modal L4, RunPod L4, and Lambda H100: 3,240/3,240 measured rows pass mechanism-specific gates; resident wins all 36 placement-cells, with observed ratios of batch-average row medians 1.19--2.39x. At the prospectively frozen `N=256, H=32` cell, the named-placement ratios are 1.71x, 2.39x, 2.06x, and 1.84x. The treatment bundles a four-byte copy, synchronization, CPU branch selection, redispatch, and a different graph topology. The horizon-ratio diagnostic is nondecreasing for every population/placement but is a post-plan operationalization. | Do not scale this microbenchmark directly. Add true invocation-tail and benchmark-process/cgroup CPU measurement under a corrected design, then reserve placement-scale inference for the first route-compacting runtime that measures online `A`; retain optimized legal baselines and powered placement-level confirmation. |

## Claims that must not appear without new evidence

- “GPUs are cheaper CPUs for agent swarms.”
- “The GPU control plane speeds up real agents end to end.”
- “The current CPU baseline is hardware-optimal.”
- “Poisson replay represents production arrival processes.”
- “The measured crossover is universal across GPU/CPU pairs.”
- “Compilation cost is negligible.”
- “Final tensor closeness guarantees trajectory equivalence.”
- “This is the first GPU agent runtime” or “first agent CPU/GPU scheduler.”

## Statistical discipline

- Keep every failure and nonpositive observation in raw ledgers.
- Report medians, all repetitions, CV, and descriptive ratio intervals.
- Do not call persistent-state repetitions independent task samples.
- Freeze confirmatory hypotheses and primary cells before rerunning.
- Use host/card/run as the replication unit for hardware generalization.
- Treat fresh provider placements, not timing rows, as the native-runtime
  sampling unit; within-placement repetitions estimate technical noise only.
- Report selection: pilots choose cells; confirmatory runs estimate effects.
- Apply multiplicity control only to confirmatory families, not retroactively to
  exploratory atlas cells.
