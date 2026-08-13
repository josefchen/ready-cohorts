# Ready Cohorts

Code and evidence for **Ready Cohorts: Bounding GPU Opportunity and Avoiding
Host Round Trips in LLM-Agent Control**.

- Paper: <https://arxiv.org/abs/2608.12123>
- Hugging Face Paper page: <https://huggingface.co/papers/2608.12123>
- Interactive results explorer: <https://huggingface.co/spaces/josefchen/ready-cohorts>
- Processed evidence mirror: <https://huggingface.co/datasets/josefchen/ready-cohorts>
- Frozen v1 source and release bundle:
  <https://github.com/josefchen/ready-cohorts/releases/tag/ready-cohorts-arxiv-v1>
- Pinned input trace dataset:
  <https://huggingface.co/datasets/Exgentic/agent-llm-traces/tree/f7c94012d0bfbf66fe4d6ed627699508bbb555ff>

The project asks when deterministic, non-neural agent control work—state
updates, routing, filtering, bookkeeping, policy checks, and batched tool-result
transitions—should execute on CPU or as a resident GPU workload. The intended
object is not “a GPU as a bag of cheap CPU cores.” It is a three-dimensional
boundary:

1. **readiness:** how many compatible events are ready inside a latency budget;
2. **regularity:** whether those events share an execution route without unsafe
   divergence;
3. **residency:** whether state and observations remain on device long enough
   to amortize launch, transfer, and synchronization.

## Current evidence

- **2,304 compiler-matched atlas observations** across a local GTX 1660 Ti and
  Modal L4, with 256–1,048,576 agents, four state/action shapes, two temporal
  horizons, CPU thread counts, and GPU visibility modes.
- **864 prospectively frozen sub-256 observations** across the same two GPUs. All
  succeeded and passed strict final action/state/budget checks.
- On L4, a resident one-step regular transition first beats the faster compiled
  CPU at **16 agents**; host-visible one-step execution does not cross by 256.
- On the GTX 1660 Ti, resident one-step execution still does not cross by 256.
- A fused 64-transition graph crosses at or below the smallest tested cohort,
  **8 agents**, on both GPUs.
- The broader atlas excludes **9 of 32 shapes on both machines** because tiny
  CPU/CUDA float differences cross discrete action thresholds and change the
  trajectory. Those failures remain in the raw data.
- A pinned public tau2 panel contains **851 sessions and 9,031 derived control
  events**. Under the frozen replay at 100,000 mean active sessions and
  `K=256`, a 25 ms window makes 85.6% of pooled events eligible but 0% of exact-
  route events eligible—the measured fixed-window regularity tax. This is an
  exact ceiling only for the frozen non-overlapping partition; the general
  sliding-deadline bounds are derived in `paper/formalism.md`.
- A complete prospectively frozen T4/L4/A10/L40S/A100-80GB/H100 sweep adds **2,592
  cloud observations**, all successful and correctness-valid. Every fused
  resident path wins at `N=8`; the one-step hypothesis fails on A100, H100,
  and L40S.
- A fresh **18-placement, 19,440-observation confirmation** (three independent
  placements per requested GPU, 30 timed repetitions per cell) completed with
  zero execution, duplication, or strict-correctness failures. H100 is fastest
  at the primary `N=256, H=64` cell, while T4 is the least expensive under the
  frozen provider prices. The 19,440 timing rows are not independent samples;
  the inferential hardware unit is the fresh placement.
- The exact equal-relative-deadline packing experiment evaluates **540
  prospectively frozen cell-seed rows from nine generated swarms**. In the
  primary route-key cell (`C=100k`,
  `K=256`, 50 ms), mean eligible share rises from **0.3019** under the frozen
  partition to the exact offline optimum **0.4300**, below the local upper bound
  **0.4585**. This closes 81.8% of the window-alignment gap under the explicitly
  limited zero-service/unlimited-capacity model.
- A field-exact native CUDA calibration contributes **12,000 measured rows on
  five distinct GPU placements**: local GTX 1660 Ti, two Modal L4s, RunPod L4,
  and Lambda H100. Every row passed the independent host-reference comparison.
  Fixed nested device launch was slower than replaying the same graph from the
  host in every tested cell, so it is a negative calibration rather than the
  proposed runtime treatment.
- A prospectively frozen decision-bearing mechanism then holds the GPU-computed
  predicate on device and tail-launches the selected uploaded route graph. On
  four named placements—local GTX 1660 Ti, Modal L4, RunPod L4, and Lambda
  H100—all **3,240 full-grid rows** pass their mechanism-specific gates. Across
  the two admissible mechanisms, **14,557,440 tested batched invocations** are
  field-exact and decision-exact against the separately implemented oracle. The device path beats
  the matched 4-byte host round-trip in all 36 placement-cells (observed
  ratios of batch-mean medians 1.19–2.39×). At the frozen `N=256, H=32` cell,
  the named-placement ratios are 1.71×, 2.39×, 2.06×, and 1.84× respectively.
  These placements establish mechanism feasibility and cross-provider
  direction, not a placement-population p-value, true invocation P99, or
  deployment-level significance.

These are pilot and framework-level results. They do **not** yet show an
optimized CPU hardware ceiling, end-to-end agent speedup, provider
generalization, energy savings, or a production arrival model. See
[`paper/claims-ledger.md`](paper/claims-ledger.md).

## Reproduce the analyses

Verify the frozen arXiv upload without cloud credentials or network access:

```bash
.venv/bin/python scripts/verify_arxiv_release.py
```

To regenerate the paper tables from the public trace source, fetch the 19
commit-pinned shards first. Existing files are never overwritten on a hash
mismatch.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,cloud]'
.venv/bin/python scripts/fetch_trace_source.py

.venv/bin/python scripts/analyze_compiler_matched.py
.venv/bin/python scripts/analyze_sub256.py
.venv/bin/python scripts/analyze_trace_ready_cohorts.py
.venv/bin/python scripts/analyze_trace_sliding_bound.py
.venv/bin/python scripts/analyze_trace_exact_packing.py
.venv/bin/python scripts/analyze_hardware_sweep.py
.venv/bin/python scripts/analyze_hardware_replication.py
.venv/bin/python scripts/analyze_native_dispatch_pilot.py --overwrite
.venv/bin/python scripts/analyze_resident_policy_pilot.py --overwrite

.venv/bin/python scripts/build_compiler_matched_notebook.py
.venv/bin/python scripts/build_sub256_notebook.py
.venv/bin/python scripts/build_trace_notebook.py
.venv/bin/python scripts/build_sliding_bound_notebook.py
.venv/bin/python scripts/build_hardware_sweep_notebook.py
.venv/bin/python scripts/build_hardware_replication_notebook.py
.venv/bin/python scripts/build_evidence_power_notebook.py
.venv/bin/python scripts/build_exact_native_notebook.py
.venv/bin/python scripts/build_resident_policy_notebook.py

.venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/04_sub256_crossover_refinement.ipynb \
  --output 04_sub256_crossover_refinement.ipynb \
  --output-dir notebooks --ExecutePreprocessor.timeout=300
```

The benchmark configurations live under `configs/`; frozen hypotheses and
exclusions live under `preregistration/`. Each run writes one append-only CSV
and one JSON manifest containing hardware, software, command, seed, and exact
configuration. Processed files contain source hashes and analysis parameters.

## Key artifacts

- [`notebooks/03_compiler_matched_analysis.ipynb`](notebooks/03_compiler_matched_analysis.ipynb)
  — full crossover atlas and correctness boundary.
- [`notebooks/04_sub256_crossover_refinement.ipynb`](notebooks/04_sub256_crossover_refinement.ipynb)
  — small-cohort hardware threshold.
- [`notebooks/02_trace_ready_cohort_analysis.ipynb`](notebooks/02_trace_ready_cohort_analysis.ipynb)
  — trace-conditioned fixed-window eligibility and regularity tax.
- [`notebooks/05_hardware_sweep_exploratory.ipynb`](notebooks/05_hardware_sweep_exploratory.ipynb)
  — cross-generation wall-time and provider-priced cost map.
- [`notebooks/06_sliding_deadline_bound.ipynb`](notebooks/06_sliding_deadline_bound.ipynb)
  — fixed-window result versus the general local sliding-deadline upper bound.
- [`notebooks/07_hardware_replication.ipynb`](notebooks/07_hardware_replication.ipynb)
  — three-placement confirmatory hardware and cost inversion.
- [`notebooks/08_evidence_and_power_audit.ipynb`](notebooks/08_evidence_and_power_audit.ipynb)
  — sampling-unit audit and placement-level power sensitivity.
- [`notebooks/09_exact_boundary_and_native_calibration.ipynb`](notebooks/09_exact_boundary_and_native_calibration.ipynb)
  — exact scheduling opportunity and five-placement native negative calibration.
- [`notebooks/10_resident_policy_pilot.ipynb`](notebooks/10_resident_policy_pilot.ipynb)
  — matched host-round-trip versus real device-resident decision epochs.
- [`paper/arxiv/main.pdf`](paper/arxiv/main.pdf) — governed 14-page arXiv
  manuscript.
- [`paper/governance/claim-evidence-map.csv`](paper/governance/claim-evidence-map.csv)
  — typed claim-to-artifact map for the manuscript.
- [`preregistration/online-route-runtime-001-Q.md`](preregistration/online-route-runtime-001-Q.md)
  — unfrozen, zero-spend qualification design for the missing finite online
  runtime; it does not authorize a cloud run.
- [`paper/formalism.md`](paper/formalism.md) — fixed-partition exactness,
  sliding-deadline packing optimum, and a valid general upper bound.
- [`preregistration/resident-policy-001.md`](preregistration/resident-policy-001.md)
  — frozen source, disclosed development smoke, mechanisms, hypotheses, and
  placement-level scale/stop rules for the decision-bearing pilot.
- [`docs/resident-policy-001-report-2026-08-12.md`](docs/resident-policy-001-report-2026-08-12.md)
  — four-placement result, provider lifecycle ledger, statistical restrictions,
  and the decision to move the next major spend to an online runtime.
- [`docs/resident-policy-002-design.md`](docs/resident-policy-002-design.md)
  — blocked unfrozen measurement-hardening draft retained as an audit record;
  it is not an executable runbook.
- [`docs/literature-map.md`](docs/literature-map.md) — nearest work and the
  defensible novelty boundary.
- [`docs/literature-audit-2026-08.md`](docs/literature-audit-2026-08.md) —
  collision-first primary-source audit and stop/demote decisions.
- [`docs/statistical-design.md`](docs/statistical-design.md) — frozen sampling
  units, effect thresholds, power targets, multiplicity, and stop rules.
- [`docs/research-decision-2026-08-11.md`](docs/research-decision-2026-08-11.md)
  — evidence cutoff, treatment decision, confirmatory provider blocks, and
  twelve-week kill/pivot gates.
- [`docs/three-month-program.md`](docs/three-month-program.md) — twelve-week
  implementation and confirmatory plan.
- [`docs/parallel-frontiers.md`](docs/parallel-frontiers.md) — ranked adjacent
  research lanes, falsification experiments, hardware gates, and publication
  boundaries.
- [`paper/outline.md`](paper/outline.md) — paper story, questions, figures, and
  baselines.

## Research rule

A GPU win is never assumed. Negative crossover regions, strict correctness
failures, cold compilation, host observation costs, route fragmentation, and
CPU-preferred regimes are part of the result. No failed row or timing outlier
is silently deleted.

## Citation

```bibtex
@misc{chen2026readycohortsboundinggpu,
  title         = {Ready Cohorts: Bounding GPU Opportunity and Avoiding Host Round Trips in LLM-Agent Control},
  author        = {Josef Liyanjun Chen},
  year          = {2026},
  eprint        = {2608.12123},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2608.12123}
}
```
