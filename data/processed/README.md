# Processed data

All files here are derived from append-only ledgers under `data/raw/` or from a
pinned public dataset. Analysis manifests record source paths, SHA-256 hashes,
parameters, and creation time.

## Compiler-matched atlas

- `compiler-matched-cell-summary.csv`: nine-repetition cell statistics.
- `compiler-matched-speedups.csv`: GPU/CPU ratios, bootstrap intervals, compile
  amortization, and stability flags.
- `compiler-matched-crossovers.csv`: smallest tested crossovers by hardware and
  workload stratum.
- `compiler-matched-shape-quality.csv`: strict correctness boundary.
- `compiler-matched-run-quality.csv`: completeness, duplicates, errors, and CV.
- `compiler-matched-analysis-manifest.json`: provenance and output hashes.

## Sub-256 refinement

- `sub256-cell-summary.csv`
- `sub256-speedups.csv`
- `sub256-crossovers.csv`
- `sub256-shape-quality.csv`
- `sub256-run-quality.csv`
- `sub256-host-penalty.csv`
- `sub256-temporal-fusion.csv`
- `sub256-analysis-manifest.json`

## Public trace features and replay

- `exgentic-tau2-span-features.parquet`: content-free derived timing/route
  features from the pinned tau2 panel.
- `exgentic-tau2-session-summary.csv`: one row per session.
- `exgentic-tau2-source-manifest.json`: dataset revision, license, excluded
  payload fields, completeness checks, and source hashes.
- `trace-ready-cohort-repetitions.csv`: every frozen replay repetition.
- `trace-ready-cohort-summary.csv`: aggregate eligibility by population,
  deadline, grouping, and threshold.
- `trace-ready-cohort-manifest.json`: simulation design and invariant checks.

## Sliding-deadline local bound

- `trace-sliding-local-bound-repetitions.csv`: repetition-level fixed-window
  eligibility and local sliding-deadline upper bounds.
- `trace-sliding-local-bound-summary.csv`: aggregate fixed-versus-local-bound
  comparison by population, deadline, grouping, and threshold.
- `trace-sliding-local-bound-manifest.json`: provenance, parameters, hashes,
  deterministic overlap checks, and frozen hypothesis outcomes.

## Exact sliding-deadline packing

- `trace-exact-packing-repetitions.csv`: all 540 preregistered replay cells
  evaluated by the exact equal-relative-deadline dynamic program.
- `trace-exact-packing-summary.csv`: repetition aggregates for the frozen
  window lower bound, exact optimum, local upper bound, and alignment-gap
  closure.
- `trace-exact-packing-manifest.json`: integer-clock contract, model scope,
  source/output hashes, and the six frozen hypothesis outcomes.

## Cross-generation hardware replication

- `hardware-replication-cell-summary.csv`: 30-repetition timing statistics for
  every placement/cell.
- `hardware-replication-speedups.csv`: matched CPU/GPU speedups.
- `hardware-replication-crossovers.csv`: tested crossover summaries.
- `hardware-replication-temporal-fusion.csv`: proportional fusion advantage.
- `hardware-replication-host-penalty.csv`: resident versus host-visible cost.
- `hardware-replication-placement-cost.csv`: placement-level frozen-price cost
  estimates.
- `hardware-replication-rank-n256-h64.csv`: primary-cell hardware ranking.
- `hardware-replication-shape-quality.csv` and
  `hardware-replication-run-quality.csv`: correctness and ledger QA.
- `hardware-replication-analysis-manifest.json`: source hashes, parameters,
  placement identities, prices, and preregistered hypothesis outcomes.

## Native dispatch calibration

- `native-dispatch-pilot-cell-summary.csv`: 240 placement-by-shape-by-mechanism
  summaries with median, P95, P99, and timing-scope metadata.
- `native-dispatch-pilot-contrasts.csv`: 60 placement-by-shape paired wall-time
  contrasts for nested-device versus host graph and ordinary host launch versus
  host graph. The placement is the analysis unit; the 50 rows within each cell
  are technical repetitions.
- `native-dispatch-pilot-manifest.json`: frozen-source semantics, source and
  provider receipt hashes, five-placement QA gates, result extrema, and explicit
  limitations. No timing-row p-values are reported.

## Device-resident policy pilot

- `resident-policy-pilot-cell-summary.csv`: placement-by-population-by-horizon
  mechanism medians, P95/P99 quantiles across batch-average rows, batch counts,
  and the number of deterministically validated invocations. The quantiles are
  not individual-invocation latency tails.
- `resident-policy-pilot-contrasts.csv`: placement-level host-round-trip over
  device-resident ratios, device-resident overhead above the no-decision oracle
  floor, and absolute wall time saved per decision epoch.
- `resident-policy-pilot-manifest.json`: frozen source/analysis hashes,
  per-placement raw/manifest hashes, provider-receipt/archive bindings,
  correctness gates, preregistered pilot outcomes, and explicit
  non-inferential caveats.

Processed tables are evidence, not hand-edited reporting surfaces. Regenerate
them through the corresponding script whenever source data or analysis changes.
