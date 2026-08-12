# Pilots 005/006 preregistration: sub-256 crossover refinement

Status: frozen before benchmark execution on 2026-08-11.

## Motivation and question

Pilots 003/004 used `N=256` as their smallest population. In the regular
`W=8, A=1` stratum, resident `H=64` GPU execution already beat the faster of
the one- and eight-thread compiled CPU cells at that boundary on both the GTX
1660 Ti and Modal L4. The experiment therefore established only that the
crossover is at or below 256.

This refinement asks where the crossover lies between 8 and 256 agents and
whether host visibility moves it. It is intentionally narrow: locating the
best-case regular-kernel threshold is more useful for joining hardware results
to the preregistered ready-cohort ceiling than rerunning the full wide/branchy
factorial.

## Design

- hardware: local GTX 1660 Ti and an ephemeral Modal L4;
- population `N`: 8, 16, 32, 64, 128, and 256;
- state width `W`: 8 float32 values;
- action-state count `A`: 1;
- total transitions per timed repetition: 64;
- observation horizon `H`: 1 and 64;
- CPU threads: 1 and 8;
- GPU visibility: resident and host-visible;
- nine repetitions after five untimed warm-ups;
- matched `torch.compile(fullgraph=True, dynamic=False)` CPU/CUDA programs;
- root seed: 20260811, with seeded randomized case order.

The primary CPU comparator is the faster median of the one- and eight-thread
compiled cells at each shape. The frozen eight-thread comparator from pilots
003/004 remains a secondary continuity result. The primary crossover is the
smallest tested population whose median synchronized GPU wall time is below
the tuned CPU median.

## Outcomes and hypotheses

Primary:

- resident and host-visible speedup over the tuned compiled CPU;
- smallest tested crossover population for each `(hardware, H, visibility)`;
- percentile-bootstrap interval for the ratio of medians.

Secondary:

- host-visibility penalty;
- `H=1` versus `H=64` temporal-fusion gain;
- cold first-call compile latency and steady-state break-even rollouts;
- within-cell coefficient of variation.

Hypotheses:

- **S1:** resident `H=64` crosses by `N=128` on both GPUs;
- **S2:** resident `H=1` crosses by `N=256` on L4 but not necessarily on the
  GTX 1660 Ti;
- **S3:** the host-visible crossover is weakly no smaller than the resident
  crossover in every comparable stratum;
- **S4:** `H=64` improves GPU wall time proportionally more than tuned CPU wall
  time at every population.

## Correctness and exclusions

The same strict rule as pilots 003/004 applies after all 64 transitions: exact
final actions and `torch.allclose` for state and budget with `atol=1e-4` and
`rtol=1e-5`. No timing outlier is removed. Failed or nonpositive cells remain
in the raw ledger and are excluded from crossover summaries. Compilation,
allocation, and initialization are excluded from steady-state timings but
retained in the manifest and compile-latency outcome.

These are framework-level PyTorch compiler results, not an optimized C++ CPU
ceiling or a custom CUDA-kernel result. A crossover below 256 narrows a
candidate batching threshold; it does not prove that real asynchronous agent
routes can assemble cohorts that large within their latency budget.
