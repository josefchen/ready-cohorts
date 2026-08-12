# Pilots 007–011 preregistration: cross-generation small-cohort sweep

Status: frozen before benchmark execution on 2026-08-11.

## Motivation and question

Pilots 005/006 show that the regular resident one-step kernel crosses the
framework-matched tuned CPU at 16 agents on an NVIDIA L4 but not by 256 on a
GTX 1660 Ti; a fused 64-transition rollout crosses at or below 8 agents on
both. The result raises a hardware-selection question that peak-FLOP tables do
not answer: **do more expensive datacenter GPUs lower the crossover or reduce
cost per agent transition in a launch-dominated control workload?**

This sweep tests five additional Modal GPU classes spanning Turing, Ampere,
Ada/Lovelace, and Hopper. It reuses the already-frozen regular-kernel design so
hardware is the only intended change.

## Design

- provider: Modal, one single-use container per GPU class;
- requested GPUs: T4, A10, L40S, A100-80GB, and exact H100 (`H100!`, no H200
  automatic upgrade);
- Modal CPU request: 8 physical cores for every container;
- population `N`: 8, 16, 32, 64, 128, and 256;
- state width `W`: 8 float32 values;
- action-state count `A`: 1;
- total transitions per timed repetition: 64;
- observation horizon `H`: 1 and 64;
- CPU threads: 1 and 8;
- GPU visibility: resident and host-visible;
- nine repetitions after five untimed warm-ups;
- matched `torch.compile(fullgraph=True, dynamic=False)` CPU/CUDA programs;
- root seed: 20260811, with seeded randomized case order;
- one append-only raw CSV and one manifest per GPU class.

The primary CPU comparator is the faster median of one- and eight-thread
compiled CPU execution on the colocated host. Absolute GPU wall time and
agent-step throughput are the primary cross-card comparators because Modal may
place different GPU classes on different CPU host models.

## Outcomes and hypotheses

Primary:

- resident and host-visible speedup over the colocated tuned CPU;
- smallest tested median crossover and smallest bootstrap-supported crossover
  for each `(GPU, H, visibility)`;
- synchronized GPU wall time and agent steps per second;
- time-stamped GPU-only marginal cost per billion agent transitions using the
  official Modal per-second price captured separately from benchmark data.

Secondary:

- host-visibility penalty;
- temporal-fusion advantage relative to the tuned CPU;
- first-call compilation latency;
- within-cell coefficient of variation.

Hypotheses:

- **G1:** every datacenter GPU has a resident `H=64` crossover at or below the
  smallest tested population, `N=8`;
- **G2:** every datacenter GPU has a resident `H=1` crossover by `N=256`;
- **G3:** `H=64` improves GPU wall time proportionally more than tuned CPU wall
  time at every population;
- **G4:** GPU price rank will not equal throughput rank in the small regular
  regime; at least one cheaper GPU has lower GPU-only cost per billion agent
  transitions than H100;
- **G5:** host-visible crossover is weakly no smaller than resident crossover
  for every GPU and horizon.

G4 is evaluated against a frozen time-stamped price table and is explicitly a
provider-specific economic result, not a timeless hardware property.

## Correctness and exclusions

The strict pilots 003–006 rule applies after all 64 transitions: exact final
actions and `torch.allclose` for state and budget with `atol=1e-4` and
`rtol=1e-5`. No timing outlier is removed. Failed or nonpositive cells remain
in the raw ledger and are excluded from crossover summaries. Compilation,
allocation, and initialization are excluded from steady-state timings but
retained in the manifest and compile-latency outcome.

These are framework/compiler measurements, not an optimized C++ CPU ceiling
or custom CUDA-kernel ceiling. Modal host identity, power, and utilization are
not assumed to be controlled. The sweep is an exploratory hardware map whose
role is to choose the confirmatory hardware set and reveal negative regimes.
