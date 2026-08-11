# Pilot 003/004 preregistration: compiler-matched residency horizons

Status: frozen before systematic benchmark execution on 2026-08-11.

Two short feasibility compilations (`N=4,096`, `W=8`, `A=8`) were run before
freezing this document. They established only that `torch.compile` works on the
local CPU and GPU. Their timings are not part of the analysis.

## Question

Does the GPU crossover survive when the CPU and GPU execute the same
whole-rollout program through the same PyTorch compiler interface, and how much
does host observation frequency move that crossover?

This experiment addresses the largest validity limitation in pilots 001/002:
CUDA Graph replay removed repeated Python dispatch only for the GPU.

## Design

One timed repetition always advances every agent by 64 transitions. The same
pure tensor program is compiled with `torch.compile(fullgraph=True,
dynamic=False)` for CPU and CUDA. Two observation horizons are tested:

- `H=1`: 64 calls to a compiled one-step transition;
- `H=64`: one call to a compiled 64-step rollout.

For CUDA, each horizon has a resident condition and a host-visible condition.
The host-visible path copies the final action vector to CPU after every compiled
call: 64 copies at `H=1`, one copy at `H=64`. CPU execution uses the same
compiled program at one and eight threads, with no device transfer.

The factorial is:

- hardware: local GTX 1660 Ti and an ephemeral Modal L4;
- population `N`: 256, 4,096, 65,536, and 1,048,576;
- state width `W`: 8 and 32 float32 values;
- action-state count `A`: 1 and 8;
- observation horizon `H`: 1 and 64;
- CPU threads: 1 and 8;
- GPU visibility: resident and host-visible;
- nine fresh timing repetitions per cell after five untimed warm-up rollouts.

Case order is seeded and randomized. Static-shape compilation is performed once
per `(device, N, W, A, H)` within a run and excluded from steady-state timing.
Cold first-call compile latency is retained as a secondary amortization metric.

## Outcomes

Primary:

- synchronized wall-clock nanoseconds per agent-step;
- GPU speedup over the matched eight-thread compiled CPU cell;
- smallest tested crossover population in each `(W, A, H, visibility)` stratum.

Secondary:

- host-visibility penalty at each horizon;
- `H=64` versus `H=1` dispatch-amortization ratio;
- CUDA-event versus synchronized wall time;
- cold first-call compile latency;
- within-cell coefficient of variation.

No timing outlier is removed. Failed cells remain in the raw ledger and are
excluded from primary crossover summaries.

## Hypotheses

- **H1 (hardware survival):** at `H=64`, compiled resident GPU execution beats
  the matched compiled eight-thread CPU in at least one valid stratum.
- **H2 (observation frontier):** host-visible `H=1` has a weakly larger
  crossover population than resident `H=1` in every stratum where either
  crosses.
- **H3 (amortization):** increasing the horizon from 1 to 64 improves GPU
  throughput more than CPU throughput at small and medium populations.
- **H4 (cross-hardware direction):** the sign of H1 and H2 replicates on the
  GTX 1660 Ti and L4, although crossover locations may differ.

These remain pilot hypotheses; effect sizes and uncertainty are descriptive.

## Correctness and exclusions

Before timing each full-shape program, compiled CPU and compiled CUDA outputs
from identical initialization are compared after all 64 transitions. Validity
requires exact final actions and `torch.allclose` for state and budget with
`atol=1e-4`, `rtol=1e-5`. The absolute tolerance was selected before this run
from the independent pilots 001/002, where the worst three-step state
difference was `3.5e-5`; it is not tuned using pilots 003/004.

A cell is invalid but retained if correctness fails, compilation or execution
raises, elapsed time is non-positive, or the configured step count is not
divisible by the observation horizon.

The hardware claim is not supported if a resident compiled GPU never beats the
compiled eight-thread CPU. The broad research program can still report a
negative crossover atlas or pivot to data-layout and persistent-kernel studies.

## Interpretation boundary

This remains a regular, dense synthetic control transition. A positive result
supports a hardware/runtime crossover under matched compilation; it does not
establish usefulness for heterogeneous, tool-blocked, or communication-heavy
agent workloads. Confirmatory work must replay traces from real agent systems
and report cloud price, power, and end-to-end latency.

## Pre-data harness amendment

At 2026-08-11 15:45 UTC, the first smoke run hit PyTorch's default limit of
eight static-shape recompilations after its first valid cells. Before either
systematic run, `recompile_limit=512` was added to permit the preregistered
static-shape factorial. The failed smoke ledger remains under `data/smoke/` and
is not analyzed. No population-scale pilot 003/004 timing had been run or
inspected when this implementation-only amendment was made.

A second smoke run showed an approximately 60 ms one-time runtime
initialization excursion after one warm-up in the newly compiled `H=4` paths.
A five-warm-up smoke rerun removed the excursion in all 16 cells. At 15:47 UTC,
before any systematic cell was run, both configs were therefore changed from
two to five warm-up rollouts. All warm-ups remain outside measured intervals.
