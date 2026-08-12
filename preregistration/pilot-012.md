# Pilots 012–029 preregistration: independent-placement hardware replication

Status: frozen before benchmark execution on 2026-08-11.

## Motivation

Pilots 006–011 produced a complete cross-generation Modal sweep with no
execution or correctness failures, but within-cell wall-time dispersion was
substantially higher on several GPU classes than in the local and L4 pilots.
The exploratory result is also non-monotonic: the A100-80GB, H100, and L40S do
not beat their colocated tuned CPU in the regular one-step regime by the
largest tested population, while every GPU wins at the smallest tested
population after fusing 64 transitions. A fresh-container replication is
required before treating either result as a hardware property.

## Frozen design

- provider: Modal;
- requested GPUs: T4, L4, A10, L40S, A100-80GB, and exact H100 (`H100!`);
- independent placements: three single-use containers per requested GPU;
- Modal CPU request: eight physical cores per container;
- population `N`: 8, 16, 32, 64, 128, and 256;
- state width `W`: 8 float32 values;
- action-state count `A`: 1;
- total transitions per timed repetition: 64;
- observation horizon `H`: 1 and 64;
- CPU threads: 1 and 8;
- GPU visibility: resident only;
- 30 timed repetitions after 20 untimed warm-up rollouts;
- matched `torch.compile(fullgraph=True, dynamic=False)` CPU/CUDA programs;
- seeded randomized case order, using a distinct seed for each placement;
- one append-only CSV and one environment manifest per placement.

The frozen workload and numerical-validity rules are unchanged from pilots
005–011. The larger repetition count addresses short-timescale dispersion;
three fresh single-use containers address placement-level variation.

## Primary estimands

For each GPU, horizon, and population:

1. the median within-placement GPU wall time;
2. the median of the three placement medians, with the full placement range;
3. speedup over the tuned colocated CPU within each placement;
4. the fraction of placements with median speedup above one;
5. synchronized CUDA-event time and the wall/device ratio;
6. time-stamped GPU-only cost per billion agent transitions.

No timing observation is deleted. Per-placement medians are the unit of
cross-placement inference, so a noisy provider placement cannot dominate by
contributing more individual repetitions.

## Confirmatory hypotheses

- **R1:** every GPU has resident `H=64` median speedup above one at `N=8` in
  all three fresh placements;
- **R2:** temporal fusion gives every GPU a proportional advantage over its
  colocated tuned CPU in every placement and population;
- **R3:** at `N=256,H=64`, at least one GPU priced below H100 has lower
  GPU-only cost per billion transitions in all three placements;
- **R4:** the rank correlation between Modal price and absolute resident GPU
  wall time at `N=256,H=64` is below 0.8 after aggregation by placement;
- **R5:** the exploratory one-step conclusion replicates directionally: at
  least two of A100-80GB, H100, and L40S fail to exceed tuned-CPU speedup one
  at `N=256` in at least two of three fresh placements.

R4 is explicitly a provider-and-date-specific economic result. R5 does not
require every card to reproduce an exact crossover because colocated CPU host
models and cloud placement are not controlled.

## Exclusions and limits

Failed and nonpositive observations remain in the raw ledger and are excluded
from timing summaries. A workload shape is timing-valid only when its exact
action outputs match and its state/budget tensors satisfy `atol=1e-4`,
`rtol=1e-5`. Compilation and allocation are excluded from steady-state timing
but retained as first-call outcomes. These results characterize PyTorch
compiler execution on Modal, not custom CUDA kernels, bare-metal power-normalized
hardware, or an optimized native CPU implementation.
