# Pilot 001 preregistration: the residency–regularity crossover

Status: frozen before benchmark execution on 2026-08-11.

## Question

At what population size, if any, does a commodity GPU execute a repeated
AI-agent control-plane transition faster than a desktop CPU, and how does that
crossover change when the host must observe every action?

This pilot is a harness and effect-size study. It is not the confirmatory study
for the paper.

## Unit of work

One **agent-step** performs the same tensor program on CPU and GPU:

1. score each agent from its dense state;
2. map the score to one of `A` action states;
3. gather the action-specific transition vector and cost;
4. update dense state and remaining budget.

Initialization, framework import, allocation, and CUDA context creation are
excluded. Repeated execution and required synchronization are included.

## Independent variables

- Population `N`: 256 through 1,048,576 on a logarithmic grid.
- State width `W`: 8 and 32 float32 values.
- Action-state count `A`: 1 and 8.
- Execution mode:
  - CPU with one physical thread;
  - CPU with eight physical threads;
  - eager GPU with resident state;
  - CUDA Graph replay with resident state;
  - eager GPU with the action vector copied to the host every step.

The host-visible condition is deliberately strict: it asks what happens when a
Python orchestrator must inspect every agent decision before the next step.

## Outcomes

Primary:

- median wall-clock nanoseconds per agent-step;
- crossover population relative to the eight-thread CPU baseline.

Secondary:

- median agent-steps per second;
- p10/p90 interval across fresh timing repetitions;
- CUDA-event time versus synchronized wall time;
- coefficient of variation;
- maximum CPU/GPU state error and exact action agreement.

No outlier is removed from the raw data. Any robust summaries or exclusions
added later must be reported as post hoc.

## Pilot hypotheses

- **H1 (resident crossover):** at least one resident GPU mode beats the
  eight-thread CPU at a tested population size.
- **H2 (residency penalty):** host-visible execution has a larger crossover
  population than eager resident execution, or never crosses in the tested
  range.
- **H3 (launch tax):** CUDA Graph replay reduces wall time relative to eager GPU
  execution in cells where median eager kernel sequences are short.
- **H4 (work intensity):** increasing `W` from 8 to 32 weakly lowers the
  crossover population for resident GPU execution.

These hypotheses are directional. Pilot confidence intervals are descriptive;
no confirmatory p-value will be presented.

## Validity and kill criteria

A cell is invalid, but retained with its error, if:

- action outputs disagree between CPU and GPU;
- state error exceeds both the absolute and relative tolerances;
- a CUDA error, allocation failure, or graph-capture failure occurs;
- elapsed time is non-positive.

The harness is revised before cloud scaling if:

- more than 5% of feasible cells are invalid;
- median within-cell timing CV exceeds 10%;
- CPU or GPU thermal throttling is observed;
- conclusions change when synchronized wall time replaces device-event time.

The research direction is **not** killed if H1 fails. It pivots to a negative
crossover atlas and investigates whether state bucketing, fusion, or persistent
kernels can move the boundary. The claim “GPU cores are cheap CPU cores” is
killed if no optimized GPU implementation wins on either synthetic kernels or
trace-replayed real agent workloads at practical swarm sizes.

## Preventing favorable benchmark selection

All configured cells are attempted in seeded randomized order. Raw rows,
failures, manifests, and analysis code are retained. Cloud hardware selection
will include at least one low-cost inference GPU and one high-bandwidth GPU;
prices will be captured at run time and separated from hardware-normalized
results.

