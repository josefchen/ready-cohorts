# Initial literature map

## Closest systems work

- FLAME GPU establishes that homogeneous, state-bucketed agent-based
  simulations can scale extremely well on GPUs. It does not study LLM-agent
  control traces, tool interruptions, or the CPU/GPU crossover under cloud
  prices.
- INFERCEPT, Continuum, and Kairos optimize LLM serving around tool pauses and
  workflow scheduling. They optimize the neural inference plane, whereas this
  project isolates the non-neural agent control plane.
- CUDA Graph work establishes that repeated small kernels can become
  launch-bound and that graph replay can reduce host submission cost.
- GPUTOK shows that a formerly CPU-side serving stage can cross over to GPU for
  sufficiently long/batched text. It motivates measuring rather than assuming
  the same for agent orchestration.
- InferScale keeps reusable agent memory state on GPU. It is evidence that
  residency is an emerging systems lever, but it targets KV retrieval and
  injection rather than general control transitions.

## Claimed gap to test

There is no identified trace-driven crossover atlas answering when an AI-agent
runtime should execute non-neural state transitions on CPU, as eager GPU work,
or as a resident captured GPU graph. Existing GPU agent-based simulation
results are not enough: LLM agents are asynchronous, tool-interrupted, ragged,
and usually orchestrated by Python.

## Candidate paper contribution

1. A simple crossover model:

   `N * (t_cpu - t_gpu) > launch + synchronization + transferred_bytes / bandwidth`.

2. A benchmark built from synthetic controls plus traces from open agent
   frameworks and tool-use benchmarks.
3. A residency–regularity phase diagram across consumer and datacenter GPUs.
4. A state-bucketing or captured-graph intervention that moves one measured
   boundary.
5. Cost, energy, latency, correctness, and downstream task-utility reporting.

