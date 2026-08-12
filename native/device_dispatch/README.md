# Native CUDA device-dispatch pilot

This directory contains a bounded microbenchmark for a deterministic,
agent-like finite-state transition. It measures four execution mechanisms:

1. optimized single-threaded C++ (`cpu_cpp`);
2. one ordinary host-issued CUDA kernel launch per transition
   (`cuda_host_launch`);
3. one host replay of a CUDA Graph containing the same transition kernels
   (`cuda_host_graph`); and
4. a host-launched parent graph whose one launcher kernel starts the transition
   graph from the device (`cuda_device_graph`).

The fourth path follows NVIDIA's device-graph execution model rather than
calling an ordinary fused kernel a scheduler. Device graphs require unified
addressing and a compatible driver/GPU/runtime combination. An unsupported or
failed device-graph setup is retained as a structured CSV failure row; it is
never silently excluded.

## Correctness and timing scope

Every measured repetition starts from the same state and must match an
independently computed C++ oracle bit-for-bit. The CSV includes both expected
and observed 64-bit checksums and an `exact_match` field.

- `wall_ns` spans dispatch through completion synchronization, excluding state
  reset, device-to-host result copy, and checksum calculation.
- `device_ns` uses CUDA events and is blank for the CPU baseline.
- The nested device-graph event covers the complete parent execution
  environment, including its fire-and-forget child, as defined by CUDA's graph
  execution-environment semantics.

Warmups and repetitions are recorded in the manifest. Measured mechanisms are
deterministically shuffled within every repetition to reduce monotonic thermal
and clock-order bias. This is still a pilot: it does not pin CPU affinity, lock
GPU clocks, or claim paper-grade causal estimates.

## Local build and smoke test

With CUDA 13 and `nvcc` installed:

```bash
make -C native/device_dispatch compile-smoke
make -C native/device_dispatch smoke
```

`compile-smoke` only runs `--help`, so it does not require a GPU. `smoke` uses
two tiny shapes, one warmup, and two measured repetitions.

For a bounded local measurement:

```bash
native/device_dispatch/build/device_dispatch_pilot \
  --experiment-id local-device-dispatch \
  --agents 32,256,2048 \
  --steps 1,8,64 \
  --warmups 10 \
  --repetitions 50 \
  --output-dir data/raw
```

Each invocation creates a new timestamp-and-process-qualified CSV and JSON
manifest. The CSV writer opens in append mode, writes each row immediately,
and flushes it before continuing. Existing local downloads are never
overwritten by the Modal runner.

## Modal paths

The runner uses `nvidia/cuda:13.0.1-devel-ubuntu24.04`; local `nvcc` is not
required. Its default action is a zero-credit plan printout:

```bash
modal run scripts/modal_device_dispatch_pilot.py
```

Compile remotely on a CPU worker (this consumes a small amount of cloud
compute, but allocates no GPU):

```bash
modal run scripts/modal_device_dispatch_pilot.py --action compile
```

Run the bounded L4 pilot only with the explicit spend acknowledgement:

```bash
modal run scripts/modal_device_dispatch_pilot.py \
  --action run --confirm-spend \
  --output-dir data/raw
```

No cloud function is invoked at module import time or by the default `plan`
action.

## Important limitations

- A device graph must still be created, instantiated, and uploaded by the
  host. This pilot measures device-side launch of a fixed graph, not a GPU
  creating arbitrary new graph topology.
- Graph topology and pointers are fixed per `(agents, steps)` cell.
- All transitions are independent across agents and have no external side
  effects. Real tool I/O needs a mailbox/authority boundary and is out of scope
  for this pilot.
- CPU and GPU clocks, NUMA placement, host virtualization, and background load
  remain provider-controlled. Placement replication is necessary before any
  inferential claim.
