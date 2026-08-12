# Resident policy native pilot

This benchmark isolates one real device-resident orchestration decision. A GPU
predicate chooses between two route graphs across multiple epochs.

- `host_roundtrip` copies the 4-byte predicate to pinned host memory,
  synchronizes, and launches the selected route graph from the host.
- `device_resident` keeps the predicate on GPU and uses a one-thread selector to
  tail-launch the selected uploaded graph.
- `no_decision_lower_bound` replays the oracle path without predicate or
  selection work; it is a structural floor, not an online scheduler.

All mechanisms use the same device predicate and route kernels. A separately
written host-only oracle validates every state field and the complete decision
trace after every invocation. Graph construction/upload, reset, result copy,
and validation are outside steady-state timing; the host predicate copy and
synchronization are inside the matched host timing.

The frozen experiment contract, source hashes, disclosed development smoke,
grid, and scale/stop rules are in
[`preregistration/resident-policy-001.md`](../../preregistration/resident-policy-001.md).

## Build

The binary requires relocatable device code and the CUDA device runtime because
child graphs are launched from device code.

```bash
make -C native/resident_policy compile-smoke NVCC=/path/to/cuda/bin/nvcc
```

The Makefile embeds the CUDA-source SHA-256, git revision, and dirty-state flag
in the binary. A portable fat binary targets SM 75, 80, 86, 89, and 90, with
PTX for compute 90.

## Bounded smoke

```bash
make -C native/resident_policy smoke NVCC=/path/to/cuda/bin/nvcc
```

Or run an explicit append-only experiment:

```bash
SOURCE_SHA256=<frozen-source-hash> \
BINARY_SHA256=<compiled-binary-hash> \
EXECUTION_PROVIDER=local \
REQUESTED_GPU=local \
PLACEMENT_ID=<fresh-placement-id> \
native/resident_policy/build/resident_policy_pilot \
  --experiment-id resident-policy-example \
  --output-dir data/raw \
  --agents 256,2048,16384 \
  --epochs 2,8,32 \
  --warmups 5 \
  --calibration-samples 3 \
  --repetitions 30 \
  --min-duration-ms 100 \
  --max-batch 20000
```

Every run writes a unique CSV and sibling JSON manifest. Existing artifacts are
never overwritten.

## Scope

This is a global binary-policy mechanism pilot over synthetic regular state.
It is not yet the route-compacting, deadline-aware ready-cohort runtime. Device
launch support requires unified addressing and pre-uploaded device-launchable
graphs. Fresh GPU placement—not CSV row or batch invocation—is the sampling
unit for any deployment-level performance claim.
