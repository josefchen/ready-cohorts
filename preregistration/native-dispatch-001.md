# Native dispatch 001 preregistration: mechanism feasibility and overhead

Status: frozen at **2026-08-11T20:57:19Z** before running the hardened source
locally or on a cloud GPU.

## Scope and prior development evidence

This is an engineering/mechanism pilot, not a confirmatory performance study.
Before this freeze, an earlier checksum-only build ran a two-repetition smoke
on one GTX 1660 Ti. All four paths executed, but nested device launch was
approximately 1.4--2.1x the host-graph wall time in those tiny cells. That
outcome is allowed design evidence and makes a fixed nested graph an overhead
calibration, not a candidate speedup claim.

The hardened source independently implements the host reference transition
and compares every final state field, while retaining checksums for provenance:

- CUDA source SHA-256:
  `a5c1f4a349075b6e76116c4f52163b488ccdae32a7bf66e95fed752e363d3ac6`;
- Makefile SHA-256:
  `9653361dcfa4d1e8681a8fa9d89ba9ba22f050575705b75ec1453fbdd7924df7`;
- Modal runner SHA-256:
  `e128e87e1bafb13cf08812f368b1b31e16fb25f5da23fee5d64bd8bfcb7939cf`.

The source compiled under CUDA 13.0 (`nvcc 13.0.88`) before this freeze. A
compile is not a timing or correctness outcome.

## Mechanisms

For identical initialized states and transition counts:

1. optimized single-threaded C++ (`cpu_cpp`);
2. one ordinary host-issued CUDA kernel per transition
   (`cuda_host_launch`);
3. one host replay of a pre-instantiated CUDA Graph containing the transition
   chain (`cuda_host_graph`);
4. one host replay of a parent graph whose launcher kernel starts the same
   pre-uploaded child graph from the device (`cuda_device_graph`).

The fourth mechanism still performs one host graph launch and adds a launcher
kernel. It does not eliminate a host decision in this workload and has no
structural reason to beat mechanism 3. Its purpose is to establish device-graph
support and measure the incremental nested-launch cost. It cannot substantiate
a “GPU orchestration speedup” claim.

## Frozen local development run

- hardware: the currently attached GTX 1660 Ti, one placement;
- agents: 32, 256, 2,048, and 16,384;
- sequential transitions: 1, 8, and 64;
- warmups: 10 per mechanism/cell;
- measured repetitions: 50 per mechanism/cell;
- CUDA block size: 256;
- root seed: 20260811;
- mechanism order: deterministic shuffle within repetition;
- reset and synchronize before timing every GPU repetition;
- allocation, graph creation/instantiation/upload, reset, result copy, and
  field-by-field comparison are outside steady-state timing and remain
  documented separately;
- no timing row is deleted.

This run supplies compatibility, order-noise, and magnitude information only.
Its independent placement count is one.

## Frozen cloud pilot layout

After local and remote compile checks pass, request two fresh placements per
available `(provider, actual GPU SKU)` cell across Modal, Lambda Cloud, and
RunPod. H100 is the cross-provider anchor. L4 and A10 are distinct cheap-card
strata and must not be pooled as one GPU type. Every launch binds to an
immediately preceding timestamped capacity/price inventory receipt.

Cloud pilots use the same grid and within-placement randomized layout. Provider,
region, instance/pod ID, exposed host ID, GPU UUID/PCI ID, exact SKU, driver,
CUDA versions, image identifier/digest, binary/source hashes, clocks, power
limit, and launch timestamps are required. Capacity failures and unsupported
device graphs remain in the launch ledger.

No performance p-value will be reported for this pilot. Its six-or-more H100
placements estimate nuisance variance for a later 90%-powered design. Provider
is a fixed blocking factor; placements from different providers are not assumed
exchangeable merely because the marketing SKU says H100.

## Outcomes and validity gates

For every mechanism/cell/placement report median, P95, P99, full repetition
distribution, randomized order, wall time, CUDA-event time, and failures.
Primary pilot contrasts are:

- host graph versus ordinary host launches;
- nested device graph versus host graph (incremental overhead);
- each GPU path versus the current C++ reference, explicitly labelled an
  untuned CPU pilot baseline.

Validity gates:

- every available measured path must match every final state field from the
  separately implemented host reference;
- any mismatch, illegal device launch, impossible timestamp, OOM, or crash is
  retained and stops that source version from performance interpretation;
- a graph reported unsupported is a platform boundary, not a removed row;
- checksums alone are insufficient for `exact_match`;
- technical repetitions do not count as independent placements.

## Decision rules

- Do not scale the fixed nested graph as the paper treatment merely because a
  noisy cell happens to beat the host graph.
- Proceed to a device-resident runtime only after adding a real on-device
  decision that eliminates a matched host synchronize/copy/dispatch epoch:
  route selection among uploaded children, or multiple decision epochs chained
  on device.
- The next treatment enters confirmation only if six fresh pilot placements in
  each intended primary stratum show at least a 10% improvement in either P99
  latency or CPU core-seconds over the strongest legal tuned baseline, with no
  correctness failure.
- The later confirmatory target is 30 fresh placements per primary hardware
  stratum, adjusted upward by blinded nuisance-variance simulation for 90%
  power. The native speedup SESOI is 15%; CPU displacement SESOI is 25%.
- Do not increase repetitions or swap the primary cell in response to a weak
  effect. A changed mechanism receives a new experiment ID and preregistration.

## Known limitations

This transition is dense, regular, integer-only, and independent across agents.
The CPU path lacks `-march=native`, SoA/SIMD, thread-pool, affinity, and NUMA
tuning. Graph setup and cold start are excluded. CUDA event values have less
resolution than their integer-nanosecond CSV representation. There is no sparse
readiness, divergent route body, external I/O, queue contention, power sampling,
or end-to-end agent utility measurement. Those limitations prevent a headline
claim regardless of the pilot timings.
