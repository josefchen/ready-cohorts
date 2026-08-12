# Resident policy 001: device-side decision epoch pilot

Status: **frozen at 2026-08-11T21:27:24Z**, after the explicitly disclosed
development smoke below and before the full local or any cloud timing run of
this source version.

## Question and claim boundary

Does keeping a GPU-computed route predicate on device and selecting a
pre-uploaded route graph there reduce steady-state decision-epoch wall time
relative to a matched implementation that returns only the 4-byte predicate to
the host, synchronizes, and selects the same route graph?

This is a mechanism pilot. It can establish that an actual device-resident
decision removes matched host control epochs. It cannot establish an
end-to-end agent-runtime speedup, a CPU-core displacement claim, broad provider
generalization, or a new CUDA graph mechanism.

## Frozen source and disclosed development evidence

- CUDA source:
  `native/resident_policy/resident_policy_pilot.cu`
- source SHA-256:
  `4b5cdcb9496a734bd7801d5c419efb8eceb72fd6962800520101e89676d204da`
- Makefile SHA-256:
  `d74935b594fb629b2113d237439289e057281becc561b6941d1134bd6a1c1351`
- local CUDA 13 binary SHA-256:
  `9c44a480e6d248c3c0d3175025e6133744f386966f3256327c6b7c61769518de`

Before this freeze, a bounded engineering smoke ran on the local GTX 1660 Ti
at `N={64,256}`, `H={2,4}`, two measured rows per mechanism/cell, and a 2 ms
minimum aggregate duration per row. It produced 24 status-`ok` rows, all with
field-exact final states and exact decision traces. Across the 24 technical
rows, the mechanism-level median invocation times were approximately 31.6 us
for the device-resident path, 49.3 us for host round-trip, and 7.5 us for the
no-decision floor. These values are allowed development evidence, not
inferential results. The archived smoke files and hashes are:

- `data/raw/resident-policy-dev-smoke-20260811T212632Z-p3952623.csv`:
  `4f1baedcf4527b3706797b9dc845c49e17dfe23cf0533f0c891e2f47bf09770b`;
- sibling manifest:
  `1956b49dd0e5f98c679f98819169919aebb59e17739562daf30a2e811f74aca3`.

No cloud run or full-grid timing was observed before this freeze. Any source
change receives a new source hash and experiment ID.

## Frozen mechanisms

All mechanisms use identical initialized state, device predicate kernels,
device route kernels, state layout, block size, and route sequence implied by
the state.

1. `host_roundtrip`: launch the GPU predicate graph, asynchronously copy one
   4-byte predicate to pinned host memory, synchronize, and launch the selected
   pre-instantiated route graph from the host at every epoch.
2. `device_resident`: launch one root graph from the host; a one-thread device
   selector reads the predicate and tail-launches one of two pre-instantiated,
   uploaded route graphs. The selected path computes the next predicate and
   continues for all epochs without returning the decision to the host.
3. `no_decision_lower_bound`: replay the host-oracle route sequence as one
   graph, omitting runtime predicate and selection. This is a structural floor,
   not a competing scheduler and not a legal online policy.

Graph construction, instantiation, and upload are outside steady-state timing
for all mechanisms. State reset, result copy, and validation are also outside
the timed region. The host predicate copy and synchronization are inside the
`host_roundtrip` timed region. CUDA event and host wall clocks are both
recorded; primary comparisons use wall time.

## Independent correctness contract

A separately written host-only oracle implements both route functions and the
global predicate without sharing the device transition functions. After every
invocation, the benchmark compares all four fields of every agent state and
the complete epoch-by-epoch decision trace. Checksums are secondary provenance,
not the definition of equality. Any unexplained state or decision mismatch,
illegal device launch, graph setup failure, OOM, crash, or nonpositive timing
is retained and blocks performance interpretation for that source version.

## Frozen first-stage layout

### Full local placement

- GPU: attached GTX 1660 Ti, one placement;
- agents `N`: 256, 2,048, 16,384;
- decision epochs `H`: 2, 8, 32;
- mechanisms: all three above;
- warmups: 5 per mechanism/cell;
- calibration samples: 3 per mechanism/cell;
- measured rows: 30 per mechanism/cell;
- minimum aggregate timed duration: 100 ms per row;
- maximum batch iterations: 20,000;
- block size: 256;
- seed: 20260811;
- deterministic mechanism shuffle within each cell/repetition.

Each cell calibrates one common initial batch count from the fastest observed
mechanism and applies that count to every mechanism. A row may extend beyond
that count only until the frozen 100 ms target is reached, subject to the common
safety cap. Report the realized batch count and whether the cap was reached.

### Bounded cloud portability stage

After the frozen source compiles in the provider image, run one fresh Modal L4
placement. If correctness and device-launch support pass, run one externally
managed replication on a named RunPod L4 or Lambda H100. Provider, region,
instance/pod ID, actual GPU name and UUID, CPU allocation, image/digest, driver,
CUDA versions, clocks/power limit when exposed, source/binary hashes, capacity
receipt, launch time, and every provisioning failure are required.

This first stage is not the six-placement nuisance pilot and has no performance
p-value. Fresh placement is the eventual sampling unit; measured rows and
batch invocations are technical repetitions.

## Outcomes and frozen contrasts

For every placement and `(N,H)` cell report status/failure counts, exactness,
batch iterations, median/P95/P99 wall time per invocation, and CUDA-event time.
Primary pilot contrast:

```text
speedup = median(host_roundtrip wall time) /
          median(device_resident wall time)
```

Secondary diagnostics are device-resident overhead above the no-decision floor,
the absolute time saved per epoch, and scaling with `H`. No technical-row
significance test is permitted. The directional engineering hypotheses are:

- `R1`: the device path is faster than host round-trip in the primary
  threshold-adjacent cell `N=256, H=32`;
- `R2`: the device advantage increases with the number of decision epochs at
  fixed `N`;
- `R3`: the device path remains slower than the no-decision floor;
- `R4`: every measured invocation is field- and decision-exact.

`R1--R3` are pilot magnitude/direction checks. `R4` is a validity gate.

## Scale/stop rule

Do not enter confirmation from one favorable placement. Run six fresh pilot
placements in each intended L4 and H100 stratum, in balanced randomized method
order, to estimate nuisance variance. A stratum enters the separately frozen
confirmation only if the implementation has zero unexplained correctness
failures and improves P99 latency or CPU core-seconds by at least 10% over the
strongest legal tuned baseline. The default confirmatory target is 30 analyzed
placements per retained hardware stratum, increased if a blinded 10,000-study
power simulation using the upper nuisance-variance bound requires it. The
confirmatory smallest worthwhile performance effect is 15%, with 90% power and
one-sided alpha 0.025.

If the device mechanism loses, that is a measured boundary. Do not change the
primary cell, lower the effect threshold, add within-process repetitions, or
replace a provider after viewing outcomes under this experiment ID.

## Known limitations

- The graph topology is host-created, fixed, and uploaded before timing.
- The predicate makes one global binary decision over a synthetic regular state
  array; this is not per-agent route compaction or a complete ready-cohort
  runtime.
- The state layout is array-of-structures rather than the final SoA engine.
- The no-decision mechanism has oracle route knowledge and intentionally omits
  real work.
- Tool/model I/O, authority checks, queueing, deadlines, isolation, failures,
  CPU fallback, power, and end-to-end task utility are outside this pilot.
- CPU affinity, NUMA placement, and accelerator clock control are not yet
  enforced.
- Device graph launch requires platform support, unified addressing, and
  uploaded device-launchable child graphs; unsupported systems remain in the
  ledger rather than being silently excluded.
