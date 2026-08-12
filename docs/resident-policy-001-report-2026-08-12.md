# Resident-policy-001 four-placement report

Evidence cutoff: **2026-08-11 22:51 UTC**. This report was written after all
four full-grid artifacts were collected and after the RunPod and Lambda
resources were confirmed absent. It is a descriptive mechanism-pilot report,
not a confirmatory analysis.

## Result first

Keeping the GPU-computed binary route decision on device was faster than
copying the same 4-byte decision to the host, synchronizing, and dispatching
the same pre-instantiated route body on every one of the 36 observed
placement-by-`(N,H)` cells.

- Four named placements: local GTX 1660 Ti, Modal L4, RunPod L4, and Lambda
  H100 SXM5.
- 3,240/3,240 measured rows passed the frozen correctness and duration gates.
- 21,974,573 batched invocations received exact state validation.
- The observed host/device ratio of within-placement batch-mean medians ranged
  from 1.1939 to 2.3877.
- At the frozen primary cell `N=256,H=32`, the named-placement ratios were
  1.7111, 2.3877, 2.0642, and 1.8366.
- No performance p-value is reported. Four placements across three hardware
  strata do not support a placement-population claim.

## Frozen treatment

- preregistration: [`preregistration/resident-policy-001.md`](../preregistration/resident-policy-001.md),
  SHA-256 `2f4da9a34135f4b3e83a0a1b25abe010e193dc3b5a30e69f2a6c937eaabf83f3`;
- CUDA source: [`native/resident_policy/resident_policy_pilot.cu`](../native/resident_policy/resident_policy_pilot.cu),
  SHA-256 `4b5cdcb9496a734bd7801d5c419efb8eceb72fd6962800520101e89676d204da`;
- Makefile SHA-256:
  `d74935b594fb629b2113d237439289e057281becc561b6941d1134bd6a1c1351`;
- full grid: `N={256,2048,16384}`, `H={2,8,32}`, three mechanisms,
  30 randomized-order measured rows per mechanism/cell, and at least 100 ms
  aggregate timed work per row.

The primary effect is:

```text
median(host_roundtrip batch-average wall time)
------------------------------------------------
median(device_resident batch-average wall time)
```

Graph creation/upload, state reset, result copy, and validation are outside
the timed interval. The host predicate copy and synchronization are inside the
host interval. The no-decision mechanism is an oracle floor and not a legal
online baseline.

## Placement results

All times below are medians of batch-average invocation durations. They are not
individual-invocation P99 values.

| Provider | Actual GPU | Speedup range over 9 cells | Primary speedup | Primary resident (us) | Primary host (us) | Primary oracle floor (us) | Derived host time saved per epoch (us) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local | NVIDIA GeForce GTX 1660 Ti | 1.1939--1.7111 | 1.7111 | 273.018 | 467.168 | 33.413 | 6.067 |
| Modal | NVIDIA L4 | 1.5855--2.3877 | 2.3877 | 261.894 | 625.311 | 39.703 | 11.357 |
| RunPod | NVIDIA L4 | 1.4807--2.0642 | 2.0642 | 258.340 | 533.269 | 37.345 | 8.592 |
| Lambda | NVIDIA H100 80GB HBM3 | 1.3426--1.8366 | 1.8366 | 308.999 | 567.492 | 45.592 | 8.078 |

The same-SKU L4 replication is directionally consistent across Modal and
RunPod. The H100 result extends mechanism feasibility to compute capability
9.0; it does not imply that L4 is generally faster than H100 because provider
host, clock, image, and placement effects are not controlled across these four
observations.

## Validity and evidence binding

The analysis requires:

- the exact frozen full configuration and source hash;
- one complete row identity for every cell/mechanism/repetition;
- status `ok`, clear error fields, positive finite timings, duration target,
  batch cap, and wall/device division consistency;
- exact state fields/checksums and observed decision traces for the legal host
  and device mechanisms;
- one GPU, unified addressing, CUDA 13 compile/runtime, requested/actual GPU
  agreement, and valid binary hashes;
- unique placement, run, raw-artifact, and physical-GPU identities; and
- cloud-byte binding to the Modal receipt, Lambda sidecar, or the RunPod
  launch receipt, collection receipt, artifact index, and retained archive.

The processed manifest reports four distinct GPU UUIDs, all placement gates
passing, and all provider bindings passing. For the oracle floor, the final
state is independently validated but the decision trace is assigned from the
oracle by construction; it must not be described as independently observed.

Primary processed evidence:

- [`data/processed/resident-policy-pilot-cell-summary.csv`](../data/processed/resident-policy-pilot-cell-summary.csv),
  108 rows, SHA-256
  `07dfd68ec14dd26b163037d239c53fcbd66ef0d0ba080c5748b6f582c46c4038`;
- [`data/processed/resident-policy-pilot-contrasts.csv`](../data/processed/resident-policy-pilot-contrasts.csv),
  36 rows, SHA-256
  `00fb518cef7b2eb935fddad6b890025fd6b4cb05368458bb317cc29a987ba851`;
- [`data/processed/resident-policy-pilot-manifest.json`](../data/processed/resident-policy-pilot-manifest.json),
  SHA-256 `7b8751b701f7ae6cb8733976b6b842943350923b99fe34aab05a27148ac14732`;
- analyzer SHA-256
  `11bc0ef19042fa84bb81ea001250862e8159f392e3763c4597c7dbc7e304ccc9`.

Reproducible presentation artifacts:

- [`notebooks/10_resident_policy_pilot.ipynb`](../notebooks/10_resident_policy_pilot.ipynb),
  executed from scratch with five executed code cells and zero error outputs;
- [`resident-policy-speedup-by-horizon.png`](../results/figures/resident-policy-speedup-by-horizon.png),
  the placement-level speedup surface; and
- [`resident-policy-primary-mechanisms.png`](../results/figures/resident-policy-primary-mechanisms.png),
  the three mechanisms at the frozen primary cell.

## Hypothesis accounting

- `R1`, device faster in the primary cell: passes descriptively on all four
  named placements.
- `R2`, advantage increases with horizon: the post-prereg operationalization
  as a nondecreasing ratio of medians passes for all 12 placement/population
  series. Because “advantage” was not operationally defined before measurement,
  this is exploratory rather than a clean preregistered result.
- `R3`, resident remains slower than the oracle floor: passes in all 36 cells.
- `R4`, exactness: all validity gates pass. Observed decision-trace exactness
  applies to host/device; the floor trace is definitional.

## Statistical interpretation

The independent deployment-relevant unit is a fresh placement, not a row or a
batch invocation. The current placement counts are L4 `n=2`, H100 `n=1`, and
GTX 1660 Ti `n=1`. Consequently:

- 36 favorable cells are not 36 independent replications;
- 21,974,573 validation invocations establish deterministic correctness under
  the tested seeds, not a tiny performance standard error;
- P95/P99 columns in the processed cell table are quantiles of 30 batch-average
  rows, not invocation-latency tails; and
- no provider, hardware-population, CPU-displacement, end-to-end agent,
  deadline, energy, or online-scheduling claim follows.

Defensible wording is:

> On four named GPU placements across local, Modal, RunPod, and Lambda, the
> frozen device-resident binary decision mechanism reduced the median of
> batch-average steady-state cohort-horizon invocation wall time relative to a
> matched host-roundtrip GPU implementation in every observed cell, with zero
> observed correctness failures.

## Preregistration deviations and dependence

The frozen bounded cloud stage authorized one Modal L4 followed by one named
RunPod L4 **or** Lambda H100 replication. Both external providers were launched
before either external replication's result was available, so analyzing all
four named placements exceeds the frozen first-stage count by one. The added
placement is retained and labeled as a descriptive scope expansion; it cannot
increase the strength of a preregistered sample-count claim.

The full local run reused the same physical GTX 1660 Ti as the disclosed
pre-freeze engineering smoke. The smoke rows are excluded from this analysis,
but the local full-grid result is post-development evidence on that device and
is not an independent fresh-device validation. The four analyzed artifacts do
have four distinct GPU UUIDs relative to one another.

## Cloud execution and lifecycle ledger

### RunPod L4

- live inventory: four low-stock L4 regions;
- allocation: one secure-cloud L4 in `EU-RO-1`, USD 0.49/hour;
- launch: 2026-08-11 22:14:12 UTC;
- validated collection: 2026-08-11 22:35:04 UTC;
- exact stop/delete and confirmed absence: 2026-08-11 22:35:35 UTC;
- elapsed launch-to-deletion: about 21 minutes 23 seconds;
- compute-price estimate through deletion: about USD 0.175, excluding storage
  and not asserted as an invoice value;
- remote volume is unrecoverable; the validated local archive remains.
- the hash-bound launch receipt is mode `0600` and Git-ignored because it
  contains an ephemeral artifact-server bearer token. A public artifact must
  use a separately redacted and re-bound export rather than editing this
  private receipt in place.

### Lambda H100 SXM5

- immediately preceding inventory: zero running instances and live one-H100
  capacity in `us-south-2` at USD 4.29/hour;
- launch accepted: 2026-08-11 22:12:11 UTC;
- the successful retained provider sidecar records explicit `sudo -n docker`
  execution and the hash of the corrected provider adapter. The immutable
  receipts do not encode the preceding controller-attempt history, so no exact
  failure count or cause is used as paper evidence. The frozen CUDA source,
  mechanisms, grid, and timing boundary did not change;
- termination requested: 2026-08-11 22:46:30 UTC;
- API-confirmed zero running instances: 2026-08-11 22:51:09 UTC;
- launch-to-termination-request price estimate: about USD 2.45 at the observed
  hourly rate, not asserted as an invoice value;
- ephemeral VM data is unrecoverable; all three validated artifacts remain
  local.

Modal used a single-use container and retained no persistent resource. The
local run allocated no cloud resource. At the evidence cutoff, Lambda reports
zero running instances and RunPod reports the exact Pod absent.

## Research decision

The mechanism survives as a component, but this microbenchmark should **not**
be scaled directly to six or 30 placements. The unfrozen
[`resident-policy-002` design](resident-policy-002-design.md) correctly calls
for raw invocation tails, CPU measurement, balanced schedules, crash-safe
chunks, and normalized provenance, but adversarial review found blockers:

1. `20k valid and continue for 30 minutes` is outcome-dependent and would
   produce hundreds of millions of rows per placement;
2. process CPU clocks and `getrusage` do not share an atomic invocation
   boundary and can perturb microsecond-scale timings;
3. provider blocks are confounded with carryover orientation;
4. completed-only P99 lacks an intention-to-run value for crashes;
5. six placements cannot guarantee the proposed 90% power through an unstable
   variance upper bound;
6. the trace's 50 ms quantity is a launch deadline, not a completion SLO;
7. `N=256` is a candidate trace threshold, not a measured `K` for this source;
   and
8. `H=32` observation-free fusion is absent from the trace opportunity model.

The corrected sequence is:

1. compact `002` into a bounded measurement-qualification experiment with a
   fixed scheduled attempt count, fixed microblocks, raw empirical quantiles,
   block-level CPU/cgroup counters, explicit wait policy, and mechanical
   intention-to-run reliability;
2. implement the first online route-compacting runtime with finite service,
   deadline-to-launch accounting, CPU fallback, and measured route/horizon-
   specific crossover `K`;
3. recompute `P*` for those measured thresholds and trace-derived permissible
   fusion horizons;
4. measure achieved online share `A`, `A-F`, and `(A-F)/(P*-F)`; and only then
5. spend placement-scale compute on nuisance estimation and a separately
   powered confirmation.

This preserves the strongest paper chain: exact opportunity, measured
mechanism boundary, online achievement, and CPU/end-to-end leverage without
semantic harm.
