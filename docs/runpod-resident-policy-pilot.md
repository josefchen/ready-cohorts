# RunPod resident-policy pilot

`scripts/runpod_resident_policy_pilot.py` is the safe provider adapter for the
frozen `resident-policy-001` CUDA source. Its default action is a local-only
plan. It never prints API keys or the ephemeral artifact token.

## Frozen contract

- CUDA source SHA-256:
  `4b5cdcb9496a734bd7801d5c419efb8eceb72fd6962800520101e89676d204da`
- Makefile SHA-256:
  `d74935b594fb629b2113d237439289e057281becc561b6941d1134bd6a1c1351`
- image: `nvidia/cuda:13.0.1-devel-ubuntu24.04`
- allocation: exactly one `NVIDIA L4`
- full grid: 810 measured rows
- smoke grid: 24 measured rows

The adapter rejects a launch if either frozen file changed, the selected data
center has no current L4 stock, the selected cloud has no current L4 price, the
bounded cost estimate exceeds the CLI cap, or either launch gate is absent.

## Safe workflow

Run the local-only plan without loading credentials:

```bash
cd /path/to/gpu-agent-crossover
env -u RUNPOD_API_KEY .venv/bin/python \
  scripts/runpod_resident_policy_pilot.py \
  --action plan --mode full --env-file /nonexistent
```

Capture a new, read-only inventory snapshot. The output path must not already
exist:

```bash
.venv/bin/python scripts/runpod_resident_policy_pilot.py \
  --action inventory \
  --inventory-output data/external/runpod-resident-policy-inventory-YYYYMMDDTHHMMSSZ.json
```

Choose a data-center ID from an `available_l4` row. Launching is the first
spending operation and requires both copies of the literal acknowledgement.
Use a new experiment ID and receipt path for every placement:

```bash
RUNPOD_ENABLE_GPU_SPEND=RUNPOD_RESIDENT_POLICY_PILOT \
.venv/bin/python scripts/runpod_resident_policy_pilot.py \
  --action launch \
  --mode full \
  --experiment-id resident-policy-001-runpod-l4-p1 \
  --data-center-id EU-RO-1 \
  --cloud-type SECURE \
  --max-run-minutes 45 \
  --max-cost-usd 1.00 \
  --receipt data/provider-runpod-launches/resident-policy-001-runpod-l4-p1.launch.json \
  --confirm-spend RUNPOD_RESIDENT_POLICY_PILOT
```

`EU-RO-1` is only an example; use a data center reported by the immediately
preceding inventory call. For a bounded engineering smoke, use a distinct
experiment ID and replace `--mode full` with `--mode smoke`.

Collect the append-only bundle:

```bash
.venv/bin/python scripts/runpod_resident_policy_pilot.py \
  --action collect \
  --receipt data/provider-runpod-launches/resident-policy-001-runpod-l4-p1.launch.json \
  --output-dir data/raw \
  --collect-timeout-seconds 3600
```

Collection verifies the archive SHA-256, artifact index, frozen source and
Makefile, compiled binary, provider/host/GPU provenance, native manifest,
configuration, exact row count, timing fields, every state field, every
decision trace, repetition coverage, and randomized order indices. A failed
bundle is retained for diagnosis but cannot authorize deletion.

After `validation_passed=true`, terminate the exact Pod and delete its remote
volume. Substitute the Pod ID from the launch receipt in both gates, and use
the collection-receipt path printed by `collect`:

```bash
RUNPOD_TERMINATE_POD_ID=<exact-pod-id> \
.venv/bin/python scripts/runpod_resident_policy_pilot.py \
  --action terminate \
  --receipt data/provider-runpod-launches/resident-policy-001-runpod-l4-p1.launch.json \
  --collection-receipt data/raw/runpod-resident-policy-001-runpod-l4-p1-<exact-pod-id>/collection-receipt.json \
  --confirm-terminate-pod-id <exact-pod-id> \
  --termination-receipt data/external/runpod-resident-policy-termination-<exact-pod-id>.json
```

Termination re-hashes and revalidates the locally retained bundle before any
provider mutation, verifies the live Pod ID and name, stops that Pod, waits for
a stopped state, deletes only that Pod, and records whether absence was
confirmed. Deleting the Pod makes its remote volume unrecoverable; the verified
local archive remains intact.

## Offline verification

These commands perform no provider call:

```bash
.venv/bin/python -m py_compile \
  scripts/runpod_resident_policy_pilot.py \
  tests/test_runpod_resident_policy_pilot.py

.venv/bin/ruff check \
  scripts/runpod_resident_policy_pilot.py \
  tests/test_runpod_resident_policy_pilot.py

env -u RUNPOD_API_KEY .venv/bin/pytest -q \
  tests/test_runpod_resident_policy_pilot.py
```
