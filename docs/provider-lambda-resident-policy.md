# Lambda runner: resident-policy-001

[`scripts/provider_lambda_resident_policy.py`](../scripts/provider_lambda_resident_policy.py)
is the guarded Lambda Cloud path for the frozen native resident-policy pilot. Its
default action is local-only. Inventory, launch, remote execution, and
termination are distinct commands so that inspecting the plan cannot allocate
or delete anything.

The runner accepts only one-GPU H100 and A10 instance types. It validates the
live type, GPU count, capacity region, hourly price, SSH key, architecture, and
boot-image availability immediately before sending a launch request. A launch
requires both a boolean spend flag and an exact confirmation phrase. The API
response must contain exactly one safe instance ID.

## Frozen inputs

- CUDA source SHA-256:
  `4b5cdcb9496a734bd7801d5c419efb8eceb72fd6962800520101e89676d204da`
- Makefile SHA-256:
  `d74935b594fb629b2113d237439289e057281becc561b6941d1134bd6a1c1351`
- CUDA build/runtime image: `nvidia/cuda:13.0.1-devel-ubuntu24.04`
- full grid: exactly 810 measured rows
- smoke grid: exactly 24 measured rows

Both local and copied remote source hashes must match before compilation. The
runner compiles in the CUDA 13 container, rejects a compiler that does not
report CUDA major version 13, hashes the resulting executable, and runs the
benchmark with container networking disabled and only GPU device 0 exposed.
Lambda Stack grants the `ubuntu` account Docker access through passwordless
`sudo`, so the adapter first proves `sudo -n` is noninteractive and then uses
that prefix only for the explicit pull, inspect, compile, run, and version
commands. This provider-access correction does not alter the frozen CUDA
source, experiment grid, timing region, or mechanism definitions and is
recorded in the provider manifest.

## Safe command sequence

Run these commands from the repository root. The examples deliberately use
placeholders; replace them only after inspecting a fresh inventory.

The default command makes no API or remote calls:

```bash
python3 scripts/provider_lambda_resident_policy.py
```

The inventory command performs authenticated GET requests only and creates a
new, non-overwriting JSON snapshot:

```bash
python3 scripts/provider_lambda_resident_policy.py inventory \
  --output data/external/lambda-resident-inventory-UNIQUE-UTC.json
```

Launch exactly one live-validated H100 instance:

```bash
python3 scripts/provider_lambda_resident_policy.py launch \
  --gpu-family H100 \
  --instance-type LIVE_1X_H100_TYPE \
  --region LIVE_CAPACITY_REGION \
  --ssh-key-name REGISTERED_KEY_NAME \
  --instance-name gpu-agent-resident-policy-001-h100-p1 \
  --image-family lambda-stack-24-04 \
  --max-hourly-usd 5.00 \
  --confirm-spend \
  --launch-confirmation LAUNCH_ONE_BILLABLE_LAMBDA_INSTANCE
```

For A10, change both `--gpu-family` and `--instance-type`; an A100 description
cannot satisfy the A10 check. Launch writes an append-only intent/response
receipt. It does not run the benchmark and does not terminate the instance.

After the exact instance becomes active, execute the full frozen grid:

```bash
python3 scripts/provider_lambda_resident_policy.py run-existing \
  --mode full \
  --gpu-family H100 \
  --instance-id EXACT_INSTANCE_ID \
  --expected-instance-name gpu-agent-resident-policy-001-h100-p1 \
  --ssh-key-name REGISTERED_KEY_NAME \
  --ssh-private-key /ABSOLUTE/PATH/TO/PRIVATE_KEY \
  --known-hosts data/external/lambda-known-hosts \
  --image-reference LIVE_BOOT_IMAGE_ID_AND_VERSION \
  --experiment-id resident-policy-001-lambda-h100-p1 \
  --confirm-remote-execution \
  --run-confirmation RUN_FROZEN_RESIDENT_POLICY_001 \
  --output-dir data/raw
```

Use `--mode smoke` only for a bounded platform-support check. Smoke output is
not a substitute for the frozen 810-row run.

After verifying that the CSV, native manifest, and provider manifest exist
locally, terminate only the exact intended instance:

```bash
python3 scripts/provider_lambda_resident_policy.py terminate \
  --instance-id EXACT_INSTANCE_ID \
  --expected-instance-name gpu-agent-resident-policy-001-h100-p1 \
  --confirm-termination \
  --termination-confirmation TERMINATE_EXACT_LAMBDA_INSTANCE_EXACT_INSTANCE_ID \
  --output-dir data/external
```

The confirmation suffix must literally be the same instance ID supplied to
`--instance-id`. Termination first re-reads that instance and checks its exact
name. It sends a one-element `instance_ids` list and creates a non-overwriting
termination receipt. Poll read-only inventory until the instance is
`terminating`, `terminated`, or absent. Ephemeral data on a terminated instance
is not recoverable.

## Artifact gates

Remote output is accepted only when there is exactly one CSV and one sibling
manifest. The full validator requires:

- exact source, Makefile, binary, provider, placement, and container provenance;
- one actual GPU whose H100/A10 family and UUID agree across `nvidia-smi` and
  the native manifest;
- CUDA 13 compile and runtime versions plus unified addressing;
- exact frozen config equality and exactly 810 rows in full mode;
- complete cell, repetition, and randomized-order coverage;
- positive, internally consistent wall and device timing;
- status `ok`, empty error fields, and the minimum duration reached;
- all state fields exact, all decision traces exact, checksum/hash equality,
  and validation of every batched invocation.

Downloaded scientific artifacts and the provider provenance manifest are
created append-only and hash-checked locally. If a scientific gate fails after
retrieval, the files are retained for diagnosis and the command exits with an
error. The remote work directory is retained until the operator terminates the
instance. API keys, IP addresses, SSH material, hostnames, and Jupyter tokens
are omitted from persisted provider metadata.

## Offline verification

These checks do not contact Lambda:

```bash
.venv/bin/python -m py_compile scripts/provider_lambda_resident_policy.py
.venv/bin/ruff check \
  scripts/provider_lambda_resident_policy.py \
  tests/test_provider_lambda_resident_policy.py
.venv/bin/ruff format --check \
  scripts/provider_lambda_resident_policy.py \
  tests/test_provider_lambda_resident_policy.py
.venv/bin/pytest -q tests/test_provider_lambda_resident_policy.py
```

The mocked tests fail if a refused launch reaches API construction, verify that
a valid launch contains one instance request, verify that A100 cannot match the
A10 allowlist, validate an adapted 810-row frozen artifact, inject and detect a
decision-trace mismatch, inspect the CUDA 13 networkless command, and exercise
an exact-target termination without making a network call.
