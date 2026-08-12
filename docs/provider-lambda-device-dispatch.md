# Lambda Cloud execution path: native device dispatch

Status date: 2026-08-11 UTC.

The executable path is `scripts/provider_lambda_device_dispatch.py`. Its
default command is a local-only plan: no API call, SSH connection, GPU
allocation, or billable resource is created.

## Live authentication and capacity result

The replacement credential authenticated successfully against Lambda's
production API. A read-only snapshot at `2026-08-11T20:54:49Z` returned 24
catalog instance types, 14 live type/region capacity pairs, zero running
instances, five SSH keys, and 259 available image records. No mutation
endpoint was called and no billable resource was created.

The exact nonempty capacity rows in that snapshot were:

| API instance type | GPU | GPUs | vCPUs | RAM GiB | Storage GiB | USD/hour | Regions with capacity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gpu_1x_a10` | A10 24 GB PCIe | 1 | 30 | 200 | 1,400 | 1.29 | `us-east-1`, `us-west-1` |
| `gpu_1x_a100_sxm4` | A100 40 GB SXM4 | 1 | 30 | 200 | 512 | 1.99 | `asia-south-1`, `us-east-1`, `us-west-2` |
| `gpu_1x_h100_pcie` | H100 80 GB PCIe | 1 | 26 | 200 | 1,024 | 3.29 | `us-west-3` |
| `gpu_1x_h100_sxm5` | H100 80 GB SXM5 | 1 | 26 | 225 | 2,816 | 4.29 | `us-south-2`, `us-south-3` |
| `gpu_2x_h100_sxm5` | H100 80 GB SXM5 | 2 | 52 | 450 | 5,632 | 8.38 | `us-south-2`, `us-south-3` |
| `gpu_8x_a100_80gb_sxm4` | A100 80 GB SXM4 | 8 | 240 | 1,800 | 20,480 | 22.32 | `us-east-1`, `us-midwest-1` |
| `gpu_8x_h100_sxm5` | H100 80 GB SXM5 | 8 | 208 | 1,800 | 22,528 | 31.92 | `us-south-2`, `us-west-3` |

All seven available types in this snapshot reported `x86_64`. A snapshot only
54 seconds earlier also showed `us-southeast-1` capacity for the one- and
two-H100 SXM5 types (16 pairs total). Its disappearance by the second request
is direct evidence that the capacity field is volatile. Every launch decision
must therefore bind to an immediately preceding timestamped inventory, not to
this table or a provider marketing page.

This command prints a fresh exact snapshot and can persist its sanitized
response:

```bash
.venv/bin/python scripts/provider_lambda_device_dispatch.py inventory \
  --output data/external/lambda-inventory-20260811T205449Z.json
```

The output path is exclusive-create: rerunning with the same path fails rather
than overwriting prior inventory.

For orientation only, Lambda's official static ODC overview has a table labeled
"As of December 2025" that lists B200 (1/2/4/8 GPU), GH200 (1), H100 SXM
(1/2/4/8), H100 PCIe (1), A100 SXM
(1/8), A100 PCIe (1/2/4), A10 (1), A6000 (1/2/4), V100 (8), and RTX 6000
(1). It lists these region identifiers: `asia-northeast-1`,
`asia-northeast-2`, `asia-south-1`, `europe-central-1`, `me-west-1`,
`us-east-1`, `us-east-2`, `us-midwest-1`, `us-south-1`, `us-south-2`,
`us-south-3`, `us-west-1`, `us-west-2`, and `us-west-3`. Those are catalog
facts, **not a claim of capacity in any type/region pair**. See Lambda's
[On-Demand Cloud overview](https://docs.lambda.ai/public-cloud/on-demand/).

## Safety contract

The runner follows Lambda's official Cloud API OpenAPI 1.10.0 contract at the
[Cloud API browser](https://docs.lambda.ai/api/cloud). It uses bearer
authentication, spaces requests to respect Lambda's documented general
one-request-per-second limit, and never prints or persists the API key.

- `plan` makes zero external calls.
- `inventory` makes only authenticated GET requests to instance types,
  instances, SSH keys, and images. Saved inventory redacts IP addresses,
  Jupyter credentials, and SSH public-key material.
- `launch` can create exactly one instance. It requires both
  `--confirm-spend` and the exact typed value
  `--launch-confirmation LAUNCH_ONE_BILLABLE_LAMBDA_INSTANCE`. It first checks
  live type/region capacity, SSH-key presence, image availability, CPU
  architecture, and a caller-supplied hourly price ceiling.
- `run-existing` cannot launch an instance. It requires an explicit existing
  instance ID, an attached SSH-key name, a private key with mode `0600` or
  stricter, an image provenance label, and `--confirm-remote-execution`.
- Pilot-shape, warmup, repetition, and estimated-transition ceilings reject
  obviously unbounded or accidentally enormous remote jobs.
- There is deliberately no terminate, restart, rebuild, delete, `apt`, or
  `sudo` path. Remote work is created in a fresh random `/tmp` directory and is
  retained for inspection.

Lambda documents that billing begins after launch health checks and continues
until termination, with one-minute billing increments. It also warns that OS
shutdown commands do not terminate an instance and billing continues. See the
[billing documentation](https://docs.lambda.ai/public-cloud/billing/) and
[instance-management documentation](https://docs.lambda.ai/public-cloud/on-demand/creating-managing-instances/).

## Launching later, after a successful inventory

Use an exact type and region pair printed by the immediately preceding live
inventory. This example is intentionally schematic; it will not run until all
required values and both gates are supplied:

```bash
.venv/bin/python scripts/provider_lambda_device_dispatch.py launch \
  --instance-type '<LIVE_API_TYPE_NAME>' \
  --region '<LIVE_CAPACITY_REGION>' \
  --ssh-key-name '<EXISTING_LAMBDA_KEY_NAME>' \
  --max-hourly-usd '<PRICE_CEILING>' \
  --confirm-spend \
  --launch-confirmation LAUNCH_ONE_BILLABLE_LAMBDA_INSTANCE
```

The default image family is `lambda-stack-24-04`. The live images endpoint
must confirm a matching image for the selected architecture and region before
launch. A JSONL launch receipt is opened with exclusive-create before the POST;
it records intent and either acceptance or a sanitized failure. SSH key names
are excluded from the receipt.

The script does **not** terminate the launched instance. The operator must
track the returned instance ID and terminate it through Lambda after artifact
verification. That omission is intentional because this provider adapter was
required to contain no destructive operation.

## Compiling and measuring on an active instance

Lambda's official [SSH instructions](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/)
use the `ubuntu` account. The default Lambda Stack image includes CUDA and
development tools, so the runner verifies `nvcc` and `nvidia-smi` rather than
modifying the system image.

```bash
.venv/bin/python scripts/provider_lambda_device_dispatch.py run-existing \
  --instance-id '<INSTANCE_ID>' \
  --ssh-key-name '<ATTACHED_KEY_NAME>' \
  --ssh-private-key '/absolute/path/to/private-key.pem' \
  --image-reference 'family:lambda-stack-24-04' \
  --confirm-remote-execution \
  --experiment-id 'device-dispatch-lambda-pilot' \
  --output-dir data/raw
```

SSH uses batch mode, a dedicated persistent `known_hosts` file,
`StrictHostKeyChecking=accept-new`, and the one explicitly selected identity.
The instance IP is taken from the authenticated instance response and is never
written to a result artifact. Source is copied only into a fresh random remote
directory. Compilation targets the installed GPU with `nvcc -arch=native` and
uses relocatable device code plus `cudadevrt`, as required by the device-graph
pilot.

Retrieval first enumerates the isolated output directory, accepts only safe
basenames, requires exactly one CSV and one native manifest, and downloads
each file by exact name. Local files and the provider manifest use exclusive-
create semantics. Existing data is never replaced.

## Recorded provenance

The native manifest already records the GPU name and UUID, PCI location,
compute capability, memory, SM count, clock properties, CUDA compile/runtime/
driver versions, host compiler, OS, CPU model, source hash, configuration,
correctness counts, and failure statuses.

The Lambda companion manifest adds:

- API spec version and collection time;
- opaque instance ID, instance name/status, region, exact instance-type
  description/specification, and attached SSH-key name;
- caller-supplied image family/ID reference with its provenance limitation;
- a hostname-presence flag (the value is redacted), kernel, architecture,
  logical CPU count, full `lscpu --json`,
  `/etc/os-release`, `nvcc --version`, and an `nvidia-smi` inventory;
- source and binary SHA-256 values, exact compile and benchmark arguments, and
  byte length/SHA-256 for each retrieved artifact; and
- the retained remote work-directory identifier.

Secrets, IP addresses, SSH private-key paths, and Jupyter credentials are not
included.

## Experimental limitations

1. The API key now authenticates, but Lambda documents API keys as having full
   access to all API operations. Keep the key out of result artifacts and
   rotate it if it is ever exposed.
2. The Lambda instances endpoint does not expose the boot image in the current
   OpenAPI schema. `run-existing` consequently requires an explicit image
   reference; a launch receipt is stronger evidence than a remembered label.
3. Lambda capacity is ephemeral. Inventory is a timestamped observation, not a
   reservation. A launch may still fail with insufficient capacity.
4. GH200 uses an ARM host CPU. The script refuses non-x86 instances by default
   because CPU/GPU crossover comparisons would otherwise confound GPU type
   with host ISA.
5. A provider instance is one experimental placement even if the native binary
   reports many timed repetitions. Placement replication, not inner-loop row
   count, determines the inferential sample size.
6. This runner does not pin CPUs, control NUMA placement, lock GPU clocks, or
   eliminate provider background load. Those fields remain blocking factors
   or measured nuisance variables in the statistical design.
7. The runner leaves remote files and the instance intact. This preserves
   auditability, but it also means the operator must explicitly terminate the
   instance through Lambda when finished to stop billing.
