# RunPod execution path for the native device-dispatch pilot

Status: implementation and authenticated read-only inventory are complete. No
Pod was created, stopped, or terminated while preparing this adapter.

The runner is `scripts/runpod_device_dispatch_pilot.py`. Its default `plan`
action is local-only. The implementation uses RunPod's current GraphQL API for
GPU/datacenter availability and the v1 REST API for a deliberately gated future
Pod creation. These interfaces and the relevant lifecycle behavior are
documented in RunPod's [GraphQL Pod guide](https://docs.runpod.io/sdks/graphql/manage-pods),
[REST API overview](https://docs.runpod.io/api-reference/overview), and
[Pod creation reference](https://docs.runpod.io/api-reference/pods/POST/pods).

## Authenticated inventory result

At `2026-08-11T20:53:39Z`, `RUNPOD_API_KEY` authenticated successfully. The
sanitized point-in-time response contained:

- 48 GPU catalog entries, including the non-hardware `unknown` entry;
- 49 datacenters;
- 22 GPU types with a `High`, `Medium`, or `Low` stock report;
- 32 datacenters with at least one such stock report; and
- 72 stocked GPU/datacenter pairs.

The complete redacted response, including all catalog entries, all 49
datacenters, all unavailable offers, prices, and the exact 72 available pairs,
is in `docs/provider-runpod-inventory-20260811T2054Z.json`. It contains no
account ID, email, balance, token, or API key.

Research-relevant stock in that exact snapshot was:

| GPU type | Reported datacenters |
|---|---|
| NVIDIA L4 | EU-RO-1 (Low), EUR-IS-1 (Low), EUR-IS-2 (Low), US-MO-2 (Low) |
| NVIDIA L40S | EUR-IS-2 (Low), US-MO-1 (Low), US-TX-3 (Low), US-TX-4 (Low) |
| NVIDIA A100 80GB PCIe | CA-MTL-3 (Low) |
| NVIDIA A100-SXM4-80GB | EUR-IS-1, US-CA-2, US-KS-2, US-MD-1, US-MO-1, US-WA-1 (all Low) |
| NVIDIA H100 80GB HBM3 | AP-IN-1 (Medium); AP-IN-2, AP-JP-1, CA-MTL-1, EU-FR-1, EU-NL-1, EUR-IS-3, EUR-NO-2, US-CA-2, US-GA-2, US-MO-1, US-NE-1 (Low) |
| NVIDIA H100 PCIe | US-KS-2 (Low) |
| NVIDIA H100 NVL | US-KS-2 (Low) |
| Tesla T4 | absent from the live GPU catalog |

Inventory is volatile: adjacent read-only queries changed individual `Low`
offers. For that reason, a future launch re-queries inventory and rejects a
GPU/datacenter pair unless it still has a nonempty `High`, `Medium`, or `Low`
stock value. A stock value is not a reservation, so allocation can still race.
The live query is more useful than the static GPU enum in the REST schema for
placement decisions.

## Safe default and read-only use

The plan action performs zero remote calls:

```bash
python3 scripts/runpod_device_dispatch_pilot.py --action plan
```

Refresh inventory without writing a file:

```bash
python3 scripts/runpod_device_dispatch_pilot.py --action inventory
```

To preserve a new immutable snapshot, give it a new filename. The runner opens
the path with exclusive-create semantics and refuses to overwrite anything:

```bash
python3 scripts/runpod_device_dispatch_pilot.py \
  --action inventory \
  --inventory-output docs/provider-runpod-inventory-YYYYMMDDTHHMMSSZ.json
```

## Deliberately double-gated launch

Pod creation remains disabled unless the same literal acknowledgement appears
in both places below. This command is documentation only; it was not run:

```bash
export RUNPOD_ENABLE_GPU_SPEND=RUNPOD_DEVICE_DISPATCH_PILOT
python3 scripts/runpod_device_dispatch_pilot.py \
  --action launch \
  --confirm-spend RUNPOD_DEVICE_DISPATCH_PILOT \
  --experiment-id device-dispatch-runpod-l4-pilot \
  --gpu-type "NVIDIA L4" \
  --data-center-id EU-RO-1 \
  --cloud-type SECURE \
  --max-run-minutes 20 \
  --max-cost-usd 1.00
```

Before the mutating call, the runner validates the current stock pair, cloud
support, source-supported GPU family, bounded benchmark shape, wall-time cap,
and an estimated compute-cost cap. The estimate uses the inventory price and
excludes storage. RunPod bills Pod compute and storage separately; see the
official [Pod pricing documentation](https://docs.runpod.io/pods/pricing).

The launch receipt is exclusive-created with mode `0600`. It records the Pod
ID, requested placement, exact image, source/archive digests, cost guard, and a
random artifact bearer token. It does not contain `RUNPOD_API_KEY`, and that
account key is never passed in the Pod request.

## Bootstrap and artifact contract

The Pod receives a deterministic gzip/tar snapshot of the reviewed native
source rather than cloning a mutable branch. It verifies both the archive hash
and the same path-aware source hash used by the Modal adapter before compiling
with CUDA 13.0.1. Every remote output lives under `/workspace`, not the
container disk.

The downloadable archive includes:

- the exact C++/CUDA source, Makefile, and native README;
- compiler command, stdout, and stderr;
- native CSV and manifest outputs when compilation and execution succeed;
- program stdout and stderr; and
- provider and collection metadata containing Pod ID, datacenter ID, Pod
  hostname, REST machine ID, GPU UUID/name/PCI ID/driver, image, CUDA source and
  binary hashes, host platform, run parameters, and return codes.

Artifacts are exposed on port 8000 through RunPod's HTTPS proxy. RunPod
documents the proxy URL and its public-access implications in
[Expose ports](https://docs.runpod.io/pods/configuration/expose-ports). The
adapter's server disables directory listing and requires the random bearer
token for its three fixed paths. The collector verifies the SHA-256 digest,
rejects path traversal, links, devices, duplicate overwrites, and unexpected
archive shapes, then exclusive-creates a new local run directory:

```bash
python3 scripts/runpod_device_dispatch_pilot.py \
  --action collect \
  --receipt data/provider-runpod-launches/LAUNCH_RECEIPT.json \
  --output-dir data/raw
```

The remote artifact server schedules a Pod **stop**, never deletion, 15 seconds
after a successful archive response. A deadline watchdog also requests a stop
after `--max-run-minutes`. Both use the Pod-scoped API key that RunPod injects;
the available Pod variables are listed in RunPod's
[environment-variable reference](https://docs.runpod.io/pods/templates/environment-variables).
The local collector itself performs only GET requests.

RunPod states that stopping releases the GPU but preserves volume data, while
stopped volume storage continues to incur a charge. Termination permanently
deletes non-network-volume data. This adapter intentionally has no terminate or
delete operation; after verifying downloaded artifacts, cleanup must be an
explicit user action following RunPod's [Pod management guide](https://docs.runpod.io/pods/manage-pods).

## Limitations that matter for the paper

- Current stock is thin and mostly `Low`; RunPod cannot be treated as a fixed
  balanced block without predeclared replacement rules.
- The inventory response does not identify cloud type per datacenter offer.
  The runner separately checks GPU-level Secure/Community support, but the
  control plane remains authoritative at allocation time.
- The 13.0.1 CUDA base image installs `g++`, Python, CA certificates, and
  `curl` during bootstrap. For a larger replication campaign, publish a
  digest-pinned image to eliminate package-mirror and setup-time variance.
- The embedded source payload is appropriate for this ~50 KB pilot, not a
  general repository transport.
- The native source currently emits SASS for SM 75/80/86/89/90 plus compute-90
  PTX. The adapter rejects Blackwell GPU IDs until native architecture coverage
  is intentionally revised and validated.
- Provider placements are independent only when GPU UUID and RunPod machine ID
  are both distinct. Repeated rows within one Pod are not independent samples.
- A stopped Pod's `/workspace` volume can be recovered, but serving it again
  requires a paid restart and may land on different hardware. Collect within
  the watchdog window whenever possible.
