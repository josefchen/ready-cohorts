"""Read-only-first RunPod adapter for the native CUDA device-dispatch pilot.

The default ``plan`` action makes no network call. ``inventory`` performs only
authenticated GraphQL queries. A Pod can be created only when the same literal
acknowledgement is supplied both through an environment variable and a CLI
argument. This module never deletes or terminates a Pod.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "native/device_dispatch"
DEFAULT_ENV_FILE = REPO_ROOT.parent / ".env"
GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"
CUDA_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
SOURCE_FILES = ("Makefile", "README.md", "device_dispatch_pilot.cu")
SPEND_ACK = "RUNPOD_DEVICE_DISPATCH_PILOT"
SPEND_ENV = "RUNPOD_ENABLE_GPU_SPEND"
ARTIFACT_PORT = 8000
KNOWN_STOCK = {"high", "medium", "low"}
EXPERIMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
GPU_IDS_SUPPORTED_BY_SOURCE = {
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A40",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA H100 PCIe",
    "NVIDIA H200",
    "NVIDIA H200 NVL",
    "NVIDIA L4",
    "NVIDIA L40",
    "NVIDIA L40S",
    "Tesla T4",
}


class RunPodError(RuntimeError):
    """A deliberately redacted RunPod control-plane error."""


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_env_file(path: Path) -> None:
    """Load simple dotenv assignments without ever echoing their values."""

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            quote = value[0]
            value = value[1:-1]
            if quote == '"':
                value = bytes(value, "utf-8").decode("unicode_escape")
        os.environ.setdefault(key, value)


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise RunPodError(
            "RUNPOD_API_KEY is not configured in the environment or the selected env file"
        )
    return key


def _redact(text: str, extra_values: tuple[str, ...] = ()) -> str:
    redacted = text
    for value in (os.environ.get("RUNPOD_API_KEY", ""), *extra_values):
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted[:300]


def _request_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    extra_redactions: tuple[str, ...] = (),
    timeout: float = 30.0,
) -> Any:
    encoded = None
    if body is not None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        method=method,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "User-Agent": "gpu-agent-crossover-runpod-adapter/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        # Never reproduce a create request or an unfiltered response body: Pod
        # errors can contain environment values, including the artifact token.
        raise RunPodError(f"RunPod HTTP request failed with status {error.code}") from error
    except urllib.error.URLError as error:
        reason = _redact(str(error.reason), extra_redactions)
        raise RunPodError(f"RunPod network request failed: {reason}") from error
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise RunPodError("RunPod returned a non-JSON response") from error


def _graphql(query: str) -> dict[str, Any]:
    payload = _request_json("POST", GRAPHQL_URL, body={"query": query})
    if not isinstance(payload, dict):
        raise RunPodError("RunPod GraphQL response had an unexpected shape")
    errors = payload.get("errors") or []
    if errors:
        message = "GraphQL query failed"
        if isinstance(errors[0], dict) and isinstance(errors[0].get("message"), str):
            message += ": " + _redact(errors[0]["message"])
        raise RunPodError(message)
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RunPodError("RunPod GraphQL response did not contain data")
    return data


def fetch_inventory() -> dict[str, Any]:
    """Return a sanitized, timestamped, read-only inventory snapshot."""

    query = """
    query DeviceDispatchInventory {
      myself { id }
      gpuTypes {
        id displayName memoryInGb secureCloud communityCloud
        securePrice communityPrice
      }
      dataCenters {
        id name location
        gpuAvailability { gpuTypeId displayName stockStatus }
      }
    }
    """
    data = _graphql(query)
    myself = data.get("myself")
    authenticated = isinstance(myself, dict) and bool(myself.get("id"))
    if not authenticated:
        raise RunPodError("RunPod authentication did not resolve an account identity")

    gpu_types = []
    for raw in data.get("gpuTypes") or []:
        if not isinstance(raw, dict):
            continue
        gpu_types.append(
            {
                "id": raw.get("id"),
                "display_name": raw.get("displayName"),
                "memory_gb": raw.get("memoryInGb"),
                "secure_cloud": bool(raw.get("secureCloud")),
                "community_cloud": bool(raw.get("communityCloud")),
                "secure_price_per_hour": raw.get("securePrice"),
                "community_price_per_hour": raw.get("communityPrice"),
            }
        )
    gpu_types.sort(key=lambda item: str(item["id"]))

    data_centers = []
    available_offers = []
    for raw_dc in data.get("dataCenters") or []:
        if not isinstance(raw_dc, dict):
            continue
        dc_id = str(raw_dc.get("id") or "")
        location = str(raw_dc.get("location") or "")
        availability = []
        for raw_offer in raw_dc.get("gpuAvailability") or []:
            if not isinstance(raw_offer, dict):
                continue
            status = str(raw_offer.get("stockStatus") or "")
            offer = {
                "gpu_type_id": raw_offer.get("gpuTypeId"),
                "display_name": raw_offer.get("displayName"),
                "stock_status": status or "none",
            }
            availability.append(offer)
            if status.strip().lower() in KNOWN_STOCK:
                available_offers.append(
                    {
                        "data_center_id": dc_id,
                        "location": location,
                        **offer,
                    }
                )
        availability.sort(key=lambda item: str(item["gpu_type_id"]))
        data_centers.append(
            {
                "id": dc_id,
                "name": raw_dc.get("name"),
                "location": location,
                "gpu_availability": availability,
            }
        )
    data_centers.sort(key=lambda item: item["id"])
    available_offers.sort(key=lambda item: (str(item["gpu_type_id"]), item["data_center_id"]))
    stocked_gpu_ids = sorted({str(item["gpu_type_id"]) for item in available_offers})
    stocked_dc_ids = sorted({item["data_center_id"] for item in available_offers})
    return {
        "schema_version": "runpod-inventory-v1",
        "captured_at_utc": _utc_now(),
        "source": {
            "api": GRAPHQL_URL,
            "operation": "DeviceDispatchInventory",
            "mutating_calls": 0,
        },
        "authentication": {"ok": True, "identity_redacted": True},
        "counts": {
            "catalog_gpu_types": len(gpu_types),
            "data_centers": len(data_centers),
            "stocked_gpu_types": len(stocked_gpu_ids),
            "data_centers_with_stock": len(stocked_dc_ids),
            "stocked_gpu_datacenter_offers": len(available_offers),
        },
        "stock_semantics": {
            "included_as_available": ["High", "Medium", "Low"],
            "excluded_as_unavailable": ["none", "blank", "missing"],
            "note": "Availability is a volatile point-in-time control-plane report.",
        },
        "stocked_gpu_type_ids": stocked_gpu_ids,
        "stocked_data_center_ids": stocked_dc_ids,
        "available_offers": available_offers,
        "gpu_types": gpu_types,
        "data_centers": data_centers,
    }


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        path = NATIVE_ROOT / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_archive() -> bytes:
    """Build a deterministic archive containing only the reviewed native source."""

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        for name in SOURCE_FILES:
            payload = (NATIVE_ROOT / name).read_bytes()
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as stream:
        stream.write(tar_buffer.getvalue())
    return compressed.getvalue()


def _write_json_new(path: Path, value: Any, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600 if private else 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parse_positive_csv(value: str, *, name: str, maximum: int) -> list[int]:
    try:
        numbers = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated list of integers") from error
    if not numbers or any(number <= 0 or number > maximum for number in numbers):
        raise ValueError(f"{name} values must be in [1, {maximum}]")
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"{name} values must not repeat")
    return numbers


def _gpu_entry(inventory: dict[str, Any], gpu_type: str) -> dict[str, Any]:
    for entry in inventory["gpu_types"]:
        if entry["id"] == gpu_type:
            return entry
    raise ValueError(f"GPU type is absent from the live RunPod catalog: {gpu_type}")


def _stock_entry(inventory: dict[str, Any], gpu_type: str, data_center_id: str) -> dict[str, Any]:
    for entry in inventory["available_offers"]:
        if entry["gpu_type_id"] == gpu_type and entry["data_center_id"] == data_center_id:
            return entry
    raise ValueError(f"live inventory reports no stock for {gpu_type!r} in {data_center_id!r}")


BOOTSTRAP = r"""set -Eeuo pipefail
umask 077

stop_self() {
  if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    if runpodctl pod stop "$RUNPOD_POD_ID" >/dev/null 2>&1; then
      return
    fi
  fi
  if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ] \
      && command -v curl >/dev/null 2>&1; then
    curl --silent --show-error --fail \
      --request POST \
      --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
      "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
      >/dev/null 2>&1 || true
  fi
}

artifact_parent="/workspace/runpod-device-dispatch/${EXPERIMENT_ID}"
artifact_root="${artifact_parent}/${RUNPOD_POD_ID:-unknown-pod}"
if [ -e "$artifact_root" ]; then
  echo "refusing to overwrite existing artifact directory" >&2
  stop_self
  exit 42
fi
mkdir -p "$artifact_parent"
mkdir "$artifact_root"
trap stop_self EXIT

( sleep "$MAX_RUN_SECONDS"; stop_self ) &

apt-get update >"$artifact_root/dependency-install.log" 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  g++ python3 ca-certificates curl >>"$artifact_root/dependency-install.log" 2>&1

mkdir "$artifact_root/source" "$artifact_root/results"
printf '%s' "$SOURCE_ARCHIVE_B64" | base64 -d >"$artifact_root/source/source.tar.gz"
printf '%s  %s\n' "$SOURCE_ARCHIVE_SHA256" "$artifact_root/source/source.tar.gz" \
  | sha256sum --check --strict

archive_listing=$(tar -tzf "$artifact_root/source/source.tar.gz")
if [ "$archive_listing" != $'Makefile\nREADME.md\ndevice_dispatch_pilot.cu' ]; then
  echo "source archive contains unexpected paths" >&2
  exit 43
fi
tar -xzf "$artifact_root/source/source.tar.gz" \
  -C "$artifact_root/source" --no-same-owner

python3 - "$artifact_root/source" "$SOURCE_SHA256" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
digest = hashlib.sha256()
for name in ("Makefile", "README.md", "device_dispatch_pilot.cu"):
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update((root / name).read_bytes())
    digest.update(b"\0")
if digest.hexdigest() != expected:
    raise SystemExit("source digest mismatch")
PY

compile_command=(
  nvcc -O3 -std=c++17 -lineinfo -rdc=true --threads 0
  -gencode arch=compute_75,code=sm_75
  -gencode arch=compute_80,code=sm_80
  -gencode arch=compute_86,code=sm_86
  -gencode arch=compute_89,code=sm_89
  -gencode arch=compute_90,code=sm_90
  -gencode arch=compute_90,code=compute_90
  "$artifact_root/source/device_dispatch_pilot.cu"
  -o "$artifact_root/device_dispatch_pilot"
  -lcudadevrt
)
printf '%q ' "${compile_command[@]}" >"$artifact_root/compile-command.txt"
printf '\n' >>"$artifact_root/compile-command.txt"

set +e
"${compile_command[@]}" >"$artifact_root/compiler.stdout.log" \
  2>"$artifact_root/compiler.stderr.log"
compile_return_code=$?
program_return_code=127
if [ "$compile_return_code" -eq 0 ]; then
  "$artifact_root/device_dispatch_pilot" \
    --experiment-id "$EXPERIMENT_ID" \
    --agents "$AGENT_COUNTS" \
    --steps "$STEP_COUNTS" \
    --warmups "$WARMUPS" \
    --repetitions "$REPETITIONS" \
    --seed "$SEED" \
    --output-dir "$artifact_root/results" \
    >"$artifact_root/program.stdout.log" \
    2>"$artifact_root/program.stderr.log"
  program_return_code=$?
else
  : >"$artifact_root/program.stdout.log"
  printf 'program not run because compilation failed\n' \
    >"$artifact_root/program.stderr.log"
fi
set -e

export ARTIFACT_ROOT="$artifact_root"
export COMPILE_RETURN_CODE="$compile_return_code"
export PROGRAM_RETURN_CODE="$program_return_code"
export BINARY_SHA256=""
if [ -f "$artifact_root/device_dispatch_pilot" ]; then
  BINARY_SHA256=$(sha256sum "$artifact_root/device_dispatch_pilot" | cut -d' ' -f1)
  export BINARY_SHA256
fi

python3 - <<'PY'
import datetime
import glob
import json
import os
import pathlib
import platform
import subprocess

root = pathlib.Path(os.environ["ARTIFACT_ROOT"])
gpu_rows = []
try:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,pci.bus_id,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5:
            gpu_rows.append(
                dict(
                    uuid=fields[0],
                    name=fields[1],
                    driver_version=fields[2],
                    pci_bus_id=fields[3],
                    memory_mib=fields[4],
                )
            )
except Exception as error:
    gpu_rows = [{"discovery_error": type(error).__name__}]

manifests = sorted(glob.glob(str(root / "results" / "*.manifest.json")))
native_summary = None
if len(manifests) == 1:
    native = json.loads(pathlib.Path(manifests[0]).read_text())
    native_summary = {
        "run_id": native.get("run_id"),
        "manifest_file": pathlib.Path(manifests[0]).name,
        "results": native.get("results"),
    }

metadata = {
    "schema_version": "runpod-device-dispatch-provider-v1",
    "captured_at_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    "provider": "runpod",
    "pod": {
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "data_center_id": os.environ.get("RUNPOD_DC_ID"),
        "pod_hostname": os.environ.get("RUNPOD_POD_HOSTNAME"),
        "provider_gpu_count": os.environ.get("RUNPOD_GPU_COUNT"),
        "provider_cpu_count": os.environ.get("RUNPOD_CPU_COUNT"),
    },
    "request": {
        "requested_gpu_type": os.environ.get("REQUESTED_GPU"),
        "requested_data_center_id": os.environ.get("REQUESTED_DATA_CENTER"),
        "requested_cloud_type": os.environ.get("REQUESTED_CLOUD_TYPE"),
        "image": os.environ.get("RUNPOD_IMAGE"),
        "allowed_cuda_version": "13.0",
    },
    "source": {
        "source_sha256": os.environ.get("SOURCE_SHA256"),
        "archive_sha256": os.environ.get("SOURCE_ARCHIVE_SHA256"),
        "binary_sha256": os.environ.get("BINARY_SHA256"),
    },
    "execution": {
        "compile_return_code": int(os.environ["COMPILE_RETURN_CODE"]),
        "program_return_code": int(os.environ["PROGRAM_RETURN_CODE"]),
        "agent_counts": os.environ.get("AGENT_COUNTS"),
        "step_counts": os.environ.get("STEP_COUNTS"),
        "warmups": int(os.environ["WARMUPS"]),
        "repetitions": int(os.environ["REPETITIONS"]),
        "seed": int(os.environ["SEED"]),
        "native_summary": native_summary,
    },
    "host": {
        "platform": platform.platform(),
        "node": platform.node(),
        "machine": platform.machine(),
    },
    "gpus": gpu_rows,
    "secrets_recorded": False,
}
with (root / "provider-metadata.json").open("x", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

tar -C "$artifact_root" -czf "$artifact_root/device-dispatch-artifacts.tar.gz.tmp" \
  source/Makefile source/README.md source/device_dispatch_pilot.cu \
  results provider-metadata.json compile-command.txt \
  dependency-install.log compiler.stdout.log compiler.stderr.log \
  program.stdout.log program.stderr.log
mv "$artifact_root/device-dispatch-artifacts.tar.gz.tmp" \
  "$artifact_root/device-dispatch-artifacts.tar.gz"
sha256sum "$artifact_root/device-dispatch-artifacts.tar.gz" \
  | sed 's# .*/#  #' \
  >"$artifact_root/device-dispatch-artifacts.tar.gz.sha256"

python3 - <<'PY'
import datetime
import json
import os
import pathlib

root = pathlib.Path(os.environ["ARTIFACT_ROOT"])
ready = {
    "ready": True,
    "captured_at_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    "compile_return_code": int(os.environ["COMPILE_RETURN_CODE"]),
    "program_return_code": int(os.environ["PROGRAM_RETURN_CODE"]),
    "source_sha256": os.environ["SOURCE_SHA256"],
}
with (root / "ready.json").open("x", encoding="utf-8") as handle:
    json.dump(ready, handle, sort_keys=True)
    handle.write("\n")
PY

export SERVER_READY=1
trap - EXIT
exec python3 - <<'PY'
import hmac
import http.server
import os
import pathlib
import subprocess
import threading
import time

root = pathlib.Path(os.environ["ARTIFACT_ROOT"])
token = os.environ["ARTIFACT_TOKEN"]
pod_id = os.environ.get("RUNPOD_POD_ID", "")
allowed = {
    "/ready.json": root / "ready.json",
    "/device-dispatch-artifacts.tar.gz": root / "device-dispatch-artifacts.tar.gz",
    "/device-dispatch-artifacts.tar.gz.sha256": root
    / "device-dispatch-artifacts.tar.gz.sha256",
}

def delayed_stop():
    time.sleep(15)
    if pod_id:
        subprocess.run(
            ["runpodctl", "pod", "stop", pod_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )

class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "device-dispatch-artifact/1"

    def authorized(self):
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, "Bearer " + token)

    def send_file(self, include_body):
        if not self.authorized():
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Bearer")
            self.end_headers()
            return
        path = allowed.get(self.path)
        if path is None or not path.is_file():
            self.send_error(404)
            return
        payload_size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(payload_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if include_body:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)
            self.wfile.flush()
            if self.path == "/device-dispatch-artifacts.tar.gz":
                threading.Thread(target=delayed_stop, daemon=True).start()

    def do_GET(self):
        self.send_file(True)

    def do_HEAD(self):
        self.send_file(False)

    def log_message(self, format, *args):
        return

http.server.ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
PY
"""


def _validate_launch(args: argparse.Namespace, inventory: dict[str, Any]) -> dict[str, Any]:
    if args.confirm_spend != SPEND_ACK or os.environ.get(SPEND_ENV) != SPEND_ACK:
        raise ValueError(
            f"launch requires both --confirm-spend {SPEND_ACK} and {SPEND_ENV}={SPEND_ACK}"
        )
    if not EXPERIMENT_RE.fullmatch(args.experiment_id):
        raise ValueError("experiment ID must be 1-96 safe filename characters")
    if args.gpu_type not in GPU_IDS_SUPPORTED_BY_SOURCE:
        raise ValueError(
            f"this source snapshot does not declare architecture coverage for {args.gpu_type!r}"
        )
    if not re.fullmatch(r"[A-Z]{2,3}-[A-Z]{2,3}-[0-9]+", args.data_center_id):
        raise ValueError("a live RunPod data-center ID is required")
    agents = _parse_positive_csv(args.agent_counts, name="agent counts", maximum=1_000_000)
    steps = _parse_positive_csv(args.step_counts, name="step counts", maximum=256)
    if not 0 <= args.warmups <= 100:
        raise ValueError("warmups must be in [0, 100]")
    if not 1 <= args.repetitions <= 500:
        raise ValueError("repetitions must be in [1, 500]")
    if len(agents) * len(steps) * args.repetitions > 12_000:
        raise ValueError("bounded pilot limit exceeded")
    if not 5 <= args.max_run_minutes <= 60:
        raise ValueError("max run minutes must be in [5, 60]")
    if not 0 < args.max_cost_usd <= 100:
        raise ValueError("max cost must be in (0, 100]")
    if not 5 <= args.volume_gb <= 100:
        raise ValueError("volume size must be in [5, 100] GB")

    gpu = _gpu_entry(inventory, args.gpu_type)
    _stock_entry(inventory, args.gpu_type, args.data_center_id)
    cloud_key = "secure_cloud" if args.cloud_type == "SECURE" else "community_cloud"
    price_key = (
        "secure_price_per_hour" if args.cloud_type == "SECURE" else "community_price_per_hour"
    )
    if not gpu[cloud_key]:
        raise ValueError(f"{args.gpu_type!r} is not listed in {args.cloud_type} cloud")
    try:
        hourly_price = float(gpu[price_key])
    except (TypeError, ValueError) as error:
        raise ValueError("live inventory did not provide a usable hourly price") from error
    estimated_max_compute = hourly_price * args.max_run_minutes / 60.0
    if estimated_max_compute > args.max_cost_usd:
        raise ValueError(
            f"estimated maximum compute ${estimated_max_compute:.4f} exceeds "
            f"--max-cost-usd ${args.max_cost_usd:.4f}"
        )
    return {
        "agents": agents,
        "steps": steps,
        "hourly_price": hourly_price,
        "estimated_max_compute": estimated_max_compute,
    }


def launch(args: argparse.Namespace) -> None:
    # Gate locally before even performing the read-only launch-time inventory.
    if args.confirm_spend != SPEND_ACK or os.environ.get(SPEND_ENV) != SPEND_ACK:
        raise ValueError(
            f"launch requires both --confirm-spend {SPEND_ACK} and {SPEND_ENV}={SPEND_ACK}"
        )
    inventory = fetch_inventory()
    validated = _validate_launch(args, inventory)
    source_archive = _source_archive()
    source_sha256 = _source_sha256()
    archive_sha256 = hashlib.sha256(source_archive).hexdigest()
    artifact_token = secrets.token_urlsafe(32)
    receipt_path = args.receipt or (
        REPO_ROOT
        / "data/provider-runpod-launches"
        / f"{args.experiment_id}-{_utc_compact()}.launch.json"
    )
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite launch receipt: {receipt_path}")

    environment = {
        "EXECUTION_PROVIDER": "runpod",
        "REQUESTED_GPU": args.gpu_type,
        "REQUESTED_DATA_CENTER": args.data_center_id,
        "REQUESTED_CLOUD_TYPE": args.cloud_type,
        "RUNPOD_IMAGE": CUDA_IMAGE,
        "SOURCE_SHA256": source_sha256,
        "SOURCE_ARCHIVE_SHA256": archive_sha256,
        "SOURCE_ARCHIVE_B64": base64.b64encode(source_archive).decode("ascii"),
        "ARTIFACT_TOKEN": artifact_token,
        "EXPERIMENT_ID": args.experiment_id,
        "AGENT_COUNTS": args.agent_counts,
        "STEP_COUNTS": args.step_counts,
        "WARMUPS": str(args.warmups),
        "REPETITIONS": str(args.repetitions),
        "SEED": str(args.seed),
        "MAX_RUN_SECONDS": str(args.max_run_minutes * 60),
    }
    payload = {
        "name": f"gpu-agent-{args.experiment_id}",
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": [args.gpu_type],
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "dataCenterIds": [args.data_center_id],
        "dataCenterPriority": "custom",
        "imageName": CUDA_IMAGE,
        "allowedCudaVersions": ["13.0"],
        "containerDiskInGb": 20,
        "volumeInGb": args.volume_gb,
        "volumeMountPath": "/workspace",
        "ports": [f"{ARTIFACT_PORT}/http"],
        "globalNetworking": False,
        "interruptible": False,
        "locked": False,
        "minVCPUPerGPU": 4,
        "minRAMPerGPU": 16,
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [BOOTSTRAP],
        "env": environment,
    }
    response = _request_json(
        "POST",
        f"{REST_URL}/pods",
        body=payload,
        extra_redactions=(artifact_token,),
        timeout=60,
    )
    if not isinstance(response, dict) or not response.get("id"):
        raise RunPodError("RunPod create response did not include a Pod ID")
    pod_id = str(response["id"])
    receipt = {
        "schema_version": "runpod-device-dispatch-launch-v1",
        "created_at_utc": _utc_now(),
        "pod_id": pod_id,
        "experiment_id": args.experiment_id,
        "artifact_url": f"https://{pod_id}-{ARTIFACT_PORT}.proxy.runpod.net",
        "artifact_token": artifact_token,
        "request": {
            "gpu_type": args.gpu_type,
            "data_center_id": args.data_center_id,
            "cloud_type": args.cloud_type,
            "image": CUDA_IMAGE,
            "agent_counts": validated["agents"],
            "step_counts": validated["steps"],
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "max_run_minutes": args.max_run_minutes,
            "volume_gb": args.volume_gb,
        },
        "cost_guard": {
            "inventory_price_per_hour": validated["hourly_price"],
            "estimated_max_compute_usd": validated["estimated_max_compute"],
            "user_cap_usd": args.max_cost_usd,
            "storage_excluded": True,
        },
        "source": {
            "source_sha256": source_sha256,
            "archive_sha256": archive_sha256,
        },
        "security": {
            "mode": "0600",
            "contains_ephemeral_artifact_token": True,
            "contains_runpod_api_key": False,
        },
        "lifecycle": {
            "automatic_stop_after_minutes": args.max_run_minutes,
            "automatic_stop_after_successful_download": True,
            "terminate_implemented_by_this_adapter": False,
            "stopped_volume_storage_can_continue_to_accrue_cost": True,
        },
    }
    _write_json_new(receipt_path, receipt, private=True)
    print(f"pod_created={pod_id}")
    print(f"launch_receipt={receipt_path.resolve()}")
    print(f"artifact_url={receipt['artifact_url']}")
    print(f"estimated_max_compute_usd={validated['estimated_max_compute']:.6f}")
    print("artifact_token_printed=false")
    print("termination_performed=false")
    print("warning=stopped volume storage persists until you explicitly terminate the Pod")


def _authenticated_download(
    url: str, token: str, timeout: float, *, max_bytes: int = 256 * 1024 * 1024
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Cache-Control": "no-store",
            "User-Agent": "gpu-agent-crossover-runpod-collector/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as error:
                    raise RunPodError(
                        "RunPod artifact returned an invalid Content-Length"
                    ) from error
                if declared_size > max_bytes:
                    raise RunPodError("RunPod artifact exceeds the collector size limit")
            payload = response.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise RunPodError("RunPod artifact exceeds the collector size limit")
            return payload
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        raise RunPodError("RunPod artifact endpoint is not ready") from error


def _sanitize_pod(raw: dict[str, Any]) -> dict[str, Any]:
    machine = raw.get("machine") if isinstance(raw.get("machine"), dict) else {}
    gpu = raw.get("gpu") if isinstance(raw.get("gpu"), dict) else {}
    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "desired_status": raw.get("desiredStatus"),
        "image": raw.get("image"),
        "interruptible": raw.get("interruptible"),
        "last_started_at": raw.get("lastStartedAt"),
        "last_status_change": raw.get("lastStatusChange"),
        "cost_per_hour": raw.get("costPerHr"),
        "adjusted_cost_per_hour": raw.get("adjustedCostPerHr"),
        "machine_id": raw.get("machineId"),
        "memory_gb": raw.get("memoryInGb"),
        "vcpu_count": raw.get("vcpuCount"),
        "gpu": {
            "id": gpu.get("id"),
            "display_name": gpu.get("displayName"),
            "count": gpu.get("count"),
        },
        "machine": {
            "gpu_type_id": machine.get("gpuTypeId"),
            "data_center_id": machine.get("dataCenterId"),
            "location": machine.get("location"),
            "secure_cloud": machine.get("secureCloud"),
            "cpu_count": machine.get("cpuCount"),
            "cpu_type_id": machine.get("cpuTypeId"),
            "disk_throughput_mbps": machine.get("diskThroughputMBps"),
            "max_download_mbps": machine.get("maxDownloadSpeedMbps"),
            "max_upload_mbps": machine.get("maxUploadSpeedMbps"),
            "current_price_per_gpu": machine.get("currentPricePerGpu"),
        },
        "environment_recorded": False,
    }


def _safe_extract(archive_bytes: bytes, destination: Path) -> list[str]:
    extracted = []
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members or len(members) > 100:
            raise ValueError("artifact archive has an invalid member count")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"unsafe artifact path: {member.name!r}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported artifact member: {member.name!r}")
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported artifact member: {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read artifact member: {member.name!r}")
            with target.open("xb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(member.name)
    return sorted(extracted)


def collect(args: argparse.Namespace) -> None:
    if args.receipt is None:
        raise ValueError("collect requires --receipt PATH")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    token = str(receipt.get("artifact_token") or "")
    pod_id = str(receipt.get("pod_id") or "")
    base_url = str(receipt.get("artifact_url") or "")
    if not token or not pod_id or not base_url:
        raise ValueError("launch receipt is missing required fields")

    raw_pod = _request_json("GET", f"{REST_URL}/pods/{pod_id}?includeMachine=true", timeout=30)
    if not isinstance(raw_pod, dict):
        raise RunPodError("RunPod Pod response had an unexpected shape")
    deadline = time.monotonic() + args.collect_timeout_seconds
    ready = None
    while time.monotonic() < deadline:
        try:
            ready = json.loads(_authenticated_download(f"{base_url}/ready.json", token, 20))
            if ready.get("ready") is True:
                break
        except (RunPodError, json.JSONDecodeError):
            pass
        time.sleep(5)
    if not isinstance(ready, dict) or ready.get("ready") is not True:
        raise RunPodError(
            "artifact readiness timeout; the Pod watchdog may have stopped it, so inspect "
            "the persisted /workspace volume before considering a paid restart"
        )

    # Refresh after readiness so machine allocation fields are not frozen from
    # an initializing-Pod response. Do this before the archive GET schedules a stop.
    raw_pod = _request_json("GET", f"{REST_URL}/pods/{pod_id}?includeMachine=true", timeout=30)
    if not isinstance(raw_pod, dict):
        raise RunPodError("RunPod Pod response had an unexpected shape after readiness")

    checksum_text = _authenticated_download(
        f"{base_url}/device-dispatch-artifacts.tar.gz.sha256", token, 30
    ).decode("ascii", errors="strict")
    expected_sha = checksum_text.split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("artifact endpoint returned an invalid checksum")
    archive_bytes = _authenticated_download(
        f"{base_url}/device-dispatch-artifacts.tar.gz", token, 100
    )
    observed_sha = hashlib.sha256(archive_bytes).hexdigest()
    if not hmac.compare_digest(expected_sha, observed_sha):
        raise ValueError("artifact checksum mismatch")

    destination = args.output_dir / f"runpod-{receipt['experiment_id']}-{pod_id}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    with (destination / "device-dispatch-artifacts.tar.gz").open("xb") as handle:
        handle.write(archive_bytes)
    extracted = _safe_extract(archive_bytes, destination)
    provider_path = destination / "provider-metadata.json"
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    if provider.get("source", {}).get("source_sha256") != receipt["source"]["source_sha256"]:
        raise ValueError("provider metadata source digest disagrees with launch receipt")
    if provider.get("pod", {}).get("pod_id") != pod_id:
        raise ValueError("provider metadata Pod ID disagrees with launch receipt")

    native_manifests = list((destination / "results").glob("*.manifest.json"))
    native_validation: dict[str, Any] = {"manifest_count": len(native_manifests)}
    if len(native_manifests) == 1:
        native = json.loads(native_manifests[0].read_text(encoding="utf-8"))
        native_validation.update(
            {
                "execution_provider": native.get("execution_provider"),
                "source_sha256_matches": native.get("source_sha256")
                == receipt["source"]["source_sha256"],
                "requested_gpu_matches": native.get("requested_gpu")
                == receipt["request"]["gpu_type"],
                "measured_rows": native.get("results", {}).get("measured_rows"),
                "failure_rows": native.get("results", {}).get("failure_rows"),
            }
        )
    collection = {
        "schema_version": "runpod-device-dispatch-collection-v1",
        "collected_at_utc": _utc_now(),
        "launch_receipt": str(args.receipt.resolve()),
        "pod": _sanitize_pod(raw_pod),
        "ready": ready,
        "archive_sha256": observed_sha,
        "archive_bytes": len(archive_bytes),
        "extracted_files": extracted,
        "native_validation": native_validation,
        "artifact_token_recorded": False,
        "runpod_api_key_recorded": False,
        "remote_mutations_by_collector": 0,
        "note": "The remote artifact server schedules a Pod stop after a successful archive response.",
    }
    _write_json_new(destination / "collection-receipt.json", collection)
    print(f"collected_to={destination.resolve()}")
    print(f"archive_sha256={observed_sha}")
    print(f"extracted_files={len(extracted)}")
    print("artifact_token_printed=false")
    print("termination_performed=false")


def print_inventory(inventory: dict[str, Any]) -> None:
    counts = inventory["counts"]
    print("authentication_ok=true")
    print(f"captured_at_utc={inventory['captured_at_utc']}")
    for key in (
        "catalog_gpu_types",
        "data_centers",
        "stocked_gpu_types",
        "data_centers_with_stock",
        "stocked_gpu_datacenter_offers",
    ):
        print(f"{key}={counts[key]}")
    for gpu_type in inventory["stocked_gpu_type_ids"]:
        offers = [
            f"{offer['data_center_id']}:{offer['stock_status']}"
            for offer in inventory["available_offers"]
            if offer["gpu_type_id"] == gpu_type
        ]
        print(f"available\t{gpu_type}\t{','.join(offers)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=("plan", "inventory", "launch", "collect"), default="plan"
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--inventory-output", type=Path)
    parser.add_argument("--confirm-spend", default="")
    parser.add_argument("--experiment-id", default="device-dispatch-runpod-pilot")
    parser.add_argument("--gpu-type", default="NVIDIA L4")
    parser.add_argument("--data-center-id", default="")
    parser.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), default="SECURE")
    parser.add_argument("--agent-counts", default="32,256,2048,16384")
    parser.add_argument("--step-counts", default="1,8,64")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-run-minutes", type=int, default=20)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--volume-gb", type=int, default=10)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/raw")
    parser.add_argument("--collect-timeout-seconds", type=int, default=1200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _load_env_file(args.env_file)
    if args.action == "plan":
        archive = _source_archive()
        plan = {
            "action": "plan",
            "remote_calls": 0,
            "runpod_api_key_configured": bool(os.environ.get("RUNPOD_API_KEY")),
            "image": CUDA_IMAGE,
            "source_sha256": _source_sha256(),
            "source_archive_sha256": hashlib.sha256(archive).hexdigest(),
            "source_archive_bytes": len(archive),
            "launch_cli_gate": f"--confirm-spend {SPEND_ACK}",
            "launch_environment_gate": f"{SPEND_ENV}={SPEND_ACK}",
            "automatic_stop_deadline_minutes": args.max_run_minutes,
            "terminate_operation_implemented": False,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.action == "inventory":
        inventory = fetch_inventory()
        print_inventory(inventory)
        if args.inventory_output is not None:
            _write_json_new(args.inventory_output, inventory)
            print(f"inventory_output={args.inventory_output.resolve()}")
        return
    if args.action == "launch":
        launch(args)
        return
    if not 30 <= args.collect_timeout_seconds <= 3600:
        raise ValueError("collect timeout must be in [30, 3600] seconds")
    collect(args)


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, OSError, RunPodError, ValueError, json.JSONDecodeError) as error:
        print(f"error={_redact(str(error))}", file=sys.stderr)
        raise SystemExit(2) from error
