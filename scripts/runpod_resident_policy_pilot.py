"""Safely run the frozen resident-policy CUDA pilot on exactly one RunPod L4.

The default ``plan`` action is local-only. ``inventory`` is read-only. Creating
a Pod requires the same literal acknowledgement in a CLI flag and an
environment variable. Deleting the exact Pod and its volume is a separate,
double-gated action that is rejected unless a locally retained collection
receipt proves that the downloaded artifact bundle passed every validity gate.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import gzip
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "native/resident_policy"
DEFAULT_ENV_FILE = REPO_ROOT.parent / ".env"

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"
CUDA_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
GPU_TYPE = "NVIDIA L4"
SCHEMA_VERSION = "resident-policy-v1"
FROZEN_SOURCE_SHA256 = "4b5cdcb9496a734bd7801d5c419efb8eceb72fd6962800520101e89676d204da"
FROZEN_MAKEFILE_SHA256 = "d74935b594fb629b2113d237439289e057281becc561b6941d1134bd6a1c1351"
SOURCE_FILES = ("Makefile", "README.md", "resident_policy_pilot.cu")
MECHANISMS = {"host_roundtrip", "device_resident", "no_decision_lower_bound"}

LAUNCH_ACK = "RUNPOD_RESIDENT_POLICY_PILOT"
LAUNCH_GATE_ENV = "RUNPOD_ENABLE_GPU_SPEND"
TERMINATE_GATE_ENV = "RUNPOD_TERMINATE_POD_ID"
ARTIFACT_PORT = 8000
KNOWN_STOCK = {"high", "medium", "low"}
EXPERIMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
DATA_CENTER_RE = re.compile(r"[A-Z]{2,3}-[A-Z]{2,3}-[0-9]+\Z")
POD_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,95}\Z")

FULL_CONFIG: dict[str, Any] = {
    "agent_counts": [256, 2048, 16384],
    "epoch_counts": [2, 8, 32],
    "warmups_per_mechanism_cell": 5,
    "calibration_samples_per_mechanism_cell": 3,
    "repetitions_per_mechanism_cell": 30,
    "min_duration_target_ns": 100_000_000,
    "max_batch_iterations": 20_000,
    "seed": 20260811,
    "block_size": 256,
}
SMOKE_CONFIG: dict[str, Any] = {
    "agent_counts": [64, 256],
    "epoch_counts": [2, 4],
    "warmups_per_mechanism_cell": 1,
    "calibration_samples_per_mechanism_cell": 2,
    "repetitions_per_mechanism_cell": 2,
    "min_duration_target_ns": 2_000_000,
    "max_batch_iterations": 1_000,
    "seed": 20260811,
    "block_size": 256,
}


class RunPodError(RuntimeError):
    """A deliberately redacted RunPod control-plane error."""


class RunPodHTTPError(RunPodError):
    """A redacted HTTP error retaining only the response status."""

    def __init__(self, status_code: int):
        super().__init__(f"RunPod HTTP request failed with status {status_code}")
        self.status_code = status_code


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_env_file(path: Path) -> None:
    """Load simple dotenv assignments without displaying secret values."""

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
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
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
            "RUNPOD_API_KEY is not configured in the environment or selected env file"
        )
    return key


def _redact(text: str, extra_values: tuple[str, ...] = ()) -> str:
    redacted = text
    for value in (os.environ.get("RUNPOD_API_KEY", ""), *extra_values):
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted[:500]


def _request_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 30.0,
    extra_redactions: tuple[str, ...] = (),
    allow_empty: bool = False,
) -> Any:
    encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=encoded,
        method=method,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "User-Agent": "gpu-agent-crossover-resident-policy/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        raise RunPodHTTPError(error.code) from error
    except urllib.error.URLError as error:
        reason = _redact(str(error.reason), extra_redactions)
        raise RunPodError(f"RunPod network request failed: {reason}") from error
    if not payload and allow_empty:
        return None
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
    """Return a sanitized point-in-time inventory; this operation is read-only."""

    query = """
    query ResidentPolicyInventory {
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
    if not isinstance(myself, dict) or not myself.get("id"):
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
                available_offers.append({"data_center_id": dc_id, "location": location, **offer})
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
    return {
        "schema_version": "runpod-resident-policy-inventory-v1",
        "captured_at_utc": _utc_now(),
        "source": {
            "api": GRAPHQL_URL,
            "operation": "ResidentPolicyInventory",
            "mutating_calls": 0,
        },
        "authentication": {"ok": True, "identity_redacted": True},
        "gpu_types": gpu_types,
        "data_centers": data_centers,
        "available_offers": available_offers,
        "counts": {
            "catalog_gpu_types": len(gpu_types),
            "data_centers": len(data_centers),
            "stocked_gpu_datacenter_offers": len(available_offers),
        },
        "stock_semantics": {
            "included_as_available": ["High", "Medium", "Low"],
            "note": "Availability is volatile and is re-read immediately before launch.",
        },
    }


def _validate_frozen_inputs() -> dict[str, str]:
    observed = {
        "source_sha256": _sha256_file(NATIVE_ROOT / "resident_policy_pilot.cu"),
        "makefile_sha256": _sha256_file(NATIVE_ROOT / "Makefile"),
    }
    expected = {
        "source_sha256": FROZEN_SOURCE_SHA256,
        "makefile_sha256": FROZEN_MAKEFILE_SHA256,
    }
    if observed != expected:
        raise ValueError(
            "frozen resident-policy inputs changed; assign a new experiment ID before running: "
            f"expected={expected}, observed={observed}"
        )
    return observed


def _source_archive() -> bytes:
    """Build a deterministic archive containing only the reviewed source files."""

    _validate_frozen_inputs()
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
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600 if private else 0o644,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _mode_config(mode: str) -> dict[str, Any]:
    if mode == "full":
        return FULL_CONFIG
    if mode == "smoke":
        return SMOKE_CONFIG
    raise ValueError("mode must be one of: smoke, full")


def _expected_row_count(config: dict[str, Any]) -> int:
    return (
        len(config["agent_counts"])
        * len(config["epoch_counts"])
        * len(MECHANISMS)
        * config["repetitions_per_mechanism_cell"]
    )


def _gpu_entry(inventory: dict[str, Any]) -> dict[str, Any]:
    for entry in inventory["gpu_types"]:
        if entry["id"] == GPU_TYPE:
            return entry
    raise ValueError(f"{GPU_TYPE!r} is absent from the live RunPod catalog")


def _stock_entry(inventory: dict[str, Any], data_center_id: str) -> dict[str, Any]:
    for entry in inventory["available_offers"]:
        if entry["gpu_type_id"] == GPU_TYPE and entry["data_center_id"] == data_center_id:
            return entry
    raise ValueError(f"live inventory reports no {GPU_TYPE!r} stock in {data_center_id!r}")


def _require_launch_gate(args: argparse.Namespace) -> None:
    if args.confirm_spend != LAUNCH_ACK or os.environ.get(LAUNCH_GATE_ENV) != LAUNCH_ACK:
        raise ValueError(
            f"launch requires both --confirm-spend {LAUNCH_ACK} and {LAUNCH_GATE_ENV}={LAUNCH_ACK}"
        )


def _validate_launch(args: argparse.Namespace, inventory: dict[str, Any]) -> dict[str, Any]:
    _require_launch_gate(args)
    _validate_frozen_inputs()
    config = _mode_config(args.mode)
    if EXPERIMENT_RE.fullmatch(args.experiment_id) is None:
        raise ValueError("experiment ID must be 1-96 safe filename characters")
    if DATA_CENTER_RE.fullmatch(args.data_center_id) is None:
        raise ValueError("a live RunPod data-center ID is required")
    if CUDA_IMAGE != "nvidia/cuda:13.0.1-devel-ubuntu24.04":
        raise ValueError("the reviewed CUDA 13 image reference changed")
    if not 5 <= args.max_run_minutes <= 60:
        raise ValueError("max run minutes must be in [5, 60]")
    if not 0 < args.max_cost_usd <= 10:
        raise ValueError("max cost must be in (0, 10]")
    if not 5 <= args.volume_gb <= 50:
        raise ValueError("volume size must be in [5, 50] GB")

    gpu = _gpu_entry(inventory)
    stock = _stock_entry(inventory, args.data_center_id)
    cloud_key = "secure_cloud" if args.cloud_type == "SECURE" else "community_cloud"
    price_key = (
        "secure_price_per_hour" if args.cloud_type == "SECURE" else "community_price_per_hour"
    )
    if not gpu[cloud_key]:
        raise ValueError(f"{GPU_TYPE!r} is not listed in {args.cloud_type} cloud")
    try:
        hourly_price = float(gpu[price_key])
    except (TypeError, ValueError) as error:
        raise ValueError("live inventory did not provide a usable L4 hourly price") from error
    if not math.isfinite(hourly_price) or hourly_price <= 0:
        raise ValueError("live inventory returned a nonpositive L4 hourly price")
    estimated_max_compute = hourly_price * args.max_run_minutes / 60.0
    if estimated_max_compute > args.max_cost_usd:
        raise ValueError(
            f"estimated maximum compute ${estimated_max_compute:.4f} exceeds "
            f"--max-cost-usd ${args.max_cost_usd:.4f}"
        )
    return {
        "config": config,
        "hourly_price": hourly_price,
        "estimated_max_compute": estimated_max_compute,
        "stock": stock,
    }


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
    curl --silent --show-error --fail --request POST \
      --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
      "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
      >/dev/null 2>&1 || true
  fi
}

artifact_parent="/workspace/runpod-resident-policy/${EXPERIMENT_ID}"
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
  g++ make python3 ca-certificates curl >>"$artifact_root/dependency-install.log" 2>&1

mkdir "$artifact_root/source" "$artifact_root/results"
printf '%s' "$SOURCE_ARCHIVE_B64" | base64 -d >"$artifact_root/source/source.tar.gz"
printf '%s  %s\n' "$SOURCE_ARCHIVE_SHA256" "$artifact_root/source/source.tar.gz" \
  | sha256sum --check --strict
archive_listing=$(tar -tzf "$artifact_root/source/source.tar.gz")
if [ "$archive_listing" != $'Makefile\nREADME.md\nresident_policy_pilot.cu' ]; then
  echo "source archive contains unexpected paths" >&2
  exit 43
fi
tar -xzf "$artifact_root/source/source.tar.gz" \
  -C "$artifact_root/source" --no-same-owner

python3 - "$artifact_root/source" "$FROZEN_SOURCE_SHA256" \
  "$FROZEN_MAKEFILE_SHA256" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_source = sys.argv[2]
expected_makefile = sys.argv[3]
source = hashlib.sha256((root / "resident_policy_pilot.cu").read_bytes()).hexdigest()
makefile = hashlib.sha256((root / "Makefile").read_bytes()).hexdigest()
if source != expected_source or makefile != expected_makefile:
    raise SystemExit("frozen source digest mismatch")
PY

compile_command=(
  make -C "$artifact_root/source"
  TARGET="$artifact_root/resident_policy_pilot"
  all
)
printf '%q ' "${compile_command[@]}" >"$artifact_root/compile-command.txt"
printf '\n' >>"$artifact_root/compile-command.txt"

set +e
"${compile_command[@]}" >"$artifact_root/compiler.stdout.log" \
  2>"$artifact_root/compiler.stderr.log"
compile_return_code=$?
program_return_code=127
binary_sha256=""
if [ "$compile_return_code" -eq 0 ]; then
  binary_sha256=$(sha256sum "$artifact_root/resident_policy_pilot" | cut -d' ' -f1)
  export BINARY_SHA256="$binary_sha256"
  program_command=(
    "$artifact_root/resident_policy_pilot"
    --experiment-id "$EXPERIMENT_ID"
    --output-dir "$artifact_root/results"
  )
  if [ "$RUN_MODE" = "smoke" ]; then
    program_command+=(--smoke)
  elif [ "$RUN_MODE" = "full" ]; then
    program_command+=(
      --agents 256,2048,16384
      --epochs 2,8,32
      --warmups 5
      --calibration-samples 3
      --repetitions 30
      --min-duration-ms 100
      --max-batch 20000
      --seed 20260811
      --block-size 256
    )
  else
    echo "unrecognized run mode" >"$artifact_root/program.stderr.log"
    program_return_code=98
    program_command=()
  fi
  if [ "${#program_command[@]}" -gt 0 ]; then
    printf '%q ' "${program_command[@]}" >"$artifact_root/program-command.txt"
    printf '\n' >>"$artifact_root/program-command.txt"
    SOURCE_SHA256="$FROZEN_SOURCE_SHA256" \
    EXECUTION_PROVIDER=runpod \
    REQUESTED_GPU="$REQUESTED_GPU" \
    PLACEMENT_ID="${RUNPOD_POD_ID:-unknown-pod}" \
    IMAGE_DIGEST="registry-ref:${RUNPOD_IMAGE}" \
      "${program_command[@]}" >"$artifact_root/program.stdout.log" \
      2>"$artifact_root/program.stderr.log"
    program_return_code=$?
  fi
else
  : >"$artifact_root/program.stdout.log"
  printf 'program not run because compilation failed\n' \
    >"$artifact_root/program.stderr.log"
  : >"$artifact_root/program-command.txt"
fi
set -e

export ARTIFACT_ROOT="$artifact_root"
export COMPILE_RETURN_CODE="$compile_return_code"
export PROGRAM_RETURN_CODE="$program_return_code"
export BINARY_SHA256="$binary_sha256"

python3 - <<'PY'
import datetime
import glob
import hashlib
import json
import os
import pathlib
import platform
import subprocess

root = pathlib.Path(os.environ["ARTIFACT_ROOT"])

def command_output(command):
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
    except Exception as error:
        return f"unavailable:{type(error).__name__}"

gpu_rows = []
query = [
    "nvidia-smi",
    "--query-gpu=uuid,name,driver_version,pci.bus_id,memory.total,"
    "compute_cap,clocks.max.sm,power.limit",
    "--format=csv,noheader,nounits",
]
output = command_output(query)
if not output.startswith("unavailable:"):
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 8:
            gpu_rows.append(
                {
                    "uuid": fields[0],
                    "name": fields[1],
                    "driver_version": fields[2],
                    "pci_bus_id": fields[3],
                    "memory_mib": fields[4],
                    "compute_capability": fields[5],
                    "max_sm_clock_mhz": fields[6],
                    "power_limit_w": fields[7],
                }
            )
else:
    gpu_rows = [{"discovery_error": output}]

cpu_model = "unknown"
try:
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
except OSError:
    pass

manifests = sorted(glob.glob(str(root / "results" / "*.manifest.json")))
native_summary = None
if len(manifests) == 1:
    native = json.loads(pathlib.Path(manifests[0]).read_text())
    native_summary = {
        "run_id": native.get("run_id"),
        "manifest_file": pathlib.Path(manifests[0]).name,
        "results": native.get("results"),
        "hardware": native.get("hardware"),
    }

metadata = {
    "schema_version": "runpod-resident-policy-provider-v1",
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
        "image_reference": os.environ.get("RUNPOD_IMAGE"),
        "allowed_cuda_version": "13.0",
        "mode": os.environ.get("RUN_MODE"),
    },
    "source": {
        "source_sha256": os.environ.get("FROZEN_SOURCE_SHA256"),
        "makefile_sha256": os.environ.get("FROZEN_MAKEFILE_SHA256"),
        "source_archive_sha256": os.environ.get("SOURCE_ARCHIVE_SHA256"),
        "binary_sha256": os.environ.get("BINARY_SHA256"),
    },
    "execution": {
        "compile_return_code": int(os.environ["COMPILE_RETURN_CODE"]),
        "program_return_code": int(os.environ["PROGRAM_RETURN_CODE"]),
        "native_summary": native_summary,
    },
    "host": {
        "platform": platform.platform(),
        "node": platform.node(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
    },
    "software": {
        "nvcc_version": command_output(["nvcc", "--version"]),
        "python_version": platform.python_version(),
    },
    "gpus": gpu_rows,
    "secrets_recorded": False,
}
with (root / "provider-metadata.json").open("x", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
    handle.write("\n")

indexed = {}
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "artifact-index.json":
        continue
    relative = path.relative_to(root).as_posix()
    if relative.endswith(".tar.gz.tmp") or relative in {
        "resident-policy-artifacts.tar.gz",
        "resident-policy-artifacts.tar.gz.sha256",
        "ready.json",
    }:
        continue
    payload = path.read_bytes()
    indexed[relative] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
index = {
    "schema_version": "runpod-resident-policy-artifact-index-v1",
    "files": indexed,
}
with (root / "artifact-index.json").open("x", encoding="utf-8") as handle:
    json.dump(index, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

tar -C "$artifact_root" -czf "$artifact_root/resident-policy-artifacts.tar.gz.tmp" \
  source resident_policy_pilot results provider-metadata.json artifact-index.json \
  compile-command.txt program-command.txt dependency-install.log \
  compiler.stdout.log compiler.stderr.log program.stdout.log program.stderr.log
mv "$artifact_root/resident-policy-artifacts.tar.gz.tmp" \
  "$artifact_root/resident-policy-artifacts.tar.gz"
sha256sum "$artifact_root/resident-policy-artifacts.tar.gz" \
  | sed 's# .*/#  #' >"$artifact_root/resident-policy-artifacts.tar.gz.sha256"

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
    "source_sha256": os.environ["FROZEN_SOURCE_SHA256"],
    "makefile_sha256": os.environ["FROZEN_MAKEFILE_SHA256"],
    "binary_sha256": os.environ.get("BINARY_SHA256", ""),
    "mode": os.environ["RUN_MODE"],
}
with (root / "ready.json").open("x", encoding="utf-8") as handle:
    json.dump(ready, handle, sort_keys=True)
    handle.write("\n")
PY

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
    "/resident-policy-artifacts.tar.gz": root / "resident-policy-artifacts.tar.gz",
    "/resident-policy-artifacts.tar.gz.sha256": root
    / "resident-policy-artifacts.tar.gz.sha256",
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
    server_version = "resident-policy-artifact/1"

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
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if include_body:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)
            self.wfile.flush()
            if self.path == "/resident-policy-artifacts.tar.gz":
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


def launch(args: argparse.Namespace) -> None:
    # Refuse locally before the read-only, launch-time inventory request.
    _require_launch_gate(args)
    inventory = fetch_inventory()
    validated = _validate_launch(args, inventory)
    source_archive = _source_archive()
    archive_sha256 = _sha256_bytes(source_archive)
    artifact_token = secrets.token_urlsafe(32)
    receipt_path = args.receipt or (
        REPO_ROOT
        / "data/provider-runpod-launches"
        / f"{args.experiment_id}-{_utc_compact()}.launch.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite launch receipt: {receipt_path}")

    environment = {
        "EXECUTION_PROVIDER": "runpod",
        "REQUESTED_GPU": GPU_TYPE,
        "REQUESTED_DATA_CENTER": args.data_center_id,
        "REQUESTED_CLOUD_TYPE": args.cloud_type,
        "RUNPOD_IMAGE": CUDA_IMAGE,
        "RUN_MODE": args.mode,
        "FROZEN_SOURCE_SHA256": FROZEN_SOURCE_SHA256,
        "FROZEN_MAKEFILE_SHA256": FROZEN_MAKEFILE_SHA256,
        "SOURCE_ARCHIVE_SHA256": archive_sha256,
        "SOURCE_ARCHIVE_B64": base64.b64encode(source_archive).decode("ascii"),
        "ARTIFACT_TOKEN": artifact_token,
        "EXPERIMENT_ID": args.experiment_id,
        "MAX_RUN_SECONDS": str(args.max_run_minutes * 60),
    }
    payload = {
        "name": f"gpu-agent-{args.experiment_id}",
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": [GPU_TYPE],
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
    if payload["gpuTypeIds"] != [GPU_TYPE] or payload["gpuCount"] != 1:
        raise AssertionError("internal safety error: launch payload is not exactly one L4")
    if payload["imageName"] != CUDA_IMAGE or payload["allowedCudaVersions"] != ["13.0"]:
        raise AssertionError(
            "internal safety error: launch image is not the reviewed CUDA 13 image"
        )

    response = _request_json(
        "POST",
        f"{REST_URL}/pods",
        body=payload,
        timeout=60,
        extra_redactions=(artifact_token,),
    )
    if not isinstance(response, dict) or not response.get("id"):
        raise RunPodError("RunPod create response did not include a Pod ID")
    pod_id = str(response["id"])
    if POD_ID_RE.fullmatch(pod_id) is None:
        raise RunPodError("RunPod create response included an unsafe Pod ID")

    receipt = {
        "schema_version": "runpod-resident-policy-launch-v1",
        "created_at_utc": _utc_now(),
        "pod_id": pod_id,
        "pod_name": payload["name"],
        "experiment_id": args.experiment_id,
        "artifact_url": f"https://{pod_id}-{ARTIFACT_PORT}.proxy.runpod.net",
        "artifact_token": artifact_token,
        "request": {
            "gpu_type": GPU_TYPE,
            "gpu_count": 1,
            "data_center_id": args.data_center_id,
            "cloud_type": args.cloud_type,
            "image": CUDA_IMAGE,
            "allowed_cuda_versions": ["13.0"],
            "mode": args.mode,
            "config": validated["config"],
            "max_run_minutes": args.max_run_minutes,
            "volume_gb": args.volume_gb,
        },
        "live_preflight": {
            "inventory_captured_at_utc": inventory["captured_at_utc"],
            "stock_status": validated["stock"]["stock_status"],
            "stock_data_center_id": validated["stock"]["data_center_id"],
            "price_per_hour": validated["hourly_price"],
        },
        "cost_guard": {
            "estimated_max_compute_usd": validated["estimated_max_compute"],
            "user_cap_usd": args.max_cost_usd,
            "storage_excluded": True,
        },
        "source": {
            "source_sha256": FROZEN_SOURCE_SHA256,
            "makefile_sha256": FROZEN_MAKEFILE_SHA256,
            "source_archive_sha256": archive_sha256,
        },
        "security": {
            "mode": "0600",
            "contains_ephemeral_artifact_token": True,
            "contains_runpod_api_key": False,
        },
        "lifecycle": {
            "automatic_stop_after_minutes": args.max_run_minutes,
            "automatic_stop_after_successful_download": True,
            "delete_requires_verified_collection": True,
            "stopped_volume_storage_can_continue_to_accrue_cost": True,
        },
    }
    _write_json_new(receipt_path, receipt, private=True)
    print(f"pod_created={pod_id}")
    print(f"launch_receipt={receipt_path.resolve()}")
    print(f"artifact_url={receipt['artifact_url']}")
    print(f"estimated_max_compute_usd={validated['estimated_max_compute']:.6f}")
    print("requested_gpu_count=1")
    print("requested_gpu=NVIDIA L4")
    print("artifact_token_printed=false")
    print("termination_performed=false")
    print("warning=stopped volume storage persists until verified explicit termination")


def _authenticated_download(
    url: str,
    token: str,
    timeout: float,
    *,
    max_bytes: int = 512 * 1024 * 1024,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Cache-Control": "no-store",
            "User-Agent": "gpu-agent-crossover-resident-policy-collector/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as error:
                    raise RunPodError("artifact returned an invalid Content-Length") from error
                if declared_size > max_bytes:
                    raise RunPodError("artifact exceeds the collector size limit")
            payload = response.read(max_bytes + 1)
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        raise RunPodError("RunPod artifact endpoint is not ready") from error
    if len(payload) > max_bytes:
        raise RunPodError("artifact exceeds the collector size limit")
    return payload


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
            "current_price_per_gpu": machine.get("currentPricePerGpu"),
        },
        "environment_recorded": False,
    }


def _get_pod_or_none(pod_id: str) -> dict[str, Any] | None:
    try:
        value = _request_json("GET", f"{REST_URL}/pods/{pod_id}?includeMachine=true")
    except RunPodHTTPError as error:
        if error.status_code == 404:
            return None
        raise
    if not isinstance(value, dict):
        raise RunPodError("RunPod Pod response had an unexpected shape")
    if str(value.get("id") or "") != pod_id:
        raise RunPodError("RunPod returned a different Pod than requested")
    return value


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


def _artifact_index_errors(destination: Path, extracted: list[str]) -> list[str]:
    errors = []
    index_path = destination / "artifact-index.json"
    try:
        index = _read_json_object(index_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return [f"artifact index is unreadable: {error}"]
    if index.get("schema_version") != "runpod-resident-policy-artifact-index-v1":
        errors.append("artifact index schema mismatch")
    files = index.get("files")
    if not isinstance(files, dict):
        return errors + ["artifact index has no file map"]
    expected_paths = {name.rstrip("/") for name in extracted if name != "artifact-index.json"}
    indexed_paths = set(files)
    if expected_paths != indexed_paths:
        errors.append("artifact index paths do not exactly match extracted regular files")
    for relative, expected in files.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"artifact index contains unsafe path {relative!r}")
            continue
        local = destination.joinpath(*path.parts)
        if not local.is_file() or not isinstance(expected, dict):
            errors.append(f"artifact index entry is missing or malformed: {relative}")
            continue
        observed_bytes = local.stat().st_size
        observed_sha = _sha256_file(local)
        if expected.get("bytes") != observed_bytes:
            errors.append(f"artifact byte count mismatch: {relative}")
        if expected.get("sha256") != observed_sha:
            errors.append(f"artifact SHA-256 mismatch: {relative}")
    return errors


def _native_validation_errors(
    *,
    csv_path: Path,
    manifest_path: Path,
    provider: dict[str, Any],
    receipt: dict[str, Any],
    ready: dict[str, Any],
    pod: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    mode = str(receipt.get("request", {}).get("mode") or "")
    try:
        config = _mode_config(mode)
    except ValueError as error:
        return [str(error)]
    expected_rows = _expected_row_count(config)
    experiment_id = str(receipt.get("experiment_id") or "")
    pod_id = str(receipt.get("pod_id") or "")

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        manifest = _read_json_object(manifest_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return [f"native manifest is unreadable: {error}"]
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = set(reader.fieldnames or [])
    except (OSError, csv.Error) as error:
        return [f"native CSV is unreadable: {error}"]

    required_columns = {
        "schema_version",
        "timestamp_utc",
        "run_id",
        "experiment_id",
        "phase",
        "mechanism",
        "agents",
        "epochs",
        "repetition",
        "order_index",
        "status",
        "failure_stage",
        "error_code",
        "error_message",
        "batch_iterations",
        "aggregate_wall_ns",
        "wall_ns_per_invocation",
        "aggregate_device_ns",
        "device_ns_per_invocation",
        "min_duration_target_ns",
        "min_duration_reached",
        "expected_state_checksum",
        "observed_state_checksum",
        "expected_decision_hash",
        "observed_decision_hash",
        "expected_decisions",
        "observed_decisions",
        "exact_state_match",
        "exact_decision_match",
        "exact_validation_count",
        "seed",
        "block_size",
        "predicate_blocks",
    }
    require(fieldnames == required_columns, "CSV columns do not exactly match the frozen schema")

    provenance = manifest.get("provenance") if isinstance(manifest.get("provenance"), dict) else {}
    hardware = manifest.get("hardware") if isinstance(manifest.get("hardware"), dict) else {}
    results = manifest.get("results") if isinstance(manifest.get("results"), dict) else {}
    manifest_config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    source = provider.get("source") if isinstance(provider.get("source"), dict) else {}
    provider_pod = provider.get("pod") if isinstance(provider.get("pod"), dict) else {}
    provider_request = provider.get("request") if isinstance(provider.get("request"), dict) else {}
    execution = provider.get("execution") if isinstance(provider.get("execution"), dict) else {}
    host = provider.get("host") if isinstance(provider.get("host"), dict) else {}
    gpus = provider.get("gpus") if isinstance(provider.get("gpus"), list) else []
    run_id = str(manifest.get("run_id") or "")
    binary_sha256 = str(source.get("binary_sha256") or "")

    require(ready.get("ready") is True, "remote readiness marker is not true")
    require(ready.get("compile_return_code") == 0, "remote compilation failed")
    require(ready.get("program_return_code") == 0, "native program failed")
    require(ready.get("mode") == mode, "readiness mode mismatch")
    require(ready.get("source_sha256") == FROZEN_SOURCE_SHA256, "readiness source mismatch")
    require(
        ready.get("makefile_sha256") == FROZEN_MAKEFILE_SHA256,
        "readiness Makefile mismatch",
    )
    require(re.fullmatch(r"[0-9a-f]{64}", binary_sha256) is not None, "binary hash is invalid")
    require(ready.get("binary_sha256") == binary_sha256, "readiness binary hash mismatch")

    require(
        provider.get("schema_version") == "runpod-resident-policy-provider-v1",
        "provider schema mismatch",
    )
    require(provider.get("provider") == "runpod", "provider is not RunPod")
    require(provider.get("secrets_recorded") is False, "provider metadata secret flag is unsafe")
    require(provider_pod.get("pod_id") == pod_id, "provider Pod ID mismatch")
    require(
        provider_pod.get("data_center_id") in {None, "", receipt["request"]["data_center_id"]},
        "provider data-center ID mismatch",
    )
    require(
        str(provider_pod.get("provider_gpu_count") or "1") == "1",
        "provider exposed more than one GPU",
    )
    require(
        provider_request.get("requested_gpu_type") == GPU_TYPE, "provider requested GPU mismatch"
    )
    require(
        provider_request.get("requested_data_center_id") == receipt["request"]["data_center_id"],
        "provider requested data center mismatch",
    )
    require(
        provider_request.get("requested_cloud_type") == receipt["request"]["cloud_type"],
        "provider cloud type mismatch",
    )
    require(provider_request.get("image_reference") == CUDA_IMAGE, "provider image mismatch")
    require(
        provider_request.get("allowed_cuda_version") == "13.0", "provider CUDA version mismatch"
    )
    require(provider_request.get("mode") == mode, "provider mode mismatch")
    require(source.get("source_sha256") == FROZEN_SOURCE_SHA256, "provider source hash mismatch")
    require(
        source.get("makefile_sha256") == FROZEN_MAKEFILE_SHA256, "provider Makefile hash mismatch"
    )
    require(
        source.get("source_archive_sha256") == receipt["source"]["source_archive_sha256"],
        "provider source archive hash mismatch",
    )
    require(execution.get("compile_return_code") == 0, "provider compile status is nonzero")
    require(execution.get("program_return_code") == 0, "provider program status is nonzero")
    require(bool(host.get("platform")), "host platform provenance is empty")
    require(bool(host.get("node")), "host node provenance is empty")
    require(bool(host.get("machine")), "host machine provenance is empty")
    require(
        isinstance(host.get("logical_cpu_count"), int) and host["logical_cpu_count"] > 0,
        "host CPU-count provenance is invalid",
    )
    require(bool(host.get("cpu_model")), "host CPU-model provenance is empty")
    require(len(gpus) == 1, "nvidia-smi did not report exactly one GPU")
    if len(gpus) == 1:
        gpu = gpus[0] if isinstance(gpus[0], dict) else {}
        require("L4" in str(gpu.get("name") or ""), "actual GPU is not an L4")
        require(str(gpu.get("uuid") or "").startswith("GPU-"), "actual GPU UUID is missing")
        require(bool(gpu.get("driver_version")), "GPU driver provenance is empty")
        require(float(gpu.get("memory_mib") or 0) > 0, "GPU memory provenance is invalid")

    require(manifest.get("schema_version") == SCHEMA_VERSION, "manifest schema mismatch")
    require(manifest.get("experiment_id") == experiment_id, "manifest experiment ID mismatch")
    require(bool(run_id), "manifest run ID is empty")
    require(manifest.get("csv_file") == csv_path.name, "manifest CSV filename mismatch")
    require(csv_path.name == f"{run_id}.csv", "CSV filename does not match run ID")
    require(
        manifest_path.name == f"{run_id}.manifest.json", "manifest filename does not match run ID"
    )
    require(provenance.get("execution_provider") == "runpod", "native provider is not RunPod")
    require(provenance.get("requested_gpu") == GPU_TYPE, "native requested GPU mismatch")
    require(provenance.get("placement_id") == pod_id, "native placement ID mismatch")
    require(
        provenance.get("image_digest") == f"registry-ref:{CUDA_IMAGE}",
        "native image provenance mismatch",
    )
    require(provenance.get("source_sha256") == FROZEN_SOURCE_SHA256, "native source hash mismatch")
    require(provenance.get("binary_sha256") == binary_sha256, "native binary hash mismatch")
    require(hardware.get("cuda_available") is True, "CUDA was unavailable")
    require(hardware.get("device_count") == 1, "native runtime did not expose one GPU")
    require("L4" in str(hardware.get("device_name") or ""), "native actual GPU is not L4")
    require(bool(hardware.get("device_uuid")), "native GPU UUID is empty")
    require(hardware.get("unified_addressing") == 1, "unified addressing is not enabled")
    require(results.get("measured_rows") == expected_rows, "unexpected manifest row count")
    require(results.get("exact_rows") == expected_rows, "not every manifest row is exact")
    require(results.get("failure_rows") == 0, "native manifest contains failures")
    require(results.get("status_counts") == {"ok": expected_rows}, "status ledger is not all-ok")
    require(len(rows) == expected_rows, f"CSV row count is not {expected_rows}")
    for key, value in config.items():
        require(manifest_config.get(key) == value, f"manifest config mismatch for {key}")
    require(set(manifest_config.get("mechanisms", [])) == MECHANISMS, "mechanism set mismatch")
    expected_cell_count = len(config["agent_counts"]) * len(config["epoch_counts"])
    require(len(manifest.get("cells", [])) == expected_cell_count, "cell-audit count mismatch")

    cell_counts: Counter[tuple[int, int, str]] = Counter()
    identities: set[tuple[int, int, str, int]] = set()
    observed_orders: dict[tuple[int, int, int], set[int]] = {}
    observed_repetitions: dict[tuple[int, int, str], set[int]] = {}
    expected_repetitions = set(range(config["repetitions_per_mechanism_cell"]))
    for index, row in enumerate(rows):
        prefix = f"row {index}"
        try:
            agents = int(row["agents"])
            epochs = int(row["epochs"])
            repetition = int(row["repetition"])
            order_index = int(row["order_index"])
            batch_iterations = int(row["batch_iterations"])
            aggregate_wall_ns = int(row["aggregate_wall_ns"])
            aggregate_device_ns = int(row["aggregate_device_ns"])
            wall_ns = float(row["wall_ns_per_invocation"])
            device_ns = float(row["device_ns_per_invocation"])
            duration_target = int(row["min_duration_target_ns"])
            validation_count = int(row["exact_validation_count"])
            seed = int(row["seed"])
            block_size = int(row["block_size"])
            predicate_blocks = int(row["predicate_blocks"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{prefix} has invalid numeric fields: {error}")
            continue
        mechanism = row.get("mechanism", "")
        cell = (agents, epochs, mechanism)
        identity = (*cell, repetition)
        cell_counts[cell] += 1
        observed_repetitions.setdefault(cell, set()).add(repetition)
        observed_orders.setdefault((agents, epochs, repetition), set()).add(order_index)
        require(identity not in identities, f"{prefix} duplicates a measured identity")
        identities.add(identity)
        require(row.get("schema_version") == SCHEMA_VERSION, f"{prefix} schema mismatch")
        require(row.get("run_id") == run_id, f"{prefix} run ID mismatch")
        require(row.get("experiment_id") == experiment_id, f"{prefix} experiment mismatch")
        require(row.get("phase") == "measure", f"{prefix} is not measured")
        require(mechanism in MECHANISMS, f"{prefix} mechanism is invalid")
        require(agents in config["agent_counts"], f"{prefix} agents are outside the grid")
        require(epochs in config["epoch_counts"], f"{prefix} epochs are outside the grid")
        require(row.get("status") == "ok", f"{prefix} status is not ok")
        require(row.get("failure_stage") == "", f"{prefix} has a failure stage")
        require(row.get("error_code") == "0", f"{prefix} has a nonzero error code")
        require(row.get("error_message") == "", f"{prefix} has an error message")
        require(batch_iterations > 0, f"{prefix} has no timed invocations")
        require(
            aggregate_wall_ns > 0 and aggregate_device_ns > 0,
            f"{prefix} aggregate timing is nonpositive",
        )
        require(math.isfinite(wall_ns) and wall_ns > 0, f"{prefix} wall timing is invalid")
        require(math.isfinite(device_ns) and device_ns > 0, f"{prefix} device timing is invalid")
        require(row.get("min_duration_reached") == "true", f"{prefix} missed duration target")
        require(aggregate_wall_ns >= duration_target, f"{prefix} is below duration target")
        require(row.get("exact_state_match") == "true", f"{prefix} state is not field-exact")
        require(row.get("exact_decision_match") == "true", f"{prefix} decision trace is not exact")
        require(
            row.get("expected_state_checksum") == row.get("observed_state_checksum"),
            f"{prefix} state checksum differs",
        )
        require(
            row.get("expected_decision_hash") == row.get("observed_decision_hash"),
            f"{prefix} decision hash differs",
        )
        require(
            row.get("expected_decisions") == row.get("observed_decisions"),
            f"{prefix} decision trace differs",
        )
        require(
            re.fullmatch(r"[01]+", row.get("expected_decisions", "")) is not None,
            f"{prefix} decision trace is malformed",
        )
        require(
            len(row.get("expected_decisions", "")) == epochs,
            f"{prefix} decision trace length differs from epochs",
        )
        require(validation_count == batch_iterations, f"{prefix} did not validate every invocation")
        require(seed == config["seed"], f"{prefix} seed mismatch")
        require(block_size == config["block_size"], f"{prefix} block size mismatch")
        require(
            predicate_blocks == math.ceil(agents / block_size),
            f"{prefix} predicate-block count mismatch",
        )
        require(
            math.isclose(
                wall_ns, aggregate_wall_ns / batch_iterations, rel_tol=1e-12, abs_tol=1e-6
            ),
            f"{prefix} wall-time division mismatch",
        )
        require(
            math.isclose(
                device_ns, aggregate_device_ns / batch_iterations, rel_tol=1e-12, abs_tol=1e-6
            ),
            f"{prefix} device-time division mismatch",
        )

    expected_cells = {
        (agents, epochs, mechanism)
        for agents in config["agent_counts"]
        for epochs in config["epoch_counts"]
        for mechanism in MECHANISMS
    }
    require(set(cell_counts) == expected_cells, "CSV does not cover the frozen cell grid")
    repetitions = config["repetitions_per_mechanism_cell"]
    for cell in expected_cells:
        require(cell_counts[cell] == repetitions, f"cell {cell} has wrong row count")
        require(
            observed_repetitions.get(cell, set()) == expected_repetitions,
            f"cell {cell} has wrong repetition indices",
        )
    for agents in config["agent_counts"]:
        for epochs in config["epoch_counts"]:
            for repetition in expected_repetitions:
                key = (agents, epochs, repetition)
                require(
                    observed_orders.get(key, set()) == {0, 1, 2},
                    f"cell/repetition {key} has invalid order indices",
                )

    require(pod.get("id") == pod_id, "collection Pod ID mismatch")
    pod_gpu = pod.get("gpu") if isinstance(pod.get("gpu"), dict) else {}
    if pod_gpu.get("id") is not None:
        require(pod_gpu.get("id") == GPU_TYPE, "control-plane GPU type mismatch")
    if pod_gpu.get("count") is not None:
        require(int(pod_gpu["count"]) == 1, "control plane did not allocate exactly one GPU")
    machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
    if machine.get("gpu_type_id") is not None:
        require(machine.get("gpu_type_id") == GPU_TYPE, "machine GPU type mismatch")
    if machine.get("data_center_id") is not None:
        require(
            machine.get("data_center_id") == receipt["request"]["data_center_id"],
            "machine data center mismatch",
        )
    if pod.get("image") is not None:
        require(pod.get("image") == CUDA_IMAGE, "control-plane image mismatch")
    return errors


def _validate_collected_bundle(
    destination: Path,
    extracted: list[str],
    receipt: dict[str, Any],
    ready: dict[str, Any],
    pod: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    errors = _artifact_index_errors(destination, extracted)
    source_dir = destination / "source"
    binary_path = destination / "resident_policy_pilot"
    provider_path = destination / "provider-metadata.json"
    try:
        provider = _read_json_object(provider_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return errors + [f"provider metadata is unreadable: {error}"], {}

    required_paths = [
        source_dir / "resident_policy_pilot.cu",
        source_dir / "Makefile",
        source_dir / "README.md",
        source_dir / "source.tar.gz",
        binary_path,
        destination / "compile-command.txt",
        destination / "program-command.txt",
        destination / "compiler.stdout.log",
        destination / "compiler.stderr.log",
        destination / "program.stdout.log",
        destination / "program.stderr.log",
    ]
    for path in required_paths:
        if not path.is_file():
            errors.append(f"required artifact is missing: {path.relative_to(destination)}")
    if all(path.is_file() for path in required_paths):
        if _sha256_file(source_dir / "resident_policy_pilot.cu") != FROZEN_SOURCE_SHA256:
            errors.append("retrieved CUDA source differs from the frozen hash")
        if _sha256_file(source_dir / "Makefile") != FROZEN_MAKEFILE_SHA256:
            errors.append("retrieved Makefile differs from the frozen hash")
        if _sha256_file(source_dir / "source.tar.gz") != receipt["source"]["source_archive_sha256"]:
            errors.append("retrieved source archive differs from the launch receipt")
        provider_binary = str(provider.get("source", {}).get("binary_sha256") or "")
        if _sha256_file(binary_path) != provider_binary:
            errors.append("retrieved binary differs from provider metadata")

    csv_files = sorted((destination / "results").glob("*.csv"))
    manifest_files = sorted((destination / "results").glob("*.manifest.json"))
    if len(csv_files) != 1 or len(manifest_files) != 1:
        errors.append(
            f"expected one CSV and one manifest, found {len(csv_files)} and {len(manifest_files)}"
        )
    else:
        errors.extend(
            _native_validation_errors(
                csv_path=csv_files[0],
                manifest_path=manifest_files[0],
                provider=provider,
                receipt=receipt,
                ready=ready,
                pod=pod,
            )
        )
    summary = {
        "mode": receipt.get("request", {}).get("mode"),
        "expected_rows": _expected_row_count(
            _mode_config(str(receipt.get("request", {}).get("mode") or ""))
        ),
        "csv_files": [path.name for path in csv_files],
        "manifest_files": [path.name for path in manifest_files],
        "source_sha256": FROZEN_SOURCE_SHA256,
        "makefile_sha256": FROZEN_MAKEFILE_SHA256,
        "binary_sha256": provider.get("source", {}).get("binary_sha256"),
        "actual_gpus": provider.get("gpus"),
        "host": provider.get("host"),
    }
    return errors, summary


def _validate_launch_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema_version") != "runpod-resident-policy-launch-v1":
        raise ValueError("launch receipt schema mismatch")
    pod_id = str(receipt.get("pod_id") or "")
    if POD_ID_RE.fullmatch(pod_id) is None:
        raise ValueError("launch receipt has an unsafe Pod ID")
    experiment_id = str(receipt.get("experiment_id") or "")
    if EXPERIMENT_RE.fullmatch(experiment_id) is None:
        raise ValueError("launch receipt has an unsafe experiment ID")
    if receipt.get("pod_name") != f"gpu-agent-{experiment_id}":
        raise ValueError("launch receipt Pod name mismatch")
    if receipt.get("request", {}).get("gpu_type") != GPU_TYPE:
        raise ValueError("launch receipt is not for the frozen L4 pilot")
    if receipt.get("request", {}).get("gpu_count") != 1:
        raise ValueError("launch receipt did not request exactly one GPU")
    if receipt.get("request", {}).get("image") != CUDA_IMAGE:
        raise ValueError("launch receipt image mismatch")
    _mode_config(str(receipt.get("request", {}).get("mode") or ""))
    if receipt.get("source", {}).get("source_sha256") != FROZEN_SOURCE_SHA256:
        raise ValueError("launch receipt source hash mismatch")
    if receipt.get("source", {}).get("makefile_sha256") != FROZEN_MAKEFILE_SHA256:
        raise ValueError("launch receipt Makefile hash mismatch")
    if (
        re.fullmatch(
            r"[0-9a-f]{64}",
            str(receipt.get("source", {}).get("source_archive_sha256") or ""),
        )
        is None
    ):
        raise ValueError("launch receipt source archive hash is invalid")


def collect(args: argparse.Namespace) -> None:
    if args.receipt is None:
        raise ValueError("collect requires --receipt PATH")
    receipt = _read_json_object(args.receipt)
    _validate_launch_receipt(receipt)
    token = str(receipt.get("artifact_token") or "")
    pod_id = str(receipt["pod_id"])
    base_url = str(receipt.get("artifact_url") or "")
    if not token or not base_url:
        raise ValueError("launch receipt is missing artifact access fields")
    raw_pod = _get_pod_or_none(pod_id)
    if raw_pod is None:
        raise RunPodError("Pod is absent before artifacts were collected")

    deadline = time.monotonic() + args.collect_timeout_seconds
    ready: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            value = json.loads(_authenticated_download(f"{base_url}/ready.json", token, 20))
            if isinstance(value, dict) and value.get("ready") is True:
                ready = value
                break
        except (RunPodError, json.JSONDecodeError):
            pass
        time.sleep(5)
    if ready is None:
        raise RunPodError(
            "artifact readiness timeout; inspect the persisted /workspace volume before restart"
        )

    refreshed = _get_pod_or_none(pod_id)
    if refreshed is not None:
        raw_pod = refreshed
    sanitized_pod = _sanitize_pod(raw_pod)
    checksum_text = _authenticated_download(
        f"{base_url}/resident-policy-artifacts.tar.gz.sha256", token, 30
    ).decode("ascii", errors="strict")
    expected_sha = checksum_text.split()[0]
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ValueError("artifact endpoint returned an invalid checksum")
    archive_bytes = _authenticated_download(
        f"{base_url}/resident-policy-artifacts.tar.gz", token, 180
    )
    observed_sha = _sha256_bytes(archive_bytes)
    if not hmac.compare_digest(expected_sha, observed_sha):
        raise ValueError("artifact checksum mismatch")

    destination = args.output_dir / f"runpod-{receipt['experiment_id']}-{pod_id}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    archive_path = destination / "resident-policy-artifacts.tar.gz"
    with archive_path.open("xb") as handle:
        handle.write(archive_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    extracted = _safe_extract(archive_bytes, destination)
    validation_errors, native_summary = _validate_collected_bundle(
        destination,
        extracted,
        receipt,
        ready,
        sanitized_pod,
    )
    collection = {
        "schema_version": "runpod-resident-policy-collection-v1",
        "collected_at_utc": _utc_now(),
        "launch_receipt": str(args.receipt.resolve()),
        "launch_receipt_sha256": _sha256_file(args.receipt),
        "pod_id": pod_id,
        "experiment_id": receipt["experiment_id"],
        "pod": sanitized_pod,
        "ready": ready,
        "archive_sha256": observed_sha,
        "archive_bytes": len(archive_bytes),
        "extracted_files": extracted,
        "validation": {
            "passed": not validation_errors,
            "errors": validation_errors,
            "native_summary": native_summary,
        },
        "artifact_token_recorded": False,
        "runpod_api_key_recorded": False,
        "remote_mutations_by_collector": 0,
        "note": "The remote server schedules an exact-Pod stop after archive download; deletion is separate.",
    }
    collection_path = destination / "collection-receipt.json"
    _write_json_new(collection_path, collection)
    print(f"collected_to={destination.resolve()}")
    print(f"collection_receipt={collection_path.resolve()}")
    print(f"archive_sha256={observed_sha}")
    print(f"extracted_files={len(extracted)}")
    print(f"validation_passed={str(not validation_errors).lower()}")
    print("artifact_token_printed=false")
    print("termination_performed=false")
    if validation_errors:
        raise ValueError(
            "collected artifacts failed validity gates and were retained; termination remains locked: "
            + "; ".join(validation_errors[:20])
        )


def _validate_termination_authority(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if args.receipt is None or args.collection_receipt is None:
        raise ValueError("terminate requires --receipt and --collection-receipt")
    receipt = _read_json_object(args.receipt)
    collection = _read_json_object(args.collection_receipt)
    _validate_launch_receipt(receipt)
    pod_id = str(receipt["pod_id"])
    if args.confirm_terminate_pod_id != pod_id or os.environ.get(TERMINATE_GATE_ENV) != pod_id:
        raise ValueError(
            "terminate requires the exact Pod ID in both --confirm-terminate-pod-id and "
            f"{TERMINATE_GATE_ENV}"
        )
    if collection.get("schema_version") != "runpod-resident-policy-collection-v1":
        raise ValueError("collection receipt schema mismatch")
    if collection.get("pod_id") != pod_id:
        raise ValueError("collection receipt Pod ID mismatch")
    if collection.get("experiment_id") != receipt.get("experiment_id"):
        raise ValueError("collection receipt experiment mismatch")
    if collection.get("launch_receipt_sha256") != _sha256_file(args.receipt):
        raise ValueError("collection receipt is not bound to this launch receipt")
    validation = collection.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("collection did not pass validity gates; refusing destructive deletion")
    if validation.get("errors") != []:
        raise ValueError("collection receipt contains validation errors")
    archive_sha = str(collection.get("archive_sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", archive_sha) is None:
        raise ValueError("collection archive hash is invalid")
    artifact_root = args.collection_receipt.parent
    archive_path = artifact_root / "resident-policy-artifacts.tar.gz"
    if not archive_path.is_file() or _sha256_file(archive_path) != archive_sha:
        raise ValueError("locally retained artifact archive is missing or has changed")
    extracted = collection.get("extracted_files")
    ready = collection.get("ready")
    pod = collection.get("pod")
    if not isinstance(extracted, list) or not all(isinstance(item, str) for item in extracted):
        raise ValueError("collection receipt has an invalid extracted-file ledger")
    if not isinstance(ready, dict) or not isinstance(pod, dict):
        raise TypeError("collection receipt lacks readiness or Pod provenance")
    revalidation_errors, _ = _validate_collected_bundle(
        artifact_root,
        extracted,
        receipt,
        ready,
        pod,
    )
    if revalidation_errors:
        raise ValueError(
            "locally retained artifacts no longer pass validity gates; refusing deletion: "
            + "; ".join(revalidation_errors[:20])
        )
    return receipt, collection, pod_id


def terminate(args: argparse.Namespace) -> None:
    receipt, collection, pod_id = _validate_termination_authority(args)
    termination_path = args.termination_receipt or (
        REPO_ROOT
        / "data/external"
        / f"runpod-resident-policy-termination-{pod_id}-{_utc_compact()}.json"
    )
    termination_path.parent.mkdir(parents=True, exist_ok=True)
    if termination_path.exists():
        raise FileExistsError(f"refusing to overwrite termination receipt: {termination_path}")

    live = _get_pod_or_none(pod_id)
    if live is None:
        outcome = {
            "schema_version": "runpod-resident-policy-termination-v1",
            "captured_at_utc": _utc_now(),
            "pod_id": pod_id,
            "experiment_id": receipt["experiment_id"],
            "validated_archive_sha256": collection["archive_sha256"],
            "initial_state": "already_absent",
            "stop_called": False,
            "delete_called": False,
            "confirmed_absent": True,
            "remote_volume_recoverable": False,
        }
        _write_json_new(termination_path, outcome)
        print(f"pod_id={pod_id}")
        print("already_absent=true")
        print(f"termination_receipt={termination_path.resolve()}")
        return
    if live.get("name") != receipt.get("pod_name"):
        raise RunPodError("live Pod name disagrees with the launch receipt; refusing mutation")

    sanitized_initial = _sanitize_pod(live)
    status = str(live.get("desiredStatus") or "").upper()
    stop_called = False
    if status not in {"EXITED", "STOPPED"}:
        _request_json(
            "POST",
            f"{REST_URL}/pods/{pod_id}/stop",
            allow_empty=True,
            timeout=30,
        )
        stop_called = True
        for _ in range(12):
            time.sleep(5)
            live = _get_pod_or_none(pod_id)
            if live is None:
                break
            status = str(live.get("desiredStatus") or "").upper()
            if status in {"EXITED", "STOPPED"}:
                break
        if live is not None and status not in {"EXITED", "STOPPED"}:
            raise RunPodError("exact Pod did not reach a stopped state; deletion was not attempted")

    delete_called = False
    if live is not None:
        _request_json(
            "DELETE",
            f"{REST_URL}/pods/{pod_id}",
            allow_empty=True,
            timeout=30,
        )
        delete_called = True
    confirmed_absent = False
    for _ in range(7):
        if _get_pod_or_none(pod_id) is None:
            confirmed_absent = True
            break
        time.sleep(5)

    outcome = {
        "schema_version": "runpod-resident-policy-termination-v1",
        "captured_at_utc": _utc_now(),
        "pod_id": pod_id,
        "experiment_id": receipt["experiment_id"],
        "validated_archive_sha256": collection["archive_sha256"],
        "initial_pod": sanitized_initial,
        "stop_called": stop_called,
        "delete_called": delete_called,
        "confirmed_absent": confirmed_absent,
        "remote_volume_recoverable": False,
        "local_artifacts_retained": str(args.collection_receipt.resolve()),
    }
    _write_json_new(termination_path, outcome)
    print(f"pod_id={pod_id}")
    print(f"stop_called={str(stop_called).lower()}")
    print(f"delete_called={str(delete_called).lower()}")
    print(f"confirmed_absent={str(confirmed_absent).lower()}")
    print(f"termination_receipt={termination_path.resolve()}")
    if not confirmed_absent:
        raise RunPodError("delete was requested but Pod absence was not confirmed")


def print_inventory(inventory: dict[str, Any]) -> None:
    print("authentication_ok=true")
    print(f"captured_at_utc={inventory['captured_at_utc']}")
    print(f"catalog_gpu_types={inventory['counts']['catalog_gpu_types']}")
    print(f"data_centers={inventory['counts']['data_centers']}")
    l4 = [offer for offer in inventory["available_offers"] if offer["gpu_type_id"] == GPU_TYPE]
    print(f"stocked_l4_offers={len(l4)}")
    for offer in l4:
        print(
            f"available_l4\t{offer['data_center_id']}\t{offer['stock_status']}\t{offer['location']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("plan", "inventory", "launch", "collect", "terminate"),
        default="plan",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--inventory-output", type=Path)
    parser.add_argument("--confirm-spend", default="")
    parser.add_argument("--experiment-id", default="resident-policy-001-runpod-l4-p1")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data-center-id", default="")
    parser.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), default="SECURE")
    parser.add_argument("--max-run-minutes", type=int, default=30)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--volume-gb", type=int, default=10)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/raw")
    parser.add_argument("--collect-timeout-seconds", type=int, default=2400)
    parser.add_argument("--collection-receipt", type=Path)
    parser.add_argument("--confirm-terminate-pod-id", default="")
    parser.add_argument("--termination-receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "plan":
        frozen = _validate_frozen_inputs()
        archive = _source_archive()
        plan = {
            "action": "plan",
            "remote_calls": 0,
            "gpu_calls": 0,
            "image": CUDA_IMAGE,
            "gpu_type": GPU_TYPE,
            "gpu_count": 1,
            "mode": args.mode,
            "expected_measured_rows": _expected_row_count(_mode_config(args.mode)),
            **frozen,
            "source_archive_sha256": _sha256_bytes(archive),
            "source_archive_bytes": len(archive),
            "launch_cli_gate": f"--confirm-spend {LAUNCH_ACK}",
            "launch_environment_gate": f"{LAUNCH_GATE_ENV}={LAUNCH_ACK}",
            "termination_cli_gate": "--confirm-terminate-pod-id <exact-pod-id>",
            "termination_environment_gate": f"{TERMINATE_GATE_ENV}=<exact-pod-id>",
            "deletion_requires_verified_collection": True,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    _load_env_file(args.env_file)
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
    if args.action == "collect":
        if not 30 <= args.collect_timeout_seconds <= 7200:
            raise ValueError("collect timeout must be in [30, 7200] seconds")
        collect(args)
        return
    terminate(args)


if __name__ == "__main__":
    try:
        main()
    except (
        FileExistsError,
        OSError,
        RunPodError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"error={_redact(str(error))}", file=sys.stderr)
        raise SystemExit(2) from error
