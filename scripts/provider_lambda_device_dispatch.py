"""Safe Lambda Cloud path for the native CUDA device-dispatch pilot.

The default command is a local-only plan. Read-only inventory is explicit.
Creating one billable instance requires two independent acknowledgements.
This module never terminates, restarts, or deletes a Lambda resource.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "native/device_dispatch"
DEFAULT_ENV_FILE = REPO_ROOT.parent / ".env"
API_ROOT = "https://cloud.lambda.ai"
API_SPEC_VERSION = "1.10.0"
DOCS_URL = "https://docs.lambda.ai/api/cloud"
IMAGE_FAMILY_DEFAULT = "lambda-stack-24-04"
LAUNCH_CONFIRMATION = "LAUNCH_ONE_BILLABLE_LAMBDA_INSTANCE"
SAFE_EXPERIMENT = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}\Z")
SAFE_CSV_INTS = re.compile(r"[1-9][0-9]*(?:,[1-9][0-9]*)*\Z")
SAFE_REMOTE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,190}\Z")
SAFE_INSTANCE_ID = re.compile(r"[A-Za-z0-9-]{8,128}\Z")
SAFE_SSH_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}\Z")


class LambdaApiError(RuntimeError):
    """A sanitized Lambda API error that never includes request headers."""

    def __init__(self, status: int, code: str, message: str, suggestion: str = ""):
        self.status = status
        self.code = code
        self.suggestion = suggestion
        detail = f"Lambda API HTTP {status}: {code}: {message}"
        if suggestion:
            detail += f" ({suggestion})"
        super().__init__(detail)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _read_env_value(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        candidate, value = line.split("=", 1)
        candidate = candidate.removeprefix("export ").strip()
        if candidate != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value or None
    return None


def lambda_api_key(env_file: Path) -> str:
    value = os.environ.get("LAMBDA_API_KEY") or _read_env_value(env_file, "LAMBDA_API_KEY")
    if not value:
        raise RuntimeError(
            "LAMBDA_API_KEY is missing; export it or add it to the selected env file"
        )
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if any(character.isspace() for character in value):
        raise RuntimeError("LAMBDA_API_KEY contains whitespace and was not used")
    return value


def source_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(NATIVE_ROOT.rglob("*")):
        if not path.is_file() or "build" in path.parts or "smoke-results" in path.parts:
            continue
        digest.update(path.relative_to(NATIVE_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_new_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}") from error


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    serialized = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    write_new_bytes(path, serialized)


class LambdaApi:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._last_request_monotonic: float | None = None

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        if not path.startswith("/api/v1/"):
            raise ValueError("Lambda API path must be below /api/v1/")
        if self._last_request_monotonic is not None:
            delay = 1.05 - (time.monotonic() - self._last_request_monotonic)
            if delay > 0:
                time.sleep(delay)
        data = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "gpu-agent-crossover-lambda-runner/0.1",
        }
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(API_ROOT + path, data=data, headers=headers, method=method)
        try:
            self._last_request_monotonic = time.monotonic()
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            try:
                payload = json.load(error)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {}
            details = payload.get("error", {}) if isinstance(payload, dict) else {}
            raise LambdaApiError(
                error.code,
                str(details.get("code", "unknown")),
                str(details.get("message", "request failed")),
                str(details.get("suggestion", "")),
            ) from None
        except urllib.error.URLError as error:
            raise RuntimeError(f"Lambda API transport failed: {error.reason}") from None
        if not isinstance(payload, dict) or "data" not in payload:
            raise RuntimeError("Lambda API response did not contain a data field")
        return payload["data"]

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str, body: dict[str, Any]) -> Any:
        return self.request("POST", path, body)


def api_from_args(args: argparse.Namespace) -> LambdaApi:
    return LambdaApi(lambda_api_key(Path(args.env_file).expanduser().resolve()))


def normalized_instance_types(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        raise TypeError("unexpected instance-types response shape")
    rows: list[dict[str, Any]] = []
    for api_name, item in sorted(data.items()):
        if not isinstance(item, dict):
            raise TypeError("unexpected instance-types item shape")
        details = item.get("instance_type", {})
        regions = item.get("regions_with_capacity_available", [])
        if not isinstance(details, dict) or not isinstance(regions, list):
            raise TypeError("unexpected instance type details")
        name = str(details.get("name", api_name))
        if name != api_name:
            raise RuntimeError(f"instance type key/name disagreement for {api_name}")
        rows.append(
            {
                "name": name,
                "description": details.get("description", ""),
                "gpu_description": details.get("gpu_description", ""),
                "price_cents_per_hour": details.get("price_cents_per_hour"),
                "architecture": details.get("architecture", ""),
                "specs": details.get("specs", {}),
                "regions_with_capacity_available": [
                    {
                        "name": region.get("name", ""),
                        "description": region.get("description", ""),
                    }
                    for region in regions
                    if isinstance(region, dict)
                ],
            }
        )
    return rows


def sanitized_instances(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise TypeError("unexpected instances response shape")
    result = []
    for instance in data:
        if not isinstance(instance, dict):
            continue
        result.append(
            {
                "id": instance.get("id"),
                "name": instance.get("name"),
                "status": instance.get("status"),
                "region": instance.get("region"),
                "instance_type": instance.get("instance_type"),
                "ssh_key_names": instance.get("ssh_key_names", []),
            }
        )
    return result


def sanitized_images(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise TypeError("unexpected images response shape")
    fields = (
        "id",
        "created_time",
        "updated_time",
        "name",
        "description",
        "family",
        "version",
        "architecture",
        "region",
    )
    return [{field: item.get(field) for field in fields} for item in data if isinstance(item, dict)]


def inventory(api: LambdaApi) -> dict[str, Any]:
    type_rows = normalized_instance_types(api.get("/api/v1/instance-types"))
    instances = sanitized_instances(api.get("/api/v1/instances"))
    ssh_keys = api.get("/api/v1/ssh-keys")
    images = sanitized_images(api.get("/api/v1/images"))
    if not isinstance(ssh_keys, list):
        raise TypeError("unexpected SSH-key response shape")
    return {
        "schema_version": "provider-lambda-inventory-v1",
        "collected_at_utc": utc_now(),
        "api_spec_version": API_SPEC_VERSION,
        "authentication": "ok",
        "instance_types": type_rows,
        "running_instances": instances,
        "ssh_keys": [
            {"id": key.get("id"), "name": key.get("name")}
            for key in ssh_keys
            if isinstance(key, dict)
        ],
        "images": images,
        "redactions": [
            "API keys",
            "SSH public-key material",
            "instance public and private IP addresses",
            "Jupyter tokens and URLs",
        ],
    }


def command_plan(args: argparse.Namespace) -> None:
    print("action=plan")
    print("api_calls=0")
    print("remote_calls=0")
    print("billable_resources_created=0")
    print(f"lambda_api_docs={DOCS_URL}")
    print(f"lambda_openapi_version={API_SPEC_VERSION}")
    print(f"source_sha256={source_sha256()}")
    print(
        "inventory: python scripts/provider_lambda_device_dispatch.py inventory "
        "--output data/external/lambda-inventory-UNIQUE-UTC.json"
    )
    print(f"launch: requires --confirm-spend and --launch-confirmation {LAUNCH_CONFIRMATION}")
    print("run-existing: requires --confirm-remote-execution and an active instance ID")
    print("termination: intentionally not implemented")


def command_inventory(args: argparse.Namespace) -> None:
    record = inventory(api_from_args(args))
    print("authentication=ok")
    print(f"collected_at_utc={record['collected_at_utc']}")
    available_pairs = 0
    for item in record["instance_types"]:
        regions = item["regions_with_capacity_available"]
        if not regions:
            print(f"capacity type={item['name']} region=NONE")
        for region in regions:
            available_pairs += 1
            print(
                f"capacity type={item['name']} region={region['name']} "
                f"gpu={item['gpu_description']!r} gpus={item['specs'].get('gpus')} "
                f"usd_per_hour={item['price_cents_per_hour'] / 100:.2f}"
            )
    print(f"instance_type_count={len(record['instance_types'])}")
    print(f"available_type_region_pair_count={available_pairs}")
    print(f"running_instance_count={len(record['running_instances'])}")
    print(f"ssh_key_count={len(record['ssh_keys'])}")
    print(f"image_count={len(record['images'])}")
    if args.output:
        destination = Path(args.output).expanduser()
        write_new_json(destination, record)
        print(f"inventory_artifact={destination.resolve()}")


def _require_launch_gates(args: argparse.Namespace) -> None:
    if not args.confirm_spend:
        raise RuntimeError("launch refused: --confirm-spend is required")
    if args.launch_confirmation != LAUNCH_CONFIRMATION:
        raise RuntimeError(
            "launch refused: --launch-confirmation must exactly equal " + LAUNCH_CONFIRMATION
        )


def _select_type(rows: list[dict[str, Any]], name: str, region: str) -> dict[str, Any]:
    selected = next((row for row in rows if row["name"] == name), None)
    if selected is None:
        raise RuntimeError(f"instance type is not in the live catalog: {name}")
    capacity_regions = {entry["name"] for entry in selected["regions_with_capacity_available"]}
    if region not in capacity_regions:
        raise RuntimeError(
            f"live inventory reports no capacity for {name} in {region}; launch refused"
        )
    return selected


def command_launch(args: argparse.Namespace) -> None:
    _require_launch_gates(args)
    if not math.isfinite(args.max_hourly_usd) or args.max_hourly_usd <= 0:
        raise RuntimeError("--max-hourly-usd must be a finite positive number")
    if len(args.instance_name) > 64:
        raise RuntimeError("--instance-name must contain at most 64 characters")
    api = api_from_args(args)
    record = inventory(api)
    selected = _select_type(record["instance_types"], args.instance_type, args.region)
    if selected["architecture"] != "x86_64" and not args.allow_non_x86:
        raise RuntimeError(
            f"{args.instance_type} uses {selected['architecture']}; "
            "pass --allow-non-x86 only after accepting the CPU-architecture confound"
        )
    price = selected["price_cents_per_hour"] / 100
    if price > args.max_hourly_usd:
        raise RuntimeError(
            f"live price ${price:.2f}/hour exceeds --max-hourly-usd "
            f"${args.max_hourly_usd:.2f}; launch refused"
        )
    ssh_names = {entry["name"] for entry in record["ssh_keys"]}
    if args.ssh_key_name not in ssh_names:
        raise RuntimeError("the selected SSH key name is not present in live inventory")
    matching_images = [
        image
        for image in record["images"]
        if image.get("family") == args.image_family
        and image.get("architecture") == selected["architecture"]
        and isinstance(image.get("region"), dict)
        and image["region"].get("name") == args.region
    ]
    if not matching_images:
        raise RuntimeError(
            f"no live {args.image_family} image for {selected['architecture']} "
            f"in {args.region}; launch refused"
        )

    output_dir = Path(args.output_dir).expanduser()
    receipt_path = output_dir / (f"lambda-launch-{compact_utc_now()}-{uuid.uuid4().hex[:8]}.jsonl")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt = receipt_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite launch receipt: {receipt_path}") from error

    request_body = {
        "region_name": args.region,
        "instance_type_name": args.instance_type,
        "ssh_key_names": [args.ssh_key_name],
        "file_system_names": [],
        "name": args.instance_name,
        "image": {"family": args.image_family},
        "tags": [
            {"key": "project", "value": "gpu-agent-crossover"},
            {"key": "experiment", "value": "device-dispatch"},
            {"key": "source-sha256", "value": source_sha256()},
        ],
    }
    intent = {
        "event": "launch_intent",
        "at_utc": utc_now(),
        "api_spec_version": API_SPEC_VERSION,
        "request_without_ssh_key_name": {
            key: value for key, value in request_body.items() if key != "ssh_key_names"
        },
        "selected_type": selected,
        "image_candidates": matching_images,
        "automatic_termination": False,
    }
    receipt.write(json.dumps(intent, sort_keys=True) + "\n")
    receipt.flush()
    os.fsync(receipt.fileno())
    try:
        response = api.post("/api/v1/instance-operations/launch", request_body)
        instance_ids = response.get("instance_ids", []) if isinstance(response, dict) else []
        if len(instance_ids) != 1:
            raise RuntimeError("launch response did not contain exactly one instance ID")
        launched = {
            "event": "launch_accepted",
            "at_utc": utc_now(),
            "instance_id": instance_ids[0],
            "warning": "billing continues until the instance is terminated through Lambda",
        }
        receipt.write(json.dumps(launched, sort_keys=True) + "\n")
        receipt.flush()
        os.fsync(receipt.fileno())
    except Exception as error:
        failed = {
            "event": "launch_failed",
            "at_utc": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        receipt.write(json.dumps(failed, sort_keys=True) + "\n")
        receipt.flush()
        os.fsync(receipt.fileno())
        raise
    finally:
        receipt.close()
    print(f"launch_receipt={receipt_path.resolve()}")
    print(f"instance_id={instance_ids[0]}")
    print("automatic_termination=false")
    print("warning=billing continues until the instance is terminated through Lambda")


def _validate_private_key(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    details = resolved.stat()
    if not stat.S_ISREG(details.st_mode):
        raise RuntimeError("SSH private key must be a regular file")
    if details.st_mode & 0o077:
        raise RuntimeError("SSH private key must not be readable or writable by group/others")
    return resolved


def _ssh_prefix(key_path: Path, known_hosts: Path, user: str, host: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=20",
        f"{user}@{host}",
    ]


def _run_process(command: list[str], *, text: bool = True) -> subprocess.CompletedProcess[Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=text,
            timeout=40 * 60,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "external SSH/SCP command timed out; connection details redacted"
        ) from None
    if completed.returncode != 0:
        raise RuntimeError(
            f"external SSH/SCP command failed with exit code {completed.returncode}; "
            "connection details and stderr redacted"
        )
    return completed


def _remote_capture(ssh: list[str], command: list[str]) -> str:
    completed = _run_process([*ssh, shlex.join(command)])
    return completed.stdout.strip()


def _get_instance(api: LambdaApi, instance_id: str) -> dict[str, Any]:
    if not SAFE_INSTANCE_ID.fullmatch(instance_id):
        raise RuntimeError("instance ID has an unsafe format")
    data = api.get(f"/api/v1/instances/{instance_id}")
    if not isinstance(data, dict):
        raise TypeError("unexpected instance response shape")
    if data.get("id") != instance_id:
        raise RuntimeError("instance ID mismatch in Lambda response")
    return data


def _validate_run_args(args: argparse.Namespace) -> None:
    if not args.confirm_remote_execution:
        raise RuntimeError("run refused: --confirm-remote-execution is required")
    if not SAFE_EXPERIMENT.fullmatch(args.experiment_id):
        raise RuntimeError("experiment ID must match [a-z0-9][a-z0-9_-]{0,47}")
    for name, value in (("agents", args.agents), ("steps", args.steps)):
        if not SAFE_CSV_INTS.fullmatch(value):
            raise RuntimeError(f"{name} must be a comma-separated list of positive integers")
    agents = [int(value) for value in args.agents.split(",")]
    steps = [int(value) for value in args.steps.split(",")]
    if len(agents) > 32 or max(agents) > 10_000_000:
        raise RuntimeError("agents exceeds the bounded pilot ceiling")
    if len(steps) > 32 or max(steps) > 1_000_000:
        raise RuntimeError("steps exceeds the bounded pilot ceiling")
    if args.warmups < 0 or args.repetitions <= 0:
        raise RuntimeError("warmups must be non-negative and repetitions must be positive")
    if args.warmups > 1_000 or args.repetitions > 10_000:
        raise RuntimeError("warmups or repetitions exceeds the bounded pilot ceiling")
    estimated_transitions = sum(agents) * sum(steps) * (args.warmups + args.repetitions) * 4
    if estimated_transitions > 1_000_000_000_000:
        raise RuntimeError("requested workload exceeds the bounded transition ceiling")
    if not args.image_reference.strip() or len(args.image_reference) > 256:
        raise RuntimeError("--image-reference is required for provenance")
    if not SAFE_SSH_USER.fullmatch(args.ssh_user):
        raise RuntimeError("SSH user has an unsafe format")


def command_run_existing(args: argparse.Namespace) -> None:
    _validate_run_args(args)
    api = api_from_args(args)
    instance = _get_instance(api, args.instance_id)
    if instance.get("status") != "active":
        raise RuntimeError(f"instance status is {instance.get('status')!r}, not 'active'")
    host = instance.get("ip")
    if not isinstance(host, str) or not host:
        raise RuntimeError("active instance does not expose a public IP")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        raise RuntimeError("Lambda returned an invalid public IP address") from None
    if args.ssh_key_name not in instance.get("ssh_key_names", []):
        raise RuntimeError("selected SSH key name is not attached to this instance")

    key_path = _validate_private_key(Path(args.ssh_private_key))
    known_hosts = Path(args.known_hosts).expanduser().resolve()
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    known_hosts.touch(mode=0o600, exist_ok=True)
    os.chmod(known_hosts, 0o600)
    ssh = _ssh_prefix(key_path, known_hosts, args.ssh_user, host)

    placement_token = uuid.uuid4().hex
    remote_root = f"/tmp/gpu-agent-crossover-{placement_token}"
    remote_native = f"{remote_root}/device_dispatch"
    remote_binary = f"{remote_root}/device_dispatch_pilot"
    remote_output = f"{remote_root}/output"
    source_digest = source_sha256()

    _remote_capture(ssh, ["mkdir", "--mode=700", remote_root])
    scp = [
        "scp",
        "-r",
        "-i",
        str(key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        str(NATIVE_ROOT),
        f"{args.ssh_user}@{host}:{remote_root}/",
    ]
    _run_process(scp)

    compile_command = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-rdc=true",
        "-arch=native",
        f"{remote_native}/device_dispatch_pilot.cu",
        "-o",
        remote_binary,
        "-lcudadevrt",
    ]
    benchmark_command = [
        "env",
        "EXECUTION_PROVIDER=lambda",
        f"REQUESTED_GPU={instance.get('instance_type', {}).get('gpu_description', '')}",
        f"SOURCE_SHA256={source_digest}",
        remote_binary,
        "--experiment-id",
        args.experiment_id,
        "--agents",
        args.agents,
        "--steps",
        args.steps,
        "--warmups",
        str(args.warmups),
        "--repetitions",
        str(args.repetitions),
        "--seed",
        str(args.seed),
        "--output-dir",
        remote_output,
    ]
    remote_script = "\n".join(
        [
            "set -euo pipefail",
            shlex.join(["command", "-v", "nvcc"]),
            shlex.join(["command", "-v", "nvidia-smi"]),
            shlex.join(["mkdir", "--mode=700", remote_output]),
            shlex.join(compile_command),
            shlex.join(benchmark_command),
        ]
    )
    _run_process([*ssh, "bash -lc " + shlex.quote(remote_script)])

    listing = _remote_capture(
        ssh,
        [
            "find",
            remote_output,
            "-maxdepth",
            "1",
            "-type",
            "f",
            "-printf",
            "%f\\n",
        ],
    ).splitlines()
    filenames = [name for name in listing if name]
    if any(not SAFE_REMOTE_FILE.fullmatch(name) for name in filenames):
        raise RuntimeError("remote output contained an unsafe filename")
    csv_names = [name for name in filenames if name.endswith(".csv")]
    manifest_names = [name for name in filenames if name.endswith(".manifest.json")]
    if len(csv_names) != 1 or len(manifest_names) != 1 or len(filenames) != 2:
        raise RuntimeError(f"expected exactly one CSV and one manifest; got {sorted(filenames)!r}")

    artifacts: dict[str, bytes] = {}
    for filename in filenames:
        completed = _run_process(
            [*ssh, shlex.join(["cat", "--", f"{remote_output}/{filename}"])], text=False
        )
        artifacts[filename] = completed.stdout
    native_manifest = json.loads(artifacts[manifest_names[0]])
    if native_manifest.get("execution_provider") != "lambda":
        raise RuntimeError("native manifest has wrong execution provider")
    if native_manifest.get("source_sha256") != source_digest:
        raise RuntimeError("native manifest has wrong source digest")
    if native_manifest.get("results", {}).get("measured_rows", 0) <= 0:
        raise RuntimeError("native benchmark emitted no measured rows")

    remote_hostname = _remote_capture(ssh, ["hostname"])
    host_metadata = {
        "hostname_present": bool(remote_hostname),
        "hostname_redacted": True,
        "kernel": _remote_capture(ssh, ["uname", "-srmo"]),
        "architecture": _remote_capture(ssh, ["uname", "-m"]),
        "logical_cpu_count": _remote_capture(ssh, ["nproc"]),
        "lscpu_json": json.loads(_remote_capture(ssh, ["lscpu", "--json"])),
        "os_release": _remote_capture(ssh, ["cat", "/etc/os-release"]),
        "nvcc_version": _remote_capture(ssh, ["nvcc", "--version"]),
        "gpu_inventory_csv": _remote_capture(
            ssh,
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total",
                "--format=csv,noheader",
            ],
        ),
        "binary_sha256": _remote_capture(ssh, ["sha256sum", remote_binary]).split()[0],
    }
    instance_metadata = {
        "id": instance.get("id"),
        "name": instance.get("name"),
        "status": instance.get("status"),
        "region": instance.get("region"),
        "instance_type": instance.get("instance_type"),
        "ssh_key_name": args.ssh_key_name,
    }
    provider_manifest = {
        "schema_version": "provider-lambda-device-dispatch-v1",
        "collected_at_utc": utc_now(),
        "api_spec_version": API_SPEC_VERSION,
        "execution_provider": "lambda",
        "placement_token": placement_token,
        "instance": instance_metadata,
        "image": {
            "reference": args.image_reference,
            "provenance": "caller-supplied; Lambda's instance response does not expose image",
        },
        "host": host_metadata,
        "source": {
            "path": "native/device_dispatch",
            "sha256": source_digest,
        },
        "benchmark": {
            "experiment_id": args.experiment_id,
            "agents": args.agents,
            "steps": args.steps,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "compile_argv": compile_command,
            "benchmark_argv_without_environment": benchmark_command[4:],
        },
        "artifacts": {
            name: {"sha256": sha256_bytes(value), "bytes": len(value)}
            for name, value in artifacts.items()
        },
        "native_manifest_summary": {
            "run_id": native_manifest.get("run_id"),
            "hardware": native_manifest.get("hardware"),
            "software": native_manifest.get("software"),
            "results": native_manifest.get("results"),
        },
        "remote_work_directory_retained": remote_root,
        "redactions": [
            "Lambda API key",
            "instance public and private IP addresses",
            "SSH private-key path and material",
            "Jupyter token and URL",
        ],
    }

    destination = Path(args.output_dir).expanduser()
    local_paths = {name: destination / name for name in filenames}
    provider_name = manifest_names[0].removesuffix(".manifest.json") + ".provider-lambda.json"
    provider_path = destination / provider_name
    conflicts = [path for path in [*local_paths.values(), provider_path] if path.exists()]
    if conflicts:
        raise RuntimeError(f"refusing to overwrite existing artifacts: {conflicts}")
    for name, value in artifacts.items():
        write_new_bytes(local_paths[name], value)
    write_new_json(provider_path, provider_manifest)

    print("remote_execution_status=ok")
    for path in local_paths.values():
        print(f"downloaded_artifact={path.resolve()}")
    print(f"provider_manifest={provider_path.resolve()}")
    print(f"remote_work_directory_retained={remote_root}")
    print("instance_termination_performed=false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file", default=str(DEFAULT_ENV_FILE), help="local file containing LAMBDA_API_KEY"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("plan", help="print a local-only plan (default)")

    inventory_parser = subparsers.add_parser(
        "inventory", help="make only authenticated GET requests"
    )
    inventory_parser.add_argument(
        "--output", help="optional new JSON path; existing files are never overwritten"
    )

    launch_parser = subparsers.add_parser("launch", help="create exactly one instance")
    launch_parser.add_argument("--instance-type", required=True)
    launch_parser.add_argument("--region", required=True)
    launch_parser.add_argument("--ssh-key-name", required=True)
    launch_parser.add_argument("--instance-name", default="gpu-agent-device-dispatch")
    launch_parser.add_argument("--image-family", default=IMAGE_FAMILY_DEFAULT)
    launch_parser.add_argument("--max-hourly-usd", required=True, type=float)
    launch_parser.add_argument("--allow-non-x86", action="store_true")
    launch_parser.add_argument("--confirm-spend", action="store_true")
    launch_parser.add_argument("--launch-confirmation", default="")
    launch_parser.add_argument("--output-dir", default="data/external")

    run_parser = subparsers.add_parser(
        "run-existing", help="compile and run on an already-active Lambda instance"
    )
    run_parser.add_argument("--instance-id", required=True)
    run_parser.add_argument("--ssh-key-name", required=True)
    run_parser.add_argument("--ssh-private-key", required=True)
    run_parser.add_argument("--ssh-user", default="ubuntu")
    run_parser.add_argument("--known-hosts", default="data/external/lambda-known-hosts")
    run_parser.add_argument("--image-reference", required=True)
    run_parser.add_argument("--confirm-remote-execution", action="store_true")
    run_parser.add_argument("--output-dir", default="data/raw")
    run_parser.add_argument("--experiment-id", default="device-dispatch-lambda-pilot")
    run_parser.add_argument("--agents", default="32,256,2048,16384")
    run_parser.add_argument("--steps", default="1,8,64")
    run_parser.add_argument("--warmups", type=int, default=10)
    run_parser.add_argument("--repetitions", type=int, default=50)
    run_parser.add_argument("--seed", type=int, default=20260811)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "plan"
    try:
        if command == "plan":
            command_plan(args)
        elif command == "inventory":
            command_inventory(args)
        elif command == "launch":
            command_launch(args)
        elif command == "run-existing":
            command_run_existing(args)
        else:
            parser.error(f"unknown command: {command}")
    except (LambdaApiError, RuntimeError, TypeError, subprocess.SubprocessError) as error:
        print(f"error={error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
