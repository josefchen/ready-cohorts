"""Safe Lambda Cloud runner for the frozen resident-policy-001 CUDA pilot.

The default command is a local-only plan. ``inventory`` performs authenticated
GET requests only. Creating one billable instance, executing on an existing
instance, and terminating an exact instance are separate, explicitly gated
operations. No command implicitly terminates or replaces a resource.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = REPO_ROOT / "native/resident_policy"
DEFAULT_ENV_FILE = REPO_ROOT.parent / ".env"

API_ROOT = "https://cloud.lambda.ai"
API_SPEC_VERSION = "1.10.0"
DOCS_URL = "https://docs.lambda.ai/public-cloud/cloud-api/"
IMAGE_FAMILY_DEFAULT = "lambda-stack-24-04"
CUDA_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
DOCKER_PREFIX = ("sudo", "-n", "docker")
FROZEN_SOURCE_SHA256 = "4b5cdcb9496a734bd7801d5c419efb8eceb72fd6962800520101e89676d204da"
FROZEN_MAKEFILE_SHA256 = "d74935b594fb629b2113d237439289e057281becc561b6941d1134bd6a1c1351"
SCHEMA_VERSION = "resident-policy-v1"
PROVIDER_SCHEMA_VERSION = "provider-lambda-resident-policy-v1"
INVENTORY_SCHEMA_VERSION = "provider-lambda-inventory-v1"
TERMINATION_SCHEMA_VERSION = "provider-lambda-termination-v1"

LAUNCH_CONFIRMATION = "LAUNCH_ONE_BILLABLE_LAMBDA_INSTANCE"
RUN_CONFIRMATION = "RUN_FROZEN_RESIDENT_POLICY_001"
TERMINATION_CONFIRMATION_PREFIX = "TERMINATE_EXACT_LAMBDA_INSTANCE_"
ALLOWED_GPU_FAMILIES = ("H100", "A10")
MECHANISMS = ("host_roundtrip", "device_resident", "no_decision_lower_bound")

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
COMMON_CONFIG: dict[str, Any] = {
    "mechanisms": list(MECHANISMS),
    "mechanism_order": "deterministically shuffled within cell and repetition",
    "batch_policy": (
        "one calibrated common lower-bound count per cell; each row extends until its "
        "aggregate wall time reaches the target, with a safety cap"
    ),
    "state_reset_in_timing": False,
    "result_copy_or_validation_in_timing": False,
    "host_predicate_copy_and_sync_in_timing": True,
    "graph_instantiation_or_upload_in_timing": False,
}
CSV_COLUMNS = (
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
)

SAFE_EXPERIMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SAFE_INSTANCE_ID = re.compile(r"[A-Za-z0-9-]{8,128}\Z")
SAFE_INSTANCE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
SAFE_REMOTE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,190}\Z")
SAFE_SSH_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}\Z")


class LambdaApiError(RuntimeError):
    """Sanitized Lambda API failure that never includes authorization headers."""

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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_bytes(path: Path, payload: bytes) -> None:
    """Create and fsync an artifact without ever replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}") from error


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    write_new_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _append_json_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


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


def validate_frozen_inputs(root: Path = NATIVE_ROOT) -> dict[str, str]:
    source = root / "resident_policy_pilot.cu"
    makefile = root / "Makefile"
    observed = {
        "source_sha256": sha256_file(source),
        "makefile_sha256": sha256_file(makefile),
    }
    expected = {
        "source_sha256": FROZEN_SOURCE_SHA256,
        "makefile_sha256": FROZEN_MAKEFILE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            "frozen resident-policy inputs changed; assign a new experiment ID before running: "
            f"expected={expected}, observed={observed}"
        )
    return observed


class LambdaApi:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._last_request_monotonic: float | None = None

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        if method not in {"GET", "POST"}:
            raise ValueError("only GET and POST are supported")
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
            "User-Agent": "gpu-agent-crossover-lambda-resident-policy/0.1",
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
    fields = ("id", "name", "status", "region", "instance_type", "ssh_key_names")
    return [
        {field: instance.get(field) for field in fields}
        for instance in data
        if isinstance(instance, dict)
    ]


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


def inventory(api: Any) -> dict[str, Any]:
    types = normalized_instance_types(api.get("/api/v1/instance-types"))
    instances = sanitized_instances(api.get("/api/v1/instances"))
    ssh_keys = api.get("/api/v1/ssh-keys")
    images = sanitized_images(api.get("/api/v1/images"))
    if not isinstance(ssh_keys, list):
        raise TypeError("unexpected SSH-key response shape")
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "collected_at_utc": utc_now(),
        "api_spec_version": API_SPEC_VERSION,
        "authentication": "ok",
        "instance_types": types,
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


def _gpu_family(description: Any) -> str | None:
    text = str(description).upper()
    for family in ALLOWED_GPU_FAMILIES:
        if re.search(rf"(?<![A-Z0-9]){re.escape(family)}(?![A-Z0-9])", text):
            return family
    return None


def _region_name(region: Any) -> str:
    if isinstance(region, dict):
        return str(region.get("name", ""))
    return str(region or "")


def _validated_launch_selection(
    record: dict[str, Any],
    *,
    instance_type: str,
    region: str,
    gpu_family: str,
    ssh_key_name: str,
    image_family: str,
    max_hourly_usd: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    if gpu_family not in ALLOWED_GPU_FAMILIES:
        raise RuntimeError(f"GPU family must be one of {ALLOWED_GPU_FAMILIES}")
    selected = next(
        (row for row in record["instance_types"] if row.get("name") == instance_type), None
    )
    if selected is None:
        raise RuntimeError(f"instance type is not in the live catalog: {instance_type}")
    actual_family = _gpu_family(selected.get("gpu_description"))
    if actual_family != gpu_family:
        raise RuntimeError(
            f"live instance type is {selected.get('gpu_description')!r}, not 1x {gpu_family}"
        )
    specs = selected.get("specs")
    if not isinstance(specs, dict) or specs.get("gpus") != 1:
        raise RuntimeError("launch is restricted to an instance type exposing exactly one GPU")
    if selected.get("architecture") != "x86_64":
        raise RuntimeError("launch is restricted to x86_64 to avoid an unplanned CPU confound")
    capacity_regions = {
        item.get("name")
        for item in selected.get("regions_with_capacity_available", [])
        if isinstance(item, dict)
    }
    if region not in capacity_regions:
        raise RuntimeError(
            f"live inventory reports no capacity for {instance_type} in {region}; launch refused"
        )
    cents = selected.get("price_cents_per_hour")
    if not isinstance(cents, (int, float)) or isinstance(cents, bool):
        raise TypeError("live instance price is missing or nonnumeric")
    price = float(cents) / 100.0
    if not math.isfinite(price) or price <= 0:
        raise RuntimeError("live instance price is not a finite positive number")
    if price > max_hourly_usd:
        raise RuntimeError(
            f"live price ${price:.2f}/hour exceeds --max-hourly-usd "
            f"${max_hourly_usd:.2f}; launch refused"
        )
    ssh_names = {item.get("name") for item in record["ssh_keys"]}
    if ssh_key_name not in ssh_names:
        raise RuntimeError("the selected SSH key name is not present in live inventory")
    matching_images = [
        image
        for image in record["images"]
        if image.get("family") == image_family
        and image.get("architecture") == selected.get("architecture")
        and _region_name(image.get("region")) == region
    ]
    if not matching_images:
        raise RuntimeError(f"no live {image_family} image for x86_64 in {region}; launch refused")
    return selected, matching_images, price


def _mode_config(mode: str) -> dict[str, Any]:
    if mode == "full":
        return FULL_CONFIG
    if mode == "smoke":
        return SMOKE_CONFIG
    raise ValueError("mode must be one of: full, smoke")


def _expected_row_count(config: dict[str, Any]) -> int:
    return (
        len(config["agent_counts"])
        * len(config["epoch_counts"])
        * len(MECHANISMS)
        * config["repetitions_per_mechanism_cell"]
    )


def _program_argv(mode: str, experiment_id: str, output_dir: str) -> list[str]:
    if not SAFE_EXPERIMENT.fullmatch(experiment_id):
        raise RuntimeError(
            "experiment ID must be 1-128 characters using letters, digits, '.', '_', or '-'"
        )
    config = _mode_config(mode)
    argv = [
        "/work/resident_policy_pilot",
        "--experiment-id",
        experiment_id,
        "--output-dir",
        output_dir,
    ]
    if mode == "smoke":
        argv.append("--smoke")
        return argv
    argv.extend(
        [
            "--agents",
            ",".join(map(str, config["agent_counts"])),
            "--epochs",
            ",".join(map(str, config["epoch_counts"])),
            "--warmups",
            str(config["warmups_per_mechanism_cell"]),
            "--calibration-samples",
            str(config["calibration_samples_per_mechanism_cell"]),
            "--repetitions",
            str(config["repetitions_per_mechanism_cell"]),
            "--min-duration-ms",
            str(config["min_duration_target_ns"] // 1_000_000),
            "--max-batch",
            str(config["max_batch_iterations"]),
            "--seed",
            str(config["seed"]),
            "--block-size",
            str(config["block_size"]),
        ]
    )
    return argv


def _canonical_gpu_uuid(value: Any) -> str:
    return re.sub(r"[^0-9a-f]", "", str(value).lower().removeprefix("gpu-"))


def artifact_validation_errors(
    *,
    csv_name: str,
    csv_text: str,
    manifest_name: str,
    manifest_text: str,
    experiment_id: str,
    mode: str,
    gpu_family: str,
    placement_id: str,
    image_digest: str,
    binary_sha256: str,
    actual_gpu_name: str,
    actual_gpu_uuid: str,
    program_returncode: int,
) -> list[str]:
    """Return every frozen-contract violation found in a downloaded result pair."""

    errors: list[str] = []
    config = _mode_config(mode)
    expected_rows = _expected_row_count(config)

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        return [f"manifest is not valid JSON: {error}"]
    if not isinstance(manifest, dict):
        return ["manifest root is not a JSON object"]
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
    except csv.Error as error:
        return [f"CSV is not parseable: {error}"]

    require(tuple(reader.fieldnames or ()) == CSV_COLUMNS, "CSV header differs from frozen schema")
    provenance = manifest.get("provenance", {})
    hardware = manifest.get("hardware", {})
    software = manifest.get("software", {})
    results = manifest.get("results", {})
    manifest_config = manifest.get("config", {})
    run_id = manifest.get("run_id")

    require(program_returncode == 0, f"program exited with status {program_returncode}")
    require(manifest.get("schema_version") == SCHEMA_VERSION, "manifest schema mismatch")
    require(manifest.get("experiment_id") == experiment_id, "manifest experiment ID mismatch")
    require(isinstance(run_id, str) and bool(run_id), "manifest run ID is empty")
    require(provenance.get("execution_provider") == "lambda", "provider is not lambda")
    require(provenance.get("requested_gpu") == gpu_family, "requested GPU family mismatch")
    require(provenance.get("placement_id") == placement_id, "placement ID mismatch")
    require(provenance.get("image_digest") == image_digest, "container image digest mismatch")
    require(
        provenance.get("source_sha256") == FROZEN_SOURCE_SHA256,
        "manifest source hash does not match the freeze",
    )
    require(
        provenance.get("binary_sha256") == binary_sha256,
        "manifest binary hash does not match the retrieved executable hash",
    )
    require(hardware.get("cuda_available") is True, "CUDA was not available")
    require(hardware.get("device_count") == 1, "placement did not expose exactly one GPU")
    require(hardware.get("unified_addressing") == 1, "unified addressing was not enabled")
    require(_gpu_family(hardware.get("device_name")) == gpu_family, "native GPU family mismatch")
    require(_gpu_family(actual_gpu_name) == gpu_family, "nvidia-smi GPU family mismatch")
    require(
        _canonical_gpu_uuid(hardware.get("device_uuid")) == _canonical_gpu_uuid(actual_gpu_uuid),
        "native and nvidia-smi GPU UUIDs differ",
    )
    compile_version = software.get("cuda_compile_version")
    runtime_version = software.get("cuda_runtime_version")
    require(
        isinstance(compile_version, int) and compile_version // 1000 == 13,
        "binary was not compiled with CUDA 13",
    )
    require(
        isinstance(runtime_version, int) and runtime_version // 1000 == 13,
        "benchmark did not use a CUDA 13 runtime",
    )
    require(len(rows) == expected_rows, f"CSV has {len(rows)} rows; expected {expected_rows}")
    if mode == "full":
        require(expected_rows == 810, "internal full-grid row-count contract is not 810")
        require(len(rows) == 810, "full run must contain exactly 810 measured rows")
    require(results.get("measured_rows") == expected_rows, "unexpected manifest measured-row count")
    require(results.get("exact_rows") == expected_rows, "not every measured row was exact")
    require(results.get("failure_rows") == 0, "native manifest contains failure rows")
    require(results.get("status_counts") == {"ok": expected_rows}, "status ledger is not all-ok")

    expected_manifest_config = {**config, **COMMON_CONFIG}
    require(
        manifest_config == expected_manifest_config, "manifest config differs from frozen config"
    )
    expected_cell_count = len(config["agent_counts"]) * len(config["epoch_counts"])
    cells = manifest.get("cells")
    require(isinstance(cells, list), "manifest cells ledger is not a list")
    if isinstance(cells, list):
        require(len(cells) == expected_cell_count, "cell-audit count mismatch")
        cell_keys: set[tuple[int, int]] = set()
        for cell_index, cell in enumerate(cells):
            if not isinstance(cell, dict):
                errors.append(f"manifest cell {cell_index} is not an object")
                continue
            try:
                cell_key = (int(cell["agents"]), int(cell["epochs"]))
                batch = int(cell["common_batch_iterations"])
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"manifest cell {cell_index} has invalid fields: {error}")
                continue
            cell_keys.add(cell_key)
            require(batch > 0, f"manifest cell {cell_key} has a nonpositive batch count")
            medians = cell.get("median_calibration_wall_ns", {})
            require(
                isinstance(medians, dict) and set(medians) == set(MECHANISMS),
                f"manifest cell {cell_key} has an invalid calibration ledger",
            )
            if isinstance(medians, dict):
                require(
                    all(isinstance(value, int) and value > 0 for value in medians.values()),
                    f"manifest cell {cell_key} has nonpositive calibration timing",
                )
        expected_cell_keys = {
            (agents, epochs)
            for agents in config["agent_counts"]
            for epochs in config["epoch_counts"]
        }
        require(cell_keys == expected_cell_keys, "manifest cells do not cover the frozen grid")

    require(manifest.get("csv_file") == csv_name, "manifest CSV filename mismatch")
    require(csv_name == f"{run_id}.csv", "CSV filename does not match the run ID")
    require(
        manifest_name == f"{run_id}.manifest.json",
        "manifest filename does not match the run ID",
    )
    semantic_contract = manifest.get("semantic_contract")
    require(
        isinstance(semantic_contract, dict) and set(semantic_contract) == {*MECHANISMS, "oracle"},
        "semantic contract is missing or incomplete",
    )
    require(isinstance(manifest.get("limitations"), list), "limitations ledger is missing")

    cell_counts: Counter[tuple[int, int, str]] = Counter()
    row_identities: set[tuple[int, int, str, int]] = set()
    observed_repetitions: dict[tuple[int, int, str], set[int]] = {}
    observed_orders: dict[tuple[int, int, int], set[int]] = {}
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
            wall_ns = float(row["wall_ns_per_invocation"])
            aggregate_device_ns = int(row["aggregate_device_ns"])
            device_ns = float(row["device_ns_per_invocation"])
            min_duration_target_ns = int(row["min_duration_target_ns"])
            exact_validation_count = int(row["exact_validation_count"])
            seed = int(row["seed"])
            block_size = int(row["block_size"])
            predicate_blocks = int(row["predicate_blocks"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{prefix} has invalid numeric fields: {error}")
            continue
        mechanism = row.get("mechanism", "")
        cell_key = (agents, epochs, mechanism)
        identity = (*cell_key, repetition)
        cell_counts[cell_key] += 1
        observed_repetitions.setdefault(cell_key, set()).add(repetition)
        observed_orders.setdefault((agents, epochs, repetition), set()).add(order_index)
        require(identity not in row_identities, f"{prefix} duplicates a measured identity")
        row_identities.add(identity)
        require(row.get("schema_version") == SCHEMA_VERSION, f"{prefix} schema mismatch")
        require(row.get("run_id") == run_id, f"{prefix} run ID mismatch")
        require(row.get("experiment_id") == experiment_id, f"{prefix} experiment mismatch")
        require(row.get("phase") == "measure", f"{prefix} is not a measured row")
        require(mechanism in MECHANISMS, f"{prefix} mechanism is invalid")
        require(agents in config["agent_counts"], f"{prefix} agent count is outside the grid")
        require(epochs in config["epoch_counts"], f"{prefix} epoch count is outside the grid")
        require(row.get("status") == "ok", f"{prefix} status is not ok")
        require(row.get("failure_stage", "") == "", f"{prefix} has a failure stage")
        require(row.get("error_code") == "0", f"{prefix} has a nonzero error code")
        require(row.get("error_message", "") == "", f"{prefix} has an error message")
        require(batch_iterations > 0, f"{prefix} has no timed invocations")
        require(aggregate_wall_ns > 0, f"{prefix} has nonpositive aggregate wall time")
        require(math.isfinite(wall_ns) and wall_ns > 0, f"{prefix} has invalid wall time")
        require(aggregate_device_ns > 0, f"{prefix} has nonpositive aggregate device time")
        require(math.isfinite(device_ns) and device_ns > 0, f"{prefix} has invalid device time")
        if batch_iterations > 0:
            require(
                math.isclose(
                    wall_ns,
                    aggregate_wall_ns / batch_iterations,
                    rel_tol=1e-12,
                    abs_tol=1e-6,
                ),
                f"{prefix} wall-time division mismatch",
            )
            require(
                math.isclose(
                    device_ns,
                    aggregate_device_ns / batch_iterations,
                    rel_tol=1e-12,
                    abs_tol=1e-6,
                ),
                f"{prefix} device-time division mismatch",
            )
        require(
            min_duration_target_ns == config["min_duration_target_ns"],
            f"{prefix} minimum duration target differs from the freeze",
        )
        require(row.get("min_duration_reached") == "true", f"{prefix} missed duration target")
        require(
            aggregate_wall_ns >= min_duration_target_ns,
            f"{prefix} aggregate wall time is below its target",
        )
        require(row.get("exact_state_match") == "true", f"{prefix} state is not field-exact")
        require(
            row.get("exact_decision_match") == "true",
            f"{prefix} decision trace is not exact",
        )
        require(
            row.get("expected_state_checksum") == row.get("observed_state_checksum"),
            f"{prefix} state checksum differs",
        )
        require(
            row.get("expected_decision_hash") == row.get("observed_decision_hash"),
            f"{prefix} decision hash differs",
        )
        expected_decisions = row.get("expected_decisions", "")
        observed_decisions = row.get("observed_decisions", "")
        require(expected_decisions == observed_decisions, f"{prefix} decision string differs")
        require(
            len(expected_decisions) == epochs and set(expected_decisions) <= {"0", "1"},
            f"{prefix} decision string is malformed",
        )
        require(
            exact_validation_count == batch_iterations,
            f"{prefix} did not validate every batched invocation",
        )
        require(seed == config["seed"], f"{prefix} seed differs from the freeze")
        require(block_size == config["block_size"], f"{prefix} block size differs from the freeze")
        require(
            predicate_blocks == (agents + block_size - 1) // block_size,
            f"{prefix} predicate block count is inconsistent",
        )

    expected_cells = {
        (agents, epochs, mechanism)
        for agents in config["agent_counts"]
        for epochs in config["epoch_counts"]
        for mechanism in MECHANISMS
    }
    require(set(cell_counts) == expected_cells, "CSV does not cover the frozen cell grid")
    for key in expected_cells:
        require(
            cell_counts[key] == config["repetitions_per_mechanism_cell"],
            f"cell {key} has the wrong row count",
        )
        require(
            observed_repetitions.get(key, set()) == expected_repetitions,
            f"cell {key} has the wrong repetition indices",
        )
    for agents in config["agent_counts"]:
        for epochs in config["epoch_counts"]:
            for repetition in expected_repetitions:
                order_key = (agents, epochs, repetition)
                require(
                    observed_orders.get(order_key, set()) == set(range(len(MECHANISMS))),
                    f"cell/repetition {order_key} has invalid mechanism order indices",
                )
    return errors


def command_plan(args: argparse.Namespace) -> None:
    frozen = validate_frozen_inputs()
    script = "python3 scripts/provider_lambda_resident_policy.py"
    print("action=plan")
    print("api_calls=0")
    print("remote_calls=0")
    print("billable_resources_created=0")
    print(f"lambda_api_docs={DOCS_URL}")
    print(f"lambda_openapi_version={API_SPEC_VERSION}")
    print(f"cuda_compile_image={CUDA_IMAGE}")
    print(f"source_sha256={frozen['source_sha256']}")
    print(f"makefile_sha256={frozen['makefile_sha256']}")
    print(
        f"inventory: {script} inventory "
        "--output data/external/lambda-resident-inventory-UNIQUE-UTC.json"
    )
    print(
        f"launch: {script} launch --gpu-family H100 --instance-type TYPE --region REGION "
        "--ssh-key-name KEY --max-hourly-usd LIMIT --confirm-spend "
        f"--launch-confirmation {LAUNCH_CONFIRMATION}"
    )
    print(
        f"run-existing: {script} run-existing --mode full --gpu-family H100 "
        "--instance-id ID --expected-instance-name NAME --ssh-key-name KEY "
        "--ssh-private-key KEYFILE --image-reference IMAGE --confirm-remote-execution "
        f"--run-confirmation {RUN_CONFIRMATION}"
    )
    print(
        f"terminate: {script} terminate --instance-id ID --expected-instance-name NAME "
        "--confirm-termination --termination-confirmation "
        f"{TERMINATION_CONFIRMATION_PREFIX}ID"
    )
    print("warning=launch does not auto-run and run-existing does not auto-terminate")


def command_inventory(args: argparse.Namespace) -> None:
    record = inventory(api_from_args(args))
    print("authentication=ok")
    print(f"collected_at_utc={record['collected_at_utc']}")
    eligible_pairs = 0
    for item in record["instance_types"]:
        family = _gpu_family(item.get("gpu_description"))
        specs = item.get("specs", {})
        eligible = family in ALLOWED_GPU_FAMILIES and specs.get("gpus") == 1
        for region in item["regions_with_capacity_available"]:
            if eligible:
                eligible_pairs += 1
            cents = item.get("price_cents_per_hour")
            price = float(cents) / 100 if isinstance(cents, (int, float)) else math.nan
            print(
                f"capacity type={item['name']} region={region['name']} "
                f"gpu={item['gpu_description']!r} gpus={specs.get('gpus')} "
                f"usd_per_hour={price:.2f} resident_policy_eligible={str(eligible).lower()}"
            )
    print(f"instance_type_count={len(record['instance_types'])}")
    print(f"eligible_type_region_pair_count={eligible_pairs}")
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


def command_launch(args: argparse.Namespace) -> None:
    validate_frozen_inputs()
    _require_launch_gates(args)
    if not SAFE_INSTANCE_NAME.fullmatch(args.instance_name):
        raise RuntimeError("instance name has an unsafe format")
    if not math.isfinite(args.max_hourly_usd) or args.max_hourly_usd <= 0:
        raise RuntimeError("--max-hourly-usd must be a finite positive number")
    api = api_from_args(args)
    record = inventory(api)
    selected, images, price = _validated_launch_selection(
        record,
        instance_type=args.instance_type,
        region=args.region,
        gpu_family=args.gpu_family,
        ssh_key_name=args.ssh_key_name,
        image_family=args.image_family,
        max_hourly_usd=args.max_hourly_usd,
    )
    output_dir = Path(args.output_dir).expanduser()
    receipt_path = output_dir / (
        f"resident-policy-lambda-launch-{compact_utc_now()}-{uuid.uuid4().hex[:8]}.jsonl"
    )
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
            {"key": "experiment", "value": "resident-policy-001"},
            {"key": "source-sha256", "value": FROZEN_SOURCE_SHA256},
        ],
    }
    _append_json_line(
        receipt,
        {
            "event": "launch_intent",
            "at_utc": utc_now(),
            "api_spec_version": API_SPEC_VERSION,
            "request_without_ssh_key_name": {
                key: value for key, value in request_body.items() if key != "ssh_key_names"
            },
            "validated_gpu_family": args.gpu_family,
            "validated_gpu_count": 1,
            "validated_price_usd_per_hour": price,
            "selected_type": selected,
            "image_candidates": images,
            "source_sha256": FROZEN_SOURCE_SHA256,
            "automatic_execution": False,
            "automatic_termination": False,
        },
    )
    instance_ids: list[Any] = []
    try:
        response = api.post("/api/v1/instance-operations/launch", request_body)
        instance_ids = response.get("instance_ids", []) if isinstance(response, dict) else []
        if len(instance_ids) != 1 or not SAFE_INSTANCE_ID.fullmatch(str(instance_ids[0])):
            raise RuntimeError("launch response did not contain exactly one safe instance ID")
        _append_json_line(
            receipt,
            {
                "event": "launch_accepted",
                "at_utc": utc_now(),
                "instance_id": instance_ids[0],
                "warning": "billing continues until this exact instance is terminated",
            },
        )
    except Exception as error:
        _append_json_line(
            receipt,
            {
                "event": "launch_failed_or_ambiguous",
                "at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "warning": "check inventory before retrying; a transport failure may be ambiguous",
            },
        )
        raise
    finally:
        receipt.close()
    print(f"launch_receipt={receipt_path.resolve()}")
    print(f"instance_id={instance_ids[0]}")
    print(f"validated_gpu_family={args.gpu_family}")
    print("validated_gpu_count=1")
    print(f"validated_price_usd_per_hour={price:.2f}")
    print("automatic_execution=false")
    print("automatic_termination=false")
    print("warning=billing continues until the exact instance is terminated")


def _get_instance(api: Any, instance_id: str) -> dict[str, Any]:
    if not SAFE_INSTANCE_ID.fullmatch(instance_id):
        raise RuntimeError("instance ID has an unsafe format")
    data = api.get(f"/api/v1/instances/{instance_id}")
    if not isinstance(data, dict):
        raise TypeError("unexpected instance response shape")
    if data.get("id") != instance_id:
        raise RuntimeError("instance ID mismatch in Lambda response")
    return data


def _validated_existing_instance(
    instance: dict[str, Any],
    *,
    expected_name: str,
    gpu_family: str,
    ssh_key_name: str | None,
    require_active: bool,
) -> dict[str, Any]:
    if instance.get("name") != expected_name:
        raise RuntimeError("instance name does not match --expected-instance-name")
    if require_active and instance.get("status") != "active":
        raise RuntimeError(f"instance status is {instance.get('status')!r}, not 'active'")
    instance_type = instance.get("instance_type")
    if not isinstance(instance_type, dict):
        raise TypeError("instance response does not contain structured type metadata")
    specs = instance_type.get("specs")
    if not isinstance(specs, dict) or specs.get("gpus") != 1:
        raise RuntimeError("existing instance does not expose exactly one GPU")
    actual_family = _gpu_family(instance_type.get("gpu_description"))
    if actual_family != gpu_family:
        raise RuntimeError(
            f"existing instance is {instance_type.get('gpu_description')!r}, not 1x {gpu_family}"
        )
    if instance_type.get("architecture") != "x86_64":
        raise RuntimeError("existing instance is not x86_64")
    if ssh_key_name is not None and ssh_key_name not in instance.get("ssh_key_names", []):
        raise RuntimeError("selected SSH key name is not attached to this instance")
    return instance_type


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


def _run_process(
    command: list[str],
    *,
    text: bool = True,
    check: bool = True,
    timeout_seconds: int = 90 * 60,
) -> subprocess.CompletedProcess[Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=text,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "external SSH/SCP command timed out; connection details redacted"
        ) from None
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"external SSH/SCP command failed with exit code {completed.returncode}; "
            "connection details and stderr redacted"
        )
    return completed


def _remote_capture(ssh: list[str], command: list[str], *, timeout_seconds: int = 600) -> str:
    completed = _run_process(
        [*ssh, shlex.join(command)], text=True, check=True, timeout_seconds=timeout_seconds
    )
    return completed.stdout.strip()


def _remote_capture_optional(ssh: list[str], command: list[str]) -> dict[str, Any]:
    completed = _run_process(
        [*ssh, shlex.join(command)], text=True, check=False, timeout_seconds=600
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr_redacted": bool(completed.stderr.strip()),
    }


def _parse_single_gpu_inventory(value: str, expected_family: str) -> dict[str, str]:
    rows = list(csv.reader(io.StringIO(value)))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("nvidia-smi did not report exactly one five-field GPU row")
    name, gpu_uuid, pci_bus_id, driver_version, memory_total = [item.strip() for item in rows[0]]
    if _gpu_family(name) != expected_family:
        raise RuntimeError(f"nvidia-smi reports {name!r}, not {expected_family}")
    if not gpu_uuid or not pci_bus_id or not driver_version or not memory_total:
        raise RuntimeError("nvidia-smi GPU provenance contains an empty field")
    return {
        "name": name,
        "uuid": gpu_uuid,
        "pci_bus_id": pci_bus_id,
        "driver_version": driver_version,
        "memory_total": memory_total,
    }


def _cuda_major(nvcc_version: str) -> int | None:
    match = re.search(r"release\s+([0-9]+)(?:\.[0-9]+)?", nvcc_version, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _docker_compile_command(remote_root: str) -> list[str]:
    inner = "\n".join(  # noqa: FLY002 - preserving one command per audited line is intentional.
        [
            "set -euo pipefail",
            "export DEBIAN_FRONTEND=noninteractive",
            "apt-get update -qq",
            ("apt-get install -y -qq --no-install-recommends g++ make git ca-certificates"),
            "nvcc --version",
            "g++ --version",
            ("make -C /work/resident_policy TARGET=/work/resident_policy_pilot NVCC=nvcc all"),
            "/work/resident_policy_pilot --help",
        ]
    )
    return [
        *DOCKER_PREFIX,
        "run",
        "--rm",
        "--network=bridge",
        "--mount",
        f"type=bind,src={remote_root},dst=/work",
        "--workdir",
        "/work/resident_policy",
        CUDA_IMAGE,
        "bash",
        "-lc",
        inner,
    ]


def _docker_program_command(
    *,
    remote_root: str,
    mode: str,
    experiment_id: str,
    gpu_family: str,
    placement_id: str,
    image_digest: str,
    binary_sha256: str,
) -> list[str]:
    command = [
        *DOCKER_PREFIX,
        "run",
        "--rm",
        "--gpus",
        "device=0",
        "--network=none",
        "--mount",
        f"type=bind,src={remote_root},dst=/work",
        "--workdir",
        "/work",
    ]
    environment = {
        "EXECUTION_PROVIDER": "lambda",
        "REQUESTED_GPU": gpu_family,
        "PLACEMENT_ID": placement_id,
        "IMAGE_DIGEST": image_digest,
        "SOURCE_SHA256": FROZEN_SOURCE_SHA256,
        "BINARY_SHA256": binary_sha256,
    }
    for key, value in environment.items():
        command.extend(["--env", f"{key}={value}"])
    command.append(CUDA_IMAGE)
    command.extend(_program_argv(mode, experiment_id, "/work/output"))
    return command


def _validate_run_gates(args: argparse.Namespace) -> None:
    if not args.confirm_remote_execution:
        raise RuntimeError("run refused: --confirm-remote-execution is required")
    if args.run_confirmation != RUN_CONFIRMATION:
        raise RuntimeError("run refused: --run-confirmation must exactly equal " + RUN_CONFIRMATION)
    if not SAFE_EXPERIMENT.fullmatch(args.experiment_id):
        raise RuntimeError("experiment ID has an unsafe format")
    if not SAFE_INSTANCE_NAME.fullmatch(args.expected_instance_name):
        raise RuntimeError("expected instance name has an unsafe format")
    if not SAFE_SSH_USER.fullmatch(args.ssh_user):
        raise RuntimeError("SSH user has an unsafe format")
    if not args.image_reference.strip() or len(args.image_reference) > 512:
        raise RuntimeError("--image-reference is required and must contain at most 512 characters")


def _instance_without_addresses(instance: dict[str, Any]) -> dict[str, Any]:
    fields = ("id", "name", "status", "region", "instance_type", "ssh_key_names")
    return {field: instance.get(field) for field in fields}


def command_run_existing(args: argparse.Namespace) -> None:
    frozen = validate_frozen_inputs()
    _validate_run_gates(args)
    config = _mode_config(args.mode)
    api = api_from_args(args)
    instance = _get_instance(api, args.instance_id)
    _validated_existing_instance(
        instance,
        expected_name=args.expected_instance_name,
        gpu_family=args.gpu_family,
        ssh_key_name=args.ssh_key_name,
        require_active=True,
    )
    host = instance.get("ip")
    if not isinstance(host, str) or not host:
        raise RuntimeError("active instance does not expose a public IP")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        raise RuntimeError("Lambda returned an invalid public IP address") from None

    key_path = _validate_private_key(Path(args.ssh_private_key))
    known_hosts = Path(args.known_hosts).expanduser().resolve()
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    known_hosts.touch(mode=0o600, exist_ok=True)
    os.chmod(known_hosts, 0o600)
    ssh = _ssh_prefix(key_path, known_hosts, args.ssh_user, host)

    placement_token = uuid.uuid4().hex
    placement_id = f"lambda-{args.gpu_family.lower()}-{args.instance_id}-{placement_token[:12]}"
    remote_root = f"/tmp/gpu-agent-resident-policy-{placement_token}"
    remote_native = f"{remote_root}/resident_policy"
    remote_binary = f"{remote_root}/resident_policy_pilot"
    remote_output = f"{remote_root}/output"
    _remote_capture(ssh, ["mkdir", "--mode=700", remote_root])
    _remote_capture(ssh, ["mkdir", "--mode=700", remote_native])

    scp = [
        "scp",
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
        str(NATIVE_ROOT / "resident_policy_pilot.cu"),
        str(NATIVE_ROOT / "Makefile"),
        f"{args.ssh_user}@{host}:{remote_native}/",
    ]
    _run_process(scp, timeout_seconds=600)
    remote_source_hash = _remote_capture(
        ssh, ["sha256sum", f"{remote_native}/resident_policy_pilot.cu"]
    ).split()[0]
    remote_makefile_hash = _remote_capture(ssh, ["sha256sum", f"{remote_native}/Makefile"]).split()[
        0
    ]
    if remote_source_hash != FROZEN_SOURCE_SHA256:
        raise RuntimeError("remote CUDA source hash differs from the frozen source")
    if remote_makefile_hash != FROZEN_MAKEFILE_SHA256:
        raise RuntimeError("remote Makefile hash differs from the frozen Makefile")

    raw_gpu_inventory = _remote_capture(
        ssh,
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total",
            "--format=csv,noheader",
        ],
    )
    gpu = _parse_single_gpu_inventory(raw_gpu_inventory, args.gpu_family)
    _remote_capture(ssh, ["command", "-v", "docker"])
    _remote_capture(ssh, ["sudo", "-n", "true"])
    docker_pull = _remote_capture(
        ssh, [*DOCKER_PREFIX, "pull", CUDA_IMAGE], timeout_seconds=20 * 60
    )
    container_image_id = _remote_capture(
        ssh, [*DOCKER_PREFIX, "image", "inspect", "--format={{.Id}}", CUDA_IMAGE]
    )
    repo_digests_json = _remote_capture(
        ssh,
        [*DOCKER_PREFIX, "image", "inspect", "--format={{json .RepoDigests}}", CUDA_IMAGE],
    )
    try:
        repo_digests = json.loads(repo_digests_json)
    except json.JSONDecodeError as error:
        raise RuntimeError("Docker returned invalid image-digest JSON") from error
    if not isinstance(repo_digests, list):
        raise TypeError("Docker image digest inventory is not a list")
    image_digest = str(repo_digests[0]) if repo_digests else container_image_id
    if not image_digest:
        raise RuntimeError("CUDA container image provenance is empty")

    compile_argv = _docker_compile_command(remote_root)
    compile_result = _run_process(
        [*ssh, shlex.join(compile_argv)],
        text=True,
        check=True,
        timeout_seconds=30 * 60,
    )
    if _cuda_major(compile_result.stdout) != 13:
        raise RuntimeError("compile container did not report CUDA 13")
    binary_sha256 = _remote_capture(ssh, ["sha256sum", remote_binary]).split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", binary_sha256):
        raise RuntimeError("remote binary SHA-256 has an invalid format")
    _remote_capture(ssh, ["mkdir", "--mode=700", remote_output])

    program_argv = _docker_program_command(
        remote_root=remote_root,
        mode=args.mode,
        experiment_id=args.experiment_id,
        gpu_family=args.gpu_family,
        placement_id=placement_id,
        image_digest=image_digest,
        binary_sha256=binary_sha256,
    )
    program_result = _run_process(
        [*ssh, shlex.join(program_argv)],
        text=True,
        check=False,
        timeout_seconds=90 * 60,
    )

    listing = _remote_capture(
        ssh,
        ["find", remote_output, "-maxdepth", "1", "-type", "f", "-printf", "%f\\n"],
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
            [*ssh, shlex.join(["cat", "--", f"{remote_output}/{filename}"])],
            text=False,
            check=True,
            timeout_seconds=600,
        )
        artifacts[filename] = completed.stdout
    try:
        csv_text = artifacts[csv_names[0]].decode("utf-8")
        manifest_text = artifacts[manifest_names[0]].decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("remote scientific artifact is not valid UTF-8") from error

    validation_errors = artifact_validation_errors(
        csv_name=csv_names[0],
        csv_text=csv_text,
        manifest_name=manifest_names[0],
        manifest_text=manifest_text,
        experiment_id=args.experiment_id,
        mode=args.mode,
        gpu_family=args.gpu_family,
        placement_id=placement_id,
        image_digest=image_digest,
        binary_sha256=binary_sha256,
        actual_gpu_name=gpu["name"],
        actual_gpu_uuid=gpu["uuid"],
        program_returncode=program_result.returncode,
    )
    native_manifest = json.loads(manifest_text)
    remote_hostname = _remote_capture(ssh, ["hostname"])
    lscpu_json = json.loads(_remote_capture(ssh, ["lscpu", "--json"]))
    detailed_gpu = _remote_capture_optional(
        ssh,
        [
            "nvidia-smi",
            (
                "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,"
                "compute_cap,pstate,power.limit,clocks.sm,clocks.mem"
            ),
            "--format=csv,noheader",
        ],
    )
    host_metadata = {
        "hostname_present": bool(remote_hostname),
        "hostname_redacted": True,
        "kernel": _remote_capture(ssh, ["uname", "-srmo"]),
        "architecture": _remote_capture(ssh, ["uname", "-m"]),
        "logical_cpu_count": _remote_capture(ssh, ["nproc"]),
        "lscpu_json": lscpu_json,
        "numa_hardware": _remote_capture_optional(ssh, ["numactl", "--hardware"]),
        "os_release": _remote_capture(ssh, ["cat", "/etc/os-release"]),
        "docker_version": _remote_capture_optional(
            ssh, [*DOCKER_PREFIX, "version", "--format", "{{json .}}"]
        ),
        "docker_access": "passwordless sudo constrained to explicit docker commands",
        "gpu_inventory": gpu,
        "gpu_extended_inventory": detailed_gpu,
        "binary_sha256": binary_sha256,
    }
    provider_manifest = {
        "schema_version": PROVIDER_SCHEMA_VERSION,
        "collected_at_utc": utc_now(),
        "api_spec_version": API_SPEC_VERSION,
        "execution_provider": "lambda",
        "provider_adapter": {
            "path": "scripts/provider_lambda_resident_policy.py",
            "sha256": sha256_file(Path(__file__).resolve()),
            "docker_access_correction": (
                "passwordless sudo prefix added before the first successful Lambda run; "
                "frozen CUDA source and experiment contract unchanged"
            ),
        },
        "mode": args.mode,
        "placement_id": placement_id,
        "instance": _instance_without_addresses(instance),
        "lambda_boot_image": {
            "reference": args.image_reference,
            "provenance": "caller-supplied because the instance response omits its boot image",
        },
        "cuda_container": {
            "reference": CUDA_IMAGE,
            "image_id": container_image_id,
            "repo_digests": repo_digests,
            "selected_digest": image_digest,
            "pull_stdout_sha256": sha256_bytes(docker_pull.encode()),
            "network_disabled_during_benchmark": True,
        },
        "host": host_metadata,
        "source": {
            "path": "native/resident_policy",
            **frozen,
            "remote_source_sha256": remote_source_hash,
            "remote_makefile_sha256": remote_makefile_hash,
        },
        "build": {
            "compile_argv": compile_argv,
            "compiler_stdout": compile_result.stdout,
            "compiler_stderr": compile_result.stderr,
            "binary_sha256": binary_sha256,
            "cuda_major_validated": 13,
        },
        "benchmark": {
            "experiment_id": args.experiment_id,
            "mode": args.mode,
            "frozen_config": config,
            "expected_rows": _expected_row_count(config),
            "program_argv": program_argv,
            "program_returncode": program_result.returncode,
            "program_stdout": program_result.stdout,
            "program_stderr": program_result.stderr,
        },
        "artifacts": {
            name: {"sha256": sha256_bytes(payload), "bytes": len(payload)}
            for name, payload in artifacts.items()
        },
        "artifact_validation": {
            "passed": not validation_errors,
            "errors": validation_errors,
            "full_mode_requires_exactly_810_rows": True,
            "field_and_decision_exactness_required": True,
        },
        "native_manifest_summary": {
            "run_id": native_manifest.get("run_id"),
            "provenance": native_manifest.get("provenance"),
            "hardware": native_manifest.get("hardware"),
            "software": native_manifest.get("software"),
            "results": native_manifest.get("results"),
        },
        "remote_work_directory_retained": remote_root,
        "automatic_termination": False,
        "redactions": [
            "Lambda API key",
            "instance public and private IP addresses",
            "SSH private-key path and material",
            "known-host key material",
            "Jupyter tokens and URLs",
            "remote hostname",
        ],
    }

    destination = Path(args.output_dir).expanduser()
    local_paths = {name: destination / name for name in filenames}
    provider_name = (
        manifest_names[0].removesuffix(".manifest.json") + ".provider-lambda-resident-policy.json"
    )
    provider_path = destination / provider_name
    conflicts = [path for path in [*local_paths.values(), provider_path] if path.exists()]
    if conflicts:
        raise RuntimeError(f"refusing to overwrite existing artifacts: {conflicts}")
    for name, payload in artifacts.items():
        write_new_bytes(local_paths[name], payload)
    write_new_json(provider_path, provider_manifest)
    for name, path in local_paths.items():
        if sha256_file(path) != sha256_bytes(artifacts[name]):
            raise RuntimeError(f"downloaded artifact hash mismatch: {path}")

    print(f"remote_execution_status={'ok' if not validation_errors else 'invalid'}")
    print(f"mode={args.mode}")
    print(f"expected_rows={_expected_row_count(config)}")
    for path in local_paths.values():
        print(f"downloaded_artifact={path.resolve()}")
    print(f"provider_manifest={provider_path.resolve()}")
    print(f"placement_id={placement_id}")
    print(f"remote_work_directory_retained={remote_root}")
    print("instance_termination_performed=false")
    if validation_errors:
        raise RuntimeError(
            "resident-policy result gates failed; append-only artifacts were retained: "
            + "; ".join(validation_errors)
        )
    print("result_gates=passed")


def _require_termination_gates(args: argparse.Namespace) -> None:
    if not args.confirm_termination:
        raise RuntimeError("termination refused: --confirm-termination is required")
    expected = TERMINATION_CONFIRMATION_PREFIX + args.instance_id
    if args.termination_confirmation != expected:
        raise RuntimeError(
            "termination refused: --termination-confirmation must exactly equal " + expected
        )
    if not SAFE_INSTANCE_NAME.fullmatch(args.expected_instance_name):
        raise RuntimeError("expected instance name has an unsafe format")


def _termination_response_summary(response: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"response_type": type(response).__name__}
    if not isinstance(response, dict):
        return summary
    for key in ("instance_ids", "status"):
        value = response.get(key)
        if isinstance(value, (str, int, float, bool, list)) or value is None:
            summary[key] = value
    instances = response.get("instances") or response.get("terminated_instances")
    if isinstance(instances, list):
        summary["instances"] = [
            {key: item.get(key) for key in ("id", "name", "status")}
            for item in instances
            if isinstance(item, dict)
        ]
    return summary


def command_terminate(args: argparse.Namespace) -> None:
    _require_termination_gates(args)
    api = api_from_args(args)
    instance = _get_instance(api, args.instance_id)
    if instance.get("name") != args.expected_instance_name:
        raise RuntimeError("termination target name does not match --expected-instance-name")
    status = str(instance.get("status", ""))
    if status in {"terminated", "terminating"}:
        raise RuntimeError(f"instance is already {status}; no termination request was sent")
    if not status:
        raise RuntimeError("termination target has no status")
    output_dir = Path(args.output_dir).expanduser()
    receipt_path = output_dir / f"resident-policy-lambda-termination-{args.instance_id}.jsonl"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt = receipt_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite termination receipt: {receipt_path}") from error
    _append_json_line(
        receipt,
        {
            "schema_version": TERMINATION_SCHEMA_VERSION,
            "event": "termination_intent",
            "at_utc": utc_now(),
            "instance": _instance_without_addresses(instance),
            "exact_target_count": 1,
            "ephemeral_instance_data_recoverable": False,
        },
    )
    try:
        response = api.post(
            "/api/v1/instance-operations/terminate", {"instance_ids": [args.instance_id]}
        )
        _append_json_line(
            receipt,
            {
                "schema_version": TERMINATION_SCHEMA_VERSION,
                "event": "termination_requested",
                "at_utc": utc_now(),
                "instance_id": args.instance_id,
                "response": _termination_response_summary(response),
            },
        )
    except Exception as error:
        _append_json_line(
            receipt,
            {
                "schema_version": TERMINATION_SCHEMA_VERSION,
                "event": "termination_failed_or_ambiguous",
                "at_utc": utc_now(),
                "instance_id": args.instance_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "warning": "check exact-instance inventory before retrying",
            },
        )
        raise
    finally:
        receipt.close()
    print(f"termination_receipt={receipt_path.resolve()}")
    print(f"termination_target={args.instance_id}")
    print("termination_target_count=1")
    print("ephemeral_instance_data_recoverable=false")
    print("warning=poll inventory until the exact instance is terminating, terminated, or absent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file", default=str(DEFAULT_ENV_FILE), help="local file containing LAMBDA_API_KEY"
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("plan", help="print a local-only plan (default)")

    inventory_parser = subparsers.add_parser(
        "inventory", help="make authenticated GET requests only"
    )
    inventory_parser.add_argument(
        "--output", help="optional new JSON path; existing files are never overwritten"
    )

    launch_parser = subparsers.add_parser("launch", help="create exactly one 1x H100 or A10")
    launch_parser.add_argument("--gpu-family", required=True, choices=ALLOWED_GPU_FAMILIES)
    launch_parser.add_argument("--instance-type", required=True)
    launch_parser.add_argument("--region", required=True)
    launch_parser.add_argument("--ssh-key-name", required=True)
    launch_parser.add_argument("--instance-name", default="gpu-agent-resident-policy-001")
    launch_parser.add_argument("--image-family", default=IMAGE_FAMILY_DEFAULT)
    launch_parser.add_argument("--max-hourly-usd", required=True, type=float)
    launch_parser.add_argument("--confirm-spend", action="store_true")
    launch_parser.add_argument("--launch-confirmation", default="")
    launch_parser.add_argument("--output-dir", default="data/external")

    run_parser = subparsers.add_parser(
        "run-existing", help="compile CUDA 13 and run on one active Lambda instance"
    )
    run_parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    run_parser.add_argument("--gpu-family", required=True, choices=ALLOWED_GPU_FAMILIES)
    run_parser.add_argument("--instance-id", required=True)
    run_parser.add_argument("--expected-instance-name", required=True)
    run_parser.add_argument("--ssh-key-name", required=True)
    run_parser.add_argument("--ssh-private-key", required=True)
    run_parser.add_argument("--ssh-user", default="ubuntu")
    run_parser.add_argument("--known-hosts", default="data/external/lambda-known-hosts")
    run_parser.add_argument("--image-reference", required=True)
    run_parser.add_argument("--experiment-id", required=True)
    run_parser.add_argument("--confirm-remote-execution", action="store_true")
    run_parser.add_argument("--run-confirmation", default="")
    run_parser.add_argument("--output-dir", default="data/raw")

    terminate_parser = subparsers.add_parser(
        "terminate", help="terminate one exact instance after two confirmations"
    )
    terminate_parser.add_argument("--instance-id", required=True)
    terminate_parser.add_argument("--expected-instance-name", required=True)
    terminate_parser.add_argument("--confirm-termination", action="store_true")
    terminate_parser.add_argument("--termination-confirmation", default="")
    terminate_parser.add_argument("--output-dir", default="data/external")
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
        elif command == "terminate":
            command_terminate(args)
        else:
            parser.error(f"unknown command: {command}")
    except (
        LambdaApiError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"error={error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
