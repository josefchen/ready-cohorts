"""Safely compile or run the frozen resident-policy CUDA pilot on Modal.

The default ``plan`` action is entirely local. The ``compile`` action uses a
CPU-only Modal worker. A GPU is requested only when both ``--action run`` and
``--confirm-spend`` are supplied, and each such invocation requests one L4.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NATIVE_ROOT = REPO_ROOT / "native/resident_policy"
REMOTE_NATIVE_ROOT = Path("/opt/gpu-agent-crossover/native/resident_policy")
REMOTE_BINARY = Path("/tmp/resident_policy_pilot")
REMOTE_OUTPUT = Path("/tmp/resident-policy-output")

CUDA_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
FROZEN_SOURCE_SHA256 = "4b5cdcb9496a734bd7801d5c419efb8eceb72fd6962800520101e89676d204da"
FROZEN_MAKEFILE_SHA256 = "d74935b594fb629b2113d237439289e057281becc561b6941d1134bd6a1c1351"
SCHEMA_VERSION = "resident-policy-v1"
GPU_TYPE = "L4"
MECHANISMS = {"host_roundtrip", "device_resident", "no_decision_lower_bound"}

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

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("g++", "make")
    .add_local_file(
        LOCAL_NATIVE_ROOT / "resident_policy_pilot.cu",
        str(REMOTE_NATIVE_ROOT / "resident_policy_pilot.cu"),
        copy=True,
    )
    .add_local_file(
        LOCAL_NATIVE_ROOT / "Makefile",
        str(REMOTE_NATIVE_ROOT / "Makefile"),
        copy=True,
    )
)

app = modal.App("gpu-agent-resident-policy-pilot", image=image)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _validate_frozen_inputs(root: Path) -> dict[str, str]:
    source_path = root / "resident_policy_pilot.cu"
    makefile_path = root / "Makefile"
    observed = {
        "source_sha256": _sha256_file(source_path),
        "makefile_sha256": _sha256_file(makefile_path),
    }
    expected = {
        "source_sha256": FROZEN_SOURCE_SHA256,
        "makefile_sha256": FROZEN_MAKEFILE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            "frozen resident-policy inputs changed; assign a new experiment ID "
            f"before running: expected={expected}, observed={observed}"
        )
    return observed


def _compile() -> dict[str, str]:
    frozen = _validate_frozen_inputs(REMOTE_NATIVE_ROOT)
    command = [
        "make",
        "-C",
        str(REMOTE_NATIVE_ROOT),
        f"TARGET={REMOTE_BINARY}",
        "all",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    help_check = subprocess.run(
        [str(REMOTE_BINARY), "--help"], check=True, text=True, capture_output=True
    )
    nvcc_version = subprocess.run(["nvcc", "--version"], check=True, text=True, capture_output=True)
    return {
        **frozen,
        "command": " ".join(command),
        "compiler_stdout": completed.stdout,
        "compiler_stderr": completed.stderr,
        "help_stdout": help_check.stdout,
        "help_stderr": help_check.stderr,
        "nvcc_version": nvcc_version.stdout,
        "binary_sha256": _sha256_file(REMOTE_BINARY),
    }


def _mode_config(mode: str) -> dict[str, Any]:
    if mode == "smoke":
        return SMOKE_CONFIG
    if mode == "full":
        return FULL_CONFIG
    raise ValueError("mode must be one of: smoke, full")


def _validate_experiment_id(experiment_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", experiment_id) is None:
        raise ValueError(
            "experiment ID must be 1-128 characters using only letters, digits, '.', '_', or '-'"
        )


def _program_command(mode: str, experiment_id: str) -> list[str]:
    config = _mode_config(mode)
    _validate_experiment_id(experiment_id)
    command = [
        str(REMOTE_BINARY),
        "--experiment-id",
        experiment_id,
        "--output-dir",
        str(REMOTE_OUTPUT),
    ]
    if mode == "smoke":
        command.append("--smoke")
        return command
    command.extend(
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
    return command


def _expected_row_count(config: dict[str, Any]) -> int:
    return (
        len(config["agent_counts"])
        * len(config["epoch_counts"])
        * len(MECHANISMS)
        * config["repetitions_per_mechanism_cell"]
    )


def _artifact_validation_errors(
    *,
    csv_name: str,
    csv_text: str,
    manifest_name: str,
    manifest_text: str,
    experiment_id: str,
    mode: str,
    binary_sha256: str,
    placement_id: str,
    program_returncode: int,
) -> list[str]:
    errors: list[str] = []
    config = _mode_config(mode)
    expected_rows = _expected_row_count(config)

    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        return [f"manifest is not valid JSON: {error}"]

    try:
        rows = list(csv.DictReader(io.StringIO(csv_text)))
    except csv.Error as error:
        return [f"CSV is not parseable: {error}"]

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    required_columns = {
        "schema_version",
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
    }
    fieldnames = set(rows[0]) if rows else set()
    require(required_columns <= fieldnames, "CSV is missing required columns")

    provenance = manifest.get("provenance", {})
    hardware = manifest.get("hardware", {})
    results = manifest.get("results", {})
    manifest_config = manifest.get("config", {})
    run_id = manifest.get("run_id")

    require(program_returncode == 0, f"program exited with status {program_returncode}")
    require(manifest.get("schema_version") == SCHEMA_VERSION, "manifest schema mismatch")
    require(manifest.get("experiment_id") == experiment_id, "manifest experiment ID mismatch")
    require(bool(run_id), "manifest run ID is empty")
    require(provenance.get("execution_provider") == "modal", "provider is not modal")
    require(provenance.get("requested_gpu") == GPU_TYPE, "requested GPU is not L4")
    require(provenance.get("placement_id") == placement_id, "placement ID mismatch")
    require(bool(provenance.get("image_digest")), "image provenance is empty")
    require(
        provenance.get("source_sha256") == FROZEN_SOURCE_SHA256,
        "manifest source hash does not match the freeze",
    )
    require(
        provenance.get("binary_sha256") == binary_sha256,
        "manifest binary hash does not match the compiled executable",
    )
    require(hardware.get("cuda_available") is True, "CUDA was not available")
    require(hardware.get("device_count") == 1, "placement did not expose exactly one GPU")
    require(GPU_TYPE in str(hardware.get("device_name", "")), "actual GPU is not an L4")
    require(results.get("measured_rows") == expected_rows, "unexpected measured-row count")
    require(results.get("exact_rows") == expected_rows, "not every measured row was exact")
    require(results.get("failure_rows") == 0, "native manifest contains failure rows")
    require(results.get("status_counts") == {"ok": expected_rows}, "status ledger is not all-ok")
    require(len(rows) == expected_rows, "CSV row count does not match the frozen grid")

    for key, value in config.items():
        require(manifest_config.get(key) == value, f"manifest config mismatch for {key}")
    require(set(manifest_config.get("mechanisms", [])) == MECHANISMS, "mechanism set mismatch")
    require(
        len(manifest.get("cells", [])) == len(config["agent_counts"]) * len(config["epoch_counts"]),
        "cell-audit count mismatch",
    )

    require(manifest.get("csv_file") == csv_name, "manifest CSV filename mismatch")
    require(csv_name == f"{run_id}.csv", "CSV filename does not match the run ID")
    require(
        manifest_name == f"{run_id}.manifest.json",
        "manifest filename does not match the run ID",
    )

    cell_counts: Counter[tuple[int, int, str]] = Counter()
    unique_rows: set[tuple[int, int, str, int]] = set()
    observed_orders: dict[tuple[int, int, int], set[int]] = {}
    expected_repetitions = set(range(config["repetitions_per_mechanism_cell"]))
    observed_repetitions: dict[tuple[int, int, str], set[int]] = {}
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
            min_duration_target_ns = int(row["min_duration_target_ns"])
            exact_validation_count = int(row["exact_validation_count"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{prefix} has invalid numeric fields: {error}")
            continue
        mechanism = row.get("mechanism", "")
        key = (agents, epochs, mechanism)
        identity = (*key, repetition)
        cell_counts[key] += 1
        observed_repetitions.setdefault(key, set()).add(repetition)
        observed_orders.setdefault((agents, epochs, repetition), set()).add(order_index)
        require(identity not in unique_rows, f"{prefix} duplicates a measured identity")
        unique_rows.add(identity)
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
        require(aggregate_wall_ns > 0 and wall_ns > 0.0, f"{prefix} has nonpositive timing")
        require(
            row.get("min_duration_reached") == "true",
            f"{prefix} did not reach the minimum duration",
        )
        require(
            aggregate_wall_ns >= min_duration_target_ns,
            f"{prefix} aggregate wall time is below its target",
        )
        require(row.get("exact_state_match") == "true", f"{prefix} state is not exact")
        require(row.get("exact_decision_match") == "true", f"{prefix} decisions are not exact")
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
            exact_validation_count == batch_iterations,
            f"{prefix} did not validate every batched invocation",
        )

    repetitions = config["repetitions_per_mechanism_cell"]
    expected_cells = {
        (agents, epochs, mechanism)
        for agents in config["agent_counts"]
        for epochs in config["epoch_counts"]
        for mechanism in MECHANISMS
    }
    require(set(cell_counts) == expected_cells, "CSV does not cover the frozen cell grid")
    for key in expected_cells:
        require(cell_counts[key] == repetitions, f"cell {key} has the wrong row count")
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


@app.function(cpu=2.0, timeout=20 * 60, retries=0, single_use_containers=True)
def compile_smoke() -> dict[str, str]:
    """Compile and execute ``--help`` without allocating a GPU."""

    return _compile()


@app.function(gpu=GPU_TYPE, cpu=4.0, timeout=60 * 60, retries=0, single_use_containers=True)
def run_one_l4(experiment_id: str, mode: str) -> dict[str, Any]:
    """Compile and execute exactly one frozen smoke or full-grid L4 placement."""

    _mode_config(mode)
    build = _compile()
    REMOTE_OUTPUT.mkdir(parents=True, exist_ok=False)
    placement_id = os.environ.get("MODAL_TASK_ID") or f"modal-l4-{uuid.uuid4()}"
    modal_image_id = os.environ.get("MODAL_IMAGE_ID")
    image_provenance = modal_image_id or f"registry-ref:{CUDA_IMAGE}"
    environment = os.environ.copy()
    environment.update(
        {
            "EXECUTION_PROVIDER": "modal",
            "REQUESTED_GPU": GPU_TYPE,
            "PLACEMENT_ID": placement_id,
            "IMAGE_DIGEST": image_provenance,
            "SOURCE_SHA256": build["source_sha256"],
            "BINARY_SHA256": build["binary_sha256"],
        }
    )
    command = _program_command(mode, experiment_id)
    completed = subprocess.run(
        command, check=False, text=True, capture_output=True, env=environment
    )

    csv_files = sorted(REMOTE_OUTPUT.glob("*.csv"))
    manifest_files = sorted(REMOTE_OUTPUT.glob("*.manifest.json"))
    if len(csv_files) != 1 or len(manifest_files) != 1:
        raise RuntimeError(
            f"expected one CSV and one manifest, got {len(csv_files)} and {len(manifest_files)}"
        )
    csv_text = csv_files[0].read_text(encoding="utf-8")
    manifest_text = manifest_files[0].read_text(encoding="utf-8")
    validation_errors = _artifact_validation_errors(
        csv_name=csv_files[0].name,
        csv_text=csv_text,
        manifest_name=manifest_files[0].name,
        manifest_text=manifest_text,
        experiment_id=experiment_id,
        mode=mode,
        binary_sha256=build["binary_sha256"],
        placement_id=placement_id,
        program_returncode=completed.returncode,
    )
    return {
        "csv_name": csv_files[0].name,
        "csv_text": csv_text,
        "csv_sha256": _sha256_bytes(csv_text.encode()),
        "manifest_name": manifest_files[0].name,
        "manifest_text": manifest_text,
        "manifest_sha256": _sha256_bytes(manifest_text.encode()),
        "program_command": " ".join(command),
        "program_returncode": completed.returncode,
        "program_stdout": completed.stdout,
        "program_stderr": completed.stderr,
        "compile_command": build["command"],
        "source_sha256": build["source_sha256"],
        "makefile_sha256": build["makefile_sha256"],
        "binary_sha256": build["binary_sha256"],
        "nvcc_version": build["nvcc_version"],
        "placement_id": placement_id,
        "image_provenance": image_provenance,
        "validation_errors": validation_errors,
    }


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}") from error


def _persist_returned_artifacts(result: dict[str, Any], destination: Path) -> tuple[Path, Path]:
    csv_path = destination / result["csv_name"]
    manifest_path = destination / result["manifest_name"]
    collisions = [path for path in (csv_path, manifest_path) if path.exists()]
    if collisions:
        raise RuntimeError(f"refusing to overwrite existing artifacts: {collisions}")
    _write_new(csv_path, result["csv_text"])
    _write_new(manifest_path, result["manifest_text"])
    if _sha256_file(csv_path) != result["csv_sha256"]:
        raise RuntimeError(f"downloaded CSV hash mismatch: {csv_path}")
    if _sha256_file(manifest_path) != result["manifest_sha256"]:
        raise RuntimeError(f"downloaded manifest hash mismatch: {manifest_path}")
    return csv_path, manifest_path


@app.local_entrypoint()
def main(
    action: str = "plan",
    mode: str = "smoke",
    confirm_spend: bool = False,
    output_dir: str = "data/raw",
    experiment_id: str = "",
) -> None:
    """Plan locally, compile CPU-only, or explicitly launch one Modal L4."""

    frozen = _validate_frozen_inputs(LOCAL_NATIVE_ROOT)
    _mode_config(mode)
    resolved_experiment_id = experiment_id or f"resident-policy-001-modal-l4-{mode}"
    _validate_experiment_id(resolved_experiment_id)

    if action == "plan":
        print("action=plan")
        print("remote_calls=0")
        print("gpu_calls=0")
        print(f"mode={mode}")
        print(f"cuda_image={CUDA_IMAGE}")
        print(f"source_sha256={frozen['source_sha256']}")
        print(f"makefile_sha256={frozen['makefile_sha256']}")
        print("compile_cpu_only: modal run scripts/modal_resident_policy_pilot.py --action compile")
        print(
            "run_one_l4: modal run scripts/modal_resident_policy_pilot.py "
            f"--action run --mode {mode} --confirm-spend"
        )
        return
    if action == "compile":
        result = compile_smoke.remote()
        print("compile_status=ok")
        print("gpu_calls=0")
        print(f"source_sha256={result['source_sha256']}")
        print(f"makefile_sha256={result['makefile_sha256']}")
        print(f"binary_sha256={result['binary_sha256']}")
        print(result["nvcc_version"], end="")
        print(result["help_stdout"], end="")
        return
    if action != "run":
        raise ValueError("action must be one of: plan, compile, run")
    if not confirm_spend:
        raise RuntimeError("run action requires --confirm-spend; no GPU function was invoked")

    result = run_one_l4.remote(resolved_experiment_id, mode)
    csv_path, manifest_path = _persist_returned_artifacts(result, Path(output_dir))
    local_errors = _artifact_validation_errors(
        csv_name=result["csv_name"],
        csv_text=result["csv_text"],
        manifest_name=result["manifest_name"],
        manifest_text=result["manifest_text"],
        experiment_id=resolved_experiment_id,
        mode=mode,
        binary_sha256=result["binary_sha256"],
        placement_id=result["placement_id"],
        program_returncode=result["program_returncode"],
    )
    validation_errors = sorted(set(result["validation_errors"] + local_errors))
    print(result["program_stdout"], end="")
    if result["program_stderr"]:
        print(result["program_stderr"], end="")
    print(f"downloaded_csv={csv_path.resolve()}")
    print(f"downloaded_manifest={manifest_path.resolve()}")
    print(f"csv_sha256={result['csv_sha256']}")
    print(f"manifest_sha256={result['manifest_sha256']}")
    print(f"source_sha256={result['source_sha256']}")
    print(f"makefile_sha256={result['makefile_sha256']}")
    print(f"binary_sha256={result['binary_sha256']}")
    print(f"placement_id={result['placement_id']}")
    if validation_errors:
        raise RuntimeError(
            "resident-policy result gates failed; append-only artifacts were retained: "
            + "; ".join(validation_errors)
        )
    print("result_gates=passed")
