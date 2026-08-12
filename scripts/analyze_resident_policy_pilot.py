from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "resident-policy-analysis-v1"
RAW_SCHEMA_VERSION = "resident-policy-v1"
FROZEN_SOURCE_SHA256 = "4b5cdcb9496a734bd7801d5c419efb8eceb72fd6962800520101e89676d204da"
FROZEN_MAKEFILE_SHA256 = "d74935b594fb629b2113d237439289e057281becc561b6941d1134bd6a1c1351"
CUDA_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
MECHANISMS = ("host_roundtrip", "device_resident", "no_decision_lower_bound")
FROZEN_CONFIG = {
    "agent_counts": [256, 2048, 16384],
    "epoch_counts": [2, 8, 32],
    "warmups_per_mechanism_cell": 5,
    "calibration_samples_per_mechanism_cell": 3,
    "repetitions_per_mechanism_cell": 30,
    "min_duration_target_ns": 100_000_000,
    "max_batch_iterations": 20_000,
    "seed": 20260811,
    "block_size": 256,
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
ALLOWED_PROVIDERS = {"local", "modal", "lambda", "runpod"}
REQUIRED_COLUMNS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize frozen resident-policy-001 placements."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--external-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("preregistration/resident-policy-001.md"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    """Render retained provenance without publishing a machine-local prefix."""

    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"external/{path.name}"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def tar_member_sha256(archive_path: Path, member_name: str) -> str:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        matches = [member for member in archive.getmembers() if member.name == member_name]
        if len(matches) != 1 or not matches[0].isfile():
            raise AssertionError(f"archive does not contain one regular {member_name!r}")
        handle = archive.extractfile(matches[0])
        if handle is None:
            raise AssertionError(f"archive member is unreadable: {member_name!r}")
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def parse_bool(series: pd.Series, *, name: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    if not values.isin({"true", "false"}).all():
        raise AssertionError(f"{name} has non-boolean values: {sorted(values.unique())}")
    return values.eq("true")


def gpu_family(value: str) -> str | None:
    normalized = value.upper()
    for family in ("GTX 1660 TI", "H100", "A10", "L40S", "L40", "L4"):
        if re.search(rf"(?<![A-Z0-9]){re.escape(family)}(?![A-Z0-9])", normalized):
            return family
    return None


def discover_inputs(raw_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(raw_dir.rglob("resident-policy-001-*.csv")):
        if path.with_suffix(".manifest.json").is_file():
            paths.append(path)
    if not paths:
        raise FileNotFoundError("no completed resident-policy-001 CSV/manifest pairs found")
    return paths


def validate_provider_binding(
    *,
    csv_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    external_dir: Path,
) -> dict[str, Any]:
    provenance = manifest["provenance"]
    provider = str(provenance.get("execution_provider") or "")
    csv_sha256 = sha256_file(csv_path)
    manifest_sha256 = sha256_file(manifest_path)
    placement_id = str(provenance.get("placement_id") or manifest.get("run_id") or "")

    if provider == "local":
        return {
            "binding_type": "native-local-manifest",
            "validation_passed": True,
            "provider_receipt_required": False,
            "paths": [],
            "sha256": [],
        }

    if provider == "modal":
        candidates = []
        for path in sorted(external_dir.glob("*modal*receipt*.json")):
            receipt = load_json(path)
            artifacts = receipt.get("artifacts")
            if not isinstance(artifacts, dict):
                continue
            artifact_by_name = {Path(str(key)).name: str(value) for key, value in artifacts.items()}
            if artifact_by_name == {
                csv_path.name: csv_sha256,
                manifest_path.name: manifest_sha256,
            }:
                candidates.append((path, receipt))
        if len(candidates) != 1:
            raise AssertionError(
                f"expected exactly one Modal receipt bound by both artifact hashes: {csv_path}"
            )
        path, receipt = candidates[0]
        if (
            receipt.get("schema_version") != "resident-policy-modal-run-v1"
            or receipt.get("result_gates_passed") is not True
            or int(receipt.get("measured_rows", -1)) != 810
            or int(receipt.get("exact_rows", -1)) != 810
            or int(receipt.get("failure_rows", -1)) != 0
            or receipt.get("source_sha256") != FROZEN_SOURCE_SHA256
            or receipt.get("makefile_sha256") != FROZEN_MAKEFILE_SHA256
            or receipt.get("binary_sha256") != provenance.get("binary_sha256")
            or receipt.get("requested_gpu") != provenance.get("requested_gpu")
            or int(receipt.get("gpu_count", -1)) != 1
            or receipt.get("placement_id") != placement_id
        ):
            raise AssertionError(f"Modal provider receipt failed its frozen binding: {path}")
        return {
            "binding_type": "modal-artifact-hash-receipt",
            "validation_passed": True,
            "provider_receipt_required": True,
            "paths": [repository_path(path)],
            "sha256": [sha256_file(path)],
            "modal_app_run_id": receipt.get("modal_app_run_id"),
        }

    if provider == "lambda":
        base = manifest_path.name.removesuffix(".manifest.json")
        path = manifest_path.with_name(base + ".provider-lambda-resident-policy.json")
        if not path.is_file():
            raise AssertionError(f"Lambda provider sidecar is missing: {path}")
        receipt = load_json(path)
        artifacts = receipt.get("artifacts")
        expected_artifacts = {
            csv_path.name: csv_sha256,
            manifest_path.name: manifest_sha256,
        }
        observed_artifacts = {
            name: str(details.get("sha256") or "")
            for name, details in (artifacts or {}).items()
            if isinstance(details, dict)
        }
        if (
            receipt.get("schema_version") != "provider-lambda-resident-policy-v1"
            or receipt.get("execution_provider") != "lambda"
            or receipt.get("mode") != "full"
            or receipt.get("placement_id") != placement_id
            or receipt.get("artifact_validation", {}).get("passed") is not True
            or not all(
                observed_artifacts.get(name) == digest
                for name, digest in expected_artifacts.items()
            )
        ):
            raise AssertionError(f"Lambda provider sidecar failed its frozen binding: {path}")
        return {
            "binding_type": "lambda-provider-sidecar",
            "validation_passed": True,
            "provider_receipt_required": True,
            "paths": [repository_path(path)],
            "sha256": [sha256_file(path)],
            "instance_id": receipt.get("instance", {}).get("id"),
            "region": receipt.get("instance", {}).get("region", {}).get("name"),
            "provider_adapter": receipt.get("provider_adapter"),
        }

    if provider == "runpod":
        bundle_root = csv_path.parent.parent
        collection_path = bundle_root / "collection-receipt.json"
        archive_path = bundle_root / "resident-policy-artifacts.tar.gz"
        artifact_index_path = bundle_root / "artifact-index.json"
        if (
            not collection_path.is_file()
            or not archive_path.is_file()
            or not artifact_index_path.is_file()
        ):
            raise AssertionError(f"RunPod collection evidence is incomplete for {csv_path}")
        receipt = load_json(collection_path)
        validation = receipt.get("validation", {})
        summary = validation.get("native_summary", {})
        pod_id = str(receipt.get("pod_id") or "")
        recorded_launch_path = Path(str(receipt.get("launch_receipt") or "")).resolve()
        launch_hash = str(receipt.get("launch_receipt_sha256") or "")
        launch_candidates = []
        if recorded_launch_path.is_file() and sha256_file(recorded_launch_path) == launch_hash:
            launch_candidates.append(recorded_launch_path)
        launch_root = external_dir.parent / "provider-runpod-launches"
        for candidate in sorted(launch_root.glob("*.json")):
            resolved_candidate = candidate.resolve()
            if (
                resolved_candidate not in launch_candidates
                and sha256_file(resolved_candidate) == launch_hash
            ):
                launch_candidates.append(resolved_candidate)
        if len(launch_candidates) != 1:
            raise AssertionError(
                f"expected one hash-bound RunPod launch receipt for {collection_path}"
            )
        launch_path = launch_candidates[0]
        launch = load_json(launch_path)
        artifact_index = load_json(artifact_index_path)
        indexed_files = artifact_index.get("files", {})
        indexed_csv = indexed_files.get(f"results/{csv_path.name}", {})
        indexed_manifest = indexed_files.get(f"results/{manifest_path.name}", {})
        if (
            receipt.get("schema_version") != "runpod-resident-policy-collection-v1"
            or receipt.get("experiment_id") != manifest.get("experiment_id")
            or pod_id != placement_id
            or receipt.get("pod", {}).get("id") != placement_id
            or validation.get("passed") is not True
            or validation.get("errors") != []
            or int(summary.get("expected_rows", -1)) != 810
            or csv_path.name not in summary.get("csv_files", [])
            or manifest_path.name not in summary.get("manifest_files", [])
            or receipt.get("archive_sha256") != sha256_file(archive_path)
            or launch_hash != sha256_file(launch_path)
            or launch.get("schema_version") != "runpod-resident-policy-launch-v1"
            or launch.get("pod_id") != placement_id
            or launch.get("experiment_id") != manifest.get("experiment_id")
            or launch.get("pod_name") != f"gpu-agent-{manifest.get('experiment_id')}"
            or launch.get("request", {}).get("mode") != "full"
            or launch.get("request", {}).get("gpu_type") != "NVIDIA L4"
            or launch.get("request", {}).get("gpu_count") != 1
            or launch.get("request", {}).get("image") != CUDA_IMAGE
            or launch.get("source", {}).get("source_sha256") != FROZEN_SOURCE_SHA256
            or launch.get("source", {}).get("makefile_sha256") != FROZEN_MAKEFILE_SHA256
            or artifact_index.get("schema_version") != "runpod-resident-policy-artifact-index-v1"
            or indexed_csv.get("sha256") != csv_sha256
            or indexed_csv.get("bytes") != csv_path.stat().st_size
            or indexed_manifest.get("sha256") != manifest_sha256
            or indexed_manifest.get("bytes") != manifest_path.stat().st_size
            or tar_member_sha256(archive_path, "artifact-index.json")
            != sha256_file(artifact_index_path)
            or tar_member_sha256(archive_path, f"results/{csv_path.name}") != csv_sha256
            or tar_member_sha256(archive_path, f"results/{manifest_path.name}") != manifest_sha256
        ):
            raise AssertionError(
                f"RunPod collection receipt failed its frozen binding: {collection_path}"
            )
        return {
            "binding_type": "runpod-validated-collection",
            "validation_passed": True,
            "provider_receipt_required": True,
            "paths": [
                repository_path(collection_path),
                repository_path(launch_path),
                repository_path(archive_path),
                repository_path(artifact_index_path),
            ],
            "sha256": [
                sha256_file(collection_path),
                sha256_file(launch_path),
                sha256_file(archive_path),
                sha256_file(artifact_index_path),
            ],
            "pod_id": pod_id,
            "data_center_id": receipt.get("pod", {}).get("machine", {}).get("data_center_id"),
        }

    raise AssertionError(f"unsupported execution provider {provider!r}: {manifest_path}")


def validate_placement(
    csv_path: Path, *, external_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = csv_path.with_suffix(".manifest.json")
    manifest = load_json(manifest_path)
    frame = pd.read_csv(
        csv_path,
        dtype={
            "run_id": "string",
            "experiment_id": "string",
            "expected_state_checksum": "string",
            "observed_state_checksum": "string",
            "expected_decision_hash": "string",
            "observed_decision_hash": "string",
            "expected_decisions": "string",
            "observed_decisions": "string",
            "exact_state_match": "string",
            "exact_decision_match": "string",
            "min_duration_reached": "string",
        },
    )
    missing = REQUIRED_COLUMNS - set(frame.columns)
    extra = set(frame.columns) - REQUIRED_COLUMNS
    if missing or extra:
        raise AssertionError(
            f"schema mismatch for {csv_path}: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if manifest["schema_version"] != RAW_SCHEMA_VERSION:
        raise AssertionError(f"unexpected manifest schema: {manifest_path}")
    if set(frame["schema_version"]) != {RAW_SCHEMA_VERSION}:
        raise AssertionError(f"unexpected row schema: {csv_path}")
    if set(frame["phase"]) != {"measure"}:
        raise AssertionError(f"non-measure rows found: {csv_path}")
    if frame["run_id"].nunique() != 1 or frame["run_id"].iloc[0] != manifest["run_id"]:
        raise AssertionError(f"CSV/manifest run mismatch: {csv_path}")
    if (
        frame["experiment_id"].nunique() != 1
        or frame["experiment_id"].iloc[0] != manifest["experiment_id"]
    ):
        raise AssertionError(f"CSV/manifest experiment mismatch: {csv_path}")
    if manifest["csv_file"] != csv_path.name:
        raise AssertionError(f"manifest csv_file mismatch: {manifest_path}")

    config = manifest["config"]
    if config != FROZEN_CONFIG:
        raise AssertionError(f"manifest config differs from the frozen full grid: {manifest_path}")
    agents = tuple(int(value) for value in config["agent_counts"])
    epochs = tuple(int(value) for value in config["epoch_counts"])
    repetitions = int(config["repetitions_per_mechanism_cell"])
    expected_rows = len(agents) * len(epochs) * len(MECHANISMS) * repetitions
    if len(frame) != expected_rows:
        raise AssertionError(f"{csv_path} has {len(frame)} rows; expected {expected_rows}")
    if set(frame["agents"]) != set(agents) or set(frame["epochs"]) != set(epochs):
        raise AssertionError(f"CSV grid differs from manifest: {csv_path}")
    if set(frame["mechanism"]) != set(MECHANISMS):
        raise AssertionError(f"unexpected mechanisms: {csv_path}")
    identity = ["agents", "epochs", "mechanism", "repetition"]
    if frame.duplicated(identity).any():
        raise AssertionError(f"duplicated measured row identity: {csv_path}")
    counts = frame.groupby(["agents", "epochs", "mechanism"]).size()
    if (
        len(counts) != len(agents) * len(epochs) * len(MECHANISMS)
        or not counts.eq(repetitions).all()
    ):
        raise AssertionError(f"incomplete mechanism cells: {csv_path}")
    expected_repetitions = tuple(range(repetitions))
    observed_repetitions = frame.groupby(["agents", "epochs", "mechanism"])["repetition"].agg(
        lambda values: tuple(sorted(values.tolist()))
    )
    if not observed_repetitions.map(lambda values: values == expected_repetitions).all():
        raise AssertionError(f"invalid repetition indices: {csv_path}")
    order_sets = frame.groupby(["agents", "epochs", "repetition"])["order_index"].agg(
        lambda values: tuple(sorted(values.tolist()))
    )
    if not order_sets.map(lambda values: values == (0, 1, 2)).all():
        raise AssertionError(f"invalid randomized-order blocks: {csv_path}")
    if not frame["seed"].eq(FROZEN_CONFIG["seed"]).all():
        raise AssertionError(f"row seed differs from the frozen seed: {csv_path}")
    if not frame["block_size"].eq(FROZEN_CONFIG["block_size"]).all():
        raise AssertionError(f"row block size differs from the frozen block size: {csv_path}")
    expected_predicate_blocks = (frame["agents"] + int(FROZEN_CONFIG["block_size"]) - 1) // int(
        FROZEN_CONFIG["block_size"]
    )
    if not frame["predicate_blocks"].eq(expected_predicate_blocks).all():
        raise AssertionError(f"predicate block count is inconsistent: {csv_path}")
    decisions_valid = frame.apply(
        lambda row: (
            len(str(row["expected_decisions"])) == int(row["epochs"])
            and set(str(row["expected_decisions"])).issubset({"0", "1"})
            and len(str(row["observed_decisions"])) == int(row["epochs"])
            and set(str(row["observed_decisions"])).issubset({"0", "1"})
        ),
        axis=1,
    )
    if not decisions_valid.all():
        raise AssertionError(f"decision strings violate length/domain contract: {csv_path}")

    frame["min_duration_reached_bool"] = parse_bool(
        frame["min_duration_reached"], name="min_duration_reached"
    )
    frame["exact_state_match_bool"] = parse_bool(
        frame["exact_state_match"], name="exact_state_match"
    )
    frame["exact_decision_match_bool"] = parse_bool(
        frame["exact_decision_match"], name="exact_decision_match"
    )
    status_ok = frame["status"].eq("ok")
    checksum_equal = frame["expected_state_checksum"].eq(frame["observed_state_checksum"])
    decision_hash_equal = frame["expected_decision_hash"].eq(frame["observed_decision_hash"])
    decisions_equal = frame["expected_decisions"].eq(frame["observed_decisions"])
    error_fields_clear = (
        frame["failure_stage"].isna() & frame["error_message"].isna() & frame["error_code"].eq(0)
    )
    strict_gates = {
        "all_status_ok": bool(status_ok.all()),
        "all_error_fields_clear": bool(error_fields_clear.all()),
        "all_min_duration_reached": bool(frame["min_duration_reached_bool"].all()),
        "all_state_fields_exact": bool(frame["exact_state_match_bool"].all()),
        "all_decision_traces_exact": bool(frame["exact_decision_match_bool"].all()),
        "all_state_checksums_equal": bool(checksum_equal.all()),
        "all_decision_hashes_equal": bool(decision_hash_equal.all()),
        "all_decision_strings_equal": bool(decisions_equal.all()),
        "all_invocations_validated": bool(
            frame["exact_validation_count"].eq(frame["batch_iterations"]).all()
        ),
        "all_positive_wall_times": bool(frame["wall_ns_per_invocation"].gt(0).all()),
        "all_positive_device_times": bool(frame["device_ns_per_invocation"].gt(0).all()),
        "all_finite_timing": bool(
            np.isfinite(
                frame[
                    [
                        "aggregate_wall_ns",
                        "wall_ns_per_invocation",
                        "aggregate_device_ns",
                        "device_ns_per_invocation",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "all_aggregate_wall_targets_met": bool(
            frame["aggregate_wall_ns"].ge(frame["min_duration_target_ns"]).all()
        ),
        "all_min_duration_targets_frozen": bool(
            frame["min_duration_target_ns"].eq(FROZEN_CONFIG["min_duration_target_ns"]).all()
        ),
        "all_batch_iterations_within_frozen_cap": bool(
            frame["batch_iterations"]
            .between(1, FROZEN_CONFIG["max_batch_iterations"], inclusive="both")
            .all()
        ),
    }
    failed = [key for key, passed in strict_gates.items() if not passed]
    if failed:
        raise AssertionError(f"validity gates failed for {csv_path}: {failed}")
    relative_error = np.abs(
        frame["wall_ns_per_invocation"] - frame["aggregate_wall_ns"] / frame["batch_iterations"]
    )
    if not np.allclose(relative_error, 0.0, atol=1e-6, rtol=1e-12):
        raise AssertionError(f"wall-time division mismatch: {csv_path}")
    device_relative_error = np.abs(
        frame["device_ns_per_invocation"] - frame["aggregate_device_ns"] / frame["batch_iterations"]
    )
    if not np.allclose(device_relative_error, 0.0, atol=1e-6, rtol=1e-12):
        raise AssertionError(f"device-time division mismatch: {csv_path}")

    results = manifest["results"]
    if (
        int(results["measured_rows"]) != len(frame)
        or int(results["exact_rows"])
        != int((frame["exact_state_match_bool"] & frame["exact_decision_match_bool"]).sum())
        or int(results["failure_rows"]) != int((~status_ok).sum())
        or results.get("status_counts") != {"ok": 810}
    ):
        raise AssertionError(f"manifest result counts disagree with CSV: {manifest_path}")
    provenance = manifest["provenance"]
    provider = str(provenance.get("execution_provider") or "")
    if provider not in ALLOWED_PROVIDERS:
        raise AssertionError(f"unexpected execution provider: {manifest_path}")
    if provenance["source_sha256"] != FROZEN_SOURCE_SHA256:
        raise AssertionError(f"source hash differs from preregistration: {manifest_path}")
    if re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("binary_sha256") or "")) is None:
        raise AssertionError(f"binary hash is invalid: {manifest_path}")
    placement_id = provenance.get("placement_id") or manifest["run_id"]
    if not isinstance(placement_id, str) or not placement_id:
        raise AssertionError(f"placement ID is empty: {manifest_path}")
    hardware = manifest["hardware"]
    software = manifest["software"]
    if (
        hardware.get("cuda_available") is not True
        or int(hardware.get("device_count", 0)) != 1
        or int(hardware.get("unified_addressing", 0)) != 1
        or not hardware.get("device_name")
        or not hardware.get("device_uuid")
        or not hardware.get("compute_capability")
        or int(software.get("cuda_compile_version", 0)) // 1000 != 13
        or int(software.get("cuda_runtime_version", 0)) // 1000 != 13
    ):
        raise AssertionError(f"hardware/software contract failed: {manifest_path}")
    requested_gpu = str(provenance.get("requested_gpu") or "")
    requested_family = gpu_family(requested_gpu)
    actual_family = gpu_family(str(hardware["device_name"]))
    if requested_family is None or requested_family != actual_family:
        raise AssertionError(f"requested GPU does not match actual device: {manifest_path}")
    if provider != "local" and not provenance.get("image_digest"):
        raise AssertionError(f"cloud placement lacks an image digest: {manifest_path}")
    device_uuid = hardware["device_uuid"]
    provider_binding = validate_provider_binding(
        csv_path=csv_path,
        manifest_path=manifest_path,
        manifest=manifest,
        external_dir=external_dir,
    )
    frame["placement_id"] = placement_id
    frame["provider"] = provider
    frame["requested_gpu"] = requested_gpu
    frame["device_name"] = hardware["device_name"]
    frame["device_uuid"] = device_uuid
    frame["source_file"] = str(csv_path)
    metadata = {
        "placement_id": placement_id,
        "provider": provider,
        "requested_gpu": requested_gpu,
        "device_name": hardware["device_name"],
        "device_uuid": device_uuid,
        "compute_capability": hardware["compute_capability"],
        "run_id": manifest["run_id"],
        "source_file": str(csv_path),
        "source_file_sha256": sha256_file(csv_path),
        "manifest_file": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "source_sha256": provenance["source_sha256"],
        "binary_sha256": provenance["binary_sha256"],
        "image_digest": provenance.get("image_digest"),
        "cuda_compile_version": software["cuda_compile_version"],
        "cuda_runtime_version": software["cuda_runtime_version"],
        "cuda_driver_version": software.get("cuda_driver_version"),
        "cpu_model": software.get("cpu_model"),
        "cpu_hardware_threads": software.get("cpu_hardware_threads"),
        "provider_binding": provider_binding,
        "rows": len(frame),
        "strict_gates": strict_gates,
    }
    return frame, metadata


def quantile(values: pd.Series, probability: float) -> float:
    return float(values.quantile(probability))


def build_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "placement_id",
        "provider",
        "requested_gpu",
        "device_name",
        "device_uuid",
        "source_file",
        "agents",
        "epochs",
        "mechanism",
    ]
    cells = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            technical_rows=("repetition", "size"),
            batch_iterations_min=("batch_iterations", "min"),
            batch_iterations_median=("batch_iterations", "median"),
            batch_iterations_max=("batch_iterations", "max"),
            wall_ns_median=("wall_ns_per_invocation", "median"),
            wall_batch_mean_ns_p95=(
                "wall_ns_per_invocation",
                lambda values: quantile(values, 0.95),
            ),
            wall_batch_mean_ns_p99=(
                "wall_ns_per_invocation",
                lambda values: quantile(values, 0.99),
            ),
            device_ns_median=("device_ns_per_invocation", "median"),
            device_batch_mean_ns_p95=(
                "device_ns_per_invocation",
                lambda values: quantile(values, 0.95),
            ),
            device_batch_mean_ns_p99=(
                "device_ns_per_invocation",
                lambda values: quantile(values, 0.99),
            ),
            validated_invocations=("exact_validation_count", "sum"),
        )
        .sort_values(["provider", "placement_id", "agents", "epochs", "mechanism"])
        .reset_index(drop=True)
    )

    index = [
        "placement_id",
        "provider",
        "requested_gpu",
        "device_name",
        "device_uuid",
        "source_file",
        "agents",
        "epochs",
    ]
    median_wide = cells.pivot(index=index, columns="mechanism", values="wall_ns_median")
    records: list[dict[str, Any]] = []
    for key, row in median_wide.iterrows():
        base = dict(zip(index, key, strict=True))
        host = float(row["host_roundtrip"])
        resident = float(row["device_resident"])
        floor = float(row["no_decision_lower_bound"])
        placement_id, _, _, _, _, source_file, agents, epochs = key
        paired = frame[
            frame["placement_id"].eq(placement_id)
            & frame["source_file"].eq(source_file)
            & frame["agents"].eq(agents)
            & frame["epochs"].eq(epochs)
        ].pivot(index="repetition", columns="mechanism", values="wall_ns_per_invocation")
        if len(paired) != 30 or set(paired.columns) != set(MECHANISMS) or paired.isna().any().any():
            raise AssertionError(
                f"incomplete technical pairing for {placement_id}, N={agents}, H={epochs}"
            )
        host_over_resident = paired["host_roundtrip"] / paired["device_resident"]
        resident_over_floor = paired["device_resident"] / paired["no_decision_lower_bound"]
        records.append(
            {
                **base,
                "analysis_unit": "placement",
                "technical_rows_paired": len(paired),
                "host_over_resident_ratio_of_medians": host / resident,
                "host_over_resident_paired_ratio_median": float(host_over_resident.median()),
                "host_over_resident_batch_mean_ratio_p95": quantile(host_over_resident, 0.95),
                "host_over_resident_batch_mean_ratio_p99": quantile(host_over_resident, 0.99),
                "resident_over_floor_ratio_of_medians": resident / floor,
                "resident_over_floor_paired_ratio_median": float(resident_over_floor.median()),
                "resident_over_floor_batch_mean_ratio_p95": quantile(resident_over_floor, 0.95),
                "resident_over_floor_batch_mean_ratio_p99": quantile(resident_over_floor, 0.99),
                "wall_ns_saved_per_invocation": host - resident,
                "wall_ns_saved_per_epoch": (host - resident) / int(epochs),
            }
        )
    contrasts = pd.DataFrame(records).sort_values(["provider", "placement_id", "agents", "epochs"])
    return cells, contrasts.reset_index(drop=True)


def write_outputs(
    output_dir: Path,
    cells: pd.DataFrame,
    contrasts: pd.DataFrame,
    metadata: list[dict[str, Any]],
    preregistration: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_path = output_dir / "resident-policy-pilot-cell-summary.csv"
    contrast_path = output_dir / "resident-policy-pilot-contrasts.csv"
    manifest_path = output_dir / "resident-policy-pilot-manifest.json"
    existing = [path for path in (cell_path, contrast_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite outputs without --overwrite: {existing}")
    cells.to_csv(cell_path, index=False)
    contrasts.to_csv(contrast_path, index=False)

    primary = contrasts[contrasts["agents"].eq(256) & contrasts["epochs"].eq(32)]
    r2_by_placement_agent = []
    for (placement_id, agents), group in contrasts.groupby(["placement_id", "agents"]):
        ordered = group.sort_values("epochs")
        ratios = ordered["host_over_resident_ratio_of_medians"].to_numpy()
        r2_by_placement_agent.append(
            {
                "placement_id": placement_id,
                "agents": int(agents),
                "nondecreasing_speedup_with_epochs": bool(np.all(np.diff(ratios) >= 0)),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "preregistration": {
            "path": str(preregistration),
            "sha256": sha256_file(preregistration),
        },
        "analysis_script": {
            "path": repository_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "design": {
            "sampling_unit": "fresh GPU placement",
            "technical_unit": "randomized-order measured row containing many validated invocations",
            "primary_contrast": "host_roundtrip median wall / device_resident median wall",
            "p_values_reported": False,
            "frozen_source_sha256": FROZEN_SOURCE_SHA256,
            "frozen_config": FROZEN_CONFIG,
            "technical_quantiles": (
                "quantiles across batch-average rows; not individual-invocation tail latency"
            ),
        },
        "placements": metadata,
        "quality_gates": {
            "all_placement_gates_pass": all(
                all(item["strict_gates"].values()) for item in metadata
            ),
            "all_provider_bindings_pass": all(
                item["provider_binding"]["validation_passed"] for item in metadata
            ),
            "distinct_placement_ids": len({item["placement_id"] for item in metadata})
            == len(metadata),
            "distinct_run_ids": len({item["run_id"] for item in metadata}) == len(metadata),
            "distinct_gpu_uuid_per_placement": len({item["device_uuid"] for item in metadata})
            == len(metadata),
            "physical_gpu_uuid_cluster_count": len({item["device_uuid"] for item in metadata}),
        },
        "pilot_outcomes": {
            "placements": len(metadata),
            "raw_rows": int(sum(item["rows"] for item in metadata)),
            "cell_summary_rows": len(cells),
            "contrast_rows": len(contrasts),
            "host_over_resident_ratio_min": float(
                contrasts["host_over_resident_ratio_of_medians"].min()
            ),
            "host_over_resident_ratio_max": float(
                contrasts["host_over_resident_ratio_of_medians"].max()
            ),
            "resident_over_floor_ratio_min": float(
                contrasts["resident_over_floor_ratio_of_medians"].min()
            ),
            "resident_over_floor_ratio_max": float(
                contrasts["resident_over_floor_ratio_of_medians"].max()
            ),
            "R1_primary_device_faster_all_observed_placements": bool(
                len(primary) == len(metadata)
                and primary["host_over_resident_ratio_of_medians"].gt(1).all()
            ),
            "R2_exploratory_ratio_monotonicity_checks": r2_by_placement_agent,
            "R3_resident_slower_than_floor_all_cells": bool(
                contrasts["resident_over_floor_ratio_of_medians"].gt(1).all()
            ),
            "R4_all_exact": all(all(item["strict_gates"].values()) for item in metadata),
        },
        "outputs": {},
        "required_caveats": [
            "This is a mechanism pilot, not a deployment-level hypothesis test.",
            "Rows and batch invocations are technical repetitions; placement is the sampling unit.",
            "P95/P99 fields summarize batch-average rows and are not individual-invocation tails.",
            "R2 is exploratory because the preregistration did not operationally define advantage.",
            "The no-decision path is an oracle floor and not a legal online scheduler.",
            "The no-decision decision trace is assigned from the oracle rather than independently observed.",
            "The global binary synthetic policy is not the final deadline-aware route-compacting runtime.",
        ],
    }
    for name, path, rows in (
        ("cell_summary", cell_path, len(cells)),
        ("contrasts", contrast_path, len(contrasts)),
    ):
        manifest["outputs"][name] = {
            "path": str(path),
            "rows": rows,
            "sha256": sha256_file(path),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return cell_path, contrast_path, manifest_path


def main() -> None:
    args = parse_args()
    frames = []
    metadata = []
    for path in discover_inputs(args.raw_dir):
        frame, placement = validate_placement(path, external_dir=args.external_dir)
        frames.append(frame)
        metadata.append(placement)
    for field in ("placement_id", "run_id", "source_file_sha256"):
        values = [item[field] for item in metadata]
        if len(set(values)) != len(values):
            raise AssertionError(f"completed placements have duplicate {field}")
    combined = pd.concat(frames, ignore_index=True)
    cells, contrasts = build_summaries(combined)
    cell_path, contrast_path, manifest_path = write_outputs(
        args.output_dir,
        cells,
        contrasts,
        metadata,
        args.preregistration,
        overwrite=args.overwrite,
    )
    print(f"placements={len(metadata)}")
    print(f"raw_rows={len(combined)}")
    print(f"cell_summary_rows={len(cells)}")
    print(f"contrast_rows={len(contrasts)}")
    print(f"cell_summary={cell_path}")
    print(f"contrasts={contrast_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
