from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "native-dispatch-pilot-analysis-v1"
RAW_SCHEMA_VERSION = "device-dispatch-v1"
EXPECTED_EXPERIMENTS = {
    "native-dispatch-001-local": "local",
    "native-dispatch-001-modal-l4-p1": "modal",
    "native-dispatch-001-modal-l4-p2": "modal",
    "native-dispatch-001-runpod-l4-p1": "runpod",
    "native-dispatch-001-lambda-h100-p1": "lambda",
}
EXPECTED_AGENTS = (32, 256, 2048, 16384)
EXPECTED_STEPS = (1, 8, 64)
EXPECTED_MECHANISMS = (
    "cpu_cpp",
    "cuda_host_launch",
    "cuda_host_graph",
    "cuda_device_graph",
)
EXPECTED_REPETITIONS = 50
EXPECTED_ROWS_PER_PLACEMENT = (
    len(EXPECTED_AGENTS) * len(EXPECTED_STEPS) * len(EXPECTED_MECHANISMS) * EXPECTED_REPETITIONS
)
FROZEN_CUDA_SOURCE_SHA256 = "a5c1f4a349075b6e76116c4f52163b488ccdae32a7bf66e95fed752e363d3ac6"
FROZEN_SOURCE_BUNDLE_SHA256 = "2ec4c6246c6c6287f443c749ebe6385ee9690a9a4eed00a9119cb96aad024395"
REQUIRED_COLUMNS = {
    "schema_version",
    "timestamp_utc",
    "run_id",
    "experiment_id",
    "phase",
    "mechanism",
    "agents",
    "steps",
    "repetition",
    "order_index",
    "status",
    "failure_stage",
    "error_code",
    "error_message",
    "wall_ns",
    "device_ns",
    "device_time_scope",
    "expected_checksum",
    "observed_checksum",
    "exact_match",
    "seed",
    "block_size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the frozen native-dispatch-001 pilot."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("preregistration/native-dispatch-001.md"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the three derived outputs after all checks pass.",
    )
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


def source_bundle_sha256(source_dir: Path) -> str:
    """Reproduce the provider runners' path-delimited source-bundle digest."""

    digest = hashlib.sha256()
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or "build" in path.parts or "smoke-results" in path.parts:
            continue
        digest.update(path.relative_to(source_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def discover_inputs(raw_dir: Path) -> list[Path]:
    flat = sorted(raw_dir.glob("native-dispatch-001-*.csv"))
    nested = sorted(raw_dir.glob("runpod-native-dispatch-001-*/results/native-dispatch-001-*.csv"))
    paths = flat + nested
    if len(paths) != len(EXPECTED_EXPERIMENTS):
        raise AssertionError(
            f"expected {len(EXPECTED_EXPERIMENTS)} placement CSVs, found {len(paths)}: "
            f"{[str(path) for path in paths]}"
        )
    return paths


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def parse_exact_match(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        values = sorted(normalized.unique().tolist())
        raise AssertionError(f"exact_match contains non-boolean values: {values}")
    return normalized.eq("true")


def validate_frozen_source(raw_dir: Path) -> dict[str, Any]:
    archived_dirs = sorted(raw_dir.glob("runpod-native-dispatch-001-*/source"))
    if len(archived_dirs) != 1:
        raise AssertionError(
            f"expected one archived RunPod source directory, found {archived_dirs}"
        )
    source_dir = archived_dirs[0]
    cuda_path = source_dir / "device_dispatch_pilot.cu"
    cuda_hash = sha256_file(cuda_path)
    bundle_hash = source_bundle_sha256(source_dir)
    if cuda_hash != FROZEN_CUDA_SOURCE_SHA256:
        raise AssertionError(
            f"archived CUDA source hash changed: {cuda_hash} != {FROZEN_CUDA_SOURCE_SHA256}"
        )
    if bundle_hash != FROZEN_SOURCE_BUNDLE_SHA256:
        raise AssertionError(
            f"archived source bundle hash changed: {bundle_hash} != {FROZEN_SOURCE_BUNDLE_SHA256}"
        )

    source = cuda_path.read_text()
    semantic_markers = {
        "separate_host_reference": "AgentState reference_transition(" in source,
        "fieldwise_comparison_function": (
            "first_state_difference(const std::vector<AgentState>& expected" in source
        ),
        "all_four_fields_compared": (
            "left.pc == right.pc && left.budget == right.budget && left.risk == right.risk"
            in source
            and "left.route == right.route" in source
        ),
        "exact_match_derived_from_field_difference": (
            "result.exact_match = !difference.has_value();" in source
        ),
        "checksum_is_secondary_guard": (
            "equal GPU states produced unequal checksums" in source
            and "equal C++ states produced unequal checksums" in source
        ),
    }
    if not all(semantic_markers.values()):
        raise AssertionError(
            f"frozen source no longer establishes field-exact semantics: {semantic_markers}"
        )
    return {
        "archived_source_dir": str(source_dir),
        "cuda_source_file": str(cuda_path),
        "cuda_source_sha256": cuda_hash,
        "source_bundle_sha256": bundle_hash,
        "semantic_markers": semantic_markers,
        "interpretation": (
            "For this frozen source, exact_match=true means every pc, budget, risk, "
            "and route field equals the separately implemented host oracle; checksum "
            "equality is an additional guard, not the definition."
        ),
    }


def validate_provider_provenance(
    csv_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    provider = str(manifest["execution_provider"])
    checks: dict[str, bool] = {}
    receipt_files: list[str] = []
    binary_sha256: str | None = None

    if provider == "lambda":
        receipt_path = csv_path.with_suffix(".provider-lambda.json")
        receipt = load_json(receipt_path)
        receipt_files.append(str(receipt_path))
        artifacts = receipt["artifacts"]
        checks["lambda_csv_receipt_hash_matches"] = artifacts[csv_path.name][
            "sha256"
        ] == sha256_file(csv_path)
        checks["lambda_manifest_receipt_hash_matches"] = artifacts[manifest_path.name][
            "sha256"
        ] == sha256_file(manifest_path)
        checks["lambda_source_hash_matches"] = (
            receipt["source"]["sha256"] == manifest["source_sha256"]
        )
        checks["lambda_native_summary_matches"] = (
            receipt["native_manifest_summary"]["run_id"] == manifest["run_id"]
            and receipt["native_manifest_summary"]["results"] == manifest["results"]
        )
        binary_sha256 = receipt["host"]["binary_sha256"]
    elif provider == "runpod":
        root = csv_path.parents[1]
        provider_path = root / "provider-metadata.json"
        collection_path = root / "collection-receipt.json"
        provider_metadata = load_json(provider_path)
        collection = load_json(collection_path)
        receipt_files.extend([str(provider_path), str(collection_path)])
        checks["runpod_program_return_code_zero"] = (
            provider_metadata["execution"]["program_return_code"] == 0
        )
        checks["runpod_compile_return_code_zero"] = (
            provider_metadata["execution"]["compile_return_code"] == 0
        )
        checks["runpod_source_hash_matches"] = (
            provider_metadata["source"]["source_sha256"] == manifest["source_sha256"]
        )
        checks["runpod_collection_native_validation_passed"] = all(
            bool(value)
            for key, value in collection["native_validation"].items()
            if key.endswith("_matches")
        )
        checks["runpod_collection_counts_match"] = (
            collection["native_validation"]["measured_rows"] == manifest["results"]["measured_rows"]
            and collection["native_validation"]["failure_rows"]
            == manifest["results"]["failure_rows"]
        )
        checks["runpod_gpu_uuid_matches"] = (
            provider_metadata["gpus"][0]["uuid"].removeprefix("GPU-").replace("-", "").lower()
            == str(manifest["hardware"]["device_uuid"]).lower()
        )
        binary_sha256 = provider_metadata["source"]["binary_sha256"]
    else:
        checks["manifest_present"] = manifest_path.is_file()

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"provider provenance failed for {csv_path}: {failed}")
    return {
        "receipt_files": receipt_files,
        "checks": checks,
        "binary_sha256": binary_sha256,
        "receipt_level": (
            "provider_receipt_and_manifest"
            if provider in {"lambda", "runpod"}
            else "native_manifest_and_frozen_source_hash"
        ),
    }


def validate_placement(
    csv_path: Path,
    frozen_source: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = csv_path.with_suffix(".manifest.json")
    manifest = load_json(manifest_path)
    frame = pd.read_csv(
        csv_path,
        dtype={
            "run_id": "string",
            "experiment_id": "string",
            "expected_checksum": "string",
            "observed_checksum": "string",
            "exact_match": "string",
        },
    )
    missing = REQUIRED_COLUMNS - set(frame.columns)
    extra = set(frame.columns) - REQUIRED_COLUMNS
    if missing or extra:
        raise AssertionError(
            f"schema mismatch in {csv_path}; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if len(frame) != EXPECTED_ROWS_PER_PLACEMENT:
        raise AssertionError(
            f"{csv_path} has {len(frame)} rows, expected {EXPECTED_ROWS_PER_PLACEMENT}"
        )

    experiment_values = frame["experiment_id"].drop_duplicates().tolist()
    run_values = frame["run_id"].drop_duplicates().tolist()
    if experiment_values != [manifest["experiment_id"]]:
        raise AssertionError(f"CSV/manifest experiment mismatch: {csv_path}")
    if run_values != [manifest["run_id"]]:
        raise AssertionError(f"CSV/manifest run mismatch: {csv_path}")
    experiment_id = str(manifest["experiment_id"])
    provider = str(manifest["execution_provider"])
    if EXPECTED_EXPERIMENTS.get(experiment_id) != provider:
        raise AssertionError(f"unexpected experiment/provider pair: {experiment_id}/{provider}")
    if manifest["csv_file"] != csv_path.name:
        raise AssertionError(f"manifest csv_file mismatch: {manifest_path}")
    if manifest["schema_version"] != RAW_SCHEMA_VERSION:
        raise AssertionError(f"unexpected manifest schema: {manifest_path}")
    if set(frame["schema_version"]) != {RAW_SCHEMA_VERSION}:
        raise AssertionError(f"unexpected row schema: {csv_path}")

    if set(frame["phase"]) != {"measure"}:
        raise AssertionError(f"non-measure rows found: {csv_path}")
    if set(frame["agents"]) != set(EXPECTED_AGENTS):
        raise AssertionError(f"unexpected agent grid: {csv_path}")
    if set(frame["steps"]) != set(EXPECTED_STEPS):
        raise AssertionError(f"unexpected step grid: {csv_path}")
    if set(frame["mechanism"]) != set(EXPECTED_MECHANISMS):
        raise AssertionError(f"unexpected mechanism grid: {csv_path}")
    if set(frame["block_size"]) != {256}:
        raise AssertionError(f"unexpected CUDA block size: {csv_path}")
    if set(frame["seed"]) != {manifest["config"]["seed"]}:
        raise AssertionError(f"CSV/manifest seed mismatch: {csv_path}")

    cell_counts = frame.groupby(["agents", "steps", "mechanism"]).size()
    if len(cell_counts) != 48 or not cell_counts.eq(EXPECTED_REPETITIONS).all():
        raise AssertionError(f"incomplete mechanism cells: {csv_path}")
    repetition_sets = frame.groupby(["agents", "steps", "mechanism"])["repetition"].agg(
        lambda values: tuple(sorted(values.tolist()))
    )
    expected_repetitions = tuple(range(EXPECTED_REPETITIONS))
    if not repetition_sets.map(lambda values: values == expected_repetitions).all():
        raise AssertionError(f"invalid repetition indices: {csv_path}")
    order_sets = frame.groupby(["agents", "steps", "repetition"])["order_index"].agg(
        lambda values: tuple(sorted(values.tolist()))
    )
    if not order_sets.map(lambda values: values == (0, 1, 2, 3)).all():
        raise AssertionError(f"invalid within-repetition order indices: {csv_path}")

    frame["field_exact"] = parse_exact_match(frame["exact_match"])
    checksum_equal = frame["expected_checksum"].eq(frame["observed_checksum"])
    status_ok = frame["status"].eq("ok")
    errors_clear = (
        frame["failure_stage"].isna() & frame["error_message"].isna() & frame["error_code"].eq(0)
    )
    if not status_ok.all():
        raise AssertionError(f"non-ok status retained in {csv_path}")
    if not errors_clear.all():
        raise AssertionError(f"failure metadata on ok row in {csv_path}")
    if not frame["field_exact"].all():
        raise AssertionError(f"field-exact mismatch in {csv_path}")
    if not checksum_equal.all():
        raise AssertionError(f"checksum mismatch in {csv_path}")
    if not frame["wall_ns"].gt(0).all():
        raise AssertionError(f"non-positive wall time in {csv_path}")

    cpu = frame["mechanism"].eq("cpu_cpp")
    if not frame.loc[cpu, "device_ns"].isna().all():
        raise AssertionError(f"CPU rows unexpectedly have CUDA event time: {csv_path}")
    if not frame.loc[~cpu, "device_ns"].notna().all():
        raise AssertionError(f"GPU rows missing CUDA event time: {csv_path}")
    if not frame.loc[~cpu, "device_ns"].gt(0).all():
        raise AssertionError(f"GPU rows have non-positive CUDA event time: {csv_path}")
    expected_scope = {
        "cpu_cpp": {np.nan},
        "cuda_host_launch": {"transition_kernels"},
        "cuda_host_graph": {"transition_kernels"},
        "cuda_device_graph": {"complete_parent_and_child_execution_environment"},
    }
    for mechanism, allowed in expected_scope.items():
        scopes = set(frame.loc[frame["mechanism"].eq(mechanism), "device_time_scope"])
        if mechanism == "cpu_cpp":
            if not all(pd.isna(value) for value in scopes):
                raise AssertionError(f"unexpected CPU device scope: {csv_path}")
        elif scopes != allowed:
            raise AssertionError(f"unexpected device scope for {mechanism}: {scopes} in {csv_path}")

    results = manifest["results"]
    if (
        results["measured_rows"] != len(frame)
        or results["exact_rows"] != int(frame["field_exact"].sum())
        or results["failure_rows"] != int((~status_ok).sum())
        or results["status_counts"] != {"ok": len(frame)}
    ):
        raise AssertionError(f"manifest result counts disagree with CSV: {manifest_path}")

    source_hash = str(manifest["source_sha256"])
    expected_source_hash = (
        frozen_source["cuda_source_sha256"]
        if provider == "local"
        else frozen_source["source_bundle_sha256"]
    )
    if source_hash != expected_source_hash:
        raise AssertionError(
            f"source hash mismatch for {experiment_id}: {source_hash} != {expected_source_hash}"
        )
    provenance = validate_provider_provenance(csv_path, manifest_path, manifest)

    frame["placement_id"] = experiment_id
    frame["provider"] = provider
    frame["requested_gpu"] = manifest["requested_gpu"]
    frame["device_name"] = manifest["hardware"]["device_name"]
    frame["device_uuid"] = manifest["hardware"]["device_uuid"]
    frame["source_file"] = str(csv_path)
    frame["source_sha256"] = source_hash
    metadata = {
        "placement_id": experiment_id,
        "provider": provider,
        "requested_gpu": manifest["requested_gpu"],
        "device_name": manifest["hardware"]["device_name"],
        "device_uuid": manifest["hardware"]["device_uuid"],
        "compute_capability": manifest["hardware"]["compute_capability"],
        "run_id": manifest["run_id"],
        "started_at_utc": manifest["started_at_utc"],
        "completed_at_utc": manifest["completed_at_utc"],
        "seed": manifest["config"]["seed"],
        "source_sha256": source_hash,
        "source_file": str(csv_path),
        "source_file_sha256": sha256_file(csv_path),
        "manifest_file": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "binary_sha256": provenance["binary_sha256"],
        "provenance": provenance,
        "measured_rows": len(frame),
        "field_exact_rows": int(frame["field_exact"].sum()),
        "checksum_equal_rows": int(checksum_equal.sum()),
        "failure_rows": int((~status_ok).sum()),
    }
    return frame, metadata


def quantile(values: pd.Series, probability: float) -> float:
    return float(np.quantile(values.to_numpy(dtype=float), probability, method="linear"))


def summarize_cells(frame: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "placement_id",
        "provider",
        "requested_gpu",
        "device_name",
        "device_uuid",
        "source_file",
        "source_sha256",
        "agents",
        "steps",
        "mechanism",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(grouping, sort=True, dropna=False):
        record = dict(zip(grouping, keys, strict=True))
        device = group["device_ns"].dropna()
        scopes = group["device_time_scope"].dropna().unique().tolist()
        record.update(
            {
                "analysis_unit": "placement",
                "technical_repetitions": len(group),
                "status_ok_rows": int(group["status"].eq("ok").sum()),
                "field_exact_rows": int(group["field_exact"].sum()),
                "checksum_equal_rows": int(
                    group["expected_checksum"].eq(group["observed_checksum"]).sum()
                ),
                "wall_ns_median": quantile(group["wall_ns"], 0.50),
                "wall_ns_p95": quantile(group["wall_ns"], 0.95),
                "wall_ns_p99": quantile(group["wall_ns"], 0.99),
                "device_time_scope": scopes[0] if len(scopes) == 1 else "",
                "device_ns_rows": len(device),
                "device_ns_median": quantile(device, 0.50) if len(device) else np.nan,
                "device_ns_p95": quantile(device, 0.95) if len(device) else np.nan,
                "device_ns_p99": quantile(device, 0.99) if len(device) else np.nan,
            }
        )
        rows.append(record)
    result = pd.DataFrame(rows)
    if len(result) != len(EXPECTED_EXPERIMENTS) * 48:
        raise AssertionError(f"expected 240 cell summaries, got {len(result)}")
    return result.sort_values(["provider", "placement_id", "agents", "steps", "mechanism"])


def paired_ratio(cell: pd.DataFrame, numerator: str, denominator: str) -> tuple[pd.Series, float]:
    left = cell.loc[cell["mechanism"].eq(numerator), ["repetition", "wall_ns"]].rename(
        columns={"wall_ns": "numerator_wall_ns"}
    )
    right = cell.loc[cell["mechanism"].eq(denominator), ["repetition", "wall_ns"]].rename(
        columns={"wall_ns": "denominator_wall_ns"}
    )
    pairs = left.merge(right, on="repetition", how="inner", validate="one_to_one")
    if len(pairs) != EXPECTED_REPETITIONS:
        raise AssertionError(
            f"expected {EXPECTED_REPETITIONS} paired repetitions for "
            f"{numerator}/{denominator}, got {len(pairs)}"
        )
    ratios = pairs["numerator_wall_ns"] / pairs["denominator_wall_ns"]
    ratio_of_medians = float(
        pairs["numerator_wall_ns"].median() / pairs["denominator_wall_ns"].median()
    )
    return ratios, ratio_of_medians


def summarize_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "placement_id",
        "provider",
        "requested_gpu",
        "device_name",
        "device_uuid",
        "source_file",
        "source_sha256",
        "agents",
        "steps",
    ]
    rows: list[dict[str, Any]] = []
    for keys, cell in frame.groupby(grouping, sort=True, dropna=False):
        record = dict(zip(grouping, keys, strict=True))
        nested_ratio, nested_ratio_of_medians = paired_ratio(
            cell, "cuda_device_graph", "cuda_host_graph"
        )
        launch_ratio, launch_ratio_of_medians = paired_ratio(
            cell, "cuda_host_launch", "cuda_host_graph"
        )
        record.update(
            {
                "analysis_unit": "placement",
                "technical_repetitions_paired": EXPECTED_REPETITIONS,
                "cuda_device_graph_over_cuda_host_graph_wall_ratio_of_medians": (
                    nested_ratio_of_medians
                ),
                "cuda_device_graph_over_cuda_host_graph_paired_wall_ratio_median": (
                    quantile(nested_ratio, 0.50)
                ),
                "cuda_device_graph_over_cuda_host_graph_paired_wall_ratio_p95": (
                    quantile(nested_ratio, 0.95)
                ),
                "cuda_device_graph_over_cuda_host_graph_paired_wall_ratio_p99": (
                    quantile(nested_ratio, 0.99)
                ),
                "cuda_host_launch_over_cuda_host_graph_wall_ratio_of_medians": (
                    launch_ratio_of_medians
                ),
                "cuda_host_launch_over_cuda_host_graph_paired_wall_ratio_median": (
                    quantile(launch_ratio, 0.50)
                ),
                "cuda_host_launch_over_cuda_host_graph_paired_wall_ratio_p95": (
                    quantile(launch_ratio, 0.95)
                ),
                "cuda_host_launch_over_cuda_host_graph_paired_wall_ratio_p99": (
                    quantile(launch_ratio, 0.99)
                ),
            }
        )
        rows.append(record)
    result = pd.DataFrame(rows)
    if len(result) != len(EXPECTED_EXPERIMENTS) * 12:
        raise AssertionError(f"expected 60 placement-cell contrasts, got {len(result)}")
    return result.sort_values(["provider", "placement_id", "agents", "steps"])


def write_outputs(
    cell_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    manifest: dict[str, Any],
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_path = output_dir / "native-dispatch-pilot-cell-summary.csv"
    contrast_path = output_dir / "native-dispatch-pilot-contrasts.csv"
    manifest_path = output_dir / "native-dispatch-pilot-manifest.json"
    paths = (cell_path, contrast_path, manifest_path)
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing outputs without --overwrite: {existing}"
        )

    cell_summary.to_csv(cell_path, index=False)
    contrasts.to_csv(contrast_path, index=False)
    manifest["outputs"] = {
        "cell_summary": {
            "path": str(cell_path),
            "rows": len(cell_summary),
            "sha256": sha256_file(cell_path),
        },
        "contrasts": {
            "path": str(contrast_path),
            "rows": len(contrasts),
            "sha256": sha256_file(contrast_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return paths


def main() -> None:
    args = parse_args()
    paths = discover_inputs(args.raw_dir)
    frozen_source = validate_frozen_source(args.raw_dir)
    frames: list[pd.DataFrame] = []
    placements: list[dict[str, Any]] = []
    for path in paths:
        frame, metadata = validate_placement(path, frozen_source)
        frames.append(frame)
        placements.append(metadata)
    combined = pd.concat(frames, ignore_index=True)

    if set(combined["placement_id"]) != set(EXPECTED_EXPERIMENTS):
        raise AssertionError("the five frozen placement IDs were not recovered exactly")
    if combined["device_uuid"].nunique() != len(EXPECTED_EXPERIMENTS):
        raise AssertionError("GPU UUIDs are not distinct across fresh placements")
    if len(combined) != EXPECTED_ROWS_PER_PLACEMENT * len(EXPECTED_EXPERIMENTS):
        raise AssertionError("combined raw row count is not 12,000")

    cell_summary = summarize_cells(combined)
    contrasts = summarize_contrasts(combined)
    nested_column = "cuda_device_graph_over_cuda_host_graph_wall_ratio_of_medians"
    launch_column = "cuda_host_launch_over_cuda_host_graph_wall_ratio_of_medians"
    placement_ranges: list[dict[str, Any]] = []
    for placement_id, group in contrasts.groupby("placement_id", sort=True):
        placement_ranges.append(
            {
                "placement_id": placement_id,
                "provider": group["provider"].iat[0],
                "device_name": group["device_name"].iat[0],
                "device_uuid": group["device_uuid"].iat[0],
                "device_graph_over_host_graph_ratio_of_medians_min": float(
                    group[nested_column].min()
                ),
                "device_graph_over_host_graph_ratio_of_medians_max": float(
                    group[nested_column].max()
                ),
                "host_launch_over_host_graph_ratio_of_medians_min": float(
                    group[launch_column].min()
                ),
                "host_launch_over_host_graph_ratio_of_medians_max": float(
                    group[launch_column].max()
                ),
            }
        )

    script_path = Path(__file__)
    quality_gates = {
        "five_expected_placements": len(placements) == 5,
        "distinct_gpu_uuid_per_placement": combined["device_uuid"].nunique() == 5,
        "rows_2400_per_placement": bool(combined.groupby("placement_id").size().eq(2400).all()),
        "rows_50_per_mechanism_cell": bool(
            combined.groupby(["placement_id", "agents", "steps", "mechanism"]).size().eq(50).all()
        ),
        "all_status_ok": bool(combined["status"].eq("ok").all()),
        "all_field_exact": bool(combined["field_exact"].all()),
        "all_checksums_equal": bool(
            combined["expected_checksum"].eq(combined["observed_checksum"]).all()
        ),
        "frozen_source_hashes_verified": True,
        "provider_provenance_checks_passed": True,
        "cell_summary_has_240_rows": len(cell_summary) == 240,
        "contrast_summary_has_60_rows": len(contrasts) == 60,
    }
    if not all(quality_gates.values()):
        raise AssertionError(f"analysis quality gate failed: {quality_gates}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "analysis_script": {
            "path": repository_path(script_path),
            "sha256": sha256_file(script_path),
        },
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": sha256_file(args.preregistration),
            "status": "engineering/mechanism pilot; no confirmatory p-value",
        },
        "design": {
            "independent_unit": "fresh GPU placement",
            "technical_repetition_unit": "within-placement timed repetition",
            "placements": len(placements),
            "agents": list(EXPECTED_AGENTS),
            "steps": list(EXPECTED_STEPS),
            "mechanisms": list(EXPECTED_MECHANISMS),
            "technical_repetitions_per_mechanism_cell": EXPECTED_REPETITIONS,
            "raw_rows": len(combined),
            "quantile_method": "NumPy linear quantile",
            "contrast_pairing": "paired by placement, agents, steps, and repetition",
            "inferential_p_values": 0,
        },
        "correctness_semantics": frozen_source,
        "quality_gates": quality_gates,
        "placements": sorted(placements, key=lambda row: row["placement_id"]),
        "pilot_results": {
            "raw_rows": len(combined),
            "status_ok_rows": int(combined["status"].eq("ok").sum()),
            "field_exact_rows": int(combined["field_exact"].sum()),
            "checksum_equal_rows": int(
                combined["expected_checksum"].eq(combined["observed_checksum"]).sum()
            ),
            "failure_rows": int(combined["status"].ne("ok").sum()),
            "placement_cell_ratio_ranges": placement_ranges,
        },
        "required_caveats": [
            "Technical repetitions are not independent placements and receive no p-value.",
            "The fixed nested device graph is an overhead calibration, not a device-resident orchestration treatment.",
            "CPU timing is an untuned single-thread pilot baseline and cannot support a CPU-versus-GPU headline.",
            "CUDA event scopes differ: device-graph events include the parent/child environment while host-graph events cover transition kernels; primary mechanism contrasts therefore use wall time.",
            "Modal and local artifacts have frozen native manifests and source hashes but no separately saved provider-level binary receipt; Lambda and RunPod do.",
        ],
    }
    cell_path, contrast_path, manifest_path = write_outputs(
        cell_summary, contrasts, manifest, args.output_dir, args.overwrite
    )
    print(f"placements={len(placements)}")
    print(f"raw_rows={len(combined)}")
    print(f"field_exact_rows={int(combined['field_exact'].sum())}")
    print(f"failure_rows={int(combined['status'].ne('ok').sum())}")
    print(f"cell_summary_rows={len(cell_summary)}")
    print(f"contrast_rows={len(contrasts)}")
    print(f"cell_summary={cell_path}")
    print(f"contrasts={contrast_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
