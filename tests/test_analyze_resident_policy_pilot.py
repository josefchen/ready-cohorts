from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest


def load_analyzer() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_resident_policy_pilot.py"
    spec = importlib.util.spec_from_file_location("analyze_resident_policy_pilot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load analyzer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyzer = load_analyzer()

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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_full_local_placement(
    root: Path,
    *,
    run_id: str,
    placement_id: str,
    device_uuid: str,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"{run_id}.csv"
    manifest_path = csv_path.with_suffix(".manifest.json")
    experiment_id = run_id.rsplit("-run", 1)[0]
    rows: list[dict[str, str]] = []
    for agents in analyzer.FROZEN_CONFIG["agent_counts"]:
        for epochs in analyzer.FROZEN_CONFIG["epoch_counts"]:
            decisions = ("01" * ((epochs + 1) // 2))[:epochs]
            for repetition in range(analyzer.FROZEN_CONFIG["repetitions_per_mechanism_cell"]):
                for order_index, mechanism in enumerate(analyzer.MECHANISMS):
                    rows.append(
                        {
                            "schema_version": analyzer.RAW_SCHEMA_VERSION,
                            "timestamp_utc": "2026-08-12T00:00:00Z",
                            "run_id": run_id,
                            "experiment_id": experiment_id,
                            "phase": "measure",
                            "mechanism": mechanism,
                            "agents": str(agents),
                            "epochs": str(epochs),
                            "repetition": str(repetition),
                            "order_index": str(order_index),
                            "status": "ok",
                            "failure_stage": "",
                            "error_code": "0",
                            "error_message": "",
                            "batch_iterations": "1",
                            "aggregate_wall_ns": "100000000",
                            "wall_ns_per_invocation": "100000000.0",
                            "aggregate_device_ns": "50000000",
                            "device_ns_per_invocation": "50000000.0",
                            "min_duration_target_ns": "100000000",
                            "min_duration_reached": "true",
                            "expected_state_checksum": "123456789",
                            "observed_state_checksum": "123456789",
                            "expected_decision_hash": "987654321",
                            "observed_decision_hash": "987654321",
                            "expected_decisions": decisions,
                            "observed_decisions": decisions,
                            "exact_state_match": "true",
                            "exact_decision_match": "true",
                            "exact_validation_count": "1",
                            "seed": str(analyzer.FROZEN_CONFIG["seed"]),
                            "block_size": str(analyzer.FROZEN_CONFIG["block_size"]),
                            "predicate_blocks": str(
                                (agents + analyzer.FROZEN_CONFIG["block_size"] - 1)
                                // analyzer.FROZEN_CONFIG["block_size"]
                            ),
                        }
                    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": analyzer.RAW_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "csv_file": csv_path.name,
        "provenance": {
            "execution_provider": "local",
            "requested_gpu": "GTX 1660 Ti",
            "placement_id": placement_id,
            "image_digest": "",
            "source_sha256": analyzer.FROZEN_SOURCE_SHA256,
            "binary_sha256": "b" * 64,
        },
        "hardware": {
            "cuda_available": True,
            "device_count": 1,
            "device_name": "NVIDIA GeForce GTX 1660 Ti",
            "device_uuid": device_uuid,
            "compute_capability": "7.5",
            "unified_addressing": 1,
        },
        "software": {
            "cuda_compile_version": 13000,
            "cuda_runtime_version": 13000,
            "cuda_driver_version": 13020,
            "cpu_model": "Mock CPU",
            "cpu_hardware_threads": 8,
        },
        "config": copy.deepcopy(analyzer.FROZEN_CONFIG),
        "results": {
            "measured_rows": 810,
            "exact_rows": 810,
            "failure_rows": 0,
            "status_counts": {"ok": 810},
        },
    }
    write_json(manifest_path, manifest)
    return csv_path, manifest_path


def test_discover_inputs_recurses_into_runpod_results(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    results = raw_dir / "runpod-resident-policy-001-runpod-l4-p1-pod123" / "results"
    results.mkdir(parents=True)
    csv_path = results / "resident-policy-001-runpod-l4-p1-run1.csv"
    manifest_path = csv_path.with_suffix(".manifest.json")
    csv_path.write_text("placeholder\n")
    manifest_path.write_text("{}\n")
    (raw_dir / "unrelated.csv").write_text("ignored\n")

    assert analyzer.discover_inputs(raw_dir) == [csv_path]


def test_validate_placement_rejects_any_nonfrozen_config(tmp_path: Path) -> None:
    csv_path, manifest_path = write_full_local_placement(
        tmp_path / "raw",
        run_id="resident-policy-001-local-config-test-run1",
        placement_id="local-config-test",
        device_uuid="11111111111111111111111111111111",
    )
    frame, metadata = analyzer.validate_placement(csv_path, external_dir=tmp_path / "external")
    assert len(frame) == 810
    assert metadata["rows"] == 810

    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["min_duration_target_ns"] = 99_000_000
    write_json(manifest_path, manifest)
    with pytest.raises(AssertionError, match="frozen full grid"):
        analyzer.validate_placement(csv_path, external_dir=tmp_path / "external")


def test_main_rejects_duplicate_placement_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    write_full_local_placement(
        raw_dir,
        run_id="resident-policy-001-local-duplicate-a-run1",
        placement_id="duplicate-placement",
        device_uuid="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    write_full_local_placement(
        raw_dir,
        run_id="resident-policy-001-local-duplicate-b-run1",
        placement_id="duplicate-placement",
        device_uuid="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(analyzer.__file__)),
            "--raw-dir",
            str(raw_dir),
            "--external-dir",
            str(tmp_path / "external"),
            "--output-dir",
            str(tmp_path / "processed"),
            "--preregistration",
            str(Path(__file__).resolve().parents[1] / "preregistration/resident-policy-001.md"),
        ],
    )
    with pytest.raises(AssertionError, match="duplicate placement_id"):
        analyzer.main()
    assert not (tmp_path / "processed/resident-policy-pilot-manifest.json").exists()


def test_lambda_sidecar_is_bound_by_both_artifact_hashes(tmp_path: Path) -> None:
    csv_path = tmp_path / "resident-policy-001-lambda-h100-p1-run1.csv"
    manifest_path = csv_path.with_suffix(".manifest.json")
    csv_path.write_bytes(b"scientific-csv")
    manifest_path.write_bytes(b'{"scientific":"manifest"}\n')
    manifest = {
        "run_id": "resident-policy-001-lambda-h100-p1-run1",
        "provenance": {"execution_provider": "lambda", "placement_id": "lambda-placement"},
    }
    sidecar_path = manifest_path.with_name(
        "resident-policy-001-lambda-h100-p1-run1.provider-lambda-resident-policy.json"
    )
    sidecar = {
        "schema_version": "provider-lambda-resident-policy-v1",
        "execution_provider": "lambda",
        "mode": "full",
        "placement_id": "lambda-placement",
        "artifact_validation": {"passed": True},
        "artifacts": {
            csv_path.name: {"sha256": analyzer.sha256_file(csv_path)},
            manifest_path.name: {"sha256": analyzer.sha256_file(manifest_path)},
        },
        "instance": {"id": "instance-123", "region": {"name": "us-south-2"}},
    }
    write_json(sidecar_path, sidecar)

    binding = analyzer.validate_provider_binding(
        csv_path=csv_path,
        manifest_path=manifest_path,
        manifest=manifest,
        external_dir=tmp_path / "external",
    )
    assert binding["binding_type"] == "lambda-provider-sidecar"
    assert binding["validation_passed"] is True

    sidecar["artifacts"][csv_path.name]["sha256"] = "0" * 64
    write_json(sidecar_path, sidecar)
    with pytest.raises(AssertionError, match="failed its frozen binding"):
        analyzer.validate_provider_binding(
            csv_path=csv_path,
            manifest_path=manifest_path,
            manifest=manifest,
            external_dir=tmp_path / "external",
        )


def make_runpod_binding(
    tmp_path: Path, *, native_placement_id: str, receipt_pod_id: str
) -> tuple[Path, Path, dict[str, Any]]:
    bundle = tmp_path / "runpod-resident-policy-001-runpod-l4-p1-pod123"
    results = bundle / "results"
    results.mkdir(parents=True)
    csv_path = results / "resident-policy-001-runpod-l4-p1-run1.csv"
    manifest_path = csv_path.with_suffix(".manifest.json")
    csv_path.write_bytes(b"scientific-csv")
    manifest_path.write_bytes(b'{"scientific":"manifest"}\n')
    archive_path = bundle / "resident-policy-artifacts.tar.gz"
    archive_path.write_bytes(b"verified-archive")
    launch_path = tmp_path / "resident-policy-001-runpod-l4-p1.launch.json"
    experiment_id = "resident-policy-001-runpod-l4-p1"
    write_json(
        launch_path,
        {
            "schema_version": "runpod-resident-policy-launch-v1",
            "pod_id": receipt_pod_id,
            "pod_name": f"gpu-agent-{experiment_id}",
            "experiment_id": experiment_id,
            "request": {
                "mode": "full",
                "gpu_type": "NVIDIA L4",
                "gpu_count": 1,
                "image": analyzer.CUDA_IMAGE,
            },
            "source": {
                "source_sha256": analyzer.FROZEN_SOURCE_SHA256,
                "makefile_sha256": analyzer.FROZEN_MAKEFILE_SHA256,
            },
        },
    )
    write_json(
        bundle / "artifact-index.json",
        {
            "schema_version": "runpod-resident-policy-artifact-index-v1",
            "files": {
                f"results/{csv_path.name}": {
                    "sha256": analyzer.sha256_file(csv_path),
                    "bytes": csv_path.stat().st_size,
                },
                f"results/{manifest_path.name}": {
                    "sha256": analyzer.sha256_file(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                },
            },
        },
    )
    collection = {
        "schema_version": "runpod-resident-policy-collection-v1",
        "experiment_id": experiment_id,
        "pod_id": receipt_pod_id,
        "launch_receipt": str(launch_path),
        "launch_receipt_sha256": analyzer.sha256_file(launch_path),
        "archive_sha256": analyzer.sha256_file(archive_path),
        "pod": {
            "id": receipt_pod_id,
            "machine": {"data_center_id": "EU-RO-1"},
        },
        "validation": {
            "passed": True,
            "errors": [],
            "native_summary": {
                "expected_rows": 810,
                "csv_files": [csv_path.name],
                "manifest_files": [manifest_path.name],
            },
        },
    }
    write_json(bundle / "collection-receipt.json", collection)
    manifest = {
        "run_id": "resident-policy-001-runpod-l4-p1-run1",
        "experiment_id": "resident-policy-001-runpod-l4-p1",
        "provenance": {
            "execution_provider": "runpod",
            "placement_id": native_placement_id,
        },
    }
    return csv_path, manifest_path, manifest


def test_runpod_receipt_must_match_native_placement_id(tmp_path: Path) -> None:
    csv_path, manifest_path, manifest = make_runpod_binding(
        tmp_path,
        native_placement_id="pod123",
        receipt_pod_id="different-pod",
    )
    with pytest.raises(AssertionError, match="RunPod.*binding|placement|Pod ID"):
        analyzer.validate_provider_binding(
            csv_path=csv_path,
            manifest_path=manifest_path,
            manifest=manifest,
            external_dir=tmp_path / "external",
        )


def technical_batch_mean_frame() -> pd.DataFrame:
    rows = []
    mechanism_scale = {
        "no_decision_lower_bound": 1.0,
        "device_resident": 2.0,
        "host_roundtrip": 4.0,
    }
    for repetition in range(30):
        for mechanism, scale in mechanism_scale.items():
            wall_ns = scale * (1_000.0 + repetition)
            rows.append(
                {
                    "placement_id": "placement-1",
                    "provider": "local",
                    "requested_gpu": "GTX 1660 Ti",
                    "device_name": "NVIDIA GeForce GTX 1660 Ti",
                    "device_uuid": "11111111111111111111111111111111",
                    "source_file": "synthetic.csv",
                    "agents": 256,
                    "epochs": 32,
                    "mechanism": mechanism,
                    "repetition": repetition,
                    "batch_iterations": 100,
                    "wall_ns_per_invocation": wall_ns,
                    "device_ns_per_invocation": wall_ns * 0.75,
                    "exact_validation_count": 100,
                }
            )
    return pd.DataFrame(rows)


def test_percentiles_are_explicitly_named_as_batch_mean_quantiles(tmp_path: Path) -> None:
    frame = technical_batch_mean_frame()
    cells, contrasts = analyzer.build_summaries(frame)
    expected_cell_columns = {
        "wall_batch_mean_ns_p95",
        "wall_batch_mean_ns_p99",
        "device_batch_mean_ns_p95",
        "device_batch_mean_ns_p99",
    }
    misleading_cell_columns = {"wall_ns_p95", "wall_ns_p99", "device_ns_p95", "device_ns_p99"}
    assert expected_cell_columns <= set(cells)
    assert misleading_cell_columns.isdisjoint(cells)
    assert {
        "host_over_resident_batch_mean_ratio_p95",
        "host_over_resident_batch_mean_ratio_p99",
        "resident_over_floor_batch_mean_ratio_p95",
        "resident_over_floor_batch_mean_ratio_p99",
    } <= set(contrasts)
    assert {
        "host_over_resident_paired_ratio_p95",
        "host_over_resident_paired_ratio_p99",
        "resident_over_floor_paired_ratio_p95",
        "resident_over_floor_paired_ratio_p99",
    }.isdisjoint(contrasts)

    host_values = frame.loc[frame["mechanism"].eq("host_roundtrip"), "wall_ns_per_invocation"]
    host_cell = cells[cells["mechanism"].eq("host_roundtrip")].iloc[0]
    assert host_cell["wall_batch_mean_ns_p99"] == pytest.approx(
        float(np.quantile(host_values, 0.99))
    )

    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text("frozen test preregistration\n")
    metadata = [
        {
            "placement_id": "placement-1",
            "run_id": "run-1",
            "device_uuid": "11111111111111111111111111111111",
            "rows": 810,
            "strict_gates": {"all_exact": True},
            "provider_binding": {"validation_passed": True},
        }
    ]
    _cell_path, _contrast_path, manifest_path = analyzer.write_outputs(
        tmp_path / "processed",
        cells,
        contrasts,
        metadata,
        preregistration,
        overwrite=False,
    )
    analysis_manifest = json.loads(manifest_path.read_text())
    assert "batch-average rows" in analysis_manifest["design"]["technical_quantiles"]
    assert any(
        "not individual-invocation tails" in caveat
        for caveat in analysis_manifest["required_caveats"]
    )
