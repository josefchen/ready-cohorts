from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/runpod_resident_policy_pilot.py"
SPEC = importlib.util.spec_from_file_location("runpod_resident_policy_pilot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def inventory_fixture() -> dict[str, Any]:
    return {
        "captured_at_utc": "2026-08-12T00:00:00Z",
        "gpu_types": [
            {
                "id": runner.GPU_TYPE,
                "display_name": "L4",
                "memory_gb": 24,
                "secure_cloud": True,
                "community_cloud": True,
                "secure_price_per_hour": 0.49,
                "community_price_per_hour": 0.39,
            }
        ],
        "available_offers": [
            {
                "gpu_type_id": runner.GPU_TYPE,
                "display_name": "L4",
                "stock_status": "Low",
                "data_center_id": "EU-RO-1",
                "location": "Romania",
            }
        ],
    }


def launch_args(tmp_path: Path, *, mode: str = "smoke") -> argparse.Namespace:
    return runner.build_parser().parse_args(
        [
            "--action",
            "launch",
            "--mode",
            mode,
            "--experiment-id",
            f"resident-policy-mocked-{mode}",
            "--data-center-id",
            "EU-RO-1",
            "--max-run-minutes",
            "30",
            "--max-cost-usd",
            "1",
            "--receipt",
            str(tmp_path / "launch.json"),
            "--confirm-spend",
            runner.LAUNCH_ACK,
        ]
    )


def valid_launch_receipt(path: Path, *, pod_id: str = "podmock123") -> dict[str, Any]:
    archive_sha = runner._sha256_bytes(runner._source_archive())
    value = {
        "schema_version": "runpod-resident-policy-launch-v1",
        "pod_id": pod_id,
        "pod_name": "gpu-agent-resident-policy-mocked-smoke",
        "experiment_id": "resident-policy-mocked-smoke",
        "artifact_url": f"https://{pod_id}-8000.proxy.runpod.net",
        "artifact_token": "mock-token",
        "request": {
            "gpu_type": runner.GPU_TYPE,
            "gpu_count": 1,
            "data_center_id": "EU-RO-1",
            "cloud_type": "SECURE",
            "image": runner.CUDA_IMAGE,
            "allowed_cuda_versions": ["13.0"],
            "mode": "smoke",
            "config": runner.SMOKE_CONFIG,
        },
        "source": {
            "source_sha256": runner.FROZEN_SOURCE_SHA256,
            "makefile_sha256": runner.FROZEN_MAKEFILE_SHA256,
            "source_archive_sha256": archive_sha,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_default_plan_is_local_only(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def forbidden() -> dict[str, Any]:
        raise AssertionError("plan attempted network inventory")

    monkeypatch.setattr(runner, "fetch_inventory", forbidden)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    runner.main(["--action", "plan", "--mode", "full", "--env-file", "/nonexistent"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["remote_calls"] == 0
    assert plan["gpu_calls"] == 0
    assert plan["gpu_type"] == "NVIDIA L4"
    assert plan["gpu_count"] == 1
    assert plan["expected_measured_rows"] == 810
    assert plan["source_sha256"] == runner.FROZEN_SOURCE_SHA256
    assert plan["makefile_sha256"] == runner.FROZEN_MAKEFILE_SHA256


def test_source_archive_is_deterministic_and_frozen() -> None:
    first = runner._source_archive()
    second = runner._source_archive()
    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        assert archive.getnames() == list(runner.SOURCE_FILES)
        source = archive.extractfile("resident_policy_pilot.cu")
        makefile = archive.extractfile("Makefile")
        assert source is not None and makefile is not None
        assert runner._sha256_bytes(source.read()) == runner.FROZEN_SOURCE_SHA256
        assert runner._sha256_bytes(makefile.read()) == runner.FROZEN_MAKEFILE_SHA256


def test_launch_requires_both_gates_before_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = launch_args(tmp_path)
    calls = 0

    def inventory() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return inventory_fixture()

    monkeypatch.setattr(runner, "fetch_inventory", inventory)
    monkeypatch.delenv(runner.LAUNCH_GATE_ENV, raising=False)
    with pytest.raises(ValueError, match="requires both"):
        runner.launch(args)
    assert calls == 0


def test_mocked_launch_requests_exactly_one_l4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = launch_args(tmp_path, mode="full")
    observed: list[tuple[str, str, dict[str, Any]]] = []

    def request(
        method: str,
        url: str,
        *,
        body: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        assert body is not None
        observed.append((method, url, body))
        return {"id": "podmock123"}

    monkeypatch.setenv(runner.LAUNCH_GATE_ENV, runner.LAUNCH_ACK)
    monkeypatch.setattr(runner, "fetch_inventory", inventory_fixture)
    monkeypatch.setattr(runner, "_request_json", request)
    runner.launch(args)

    assert len(observed) == 1
    method, url, payload = observed[0]
    assert (method, url) == ("POST", f"{runner.REST_URL}/pods")
    assert payload["gpuTypeIds"] == ["NVIDIA L4"]
    assert payload["gpuCount"] == 1
    assert payload["dataCenterIds"] == ["EU-RO-1"]
    assert payload["imageName"] == runner.CUDA_IMAGE
    assert payload["allowedCudaVersions"] == ["13.0"]
    assert payload["interruptible"] is False
    assert "RUNPOD_API_KEY" not in payload["env"]
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    assert receipt["request"]["mode"] == "full"
    assert receipt["request"]["config"] == runner.FULL_CONFIG
    assert receipt["security"]["contains_runpod_api_key"] is False
    assert args.receipt.stat().st_mode & 0o777 == 0o600


def test_live_cost_and_stock_are_hard_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = launch_args(tmp_path)
    monkeypatch.setenv(runner.LAUNCH_GATE_ENV, runner.LAUNCH_ACK)
    no_stock = deepcopy(inventory_fixture())
    no_stock["available_offers"] = []
    with pytest.raises(ValueError, match="no 'NVIDIA L4' stock"):
        runner._validate_launch(args, no_stock)

    expensive = deepcopy(inventory_fixture())
    expensive["gpu_types"][0]["secure_price_per_hour"] = 20.0
    with pytest.raises(ValueError, match="exceeds --max-cost-usd"):
        runner._validate_launch(args, expensive)


def full_native_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    experiment_id = "resident-policy-mocked-full"
    run_id = "resident-policy-mocked-full-20260812T000000Z-p1"
    csv_path = tmp_path / f"{run_id}.csv"
    manifest_path = tmp_path / f"{run_id}.manifest.json"
    binary_sha = "a" * 64
    fieldnames = [
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
    ]
    rows = []
    mechanisms = sorted(runner.MECHANISMS)
    for agents in runner.FULL_CONFIG["agent_counts"]:
        for epochs in runner.FULL_CONFIG["epoch_counts"]:
            decisions = "0" * epochs
            for repetition in range(runner.FULL_CONFIG["repetitions_per_mechanism_cell"]):
                for order_index, mechanism in enumerate(mechanisms):
                    rows.append(
                        {
                            "schema_version": runner.SCHEMA_VERSION,
                            "timestamp_utc": "2026-08-12T00:00:00Z",
                            "run_id": run_id,
                            "experiment_id": experiment_id,
                            "phase": "measure",
                            "mechanism": mechanism,
                            "agents": agents,
                            "epochs": epochs,
                            "repetition": repetition,
                            "order_index": order_index,
                            "status": "ok",
                            "failure_stage": "",
                            "error_code": 0,
                            "error_message": "",
                            "batch_iterations": 10,
                            "aggregate_wall_ns": 100_000_000,
                            "wall_ns_per_invocation": 10_000_000.0,
                            "aggregate_device_ns": 90_000_000,
                            "device_ns_per_invocation": 9_000_000.0,
                            "min_duration_target_ns": 100_000_000,
                            "min_duration_reached": "true",
                            "expected_state_checksum": "123",
                            "observed_state_checksum": "123",
                            "expected_decision_hash": "456",
                            "observed_decision_hash": "456",
                            "expected_decisions": decisions,
                            "observed_decisions": decisions,
                            "exact_state_match": "true",
                            "exact_decision_match": "true",
                            "exact_validation_count": 10,
                            "seed": runner.FULL_CONFIG["seed"],
                            "block_size": runner.FULL_CONFIG["block_size"],
                            "predicate_blocks": (agents + 255) // 256,
                        }
                    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "csv_file": csv_path.name,
        "provenance": {
            "execution_provider": "runpod",
            "requested_gpu": runner.GPU_TYPE,
            "placement_id": "podmock123",
            "image_digest": f"registry-ref:{runner.CUDA_IMAGE}",
            "source_sha256": runner.FROZEN_SOURCE_SHA256,
            "binary_sha256": binary_sha,
        },
        "hardware": {
            "cuda_available": True,
            "device_count": 1,
            "device_name": "NVIDIA L4",
            "device_uuid": "00112233445566778899aabbccddeeff",
            "unified_addressing": 1,
        },
        "config": {**runner.FULL_CONFIG, "mechanisms": mechanisms},
        "cells": [
            {"agents": agents, "epochs": epochs}
            for agents in runner.FULL_CONFIG["agent_counts"]
            for epochs in runner.FULL_CONFIG["epoch_counts"]
        ],
        "results": {
            "measured_rows": 810,
            "exact_rows": 810,
            "failure_rows": 0,
            "status_counts": {"ok": 810},
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = {
        "pod_id": "podmock123",
        "experiment_id": experiment_id,
        "request": {
            "mode": "full",
            "data_center_id": "EU-RO-1",
            "cloud_type": "SECURE",
        },
        "source": {"source_archive_sha256": "b" * 64},
    }
    provider = {
        "schema_version": "runpod-resident-policy-provider-v1",
        "provider": "runpod",
        "secrets_recorded": False,
        "pod": {
            "pod_id": "podmock123",
            "data_center_id": "EU-RO-1",
            "provider_gpu_count": "1",
        },
        "request": {
            "requested_gpu_type": runner.GPU_TYPE,
            "requested_data_center_id": "EU-RO-1",
            "requested_cloud_type": "SECURE",
            "image_reference": runner.CUDA_IMAGE,
            "allowed_cuda_version": "13.0",
            "mode": "full",
        },
        "source": {
            "source_sha256": runner.FROZEN_SOURCE_SHA256,
            "makefile_sha256": runner.FROZEN_MAKEFILE_SHA256,
            "source_archive_sha256": "b" * 64,
            "binary_sha256": binary_sha,
        },
        "execution": {"compile_return_code": 0, "program_return_code": 0},
        "host": {
            "platform": "Linux-test",
            "node": "mock-node",
            "machine": "x86_64",
            "logical_cpu_count": 8,
            "cpu_model": "mock CPU",
        },
        "gpus": [
            {
                "uuid": "GPU-mock-l4",
                "name": "NVIDIA L4",
                "driver_version": "999.0",
                "memory_mib": "23034",
            }
        ],
    }
    ready = {
        "ready": True,
        "compile_return_code": 0,
        "program_return_code": 0,
        "mode": "full",
        "source_sha256": runner.FROZEN_SOURCE_SHA256,
        "makefile_sha256": runner.FROZEN_MAKEFILE_SHA256,
        "binary_sha256": binary_sha,
    }
    pod = {
        "id": "podmock123",
        "image": runner.CUDA_IMAGE,
        "gpu": {"id": runner.GPU_TYPE, "count": 1},
        "machine": {"gpu_type_id": runner.GPU_TYPE, "data_center_id": "EU-RO-1"},
    }
    return csv_path, manifest_path, provider, receipt, ready, pod


def test_full_810_row_field_and_decision_validator(tmp_path: Path) -> None:
    csv_path, manifest_path, provider, receipt, ready, pod = full_native_fixture(tmp_path)
    errors = runner._native_validation_errors(
        csv_path=csv_path,
        manifest_path=manifest_path,
        provider=provider,
        receipt=receipt,
        ready=ready,
        pod=pod,
    )
    assert errors == []


def test_validator_rejects_one_decision_trace_mismatch(tmp_path: Path) -> None:
    csv_path, manifest_path, provider, receipt, ready, pod = full_native_fixture(tmp_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows[0]["observed_decisions"] = "1" * int(rows[0]["epochs"])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    errors = runner._native_validation_errors(
        csv_path=csv_path,
        manifest_path=manifest_path,
        provider=provider,
        receipt=receipt,
        ready=ready,
        pod=pod,
    )
    assert any("decision trace differs" in error for error in errors)


def test_termination_is_locked_without_verified_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = tmp_path / "launch.json"
    receipt = valid_launch_receipt(launch_path)
    collection_path = tmp_path / "collection.json"
    collection = {
        "schema_version": "runpod-resident-policy-collection-v1",
        "pod_id": receipt["pod_id"],
        "experiment_id": receipt["experiment_id"],
        "launch_receipt_sha256": runner._sha256_file(launch_path),
        "archive_sha256": "c" * 64,
        "validation": {"passed": False, "errors": ["mock failure"]},
    }
    collection_path.write_text(json.dumps(collection), encoding="utf-8")
    args = runner.build_parser().parse_args(
        [
            "--action",
            "terminate",
            "--receipt",
            str(launch_path),
            "--collection-receipt",
            str(collection_path),
            "--confirm-terminate-pod-id",
            receipt["pod_id"],
        ]
    )
    monkeypatch.setenv(runner.TERMINATE_GATE_ENV, receipt["pod_id"])
    monkeypatch.setattr(
        runner,
        "_get_pod_or_none",
        lambda _: pytest.fail("termination performed a network lookup before validation"),
    )
    with pytest.raises(ValueError, match="did not pass validity gates"):
        runner.terminate(args)


def test_termination_is_locked_if_retained_archive_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = tmp_path / "launch.json"
    receipt = valid_launch_receipt(launch_path)
    collection_path = tmp_path / "collection.json"
    collection = {
        "schema_version": "runpod-resident-policy-collection-v1",
        "pod_id": receipt["pod_id"],
        "experiment_id": receipt["experiment_id"],
        "launch_receipt_sha256": runner._sha256_file(launch_path),
        "archive_sha256": "c" * 64,
        "validation": {"passed": True, "errors": []},
    }
    collection_path.write_text(json.dumps(collection), encoding="utf-8")
    args = runner.build_parser().parse_args(
        [
            "--action",
            "terminate",
            "--receipt",
            str(launch_path),
            "--collection-receipt",
            str(collection_path),
            "--confirm-terminate-pod-id",
            receipt["pod_id"],
        ]
    )
    monkeypatch.setenv(runner.TERMINATE_GATE_ENV, receipt["pod_id"])
    monkeypatch.setattr(
        runner,
        "_get_pod_or_none",
        lambda _: pytest.fail("termination touched RunPod before checking local evidence"),
    )
    with pytest.raises(ValueError, match="archive is missing or has changed"):
        runner.terminate(args)


def test_mocked_termination_stops_then_deletes_exact_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = tmp_path / "launch.json"
    receipt = valid_launch_receipt(launch_path)
    pod_id = receipt["pod_id"]
    collection_path = tmp_path / "collection.json"
    archive_path = tmp_path / "resident-policy-artifacts.tar.gz"
    archive_path.write_bytes(b"mock-verified-archive")
    collection = {
        "schema_version": "runpod-resident-policy-collection-v1",
        "pod_id": pod_id,
        "experiment_id": receipt["experiment_id"],
        "launch_receipt_sha256": runner._sha256_file(launch_path),
        "archive_sha256": runner._sha256_file(archive_path),
        "extracted_files": ["artifact-index.json"],
        "ready": {"ready": True},
        "pod": {"id": pod_id},
        "validation": {"passed": True, "errors": []},
    }
    collection_path.write_text(json.dumps(collection), encoding="utf-8")
    termination_path = tmp_path / "termination.json"
    args = runner.build_parser().parse_args(
        [
            "--action",
            "terminate",
            "--receipt",
            str(launch_path),
            "--collection-receipt",
            str(collection_path),
            "--confirm-terminate-pod-id",
            pod_id,
            "--termination-receipt",
            str(termination_path),
        ]
    )
    pod_values = iter(
        [
            {"id": pod_id, "name": receipt["pod_name"], "desiredStatus": "RUNNING"},
            {"id": pod_id, "name": receipt["pod_name"], "desiredStatus": "EXITED"},
            None,
        ]
    )
    calls: list[tuple[str, str]] = []

    def request(method: str, url: str, **_: Any) -> None:
        calls.append((method, url))

    monkeypatch.setenv(runner.TERMINATE_GATE_ENV, pod_id)
    monkeypatch.setattr(runner, "_validate_collected_bundle", lambda *_: ([], {}))
    monkeypatch.setattr(runner, "_get_pod_or_none", lambda _: next(pod_values))
    monkeypatch.setattr(runner, "_request_json", request)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    runner.terminate(args)

    assert calls == [
        ("POST", f"{runner.REST_URL}/pods/{pod_id}/stop"),
        ("DELETE", f"{runner.REST_URL}/pods/{pod_id}"),
    ]
    outcome = json.loads(termination_path.read_text(encoding="utf-8"))
    assert outcome["pod_id"] == pod_id
    assert outcome["stop_called"] is True
    assert outcome["delete_called"] is True
    assert outcome["confirmed_absent"] is True
    assert outcome["remote_volume_recoverable"] is False


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        payload = b"unsafe"
        info = tarfile.TarInfo("../escape")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe artifact path"):
        runner._safe_extract(stream.getvalue(), tmp_path)
    assert not (tmp_path.parent / "escape").exists()
