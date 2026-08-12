from __future__ import annotations

import copy
import csv
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts/provider_lambda_resident_policy.py"
    spec = importlib.util.spec_from_file_location("provider_lambda_resident_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load runner module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


class FakeLambdaApi:
    def __init__(self, responses: dict[str, Any]):
        self.responses = responses
        self.get_paths: list[str] = []
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def get(self, path: str) -> Any:
        self.get_paths.append(path)
        return copy.deepcopy(self.responses[path])

    def post(self, path: str, body: dict[str, Any]) -> Any:
        self.posts.append((path, copy.deepcopy(body)))
        if path == "/api/v1/instance-operations/launch":
            return {"instance_ids": ["12345678abcdef00"]}
        if path == "/api/v1/instance-operations/terminate":
            return {"instance_ids": body["instance_ids"], "status": "terminating"}
        raise AssertionError(f"unexpected mocked POST: {path}")


def launch_inventory_responses() -> dict[str, Any]:
    return {
        "/api/v1/instance-types": {
            "gpu_1x_h100_sxm5": {
                "instance_type": {
                    "name": "gpu_1x_h100_sxm5",
                    "description": "1x H100 (80 GB SXM5)",
                    "gpu_description": "H100 (80 GB SXM5)",
                    "price_cents_per_hour": 429,
                    "architecture": "x86_64",
                    "specs": {
                        "gpus": 1,
                        "vcpus": 26,
                        "memory_gib": 225,
                        "storage_gib": 2816,
                    },
                },
                "regions_with_capacity_available": [
                    {"name": "us-south-2", "description": "North Texas, USA"}
                ],
            }
        },
        "/api/v1/instances": [],
        "/api/v1/ssh-keys": [{"id": "key-1", "name": "research-key"}],
        "/api/v1/images": [
            {
                "id": "image-1",
                "name": "Lambda Stack 24.04",
                "family": "lambda-stack-24-04",
                "version": "24.4.4-2141",
                "architecture": "x86_64",
                "region": {"name": "us-south-2", "description": "North Texas, USA"},
            }
        ],
    }


def launch_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        env_file="unused.env",
        confirm_spend=True,
        launch_confirmation=runner.LAUNCH_CONFIRMATION,
        instance_name="gpu-agent-resident-policy-001-test",
        max_hourly_usd=5.0,
        instance_type="gpu_1x_h100_sxm5",
        region="us-south-2",
        gpu_family="H100",
        ssh_key_name="research-key",
        image_family="lambda-stack-24-04",
        output_dir=str(tmp_path),
    )


def test_default_plan_is_local_only(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def fail_if_api_is_built(_args: Any) -> None:
        raise AssertionError("the local plan attempted to construct an API client")

    monkeypatch.setattr(runner, "api_from_args", fail_if_api_is_built)
    runner.command_plan(SimpleNamespace())
    output = capsys.readouterr().out
    assert "api_calls=0" in output
    assert "remote_calls=0" in output
    assert "billable_resources_created=0" in output
    assert runner.FROZEN_SOURCE_SHA256 in output
    assert runner.FROZEN_MAKEFILE_SHA256 in output


def test_launch_gate_fails_before_api_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = launch_args(tmp_path)
    args.confirm_spend = False

    def fail_if_api_is_built(_args: Any) -> None:
        raise AssertionError("a refused launch attempted to construct an API client")

    monkeypatch.setattr(runner, "api_from_args", fail_if_api_is_built)
    with pytest.raises(RuntimeError, match="--confirm-spend"):
        runner.command_launch(args)


def test_mocked_launch_creates_exactly_one_validated_h100(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLambdaApi(launch_inventory_responses())
    monkeypatch.setattr(runner, "api_from_args", lambda _args: fake)
    runner.command_launch(launch_args(tmp_path))

    assert fake.get_paths == [
        "/api/v1/instance-types",
        "/api/v1/instances",
        "/api/v1/ssh-keys",
        "/api/v1/images",
    ]
    assert len(fake.posts) == 1
    path, body = fake.posts[0]
    assert path == "/api/v1/instance-operations/launch"
    assert body["instance_type_name"] == "gpu_1x_h100_sxm5"
    assert body["file_system_names"] == []
    receipts = list(tmp_path.glob("resident-policy-lambda-launch-*.jsonl"))
    assert len(receipts) == 1
    events = [json.loads(line) for line in receipts[0].read_text().splitlines()]
    assert [event["event"] for event in events] == ["launch_intent", "launch_accepted"]
    assert events[0]["validated_gpu_count"] == 1
    assert events[1]["instance_id"] == "12345678abcdef00"


def test_a100_cannot_pass_as_a10() -> None:
    record = runner.inventory(FakeLambdaApi(launch_inventory_responses()))
    record["instance_types"][0]["gpu_description"] = "A100 (80 GB SXM4)"
    with pytest.raises(RuntimeError, match="not 1x A10"):
        runner._validated_launch_selection(
            record,
            instance_type="gpu_1x_h100_sxm5",
            region="us-south-2",
            gpu_family="A10",
            ssh_key_name="research-key",
            image_family="lambda-stack-24-04",
            max_hourly_usd=5.0,
        )


def _lambda_full_artifacts() -> tuple[str, str, str, str]:
    run_id = "resident-policy-test-run"
    experiment_id = "resident-policy-001-local-p1"
    rows: list[dict[str, str]] = []
    for agents in runner.FULL_CONFIG["agent_counts"]:
        for epochs in runner.FULL_CONFIG["epoch_counts"]:
            decisions = ("01" * ((epochs + 1) // 2))[:epochs]
            for repetition in range(runner.FULL_CONFIG["repetitions_per_mechanism_cell"]):
                for order_index, mechanism in enumerate(runner.MECHANISMS):
                    rows.append(
                        {
                            "schema_version": runner.SCHEMA_VERSION,
                            "timestamp_utc": "2026-08-11T21:28:37Z",
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
                            "batch_iterations": "2",
                            "aggregate_wall_ns": "200000000",
                            "wall_ns_per_invocation": "100000000.0",
                            "aggregate_device_ns": "100000000",
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
                            "exact_validation_count": "2",
                            "seed": str(runner.FULL_CONFIG["seed"]),
                            "block_size": str(runner.FULL_CONFIG["block_size"]),
                            "predicate_blocks": str(
                                (agents + runner.FULL_CONFIG["block_size"] - 1)
                                // runner.FULL_CONFIG["block_size"]
                            ),
                        }
                    )
    csv_output = io.StringIO()
    writer = csv.DictWriter(csv_output, fieldnames=runner.CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    manifest = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "csv_file": f"{run_id}.csv",
        "provenance": {
            "execution_provider": "lambda",
            "requested_gpu": "H100",
            "placement_id": "lambda-h100-test-placement",
            "image_digest": "nvidia/cuda@sha256:test",
            "source_sha256": runner.FROZEN_SOURCE_SHA256,
            "binary_sha256": "b" * 64,
        },
        "hardware": {
            "cuda_available": True,
            "device_count": 1,
            "device_name": "NVIDIA H100 80GB HBM3",
            "device_uuid": "1234567890abcdef1234567890abcdef",
            "unified_addressing": 1,
        },
        "software": {"cuda_compile_version": 13000, "cuda_runtime_version": 13000},
        "config": {**runner.FULL_CONFIG, **runner.COMMON_CONFIG},
        "cells": [
            {
                "agents": agents,
                "epochs": epochs,
                "common_batch_iterations": 2,
                "batch_cap_reached": False,
                "median_calibration_wall_ns": {
                    mechanism: 100_000_000 for mechanism in runner.MECHANISMS
                },
            }
            for agents in runner.FULL_CONFIG["agent_counts"]
            for epochs in runner.FULL_CONFIG["epoch_counts"]
        ],
        "results": {
            "measured_rows": 810,
            "exact_rows": 810,
            "failure_rows": 0,
            "status_counts": {"ok": 810},
        },
        "semantic_contract": {
            "host_roundtrip": "mocked test contract",
            "device_resident": "mocked test contract",
            "no_decision_lower_bound": "mocked test contract",
            "oracle": "mocked test contract",
        },
        "limitations": [],
    }
    return (
        f"{run_id}.csv",
        csv_output.getvalue(),
        f"{run_id}.manifest.json",
        json.dumps(manifest),
    )


def validate_full_artifacts(csv_text: str, manifest_text: str) -> list[str]:
    csv_name, _original_csv, manifest_name, _original_manifest = _lambda_full_artifacts()
    return runner.artifact_validation_errors(
        csv_name=csv_name,
        csv_text=csv_text,
        manifest_name=manifest_name,
        manifest_text=manifest_text,
        experiment_id="resident-policy-001-local-p1",
        mode="full",
        gpu_family="H100",
        placement_id="lambda-h100-test-placement",
        image_digest="nvidia/cuda@sha256:test",
        binary_sha256="b" * 64,
        actual_gpu_name="NVIDIA H100 80GB HBM3",
        actual_gpu_uuid="GPU-12345678-90ab-cdef-1234-567890abcdef",
        program_returncode=0,
    )


def test_full_artifact_validator_accepts_exact_810_row_contract() -> None:
    _csv_name, csv_text, _manifest_name, manifest_text = _lambda_full_artifacts()
    assert validate_full_artifacts(csv_text, manifest_text) == []


def test_full_artifact_validator_rejects_decision_mismatch() -> None:
    _csv_name, csv_text, _manifest_name, manifest_text = _lambda_full_artifacts()
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    rows[0]["observed_decisions"] = "10"
    rows[0]["exact_decision_match"] = "false"
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    errors = validate_full_artifacts(output.getvalue(), manifest_text)
    assert any("decision trace is not exact" in error for error in errors)
    assert any("decision string differs" in error for error in errors)


def test_cuda13_program_command_is_frozen_and_networkless() -> None:
    command = runner._docker_program_command(
        remote_root="/tmp/gpu-agent-resident-policy-0123456789abcdef",
        mode="full",
        experiment_id="resident-policy-001-lambda-h100-p1",
        gpu_family="H100",
        placement_id="lambda-h100-placement",
        image_digest="nvidia/cuda@sha256:test",
        binary_sha256="c" * 64,
    )
    assert runner.CUDA_IMAGE in command
    assert command[:3] == ["sudo", "-n", "docker"]
    assert "--network=none" in command
    assert "device=0" in command
    assert "--agents" in command
    assert command[command.index("--agents") + 1] == "256,2048,16384"
    assert command[command.index("--repetitions") + 1] == "30"
    assert f"SOURCE_SHA256={runner.FROZEN_SOURCE_SHA256}" in command
    assert runner._expected_row_count(runner.FULL_CONFIG) == 810
    assert runner._cuda_major("Cuda compilation tools, release 13.0, V13.0.88") == 13


def test_mocked_exact_target_termination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    instance_id = "12345678abcdef00"
    expected_name = "gpu-agent-resident-policy-001-test"
    fake = FakeLambdaApi(
        {
            f"/api/v1/instances/{instance_id}": {
                "id": instance_id,
                "name": expected_name,
                "status": "active",
                "region": {"name": "us-south-2"},
                "instance_type": {"name": "gpu_1x_h100_sxm5"},
                "ssh_key_names": ["research-key"],
                "ip": "192.0.2.1",
            }
        }
    )
    monkeypatch.setattr(runner, "api_from_args", lambda _args: fake)
    args = SimpleNamespace(
        env_file="unused.env",
        instance_id=instance_id,
        expected_instance_name=expected_name,
        confirm_termination=True,
        termination_confirmation=runner.TERMINATION_CONFIRMATION_PREFIX + instance_id,
        output_dir=str(tmp_path),
    )
    runner.command_terminate(args)

    assert fake.posts == [
        ("/api/v1/instance-operations/terminate", {"instance_ids": [instance_id]})
    ]
    receipt = tmp_path / f"resident-policy-lambda-termination-{instance_id}.jsonl"
    events = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "termination_intent",
        "termination_requested",
    ]
    assert events[0]["exact_target_count"] == 1
    assert "ip" not in events[0]["instance"]
