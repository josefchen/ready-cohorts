"""Explicitly gated Modal compiler/single-L4 runner for the native dispatch pilot.

The default ``plan`` action is local-only. Neither importing this module nor
running its default entry point invokes a remote function.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NATIVE_ROOT = REPO_ROOT / "native/device_dispatch"
REMOTE_NATIVE_ROOT = Path("/opt/gpu-agent-crossover/native/device_dispatch")
REMOTE_BINARY = Path("/tmp/device_dispatch_pilot")
REMOTE_OUTPUT = Path("/tmp/device-dispatch-output")
CUDA_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("g++")
    .add_local_dir(LOCAL_NATIVE_ROOT, str(REMOTE_NATIVE_ROOT), copy=True)
)

app = modal.App("gpu-agent-device-dispatch-pilot", image=image)


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(LOCAL_NATIVE_ROOT.rglob("*")):
        if not path.is_file() or "build" in path.parts or "smoke-results" in path.parts:
            continue
        digest.update(path.relative_to(LOCAL_NATIVE_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _compile() -> dict[str, str]:
    architectures = ("75", "80", "86", "89", "90")
    command = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-rdc=true",
        "--threads",
        "0",
    ]
    for architecture in architectures:
        command.extend(
            ["-gencode", f"arch=compute_{architecture},code=sm_{architecture}"]
        )
    command.extend(
        [
            "-gencode",
            "arch=compute_90,code=compute_90",
            str(REMOTE_NATIVE_ROOT / "device_dispatch_pilot.cu"),
            "-o",
            str(REMOTE_BINARY),
            "-lcudadevrt",
        ]
    )
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    help_check = subprocess.run(
        [str(REMOTE_BINARY), "--help"], check=True, text=True, capture_output=True
    )
    binary_sha256 = hashlib.sha256(REMOTE_BINARY.read_bytes()).hexdigest()
    return {
        "command": " ".join(command),
        "compiler_stdout": completed.stdout,
        "compiler_stderr": completed.stderr,
        "help_stdout": help_check.stdout,
        "binary_sha256": binary_sha256,
    }


@app.function(cpu=2.0, timeout=20 * 60, single_use_containers=True)
def compile_smoke() -> dict[str, str]:
    """Compile and run ``--help`` on a CPU worker; no GPU is allocated."""

    return _compile()


@app.function(gpu="L4", cpu=4.0, timeout=30 * 60, single_use_containers=True)
def run_l4(
    experiment_id: str,
    agent_counts: str,
    step_counts: str,
    warmups: int,
    repetitions: int,
    seed: int,
) -> dict[str, str]:
    """Compile, execute one bounded L4 placement, and return immutable artifacts."""

    build = _compile()
    REMOTE_OUTPUT.mkdir(parents=True, exist_ok=True)
    source_sha256 = hashlib.sha256()
    for path in sorted(REMOTE_NATIVE_ROOT.rglob("*")):
        if path.is_file():
            source_sha256.update(path.relative_to(REMOTE_NATIVE_ROOT).as_posix().encode())
            source_sha256.update(b"\0")
            source_sha256.update(path.read_bytes())
            source_sha256.update(b"\0")
    environment = os.environ.copy()
    environment.update(
        {
            "EXECUTION_PROVIDER": "modal",
            "REQUESTED_GPU": "L4",
            "SOURCE_SHA256": source_sha256.hexdigest(),
        }
    )
    command = [
        str(REMOTE_BINARY),
        "--experiment-id",
        experiment_id,
        "--agents",
        agent_counts,
        "--steps",
        step_counts,
        "--warmups",
        str(warmups),
        "--repetitions",
        str(repetitions),
        "--seed",
        str(seed),
        "--output-dir",
        str(REMOTE_OUTPUT),
    ]
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment
    )
    csv_files = list(REMOTE_OUTPUT.glob("*.csv"))
    manifest_files = list(REMOTE_OUTPUT.glob("*.manifest.json"))
    if len(csv_files) != 1 or len(manifest_files) != 1:
        raise RuntimeError(
            f"expected one CSV and one manifest, got {len(csv_files)} and {len(manifest_files)}"
        )
    manifest = json.loads(manifest_files[0].read_text())
    if manifest["results"]["measured_rows"] <= 0:
        raise RuntimeError("native pilot emitted no measured rows")
    return {
        "csv_name": csv_files[0].name,
        "csv_text": csv_files[0].read_text(),
        "manifest_name": manifest_files[0].name,
        "manifest_text": manifest_files[0].read_text(),
        "program_stdout": completed.stdout,
        "program_stderr": completed.stderr,
        "compile_command": build["command"],
        "binary_sha256": build["binary_sha256"],
    }


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite existing artifact: {path}") from error


@app.local_entrypoint()
def main(
    action: str = "plan",
    confirm_spend: bool = False,
    output_dir: str = "data/raw",
    experiment_id: str = "device-dispatch-modal-l4-pilot",
    agent_counts: str = "32,256,2048,16384",
    step_counts: str = "1,8,64",
    warmups: int = 10,
    repetitions: int = 50,
    seed: int = 20260811,
) -> None:
    """Plan locally, compile remotely, or explicitly launch one bounded L4 run."""

    if action == "plan":
        print("action=plan")
        print("remote_calls=0")
        print(f"cuda_image={CUDA_IMAGE}")
        print(f"source_sha256={_source_sha256()}")
        print("compile: modal run scripts/modal_device_dispatch_pilot.py --action compile")
        print(
            "run: modal run scripts/modal_device_dispatch_pilot.py "
            "--action run --confirm-spend"
        )
        return
    if action == "compile":
        result = compile_smoke.remote()
        print("compile_status=ok")
        print(f"binary_sha256={result['binary_sha256']}")
        print(result["help_stdout"], end="")
        return
    if action != "run":
        raise ValueError("action must be one of: plan, compile, run")
    if not confirm_spend:
        raise RuntimeError(
            "run action requires --confirm-spend; no GPU function was invoked"
        )
    if warmups < 0 or repetitions <= 0:
        raise ValueError("warmups must be non-negative and repetitions must be positive")

    result = run_l4.remote(
        experiment_id,
        agent_counts,
        step_counts,
        warmups,
        repetitions,
        seed,
    )
    destination = Path(output_dir)
    csv_path = destination / result["csv_name"]
    manifest_path = destination / result["manifest_name"]
    _write_new(csv_path, result["csv_text"])
    _write_new(manifest_path, result["manifest_text"])
    print(result["program_stdout"], end="")
    if result["program_stderr"]:
        print(result["program_stderr"], end="")
    print(f"downloaded_csv={csv_path.resolve()}")
    print(f"downloaded_manifest={manifest_path.resolve()}")
    print(f"binary_sha256={result['binary_sha256']}")
