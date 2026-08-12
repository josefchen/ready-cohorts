from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shlex
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Case:
    agent_count: int
    state_width: int
    action_count: int
    mode: str
    threads: int | None = None

    @property
    def case_id(self) -> str:
        raw = (
            f"n={self.agent_count}|w={self.state_width}|a={self.action_count}|"
            f"mode={self.mode}|threads={self.threads}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


class AgentTransition:
    """A static-shape control transition shared by CPU and GPU backends."""

    def __init__(
        self,
        agent_count: int,
        state_width: int,
        action_count: int,
        device: torch.device,
        seed: int,
    ) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        initial_state = torch.randn(
            (agent_count, state_width), generator=generator, dtype=torch.float32
        )
        weights = torch.randn((state_width,), generator=generator, dtype=torch.float32)
        weights /= weights.norm().clamp_min(1.0e-12)
        action_deltas = torch.randn(
            (action_count, state_width), generator=generator, dtype=torch.float32
        )
        action_deltas *= 0.05 / math.sqrt(state_width)
        action_costs = torch.linspace(0.25, 1.0, action_count, dtype=torch.float32)

        if action_count > 1:
            thresholds = torch.linspace(-1.25, 1.25, action_count + 1, dtype=torch.float32)[1:-1]
        else:
            thresholds = torch.empty((0,), dtype=torch.float32)

        self.state = initial_state.to(device)
        self.weights = weights.to(device)
        self.action_deltas = action_deltas.to(device)
        self.action_costs = action_costs.to(device)
        self.thresholds = thresholds.to(device)
        self.budget = torch.full((agent_count,), 100.0, device=device, dtype=torch.float32)

    def step(self) -> torch.Tensor:
        score = (self.state * self.weights).sum(dim=1)
        actions = torch.bucketize(score, self.thresholds)
        deltas = self.action_deltas.index_select(0, actions)
        costs = self.action_costs.index_select(0, actions)
        self.state.mul_(0.9995).add_(deltas, alpha=0.001)
        self.budget.sub_(costs)
        self.budget.add_((self.budget < 0).to(self.budget.dtype) * 100.0)
        return actions

    def checksum(self) -> float:
        return float(self.state[: min(32, self.state.shape[0])].sum().item())


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def nvidia_query() -> list[dict[str, str]]:
    fields = [
        "name",
        "uuid",
        "driver_version",
        "memory.total",
        "power.limit",
        "temperature.gpu",
    ]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        rows.append(dict(zip(fields, values, strict=False)))
    return rows


def cpu_query() -> dict[str, Any]:
    processor: dict[str, str] = {}
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(errors="replace").splitlines():
            if not line.strip() and processor:
                break
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if key in {
                "vendor_id",
                "model name",
                "cpu family",
                "model",
                "stepping",
                "microcode",
            }:
                processor[key.replace(" ", "_")] = value

    governors: set[str] = set()
    for governor_path in Path("/sys/devices/system/cpu").glob(
        "cpu[0-9]*/cpufreq/scaling_governor"
    ):
        try:
            governors.add(governor_path.read_text().strip())
        except OSError:
            continue
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    return {
        **processor,
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "process_affinity_cpus": affinity,
        "frequency_governors": sorted(governors),
        "torch_num_threads_at_manifest": torch.get_num_threads(),
        "torch_num_interop_threads_at_manifest": torch.get_num_interop_threads(),
    }


def hardware_manifest(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                    "multiprocessor_count": properties.multi_processor_count,
                }
            )
    return {
        "created_at": utc_now(),
        "argv": [sys.executable, *sys.argv],
        "command_shell_escaped": shlex.join([sys.executable, *sys.argv]),
        "git_revision": git_revision(repo_root),
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "cpu": cpu_query(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_devices": cuda_devices,
        "nvidia_smi": nvidia_query(),
        "config": config,
    }


def correctness_check(
    case: Case,
    seed: int,
    steps: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "correctness_valid": None,
            "correctness_action_match": None,
            "correctness_max_abs_error": None,
            "correctness_max_rel_error": None,
        }

    cpu = AgentTransition(
        case.agent_count, case.state_width, case.action_count, torch.device("cpu"), seed
    )
    gpu = AgentTransition(
        case.agent_count, case.state_width, case.action_count, torch.device("cuda"), seed
    )
    cpu_actions = None
    gpu_actions = None
    with torch.inference_mode():
        for _ in range(steps):
            cpu_actions = cpu.step()
            gpu_actions = gpu.step()
    torch.cuda.synchronize()
    assert cpu_actions is not None and gpu_actions is not None
    gpu_state = gpu.state.cpu()
    absolute = (cpu.state - gpu_state).abs()
    relative = absolute / cpu.state.abs().clamp_min(atol)
    max_abs = float(absolute.max().item())
    max_rel = float(relative.max().item())
    action_match = bool(torch.equal(cpu_actions, gpu_actions.cpu()))
    valid = action_match and bool(torch.allclose(cpu.state, gpu_state, atol=atol, rtol=rtol))
    return {
        "correctness_valid": valid,
        "correctness_action_match": action_match,
        "correctness_max_abs_error": max_abs,
        "correctness_max_rel_error": max_rel,
    }


def timed_cpu(
    case: Case,
    seed: int,
    steps: int,
    warmup_steps: int,
    repetitions: int,
) -> Iterable[dict[str, Any]]:
    assert case.threads is not None
    torch.set_num_threads(case.threads)
    transition = AgentTransition(
        case.agent_count,
        case.state_width,
        case.action_count,
        torch.device("cpu"),
        seed,
    )
    with torch.inference_mode():
        for _ in range(warmup_steps):
            transition.step()
        for repetition in range(repetitions):
            started = time.perf_counter_ns()
            for _ in range(steps):
                transition.step()
            elapsed_ns = time.perf_counter_ns() - started
            yield {
                "repetition": repetition,
                "wall_ms": elapsed_ns / 1.0e6,
                "device_ms": None,
                "checksum": transition.checksum(),
            }


def _warm_cuda(transition: AgentTransition, warmup_steps: int) -> None:
    with torch.inference_mode():
        for _ in range(warmup_steps):
            transition.step()
    torch.cuda.synchronize()


def timed_gpu_eager(
    case: Case,
    seed: int,
    steps: int,
    warmup_steps: int,
    repetitions: int,
    host_visible: bool,
) -> Iterable[dict[str, Any]]:
    transition = AgentTransition(
        case.agent_count,
        case.state_width,
        case.action_count,
        torch.device("cuda"),
        seed,
    )
    _warm_cuda(transition, warmup_steps)
    with torch.inference_mode():
        for repetition in range(repetitions):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            start_event.record()
            for _ in range(steps):
                actions = transition.step()
                if host_visible:
                    actions.cpu()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ns = time.perf_counter_ns() - started
            yield {
                "repetition": repetition,
                "wall_ms": elapsed_ns / 1.0e6,
                "device_ms": float(start_event.elapsed_time(end_event)),
                "checksum": transition.checksum(),
            }


def timed_gpu_graph(
    case: Case,
    seed: int,
    steps: int,
    warmup_steps: int,
    repetitions: int,
) -> Iterable[dict[str, Any]]:
    transition = AgentTransition(
        case.agent_count,
        case.state_width,
        case.action_count,
        torch.device("cuda"),
        seed,
    )
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.inference_mode():
        for _ in range(warmup_steps):
            transition.step()
    capture_stream.synchronize()
    torch.cuda.current_stream().wait_stream(capture_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        for _ in range(steps):
            transition.step()

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    with torch.inference_mode():
        for repetition in range(repetitions):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            start_event.record()
            graph.replay()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ns = time.perf_counter_ns() - started
            yield {
                "repetition": repetition,
                "wall_ms": elapsed_ns / 1.0e6,
                "device_ms": float(start_event.elapsed_time(end_event)),
                "checksum": transition.checksum(),
            }


def make_cases(config: dict[str, Any]) -> list[Case]:
    experiment = config["experiment"]
    cases: list[Case] = []
    for agent_count in experiment["agent_counts"]:
        for state_width in experiment["state_widths"]:
            for action_count in experiment["action_counts"]:
                for threads in experiment["cpu_threads"]:
                    cases.append(Case(agent_count, state_width, action_count, "cpu", int(threads)))
                for mode in experiment["gpu_modes"]:
                    cases.append(Case(agent_count, state_width, action_count, str(mode)))
    if experiment.get("randomize_case_order", True):
        random.Random(int(experiment["seed"])).shuffle(cases)
    return cases


FIELDNAMES = [
    "experiment_id",
    "run_id",
    "recorded_at",
    "case_id",
    "status",
    "error_type",
    "error_message",
    "mode",
    "backend",
    "threads",
    "agent_count",
    "state_width",
    "action_count",
    "steps",
    "repetition",
    "wall_ms",
    "device_ms",
    "agent_steps_per_second",
    "ns_per_agent_step",
    "checksum",
    "correctness_valid",
    "correctness_action_match",
    "correctness_max_abs_error",
    "correctness_max_rel_error",
]


def normalized_error(error: Exception) -> tuple[str, str]:
    message = " ".join(str(error).split())
    return type(error).__name__, message[:500]


def run(config_path: Path, output_dir: Path) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    experiment = config["experiment"]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{experiment['id']}-{run_id}.csv"
    manifest_path = output_dir / f"{experiment['id']}-{run_id}.manifest.json"
    manifest = hardware_manifest(repo_root, config)
    manifest["run_id"] = run_id
    manifest["config_path"] = str(config_path.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    cases = make_cases(config)
    correctness_cache: dict[tuple[int, int, int], dict[str, Any]] = {}
    print(f"run_id={run_id} cases={len(cases)} cuda={torch.cuda.is_available()}", flush=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for case_index, case in enumerate(cases, start=1):
            print(
                f"[{case_index}/{len(cases)}] {case.mode} n={case.agent_count} "
                f"w={case.state_width} a={case.action_count} threads={case.threads}",
                flush=True,
            )
            correctness_key = (case.agent_count, case.state_width, case.action_count)
            if correctness_key not in correctness_cache:
                correctness_cache[correctness_key] = correctness_check(
                    case,
                    int(experiment["seed"]),
                    int(experiment["correctness_steps"]),
                    float(experiment["absolute_tolerance"]),
                    float(experiment["relative_tolerance"]),
                )
            correctness = correctness_cache[correctness_key]
            base = {
                "experiment_id": experiment["id"],
                "run_id": run_id,
                "recorded_at": utc_now(),
                "case_id": case.case_id,
                "mode": case.mode,
                "backend": "cpu" if case.mode == "cpu" else "cuda",
                "threads": case.threads,
                "agent_count": case.agent_count,
                "state_width": case.state_width,
                "action_count": case.action_count,
                "steps": int(experiment["steps"]),
                **correctness,
            }
            try:
                if case.mode == "cpu":
                    observations = timed_cpu(
                        case,
                        int(experiment["seed"]),
                        int(experiment["steps"]),
                        int(experiment["warmup_steps"]),
                        int(experiment["repetitions"]),
                    )
                elif case.mode == "eager-resident":
                    observations = timed_gpu_eager(
                        case,
                        int(experiment["seed"]),
                        int(experiment["steps"]),
                        int(experiment["warmup_steps"]),
                        int(experiment["repetitions"]),
                        host_visible=False,
                    )
                elif case.mode == "eager-host-visible":
                    observations = timed_gpu_eager(
                        case,
                        int(experiment["seed"]),
                        int(experiment["steps"]),
                        int(experiment["warmup_steps"]),
                        int(experiment["repetitions"]),
                        host_visible=True,
                    )
                elif case.mode == "graph-resident":
                    observations = timed_gpu_graph(
                        case,
                        int(experiment["seed"]),
                        int(experiment["steps"]),
                        int(experiment["warmup_steps"]),
                        int(experiment["repetitions"]),
                    )
                else:
                    raise ValueError(f"unknown mode: {case.mode}")

                for observation in observations:
                    wall_ms = float(observation["wall_ms"])
                    agent_steps = case.agent_count * int(experiment["steps"])
                    row = {
                        **base,
                        **observation,
                        "status": "ok",
                        "error_type": None,
                        "error_message": None,
                        "agent_steps_per_second": agent_steps / (wall_ms / 1000.0),
                        "ns_per_agent_step": wall_ms * 1.0e6 / agent_steps,
                    }
                    writer.writerow(row)
                    handle.flush()
            except Exception as error:  # noqa: BLE001 -- every failed cell enters the ledger
                error_type, error_message = normalized_error(error)
                writer.writerow(
                    {
                        **base,
                        "status": "error",
                        "error_type": error_type,
                        "error_message": error_message,
                    }
                )
                handle.flush()
                print(f"  ERROR {error_type}: {error_message}", flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

    print(f"raw_csv={csv_path}", flush=True)
    print(f"manifest={manifest_path}", flush=True)
    return csv_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
