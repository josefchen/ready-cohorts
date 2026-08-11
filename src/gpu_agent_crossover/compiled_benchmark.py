from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from gpu_agent_crossover.benchmark import (
    AgentTransition,
    hardware_manifest,
    normalized_error,
    utc_now,
)

TensorTuple = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
RolloutOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Rollout = Callable[..., RolloutOutput]


@dataclass(frozen=True)
class CompiledCase:
    agent_count: int
    state_width: int
    action_count: int
    observation_horizon: int
    mode: str
    threads: int | None = None

    @property
    def backend(self) -> str:
        return "cpu" if self.mode == "compiled-cpu" else "cuda"

    @property
    def host_visible(self) -> bool:
        return self.mode == "compiled-gpu-host-visible"

    @property
    def case_id(self) -> str:
        raw = (
            f"n={self.agent_count}|w={self.state_width}|a={self.action_count}|"
            f"h={self.observation_horizon}|mode={self.mode}|threads={self.threads}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def make_rollout(observation_horizon: int) -> Rollout:
    if observation_horizon < 1:
        raise ValueError("observation_horizon must be positive")

    def rollout(
        state: torch.Tensor,
        budget: torch.Tensor,
        weights: torch.Tensor,
        action_deltas: torch.Tensor,
        action_costs: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> RolloutOutput:
        actions = torch.empty((state.shape[0],), dtype=torch.int64, device=state.device)
        for _ in range(observation_horizon):
            score = (state * weights).sum(dim=1)
            actions = torch.bucketize(score, thresholds)
            deltas = action_deltas.index_select(0, actions)
            costs = action_costs.index_select(0, actions)
            state = state * 0.9995 + deltas * 0.001
            budget = budget - costs
            budget = budget + (budget < 0).to(budget.dtype) * 100.0
        return state, budget, actions

    return rollout


def initial_inputs(case: CompiledCase, device: torch.device, seed: int) -> TensorTuple:
    transition = AgentTransition(
        case.agent_count,
        case.state_width,
        case.action_count,
        torch.device("cpu"),
        seed,
    )
    return (
        transition.state.to(device),
        transition.budget.to(device),
        transition.weights.to(device),
        transition.action_deltas.to(device),
        transition.action_costs.to(device),
        transition.thresholds.to(device),
    )


class ProgramCache:
    def __init__(self, compile_threads: int) -> None:
        self.compile_threads = compile_threads
        self.programs: dict[tuple[Any, ...], Rollout] = {}
        self.compile_ms: dict[tuple[Any, ...], float] = {}

    @staticmethod
    def key(case: CompiledCase, device: torch.device) -> tuple[Any, ...]:
        return (
            device.type,
            case.agent_count,
            case.state_width,
            case.action_count,
            case.observation_horizon,
        )

    def get(
        self,
        case: CompiledCase,
        device: torch.device,
        inputs: TensorTuple,
    ) -> tuple[Rollout, float]:
        key = self.key(case, device)
        if key in self.programs:
            return self.programs[key], self.compile_ms[key]

        previous_threads = torch.get_num_threads()
        if device.type == "cpu":
            torch.set_num_threads(self.compile_threads)
        program = torch.compile(
            make_rollout(case.observation_horizon),
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        with torch.inference_mode():
            program(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
        if device.type == "cpu":
            torch.set_num_threads(previous_threads)
        self.programs[key] = program
        self.compile_ms[key] = elapsed_ms
        print(
            f"  compiled device={device.type} n={case.agent_count} "
            f"w={case.state_width} a={case.action_count} "
            f"h={case.observation_horizon} first_call_ms={elapsed_ms:.3f}",
            flush=True,
        )
        return program, elapsed_ms


def make_cases(config: dict[str, Any]) -> list[CompiledCase]:
    experiment = config["experiment"]
    cases: list[CompiledCase] = []
    for agent_count in experiment["agent_counts"]:
        for state_width in experiment["state_widths"]:
            for action_count in experiment["action_counts"]:
                for observation_horizon in experiment["observation_horizons"]:
                    for threads in experiment["cpu_threads"]:
                        cases.append(
                            CompiledCase(
                                int(agent_count),
                                int(state_width),
                                int(action_count),
                                int(observation_horizon),
                                "compiled-cpu",
                                int(threads),
                            )
                        )
                    for visibility in experiment["gpu_visibility"]:
                        cases.append(
                            CompiledCase(
                                int(agent_count),
                                int(state_width),
                                int(action_count),
                                int(observation_horizon),
                                f"compiled-gpu-{visibility}",
                            )
                        )
    if experiment.get("randomize_case_order", True):
        random.Random(int(experiment["seed"])).shuffle(cases)
    return cases


def run_chunks(
    program: Rollout,
    inputs: TensorTuple,
    chunks: int,
    host_visible: bool,
) -> tuple[RolloutOutput, torch.Tensor | None]:
    state, budget, weights, action_deltas, action_costs, thresholds = inputs
    observed_actions = None
    for _ in range(chunks):
        state, budget, actions = program(
            state,
            budget,
            weights,
            action_deltas,
            action_costs,
            thresholds,
        )
        if host_visible:
            observed_actions = actions.cpu()
    return (state, budget, actions), observed_actions


def correctness_check(
    case: CompiledCase,
    cache: ProgramCache,
    seed: int,
    total_steps: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "correctness_valid": None,
            "correctness_action_match": None,
            "correctness_state_max_abs_error": None,
            "correctness_state_max_rel_error": None,
            "correctness_budget_max_abs_error": None,
        }
    if total_steps % case.observation_horizon:
        raise ValueError("total_steps must be divisible by observation_horizon")

    chunks = total_steps // case.observation_horizon
    cpu_inputs = initial_inputs(case, torch.device("cpu"), seed)
    gpu_inputs = initial_inputs(case, torch.device("cuda"), seed)
    cpu_program, _ = cache.get(case, torch.device("cpu"), cpu_inputs)
    gpu_program, _ = cache.get(case, torch.device("cuda"), gpu_inputs)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(cache.compile_threads)
    with torch.inference_mode():
        cpu_output, _ = run_chunks(cpu_program, cpu_inputs, chunks, host_visible=False)
        gpu_output, _ = run_chunks(gpu_program, gpu_inputs, chunks, host_visible=False)
    torch.cuda.synchronize()
    torch.set_num_threads(previous_threads)

    cpu_state, cpu_budget, cpu_actions = cpu_output
    gpu_state, gpu_budget, gpu_actions = (value.cpu() for value in gpu_output)
    state_absolute = (cpu_state - gpu_state).abs()
    state_relative = state_absolute / cpu_state.abs().clamp_min(atol)
    budget_absolute = (cpu_budget - gpu_budget).abs()
    action_match = bool(torch.equal(cpu_actions, gpu_actions))
    state_valid = bool(torch.allclose(cpu_state, gpu_state, atol=atol, rtol=rtol))
    budget_valid = bool(torch.allclose(cpu_budget, gpu_budget, atol=atol, rtol=rtol))
    return {
        "correctness_valid": action_match and state_valid and budget_valid,
        "correctness_action_match": action_match,
        "correctness_state_max_abs_error": float(state_absolute.max().item()),
        "correctness_state_max_rel_error": float(state_relative.max().item()),
        "correctness_budget_max_abs_error": float(budget_absolute.max().item()),
    }


def timed_cpu(
    case: CompiledCase,
    cache: ProgramCache,
    seed: int,
    total_steps: int,
    warmup_rollouts: int,
    repetitions: int,
) -> tuple[Iterable[dict[str, Any]], float]:
    assert case.threads is not None
    torch.set_num_threads(case.threads)
    inputs = initial_inputs(case, torch.device("cpu"), seed)
    program, compile_ms = cache.get(case, torch.device("cpu"), inputs)
    chunks = total_steps // case.observation_horizon

    def observations() -> Iterable[dict[str, Any]]:
        nonlocal inputs
        with torch.inference_mode():
            for _ in range(warmup_rollouts):
                output, _ = run_chunks(program, inputs, chunks, host_visible=False)
                inputs = (*output[:2], *inputs[2:])
            for repetition in range(repetitions):
                started = time.perf_counter_ns()
                output, _ = run_chunks(program, inputs, chunks, host_visible=False)
                elapsed_ns = time.perf_counter_ns() - started
                inputs = (*output[:2], *inputs[2:])
                yield {
                    "repetition": repetition,
                    "wall_ms": elapsed_ns / 1.0e6,
                    "device_ms": None,
                    "checksum": float(output[0][: min(32, case.agent_count)].sum().item()),
                }

    return observations(), compile_ms


def timed_gpu(
    case: CompiledCase,
    cache: ProgramCache,
    seed: int,
    total_steps: int,
    warmup_rollouts: int,
    repetitions: int,
) -> tuple[Iterable[dict[str, Any]], float]:
    inputs = initial_inputs(case, torch.device("cuda"), seed)
    program, compile_ms = cache.get(case, torch.device("cuda"), inputs)
    chunks = total_steps // case.observation_horizon

    def observations() -> Iterable[dict[str, Any]]:
        nonlocal inputs
        with torch.inference_mode():
            for _ in range(warmup_rollouts):
                output, _ = run_chunks(program, inputs, chunks, case.host_visible)
                inputs = (*output[:2], *inputs[2:])
            torch.cuda.synchronize()
            for repetition in range(repetitions):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                start_event.record()
                output, observed_actions = run_chunks(
                    program,
                    inputs,
                    chunks,
                    case.host_visible,
                )
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ns = time.perf_counter_ns() - started
                inputs = (*output[:2], *inputs[2:])
                checksum = float(output[0][: min(32, case.agent_count)].sum().item())
                if observed_actions is not None:
                    checksum += float(observed_actions[: min(32, case.agent_count)].sum().item())
                yield {
                    "repetition": repetition,
                    "wall_ms": elapsed_ns / 1.0e6,
                    "device_ms": float(start_event.elapsed_time(end_event)),
                    "checksum": checksum,
                }

    return observations(), compile_ms


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
    "total_steps",
    "observation_horizon",
    "host_visible",
    "compiler_backend",
    "compile_first_call_ms",
    "repetition",
    "wall_ms",
    "device_ms",
    "agent_steps_per_second",
    "ns_per_agent_step",
    "checksum",
    "correctness_valid",
    "correctness_action_match",
    "correctness_state_max_abs_error",
    "correctness_state_max_rel_error",
    "correctness_budget_max_abs_error",
]


def run(config_path: Path, output_dir: Path) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    experiment = config["experiment"]
    recompile_limit = int(experiment["recompile_limit"])
    torch._dynamo.config.recompile_limit = recompile_limit
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        torch._dynamo.config.accumulated_recompile_limit = recompile_limit
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{experiment['id']}-{run_id}.csv"
    manifest_path = output_dir / f"{experiment['id']}-{run_id}.manifest.json"
    compile_cache_dir = tempfile.mkdtemp(prefix=f"gpu-agent-inductor-{run_id}-")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = compile_cache_dir
    manifest = hardware_manifest(repo_root, config)
    manifest.update(
        {
            "run_id": run_id,
            "config_path": str(config_path.resolve()),
            "compiler_backend": "inductor",
            "torch_compile": {
                "fullgraph": True,
                "dynamic": False,
                "cache_directory": compile_cache_dir,
                "recompile_limit": recompile_limit,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if not torch.cuda.is_available():
        raise RuntimeError("compiled crossover pilot requires CUDA")
    total_steps = int(experiment["total_steps"])
    cases = make_cases(config)
    cache = ProgramCache(int(experiment["compile_threads"]))
    correctness_cache: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    print(f"run_id={run_id} cases={len(cases)} cuda=True", flush=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for case_index, case in enumerate(cases, start=1):
            print(
                f"[{case_index}/{len(cases)}] {case.mode} n={case.agent_count} "
                f"w={case.state_width} a={case.action_count} "
                f"h={case.observation_horizon} threads={case.threads}",
                flush=True,
            )
            correctness_key = (
                case.agent_count,
                case.state_width,
                case.action_count,
                case.observation_horizon,
            )
            base = {
                "experiment_id": experiment["id"],
                "run_id": run_id,
                "recorded_at": utc_now(),
                "case_id": case.case_id,
                "mode": case.mode,
                "backend": case.backend,
                "threads": case.threads,
                "agent_count": case.agent_count,
                "state_width": case.state_width,
                "action_count": case.action_count,
                "total_steps": total_steps,
                "observation_horizon": case.observation_horizon,
                "host_visible": case.host_visible,
                "compiler_backend": "inductor",
            }
            try:
                if total_steps % case.observation_horizon:
                    raise ValueError("total_steps must be divisible by observation_horizon")
                if correctness_key not in correctness_cache:
                    correctness_cache[correctness_key] = correctness_check(
                        case,
                        cache,
                        int(experiment["seed"]),
                        total_steps,
                        float(experiment["absolute_tolerance"]),
                        float(experiment["relative_tolerance"]),
                    )
                correctness = correctness_cache[correctness_key]
                if case.backend == "cpu":
                    observations, compile_ms = timed_cpu(
                        case,
                        cache,
                        int(experiment["seed"]),
                        total_steps,
                        int(experiment["warmup_rollouts"]),
                        int(experiment["repetitions"]),
                    )
                else:
                    observations, compile_ms = timed_gpu(
                        case,
                        cache,
                        int(experiment["seed"]),
                        total_steps,
                        int(experiment["warmup_rollouts"]),
                        int(experiment["repetitions"]),
                    )
                for observation in observations:
                    wall_ms = float(observation["wall_ms"])
                    agent_steps = case.agent_count * total_steps
                    writer.writerow(
                        {
                            **base,
                            **correctness,
                            **observation,
                            "status": "ok",
                            "error_type": None,
                            "error_message": None,
                            "compile_first_call_ms": compile_ms,
                            "agent_steps_per_second": agent_steps / (wall_ms / 1000.0),
                            "ns_per_agent_step": wall_ms * 1.0e6 / agent_steps,
                        }
                    )
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
