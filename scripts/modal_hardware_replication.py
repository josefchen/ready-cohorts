import json
import re
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/gpu-agent-crossover")
REMOTE_TEMPLATE = REMOTE_ROOT / "configs/pilot-012-modal-replication.toml"
REMOTE_OUTPUT = Path("/tmp/benchmark-output")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("g++")
    .pip_install("torch==2.13.0")
    .add_local_dir(REPO_ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
    .add_local_file(
        REPO_ROOT / "configs/pilot-012-modal-replication.toml",
        str(REMOTE_TEMPLATE),
        copy=True,
    )
)

app = modal.App("gpu-agent-crossover-hardware-replication", image=image)


def execute(experiment_id: str, seed: int) -> dict[str, str]:
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from gpu_agent_crossover.compiled_benchmark import run

    config_text = REMOTE_TEMPLATE.read_text()
    config_text = re.sub(
        r'^id = "[^"]+"$',
        f'id = "{experiment_id}"',
        config_text,
        count=1,
        flags=re.MULTILINE,
    )
    config_text = re.sub(
        r"^seed = \d+$",
        f"seed = {seed}",
        config_text,
        count=1,
        flags=re.MULTILINE,
    )
    config_path = Path("/tmp") / f"{experiment_id}.toml"
    config_path.write_text(config_text)
    csv_path, manifest_path = run(config_path, REMOTE_OUTPUT)
    return {
        "csv_name": csv_path.name,
        "csv_text": csv_path.read_text(),
        "manifest_name": manifest_path.name,
        "manifest_text": manifest_path.read_text(),
    }


@app.function(gpu="T4", cpu=8.0, timeout=45 * 60, single_use_containers=True)
def run_t4(experiment_id: str, seed: int) -> dict[str, str]:
    return execute(experiment_id, seed)


@app.function(gpu="L4", cpu=8.0, timeout=45 * 60, single_use_containers=True)
def run_l4(experiment_id: str, seed: int) -> dict[str, str]:
    return execute(experiment_id, seed)


@app.function(gpu="A10", cpu=8.0, timeout=45 * 60, single_use_containers=True)
def run_a10(experiment_id: str, seed: int) -> dict[str, str]:
    return execute(experiment_id, seed)


@app.function(gpu="L40S", cpu=8.0, timeout=45 * 60, single_use_containers=True)
def run_l40s(experiment_id: str, seed: int) -> dict[str, str]:
    return execute(experiment_id, seed)


@app.function(gpu="A100-80GB", cpu=8.0, timeout=45 * 60, single_use_containers=True)
def run_a100(experiment_id: str, seed: int) -> dict[str, str]:
    return execute(experiment_id, seed)


@app.function(gpu="H100!", cpu=8.0, timeout=45 * 60, single_use_containers=True)
def run_h100(experiment_id: str, seed: int) -> dict[str, str]:
    return execute(experiment_id, seed)


GPU_JOBS = [
    ("T4", run_t4),
    ("L4", run_l4),
    ("A10", run_a10),
    ("L40S", run_l40s),
    ("A100-80GB", run_a100),
    ("H100!", run_h100),
]


@app.local_entrypoint()
def main(output_dir: str = "data/raw") -> None:
    jobs: list[tuple[str, int, str, modal.functions.FunctionCall]] = []
    pilot = 12
    for requested_gpu, function in GPU_JOBS:
        for placement in range(1, 4):
            experiment_id = (
                f"pilot-{pilot:03d}-modal-{requested_gpu.lower().replace('!', '').replace('-80gb', '')}"
                f"-rep{placement}"
            )
            seed = 20260811 + placement
            jobs.append(
                (
                    requested_gpu,
                    placement,
                    experiment_id,
                    function.spawn(experiment_id, seed),
                )
            )
            pilot += 1

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for requested_gpu, placement, experiment_id, job in jobs:
        try:
            result = job.get()
        except modal.exception.Error as error:
            failures.append(
                f"{requested_gpu}/placement-{placement}: {type(error).__name__}: {error}"
            )
            continue
        csv_path = destination / result["csv_name"]
        manifest_path = destination / result["manifest_name"]
        csv_path.write_text(result["csv_text"])
        manifest = json.loads(result["manifest_text"])
        manifest["execution_provider"] = "modal"
        manifest["requested_gpu"] = requested_gpu
        manifest["placement_replicate"] = placement
        manifest["preregistration"] = "preregistration/pilot-012.md"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"requested_gpu={requested_gpu}")
        print(f"placement_replicate={placement}")
        print(f"experiment_id={experiment_id}")
        print(f"downloaded_csv={csv_path}")
        print(f"downloaded_manifest={manifest_path}")
    if failures:
        raise RuntimeError("; ".join(failures))
