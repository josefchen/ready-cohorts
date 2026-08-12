import json
import re
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/gpu-agent-crossover")
REMOTE_TEMPLATE = REMOTE_ROOT / "configs/pilot-006-modal-l4-sub256.toml"
REMOTE_OUTPUT = Path("/tmp/benchmark-output")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("g++")
    .pip_install("torch==2.13.0")
    .add_local_dir(REPO_ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
    .add_local_file(
        REPO_ROOT / "configs/pilot-006-modal-l4-sub256.toml",
        str(REMOTE_TEMPLATE),
        copy=True,
    )
)

app = modal.App("gpu-agent-crossover-hardware-sweep", image=image)


def execute(experiment_id: str) -> dict[str, str]:
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
    config_path = Path("/tmp") / f"{experiment_id}.toml"
    config_path.write_text(config_text)
    csv_path, manifest_path = run(config_path, REMOTE_OUTPUT)
    return {
        "csv_name": csv_path.name,
        "csv_text": csv_path.read_text(),
        "manifest_name": manifest_path.name,
        "manifest_text": manifest_path.read_text(),
    }


@app.function(gpu="T4", cpu=8.0, timeout=30 * 60, single_use_containers=True)
def run_t4() -> dict[str, str]:
    return execute("pilot-007-modal-t4-sub256")


@app.function(gpu="A10", cpu=8.0, timeout=30 * 60, single_use_containers=True)
def run_a10() -> dict[str, str]:
    return execute("pilot-008-modal-a10-sub256")


@app.function(gpu="L40S", cpu=8.0, timeout=30 * 60, single_use_containers=True)
def run_l40s() -> dict[str, str]:
    return execute("pilot-009-modal-l40s-sub256")


@app.function(gpu="A100-80GB", cpu=8.0, timeout=30 * 60, single_use_containers=True)
def run_a100() -> dict[str, str]:
    return execute("pilot-010-modal-a100-80gb-sub256")


@app.function(gpu="H100!", cpu=8.0, timeout=30 * 60, single_use_containers=True)
def run_h100() -> dict[str, str]:
    return execute("pilot-011-modal-h100-sub256")


@app.local_entrypoint()
def main(output_dir: str = "data/raw") -> None:
    jobs = [
        ("T4", run_t4.spawn()),
        ("A10", run_a10.spawn()),
        ("L40S", run_l40s.spawn()),
        ("A100-80GB", run_a100.spawn()),
        ("H100!", run_h100.spawn()),
    ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for requested_gpu, job in jobs:
        try:
            result = job.get()
        except modal.exception.Error as error:  # keep successful ledgers if one GPU class fails
            failures.append(f"{requested_gpu}: {type(error).__name__}: {error}")
            continue
        csv_path = destination / result["csv_name"]
        manifest_path = destination / result["manifest_name"]
        csv_path.write_text(result["csv_text"])
        manifest = json.loads(result["manifest_text"])
        manifest["execution_provider"] = "modal"
        manifest["requested_gpu"] = requested_gpu
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"requested_gpu={requested_gpu}")
        print(f"downloaded_csv={csv_path}")
        print(f"downloaded_manifest={manifest_path}")
    if failures:
        raise RuntimeError("; ".join(failures))
