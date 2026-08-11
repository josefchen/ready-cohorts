import json
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/gpu-agent-crossover")
REMOTE_CONFIG = REMOTE_ROOT / "configs/pilot-004-modal-l4-compiled.toml"
REMOTE_OUTPUT = Path("/tmp/benchmark-output")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("g++")
    .pip_install("torch==2.13.0")
    .add_local_dir(REPO_ROOT / "src", str(REMOTE_ROOT / "src"), copy=True)
    .add_local_file(
        REPO_ROOT / "configs/pilot-004-modal-l4-compiled.toml",
        str(REMOTE_CONFIG),
        copy=True,
    )
)

app = modal.App("gpu-agent-crossover-pilot-004", image=image)


@app.function(gpu="L4", cpu=8.0, timeout=60 * 60, single_use_containers=True)
def run_l4_pilot() -> dict[str, str]:
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from gpu_agent_crossover.compiled_benchmark import run

    csv_path, manifest_path = run(REMOTE_CONFIG, REMOTE_OUTPUT)
    return {
        "csv_name": csv_path.name,
        "csv_text": csv_path.read_text(),
        "manifest_name": manifest_path.name,
        "manifest_text": manifest_path.read_text(),
    }


@app.local_entrypoint()
def main(output_dir: str = "data/raw") -> None:
    result = run_l4_pilot.remote()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / result["csv_name"]
    manifest_path = destination / result["manifest_name"]
    csv_path.write_text(result["csv_text"])
    manifest = json.loads(result["manifest_text"])
    manifest["execution_provider"] = "modal"
    manifest["requested_gpu"] = "L4"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"downloaded_csv={csv_path}")
    print(f"downloaded_manifest={manifest_path}")
