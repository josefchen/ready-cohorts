"""Build the public, path-redacted Hugging Face evidence mirror."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "release" / "huggingface"
OUTPUT = RELEASE_ROOT / "ready-cohorts"
TEMPLATE = RELEASE_ROOT / "README.template.md"
LOCAL_PREFIX = f"{ROOT.resolve()}/"
MACHINE_PATH = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\\\|file://|localhost)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET = re.compile(
    rb"(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|artifact_token)",
    re.IGNORECASE,
)

FILES: tuple[tuple[str, str, str | None], ...] = (
    ("data/processed/exgentic-tau2-source-manifest.json", "source/source-manifest.json", None),
    ("data/processed/exgentic-tau2-session-summary.csv", "source/session-summary.csv", None),
    ("data/processed/exgentic-tau2-span-features.parquet", "source/span-features.parquet", None),
    ("data/processed/trace-exact-packing-manifest.json", "trace/manifest.json", None),
    ("data/processed/trace-exact-packing-repetitions.csv", "trace/cell-seed-evaluations.csv", None),
    ("data/processed/trace-exact-packing-summary.csv", "trace/summary.csv", None),
    (
        "data/processed/resident-policy-pilot-manifest.json",
        "resident/manifest.json",
        "redact_paths",
    ),
    ("data/processed/resident-policy-pilot-cell-summary.csv", "resident/cell-summary.csv", None),
    ("data/processed/resident-policy-pilot-contrasts.csv", "resident/contrasts.csv", None),
    ("data/processed/native-dispatch-pilot-manifest.json", "native/manifest.json", "redact_paths"),
    ("data/processed/native-dispatch-pilot-cell-summary.csv", "native/cell-summary.csv", None),
    ("data/processed/native-dispatch-pilot-contrasts.csv", "native/contrasts.csv", None),
    ("paper/arxiv/generated/paper-data-manifest.json", "paper/paper-data-manifest.json", None),
    ("paper/governance/claim-evidence-map.csv", "paper/claim-evidence-map.csv", None),
    ("paper/arxiv/main.pdf", "paper/ready-cohorts.pdf", None),
    (
        "results/figures/trace-exact-opportunity-frontier.png",
        "figures/trace-exact-opportunity-frontier.png",
        None,
    ),
    (
        "results/figures/resident-policy-speedup-by-horizon.png",
        "figures/resident-policy-speedup-by-horizon.png",
        None,
    ),
    (
        "results/figures/resident-policy-primary-mechanisms.png",
        "figures/resident-policy-primary-mechanisms.png",
        None,
    ),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def redact_json_paths(data: bytes) -> bytes:
    parsed = json.loads(data.decode("utf-8"))

    def visit(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            return value.replace(LOCAL_PREFIX, "")
        return value

    return (json.dumps(visit(parsed), indent=2, sort_keys=True) + "\n").encode("utf-8")


def scan_public_file(path: Path, data: bytes) -> None:
    if SECRET.search(data):
        raise RuntimeError(f"possible credential in public artifact: {path}")
    if path.suffix.lower() in {".md", ".json", ".csv", ".txt"}:
        text = data.decode("utf-8")
        if EMAIL.search(text):
            raise RuntimeError(f"email address in public artifact: {path}")
        if MACHINE_PATH.search(text):
            raise RuntimeError(f"machine-local path in public artifact: {path}")


def main() -> None:
    staging = Path(tempfile.mkdtemp(prefix="ready-cohorts-hf.", dir=RELEASE_ROOT))
    records: list[dict[str, object]] = []
    try:
        readme = TEMPLATE.read_bytes()
        scan_public_file(Path("README.md"), readme)
        (staging / "README.md").write_bytes(readme)
        os.chmod(staging / "README.md", 0o644)

        for source_name, target_name, transform in FILES:
            source = ROOT / source_name
            if not source.is_file():
                raise RuntimeError(f"missing public-artifact input: {source_name}")
            original = source.read_bytes()
            public = redact_json_paths(original) if transform == "redact_paths" else original
            if transform not in {None, "redact_paths"}:
                raise RuntimeError(f"unknown transform for {source_name}: {transform}")
            target = staging / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            scan_public_file(target, public)
            target.write_bytes(public)
            os.chmod(target, 0o644)
            records.append(
                {
                    "bytes": len(public),
                    "path": target_name,
                    "public_sha256": sha256(public),
                    "source_path": source_name,
                    "source_sha256": sha256(original),
                    "transformation": transform or "none",
                }
            )

        manifest = {
            "files": sorted(records, key=lambda item: str(item["path"])),
            "release_date": "2026-08-12",
            "release_id": "ready-cohorts-public-evidence-v1",
            "schema_version": "ready-cohorts-public-evidence-manifest-v1",
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (staging / "MANIFEST.json").write_bytes(manifest_bytes)
        os.chmod(staging / "MANIFEST.json", 0o644)

        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        os.replace(staging, OUTPUT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    total_bytes = sum(path.stat().st_size for path in OUTPUT.rglob("*") if path.is_file())
    print(f"built {OUTPUT.relative_to(ROOT)}: {len(records) + 2} files, {total_bytes} bytes")


if __name__ == "__main__":
    main()
