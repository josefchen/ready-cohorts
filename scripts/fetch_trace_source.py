"""Fetch and verify the 19 commit-pinned public trace shards used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "exgentic-tau2-source-manifest.json"
EXPECTED_DATASET = "Exgentic/agent-llm-traces"
EXPECTED_REVISION = "f7c94012d0bfbf66fe4d6ed627699508bbb555ff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing shards without downloading missing files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_record(record: dict[str, Any]) -> tuple[Path, str, int]:
    target = (ROOT / str(record["path"])).resolve()
    external_root = (ROOT / "data" / "external" / "exgentic-agent-llm-traces").resolve()
    if target.parent != external_root:
        raise RuntimeError(f"unexpected target path in source manifest: {target}")

    url = str(record["url"])
    parsed = urllib.parse.urlparse(url)
    expected_prefix = f"/datasets/{EXPECTED_DATASET}/resolve/{EXPECTED_REVISION}/default/train/"
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise RuntimeError(f"unexpected source host: {url}")
    if not parsed.path.startswith(expected_prefix) or Path(parsed.path).name != target.name:
        raise RuntimeError(f"source URL is not pinned to the expected shard: {url}")
    return target, str(record["sha256"]), int(record["bytes"])


def download(url: str, target: Path, expected_sha256: str, expected_bytes: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    written = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ready-cohorts/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if written != expected_bytes or digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"downloaded shard failed its size or SHA-256 gate: {target.name}")
        os.replace(partial, target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("source_dataset") != EXPECTED_DATASET:
        raise RuntimeError("source dataset changed")
    if manifest.get("source_conversion_revision") != EXPECTED_REVISION:
        raise RuntimeError("source conversion revision changed")

    records = manifest.get("source_files")
    if not isinstance(records, list) or len(records) != 19:
        raise RuntimeError("expected exactly 19 source shards")

    downloaded = 0
    verified = 0
    for record in records:
        target, expected_sha256, expected_bytes = validate_source_record(record)
        if target.is_file():
            if target.stat().st_size != expected_bytes or sha256_file(target) != expected_sha256:
                raise RuntimeError(f"existing shard failed verification: {target}")
        else:
            if args.verify_only:
                raise FileNotFoundError(f"missing source shard: {target}")
            download(str(record["url"]), target, expected_sha256, expected_bytes)
            downloaded += 1
        verified += 1

    print(f"verified {verified}/19 pinned trace shards; downloaded {downloaded}")


if __name__ == "__main__":
    main()
