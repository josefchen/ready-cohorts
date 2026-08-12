from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gpu_agent_crossover.trace_features import extract_session_features

DEFAULT_DATASET = "Exgentic/agent-llm-traces"
DEFAULT_REVISION = "70036b93a04e61b0ea2706a68b962f4f26774587"
DEFAULT_CONVERSION_REVISION = "f7c94012d0bfbf66fe4d6ed627699508bbb555ff"
DEFAULT_BENCHMARKS = ("tau2_airline", "tau2_retail", "tau2_telecom")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/external/exgentic-agent-llm-traces"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--conversion-revision", default=DEFAULT_CONVERSION_REVISION)
    parser.add_argument("--license", default="cdla-permissive-2.0")
    parser.add_argument("--benchmarks", nargs="+", default=list(DEFAULT_BENCHMARKS))
    parser.add_argument("--expected-sessions", type=int, default=851)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("*.parquet"))
    if not input_paths:
        raise FileNotFoundError(f"no parquet inputs under {args.input_dir}")

    selected = set(args.benchmarks)
    span_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    source_files: list[dict[str, Any]] = []
    for input_path in input_paths:
        source_files.append(
            {
                "path": str(input_path),
                "bytes": input_path.stat().st_size,
                "sha256": sha256_file(input_path),
                "url": (
                    f"https://huggingface.co/datasets/{args.dataset}/resolve/"
                    f"{args.conversion_revision}/default/train/{input_path.name}"
                ),
            }
        )
        table = pq.read_table(
            input_path,
            columns=["harness", "benchmark", "models", "session_id", "spans", "collected_at"],
        )
        for source_row, row in enumerate(table.to_pylist()):
            if row["benchmark"] not in selected:
                continue
            session_id = str(row["session_id"])
            if session_id in seen_sessions:
                raise ValueError(f"duplicate session_id: {session_id}")
            seen_sessions.add(session_id)
            extracted_spans, extracted_session = extract_session_features(
                row,
                source_file=input_path.name,
                source_row=source_row,
            )
            span_rows.extend(extracted_spans)
            session_rows.append(extracted_session)

    if args.expected_sessions is not None and len(session_rows) != args.expected_sessions:
        raise ValueError(
            f"expected {args.expected_sessions} selected sessions, observed {len(session_rows)}"
        )

    span_frame = pd.DataFrame(span_rows).sort_values(
        ["benchmark", "harness", "session_id", "span_index"]
    )
    session_frame = pd.DataFrame(session_rows).sort_values(["benchmark", "harness", "session_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "exgentic-tau2"
    span_path = args.output_dir / f"{stem}-span-features.parquet"
    session_path = args.output_dir / f"{stem}-session-summary.csv"
    manifest_path = args.output_dir / f"{stem}-source-manifest.json"
    pq.write_table(
        pa.Table.from_pandas(span_frame, preserve_index=False),
        span_path,
        compression="zstd",
    )
    session_frame.to_csv(session_path, index=False)

    source_fingerprint = hashlib.sha256(
        "".join(item["sha256"] for item in source_files).encode()
    ).hexdigest()
    quality = {
        "session_rows": len(session_frame),
        "span_rows": len(span_frame),
        "duplicate_session_ids": int(session_frame["session_id"].duplicated().sum()),
        "benchmarks": Counter(session_frame["benchmark"]).most_common(),
        "harnesses": Counter(session_frame["harness"]).most_common(),
        "invalid_json_fields": int(session_frame["invalid_json_fields"].sum()),
        "nonpositive_duration_spans": int(session_frame["nonpositive_duration_spans"].sum()),
        "failed_status_spans": int(session_frame["failed_status_spans"].sum()),
        "overlapping_span_starts": int(session_frame["overlapping_span_starts"].sum()),
        "required_nulls": {
            column: int(span_frame[column].isna().sum())
            for column in ("session_id", "benchmark", "harness", "start_time", "route_key")
        },
    }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_dataset": args.dataset,
        "source_revision": args.revision,
        "source_conversion_revision": args.conversion_revision,
        "source_license": args.license,
        "selected_benchmarks": sorted(selected),
        "source_files": source_files,
        "source_fingerprint_sha256": source_fingerprint,
        "content_policy": (
            "Derived outputs retain timestamps, counts, lengths, route labels, and public IDs only; "
            "prompt text, tool arguments, and tool results are not emitted."
        ),
        "quality": quality,
        "outputs": {
            "span_features": {
                "path": str(span_path),
                "bytes": span_path.stat().st_size,
                "sha256": sha256_file(span_path),
            },
            "session_summary": {
                "path": str(session_path),
                "bytes": session_path.stat().st_size,
                "sha256": sha256_file(session_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"span_features={span_path} rows={len(span_frame)}")
    print(f"session_summary={session_path} rows={len(session_frame)}")
    print(f"manifest={manifest_path}")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
