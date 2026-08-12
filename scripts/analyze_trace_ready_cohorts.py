from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_agent_crossover.ready_cohort import cohort_metrics, simulate_stationary_swarm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--span-path",
        type=Path,
        default=Path("data/processed/exgentic-tau2-span-features.parquet"),
    )
    parser.add_argument(
        "--session-path",
        type=Path,
        default=Path("data/processed/exgentic-tau2-session-summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--active-sessions", nargs="+", type=int, default=[100, 1000, 10000, 100000]
    )
    parser.add_argument(
        "--windows-ms",
        nargs="+",
        type=float,
        default=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256, 512, 1024, 4096],
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--horizon-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spans = pd.read_parquet(args.span_path)
    sessions = pd.read_csv(args.session_path)
    rows: list[dict[str, object]] = []
    grouping_names = ("pooled", "event_class", "route_key")

    for active_sessions in args.active_sessions:
        for repetition in range(args.repetitions):
            seed_sequence = np.random.SeedSequence([args.seed, active_sessions, repetition])
            rng = np.random.default_rng(seed_sequence)
            replay = simulate_stationary_swarm(
                spans,
                sessions,
                target_active_sessions=active_sessions,
                horizon_s=args.horizon_s,
                rng=rng,
            )
            grouping_ids = {
                "pooled": np.zeros(len(replay.event_times_s), dtype=np.int32),
                "event_class": replay.event_class_ids,
                "route_key": replay.route_ids,
            }
            for window_ms in args.windows_ms:
                cell_rows: dict[str, dict[str, object]] = {}
                for grouping in grouping_names:
                    metrics = cohort_metrics(
                        replay.event_times_s,
                        grouping_ids[grouping],
                        window_ms=window_ms,
                        thresholds=args.thresholds,
                    )
                    row = {
                        "target_active_sessions": active_sessions,
                        "repetition": repetition,
                        "window_ms": window_ms,
                        "grouping": grouping,
                        "horizon_s": args.horizon_s,
                        "mean_active_sessions": replay.mean_active_sessions,
                        "active_population_ratio": replay.mean_active_sessions / active_sessions,
                        "arrival_count": replay.arrival_count,
                        "event_rate_hz": len(replay.event_times_s) / args.horizon_s,
                        **metrics,
                    }
                    cell_rows[grouping] = row
                    rows.append(row)

                for threshold in args.thresholds:
                    column = f"eligible_share_k{threshold}"
                    pooled = float(cell_rows["pooled"][column])
                    coarse = float(cell_rows["event_class"][column])
                    exact = float(cell_rows["route_key"][column])
                    if pooled + 1e-12 < coarse or coarse + 1e-12 < exact:
                        raise AssertionError(
                            "grouping hierarchy violated at "
                            f"C={active_sessions}, repetition={repetition}, "
                            f"window_ms={window_ms}, K={threshold}: "
                            f"{pooled}, {coarse}, {exact}"
                        )
            print(
                f"active_sessions={active_sessions} repetition={repetition} "
                f"events={len(replay.event_times_s)} "
                f"realized_active={replay.mean_active_sessions:.1f}",
                flush=True,
            )

    wide = pd.DataFrame(rows).sort_values(
        ["target_active_sessions", "repetition", "window_ms", "grouping"]
    )
    id_columns = [
        "target_active_sessions",
        "repetition",
        "window_ms",
        "grouping",
        "horizon_s",
        "mean_active_sessions",
        "active_population_ratio",
        "arrival_count",
        "event_rate_hz",
        "event_count",
        "cohort_count",
        "cohort_size_event_mean",
        "cohort_size_event_p50",
        "cohort_size_event_p90",
        "cohort_size_event_p99",
        "cohort_size_max",
        "wait_ms_mean",
        "wait_ms_p95",
    ]
    long = wide.melt(
        id_vars=id_columns,
        value_vars=[f"eligible_share_k{value}" for value in args.thresholds],
        var_name="threshold_label",
        value_name="eligible_share",
    )
    long["threshold_k"] = long["threshold_label"].str.removeprefix(
        "eligible_share_k"
    ).astype(int)
    long = long.drop(columns="threshold_label").sort_values(
        [
            "target_active_sessions",
            "repetition",
            "window_ms",
            "grouping",
            "threshold_k",
        ]
    )
    group_columns = ["target_active_sessions", "window_ms", "grouping", "threshold_k"]
    summary = (
        long.groupby(group_columns, as_index=False)
        .agg(
            repetitions=("repetition", "nunique"),
            eligible_share_mean=("eligible_share", "mean"),
            eligible_share_std=("eligible_share", "std"),
            eligible_share_min=("eligible_share", "min"),
            eligible_share_max=("eligible_share", "max"),
            cohort_size_event_mean=("cohort_size_event_mean", "mean"),
            cohort_size_event_p50=("cohort_size_event_p50", "mean"),
            cohort_size_event_p90=("cohort_size_event_p90", "mean"),
            cohort_size_event_p99=("cohort_size_event_p99", "mean"),
            cohort_size_max=("cohort_size_max", "max"),
            wait_ms_mean=("wait_ms_mean", "mean"),
            wait_ms_p95=("wait_ms_p95", "mean"),
            mean_active_sessions=("mean_active_sessions", "mean"),
            active_population_ratio=("active_population_ratio", "mean"),
            event_rate_hz=("event_rate_hz", "mean"),
        )
        .sort_values(group_columns)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repetitions_path = args.output_dir / "trace-ready-cohort-repetitions.csv"
    summary_path = args.output_dir / "trace-ready-cohort-summary.csv"
    manifest_path = args.output_dir / "trace-ready-cohort-manifest.json"
    long.to_csv(repetitions_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "preregistration": "preregistration/trace-replay-001.md",
        "source_files": {
            "span_features": {
                "path": str(args.span_path),
                "sha256": sha256_file(args.span_path),
            },
            "session_summary": {
                "path": str(args.session_path),
                "sha256": sha256_file(args.session_path),
            },
        },
        "configuration": {
            "active_sessions": args.active_sessions,
            "windows_ms": args.windows_ms,
            "thresholds": args.thresholds,
            "repetitions": args.repetitions,
            "horizon_s": args.horizon_s,
            "seed": args.seed,
            "arrival_model": "stationary homogeneous Poisson",
            "template_sampling": "uniform over empirical sessions",
            "event_time": "recorded span completion offset",
            "groupings": list(grouping_names),
        },
        "quality": {
            "input_sessions": int(sessions["session_id"].nunique()),
            "input_events": len(spans),
            "output_repetition_rows": len(long),
            "output_summary_rows": len(summary),
            "active_population_ratio_min": float(long["active_population_ratio"].min()),
            "active_population_ratio_max": float(long["active_population_ratio"].max()),
            "grouping_hierarchy_checked": True,
        },
        "outputs": {
            "repetitions": {
                "path": str(repetitions_path),
                "sha256": sha256_file(repetitions_path),
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"repetitions={repetitions_path} rows={len(long)}")
    print(f"summary={summary_path} rows={len(summary)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
