from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_agent_crossover.ready_cohort import (
    cohort_metrics,
    simulate_stationary_swarm,
    sliding_deadline_local_metrics,
)


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
        "--active-sessions", nargs="+", type=int, default=[1000, 10000, 100000]
    )
    parser.add_argument(
        "--deadlines-ms", nargs="+", type=float, default=[10, 25, 50, 100, 250]
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=int, default=[32, 64, 128, 256]
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--horizon-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spans = pd.read_parquet(args.span_path)
    sessions = pd.read_csv(args.session_path)
    grouping_names = ("pooled", "event_class", "route_key")
    rows: list[dict[str, object]] = []

    for active_sessions in args.active_sessions:
        for repetition in range(args.repetitions):
            rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, active_sessions, repetition])
            )
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
            for deadline_ms in args.deadlines_ms:
                grouping_rows: dict[str, dict[str, object]] = {}
                for grouping in grouping_names:
                    fixed = cohort_metrics(
                        replay.event_times_s,
                        grouping_ids[grouping],
                        window_ms=deadline_ms,
                        thresholds=args.thresholds,
                    )
                    local = sliding_deadline_local_metrics(
                        replay.event_times_s,
                        grouping_ids[grouping],
                        deadline_ms=deadline_ms,
                        thresholds=args.thresholds,
                    )
                    for threshold in args.thresholds:
                        fixed_share = float(fixed[f"eligible_share_k{threshold}"])
                        upper_share = float(local[f"local_upper_share_k{threshold}"])
                        if fixed_share > upper_share + 1e-12:
                            raise AssertionError(
                                "fixed-window eligibility exceeded local bound at "
                                f"C={active_sessions}, rep={repetition}, "
                                f"deadline={deadline_ms}, grouping={grouping}, "
                                f"K={threshold}: {fixed_share} > {upper_share}"
                            )
                        row = {
                            "target_active_sessions": active_sessions,
                            "repetition": repetition,
                            "deadline_ms": deadline_ms,
                            "grouping": grouping,
                            "threshold_k": threshold,
                            "horizon_s": args.horizon_s,
                            "mean_active_sessions": replay.mean_active_sessions,
                            "event_count": len(replay.event_times_s),
                            "fixed_window_eligible_share": fixed_share,
                            "local_upper_share": upper_share,
                            "boundary_alignment_gap": upper_share - fixed_share,
                            "fixed_cohort_p90": fixed["cohort_size_event_p90"],
                            "local_cohort_p90": local["local_cohort_size_event_p90"],
                            "local_cohort_max": local["local_cohort_size_max"],
                        }
                        rows.append(row)
                        grouping_rows[f"{grouping}:{threshold}"] = row

                for threshold in args.thresholds:
                    pooled = grouping_rows[f"pooled:{threshold}"]["local_upper_share"]
                    coarse = grouping_rows[f"event_class:{threshold}"][
                        "local_upper_share"
                    ]
                    exact = grouping_rows[f"route_key:{threshold}"]["local_upper_share"]
                    if (
                        float(pooled) + 1e-12 < float(coarse)
                        or float(coarse) + 1e-12 < float(exact)
                    ):
                        raise AssertionError(
                            "local-bound grouping hierarchy violated at "
                            f"C={active_sessions}, rep={repetition}, "
                            f"deadline={deadline_ms}, K={threshold}: "
                            f"{pooled}, {coarse}, {exact}"
                        )
            print(
                f"active_sessions={active_sessions} repetition={repetition} "
                f"events={len(replay.event_times_s)}",
                flush=True,
            )

    repetitions = pd.DataFrame(rows).sort_values(
        [
            "target_active_sessions",
            "repetition",
            "deadline_ms",
            "grouping",
            "threshold_k",
        ]
    )
    group_columns = ["target_active_sessions", "deadline_ms", "grouping", "threshold_k"]
    summary = (
        repetitions.groupby(group_columns, as_index=False)
        .agg(
            repetitions=("repetition", "nunique"),
            fixed_window_eligible_share_mean=("fixed_window_eligible_share", "mean"),
            local_upper_share_mean=("local_upper_share", "mean"),
            local_upper_share_min=("local_upper_share", "min"),
            local_upper_share_max=("local_upper_share", "max"),
            boundary_alignment_gap_mean=("boundary_alignment_gap", "mean"),
            boundary_alignment_gap_max=("boundary_alignment_gap", "max"),
            event_count_mean=("event_count", "mean"),
            mean_active_sessions=("mean_active_sessions", "mean"),
            fixed_cohort_p90=("fixed_cohort_p90", "mean"),
            local_cohort_p90=("local_cohort_p90", "mean"),
            local_cohort_max=("local_cohort_max", "max"),
        )
        .sort_values(group_columns)
    )

    b4 = summary[
        summary["target_active_sessions"].eq(100000)
        & summary["threshold_k"].eq(256)
        & summary["deadline_ms"].isin([50, 100])
        & summary["grouping"].eq("route_key")
    ]
    hypotheses = {
        "B1_fixed_window_never_exceeds_local_upper": bool(
            repetitions["boundary_alignment_gap"].ge(-1e-12).all()
        ),
        "B2_grouping_hierarchy_checked": True,
        "B3_positive_alignment_gap_exists": bool(
            repetitions["boundary_alignment_gap"].gt(1e-12).any()
        ),
        "B4_exact_route_upper_exceeds_fixed_at_50_or_100ms": bool(
            len(b4) == 2
            and b4["local_upper_share_mean"].gt(
                b4["fixed_window_eligible_share_mean"]
            ).all()
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repetitions_path = args.output_dir / "trace-sliding-local-bound-repetitions.csv"
    summary_path = args.output_dir / "trace-sliding-local-bound-summary.csv"
    manifest_path = args.output_dir / "trace-sliding-local-bound-manifest.json"
    repetitions.to_csv(repetitions_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "preregistration": "preregistration/trace-replay-002.md",
        "bound": (
            "per-event maximum compatible interval overlap; valid upper bound, "
            "not necessarily jointly achievable"
        ),
        "configuration": {
            "active_sessions": args.active_sessions,
            "deadlines_ms": args.deadlines_ms,
            "thresholds": args.thresholds,
            "repetitions": args.repetitions,
            "horizon_s": args.horizon_s,
            "seed": args.seed,
            "groupings": list(grouping_names),
        },
        "hypotheses": hypotheses,
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
        "outputs": {
            "repetitions": {
                "path": str(repetitions_path),
                "rows": len(repetitions),
                "sha256": sha256_file(repetitions_path),
            },
            "summary": {
                "path": str(summary_path),
                "rows": len(summary),
                "sha256": sha256_file(summary_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(hypotheses, indent=2, sort_keys=True))
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
