from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_agent_crossover.ready_cohort import (
    cohort_metrics,
    exact_sliding_deadline_packing,
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
    parser.add_argument(
        "--prior-path",
        type=Path,
        default=Path("data/processed/trace-sliding-local-bound-repetitions.csv"),
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


def verify_prior_reproduction(observed: pd.DataFrame, prior_path: Path) -> None:
    prior = pd.read_csv(prior_path)
    keys = [
        "target_active_sessions",
        "repetition",
        "deadline_ms",
        "grouping",
        "threshold_k",
    ]
    expected = prior[
        keys + ["fixed_window_eligible_share", "local_upper_share"]
    ]
    matched = observed.merge(
        expected,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("", "_prior"),
        indicator=True,
    )
    if not matched["_merge"].eq("both").all():
        raise AssertionError("exact replay does not have the same cell grid as replay 002")
    for column in ["fixed_window_eligible_share", "local_upper_share"]:
        if not np.allclose(
            matched[column],
            matched[f"{column}_prior"],
            rtol=0.0,
            atol=1e-12,
        ):
            difference = np.abs(matched[column] - matched[f"{column}_prior"])
            raise AssertionError(
                f"failed deterministic replay of {column}; max difference={difference.max()}"
            )


def check_invariants(rows: pd.DataFrame) -> dict[str, bool]:
    tolerance = 1e-12
    e1 = bool(
        (
            rows["fixed_window_eligible_share"]
            <= rows["exact_optimal_share"] + tolerance
        ).all()
        and (
            rows["exact_optimal_share"] <= rows["local_upper_share"] + tolerance
        ).all()
    )

    e2 = True
    hierarchy_keys = [
        "target_active_sessions",
        "repetition",
        "deadline_ms",
        "threshold_k",
    ]
    for _, group in rows.groupby(hierarchy_keys, sort=False):
        values = group.set_index("grouping")["exact_optimal_share"]
        e2 &= bool(
            values["pooled"] + tolerance >= values["event_class"]
            and values["event_class"] + tolerance >= values["route_key"]
        )

    e3 = True
    deadline_keys = [
        "target_active_sessions",
        "repetition",
        "grouping",
        "threshold_k",
    ]
    for _, group in rows.groupby(deadline_keys, sort=False):
        values = group.sort_values("deadline_ms")["exact_optimal_share"].to_numpy()
        e3 &= bool(np.all(np.diff(values) >= -tolerance))

    e4 = True
    threshold_keys = [
        "target_active_sessions",
        "repetition",
        "deadline_ms",
        "grouping",
    ]
    for _, group in rows.groupby(threshold_keys, sort=False):
        values = group.sort_values("threshold_k")["exact_optimal_share"].to_numpy()
        e4 &= bool(np.all(np.diff(values) <= tolerance))

    closed = np.isclose(
        rows["fixed_window_eligible_share"],
        rows["local_upper_share"],
        rtol=0.0,
        atol=tolerance,
    )
    e5 = bool(
        np.isclose(
            rows.loc[closed, "exact_optimal_share"],
            rows.loc[closed, "fixed_window_eligible_share"],
            rtol=0.0,
            atol=tolerance,
        ).all()
    )

    primary = rows[
        rows["target_active_sessions"].eq(100000)
        & rows["deadline_ms"].eq(50)
        & rows["grouping"].eq("route_key")
        & rows["threshold_k"].eq(256)
    ]
    e6 = bool(
        len(primary) == 3
        and primary["exact_optimal_share"].mean()
        > primary["fixed_window_eligible_share"].mean()
    )
    return {
        "E1_fixed_le_exact_le_local": e1,
        "E2_compatibility_coarsening_monotone": e2,
        "E3_deadline_monotone": e3,
        "E4_threshold_monotone": e4,
        "E5_closed_bounds_force_exact_equality": e5,
        "E6_primary_exact_strictly_above_fixed": e6,
    }


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
                        started = time.perf_counter()
                        exact = exact_sliding_deadline_packing(
                            replay.event_times_s,
                            grouping_ids[grouping],
                            deadline_ms=deadline_ms,
                            threshold=threshold,
                        )
                        elapsed_s = time.perf_counter() - started
                        fixed_share = float(fixed[f"eligible_share_k{threshold}"])
                        upper_share = float(local[f"local_upper_share_k{threshold}"])
                        gap = upper_share - fixed_share
                        closure = (
                            (exact.accelerated_share - fixed_share) / gap
                            if gap > 1e-15
                            else np.nan
                        )
                        rows.append(
                            {
                                "target_active_sessions": active_sessions,
                                "repetition": repetition,
                                "deadline_ms": deadline_ms,
                                "grouping": grouping,
                                "threshold_k": threshold,
                                "horizon_s": args.horizon_s,
                                "mean_active_sessions": replay.mean_active_sessions,
                                "event_count": len(replay.event_times_s),
                                "fixed_window_eligible_share": fixed_share,
                                "exact_optimal_share": exact.accelerated_share,
                                "local_upper_share": upper_share,
                                "exact_accelerated_event_count": (
                                    exact.accelerated_event_count
                                ),
                                "exact_batch_count": exact.batch_count,
                                "alignment_gap_closure": closure,
                                "solver_elapsed_s": elapsed_s,
                                "algorithm": exact.algorithm,
                            }
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
    verify_prior_reproduction(repetitions, args.prior_path)
    hypotheses = check_invariants(repetitions)
    validity_keys = [key for key in hypotheses if key.startswith("E") and key[1] < "6"]
    if not all(hypotheses[key] for key in validity_keys):
        raise AssertionError(f"exact-packing validity gate failed: {hypotheses}")

    group_columns = ["target_active_sessions", "deadline_ms", "grouping", "threshold_k"]
    summary = (
        repetitions.groupby(group_columns, as_index=False)
        .agg(
            repetitions=("repetition", "nunique"),
            event_count_mean=("event_count", "mean"),
            fixed_window_eligible_share_mean=("fixed_window_eligible_share", "mean"),
            exact_optimal_share_mean=("exact_optimal_share", "mean"),
            exact_optimal_share_min=("exact_optimal_share", "min"),
            exact_optimal_share_max=("exact_optimal_share", "max"),
            local_upper_share_mean=("local_upper_share", "mean"),
            exact_batch_count_mean=("exact_batch_count", "mean"),
            alignment_gap_closure_mean=("alignment_gap_closure", "mean"),
            solver_elapsed_s_mean=("solver_elapsed_s", "mean"),
        )
        .sort_values(group_columns)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repetitions_path = args.output_dir / "trace-exact-packing-repetitions.csv"
    summary_path = args.output_dir / "trace-exact-packing-summary.csv"
    manifest_path = args.output_dir / "trace-exact-packing-manifest.json"
    for path in [repetitions_path, summary_path, manifest_path]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    repetitions.to_csv(repetitions_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "preregistration": "preregistration/trace-replay-003.md",
        "clock_contract": "release/deadline rounded to nearest integer nanosecond",
        "model": "equal relative deadline, zero service, unlimited capacity",
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
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "source_files": {
            "solver": {
                "path": "src/gpu_agent_crossover/ready_cohort.py",
                "sha256": sha256_file(
                    Path("src/gpu_agent_crossover/ready_cohort.py")
                ),
            },
            "span_features": {
                "path": str(args.span_path),
                "sha256": sha256_file(args.span_path),
            },
            "session_summary": {
                "path": str(args.session_path),
                "sha256": sha256_file(args.session_path),
            },
            "prior_replay": {
                "path": str(args.prior_path),
                "sha256": sha256_file(args.prior_path),
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
