from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SHAPE_COLUMNS = ["agent_count", "state_width", "action_count", "observation_horizon"]
CELL_COLUMNS = [*SHAPE_COLUMNS, "mode", "threads"]
RUN_COLUMNS = ["source_file", "run_id", "hardware", "provider"]
GPU_MODES = ["compiled-gpu-resident", "compiled-gpu-host-visible"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(root_seed: int, values: tuple[object, ...]) -> int:
    payload = "|".join(str(value) for value in (root_seed, *values)).encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def bootstrap_speedup_interval(
    cpu_values: np.ndarray,
    gpu_values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    cpu_indices = rng.integers(0, len(cpu_values), size=(replicates, len(cpu_values)))
    gpu_indices = rng.integers(0, len(gpu_values), size=(replicates, len(gpu_values)))
    cpu_medians = np.median(cpu_values[cpu_indices], axis=1)
    gpu_medians = np.median(gpu_values[gpu_indices], axis=1)
    ratios = cpu_medians / gpu_medians
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def load_runs(paths: list[Path]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for csv_path in paths:
        manifest_path = csv_path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        cuda_devices = manifest.get("cuda_devices") or []
        hardware = cuda_devices[0]["name"] if cuda_devices else "no-cuda-device"
        provider = manifest.get("execution_provider") or "local"
        frame = pd.read_csv(csv_path)
        frame["source_file"] = csv_path.name
        frame["hardware"] = hardware
        frame["provider"] = provider
        frames.append(frame)
        sources.append(
            {
                "csv_path": str(csv_path),
                "csv_sha256": sha256_file(csv_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "run_id": manifest["run_id"],
                "experiment_id": manifest["config"]["experiment"]["id"],
                "hardware": hardware,
                "provider": provider,
                "cpu": manifest.get("cpu"),
                "git_revision": manifest.get("git_revision"),
            }
        )
    return pd.concat(frames, ignore_index=True), sources


def first_sustained_win(group: pd.DataFrame, column: str) -> float:
    ordered = group.sort_values("agent_count")
    wins = ordered[column].gt(1).to_numpy()
    populations = ordered["agent_count"].to_numpy()
    for index, population in enumerate(populations):
        if wins[index:].all():
            return float(population)
    return float("nan")


def crossover_summary(speedups: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouping = [
        *RUN_COLUMNS,
        "state_width",
        "action_count",
        "observation_horizon",
        "mode",
    ]
    valid = speedups[speedups["shape_valid"]].copy()
    for keys, group in valid.groupby(grouping, dropna=False, sort=False):
        row = dict(zip(grouping, keys, strict=True))
        median_wins = group["speedup_vs_best_cpu"].gt(1)
        interval_wins = group["speedup_vs_best_cpu_bootstrap_ci_low"].gt(1)
        row.update(
            {
                "smallest_tested_crossover_n_best_cpu": (
                    float(group.loc[median_wins, "agent_count"].min())
                    if median_wins.any()
                    else np.nan
                ),
                "smallest_tested_ci_supported_crossover_n_best_cpu": (
                    float(group.loc[interval_wins, "agent_count"].min())
                    if interval_wins.any()
                    else np.nan
                ),
                "smallest_sustained_crossover_n_best_cpu": first_sustained_win(
                    group, "speedup_vs_best_cpu"
                ),
                "max_valid_speedup_best_cpu": float(group["speedup_vs_best_cpu"].max()),
                "valid_populations": int(group["agent_count"].nunique()),
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records).sort_values(
        ["hardware", "observation_horizon", "mode"]
    )


def host_penalty_summary(speedups: pd.DataFrame) -> pd.DataFrame:
    join_columns = [*RUN_COLUMNS, *SHAPE_COLUMNS]
    resident = speedups[speedups["mode"].eq("compiled-gpu-resident")][
        [*join_columns, "wall_ms_median", "wall_ms_cv", "shape_valid"]
    ].rename(
        columns={
            "wall_ms_median": "resident_wall_ms_median",
            "wall_ms_cv": "resident_wall_ms_cv",
        }
    )
    host = speedups[speedups["mode"].eq("compiled-gpu-host-visible")][
        [*join_columns, "wall_ms_median", "wall_ms_cv"]
    ].rename(
        columns={
            "wall_ms_median": "host_visible_wall_ms_median",
            "wall_ms_cv": "host_visible_wall_ms_cv",
        }
    )
    paired = resident.merge(host, on=join_columns, how="inner")
    paired["host_visibility_penalty"] = (
        paired["host_visible_wall_ms_median"] / paired["resident_wall_ms_median"]
    )
    return paired.sort_values(["hardware", "observation_horizon", "agent_count"])


def temporal_fusion_summary(speedups: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        *RUN_COLUMNS,
        "agent_count",
        "state_width",
        "action_count",
        "mode",
    ]
    columns = [
        *index_columns,
        "observation_horizon",
        "wall_ms_median",
        "best_cpu_wall_ms_median",
        "shape_valid",
    ]
    source = speedups[columns].copy()
    h1 = source[source["observation_horizon"].eq(1)].drop(
        columns="observation_horizon"
    )
    h64 = source[source["observation_horizon"].eq(64)].drop(
        columns="observation_horizon"
    )
    paired = h1.merge(h64, on=index_columns, suffixes=("_h1", "_h64"), how="inner")
    paired["gpu_temporal_fusion_gain"] = (
        paired["wall_ms_median_h1"] / paired["wall_ms_median_h64"]
    )
    paired["cpu_temporal_fusion_gain"] = (
        paired["best_cpu_wall_ms_median_h1"]
        / paired["best_cpu_wall_ms_median_h64"]
    )
    paired["gpu_proportional_fusion_advantage"] = (
        paired["gpu_temporal_fusion_gain"] / paired["cpu_temporal_fusion_gain"]
    )
    paired["shape_valid"] = paired["shape_valid_h1"] & paired["shape_valid_h64"]
    return paired.sort_values(["hardware", "mode", "agent_count"])


def build_tables(
    raw: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    raw = raw.copy()
    raw["correctness_valid"] = raw["correctness_valid"].astype("boolean")
    shape_quality = (
        raw.groupby([*RUN_COLUMNS, *SHAPE_COLUMNS], as_index=False, dropna=False)
        .agg(
            shape_valid=("correctness_valid", "all"),
            action_match=("correctness_action_match", "all"),
            max_state_abs_error=("correctness_state_max_abs_error", "max"),
            max_state_rel_error=("correctness_state_max_rel_error", "max"),
            max_budget_abs_error=("correctness_budget_max_abs_error", "max"),
            execution_error_rows=("status", lambda values: int((values != "ok").sum())),
        )
        .sort_values(["hardware", *SHAPE_COLUMNS])
    )
    shape_quality["shape_valid"] &= shape_quality["execution_error_rows"].eq(0)

    valid_timing = raw[raw["status"].eq("ok") & raw["wall_ms"].gt(0)].copy()
    cells = (
        valid_timing.groupby([*RUN_COLUMNS, *CELL_COLUMNS], as_index=False, dropna=False)
        .agg(
            observations=("wall_ms", "size"),
            wall_ms_mean=("wall_ms", "mean"),
            wall_ms_std=("wall_ms", "std"),
            wall_ms_median=("wall_ms", "median"),
            wall_ms_min=("wall_ms", "min"),
            wall_ms_max=("wall_ms", "max"),
            device_ms_median=("device_ms", "median"),
            compile_first_call_ms=("compile_first_call_ms", "first"),
            ns_per_agent_step_median=("ns_per_agent_step", "median"),
        )
        .merge(shape_quality, on=[*RUN_COLUMNS, *SHAPE_COLUMNS], how="left")
    )
    cells["wall_ms_cv"] = cells["wall_ms_std"] / cells["wall_ms_mean"]

    cpu8 = cells[cells["mode"].eq("compiled-cpu") & cells["threads"].eq(8)][
        [
            *RUN_COLUMNS,
            *SHAPE_COLUMNS,
            "wall_ms_median",
            "wall_ms_cv",
            "compile_first_call_ms",
        ]
    ].rename(
        columns={
            "wall_ms_median": "cpu8_wall_ms_median",
            "wall_ms_cv": "cpu8_wall_ms_cv",
            "compile_first_call_ms": "cpu8_compile_first_call_ms",
        }
    )
    best_cpu = (
        cells[cells["mode"].eq("compiled-cpu")]
        .sort_values("wall_ms_median")
        .groupby([*RUN_COLUMNS, *SHAPE_COLUMNS], as_index=False, dropna=False)
        .first()[
            [
                *RUN_COLUMNS,
                *SHAPE_COLUMNS,
                "wall_ms_median",
                "wall_ms_cv",
                "compile_first_call_ms",
                "threads",
            ]
        ]
        .rename(
            columns={
                "wall_ms_median": "best_cpu_wall_ms_median",
                "wall_ms_cv": "best_cpu_wall_ms_cv",
                "compile_first_call_ms": "best_cpu_compile_first_call_ms",
                "threads": "best_cpu_threads",
            }
        )
    )
    gpu = cells[cells["mode"].isin(GPU_MODES)].copy()
    speedups = gpu.merge(cpu8, on=[*RUN_COLUMNS, *SHAPE_COLUMNS], how="left").merge(
        best_cpu, on=[*RUN_COLUMNS, *SHAPE_COLUMNS], how="left"
    )
    speedups["speedup_vs_cpu8"] = (
        speedups["cpu8_wall_ms_median"] / speedups["wall_ms_median"]
    )
    speedups["speedup_vs_best_cpu"] = (
        speedups["best_cpu_wall_ms_median"] / speedups["wall_ms_median"]
    )
    speedups["steady_state_saved_ms_best_cpu"] = (
        speedups["best_cpu_wall_ms_median"] - speedups["wall_ms_median"]
    )
    speedups["extra_compile_ms_vs_best_cpu"] = (
        speedups["compile_first_call_ms"] - speedups["best_cpu_compile_first_call_ms"]
    ).clip(lower=0)
    speedups["break_even_rollouts_best_cpu"] = np.where(
        speedups["steady_state_saved_ms_best_cpu"].gt(0),
        speedups["extra_compile_ms_vs_best_cpu"]
        / speedups["steady_state_saved_ms_best_cpu"],
        np.nan,
    )
    speedups["break_even_agent_steps_best_cpu"] = (
        speedups["break_even_rollouts_best_cpu"]
        * speedups["agent_count"]
        * speedups["observation_horizon"]
    )
    speedups["stable_under_10pct_cv"] = (
        speedups["wall_ms_cv"].le(0.10) & speedups["best_cpu_wall_ms_cv"].le(0.10)
    )

    raw_lookup = {
        tuple(key if isinstance(key, tuple) else (key,)): group["wall_ms"].to_numpy()
        for key, group in valid_timing.groupby(
            [*RUN_COLUMNS, *CELL_COLUMNS], dropna=False, sort=False
        )
    }
    ci_low: list[float] = []
    ci_high: list[float] = []
    best_ci_low: list[float] = []
    best_ci_high: list[float] = []
    for row in speedups.itertuples(index=False):
        common = tuple(getattr(row, column) for column in [*RUN_COLUMNS, *SHAPE_COLUMNS])
        cpu8_key = (*common, "compiled-cpu", 8.0)
        best_cpu_key = (*common, "compiled-cpu", float(row.best_cpu_threads))
        gpu_values = valid_timing[
            np.logical_and.reduce(
                [
                    valid_timing[column].eq(getattr(row, column))
                    for column in [*RUN_COLUMNS, *SHAPE_COLUMNS, "mode"]
                ]
            )
        ]["wall_ms"].to_numpy()
        low, high = bootstrap_speedup_interval(
            raw_lookup[cpu8_key],
            gpu_values,
            replicates=bootstrap_replicates,
            seed=stable_seed(seed, (*common, row.mode, "cpu8")),
        )
        ci_low.append(low)
        ci_high.append(high)
        low, high = bootstrap_speedup_interval(
            raw_lookup[best_cpu_key],
            gpu_values,
            replicates=bootstrap_replicates,
            seed=stable_seed(seed, (*common, row.mode, "best-cpu")),
        )
        best_ci_low.append(low)
        best_ci_high.append(high)
    speedups["speedup_bootstrap_ci_low"] = ci_low
    speedups["speedup_bootstrap_ci_high"] = ci_high
    speedups["speedup_vs_best_cpu_bootstrap_ci_low"] = best_ci_low
    speedups["speedup_vs_best_cpu_bootstrap_ci_high"] = best_ci_high

    quality = (
        raw.groupby(RUN_COLUMNS, as_index=False)
        .agg(
            rows=("case_id", "size"),
            cases=("case_id", "nunique"),
            execution_error_rows=("status", lambda values: int((values != "ok").sum())),
            duplicate_case_repetitions=(
                "case_id",
                lambda _values: int(
                    raw.loc[_values.index].duplicated(["case_id", "repetition"]).sum()
                ),
            ),
        )
        .merge(
            shape_quality.groupby(RUN_COLUMNS, as_index=False).agg(
                shapes=("shape_valid", "size"),
                invalid_shapes=("shape_valid", lambda values: int((~values).sum())),
            ),
            on=RUN_COLUMNS,
        )
        .merge(
            cells.groupby(RUN_COLUMNS, as_index=False).agg(
                median_cell_cv=("wall_ms_cv", "median"),
                p90_cell_cv=("wall_ms_cv", lambda values: values.quantile(0.90)),
                cells_cv_over_10pct=("wall_ms_cv", lambda values: int((values > 0.10).sum())),
            ),
            on=RUN_COLUMNS,
        )
    )

    return {
        "sub256-cell-summary.csv": cells,
        "sub256-speedups.csv": speedups,
        "sub256-crossovers.csv": crossover_summary(speedups),
        "sub256-shape-quality.csv": shape_quality,
        "sub256-run-quality.csv": quality,
        "sub256-host-penalty.csv": host_penalty_summary(speedups),
        "sub256-temporal-fusion.csv": temporal_fusion_summary(speedups),
    }


def evaluate_hypotheses(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    crossovers = tables["sub256-crossovers.csv"]
    fusion = tables["sub256-temporal-fusion.csv"]
    resident = crossovers[crossovers["mode"].eq("compiled-gpu-resident")]
    h64 = resident[resident["observation_horizon"].eq(64)]
    l4_h1 = resident[
        resident["hardware"].eq("NVIDIA L4")
        & resident["observation_horizon"].eq(1)
    ]

    comparisons: list[bool] = []
    for _, resident_row in resident.iterrows():
        host_row = crossovers[
            crossovers["hardware"].eq(resident_row["hardware"])
            & crossovers["observation_horizon"].eq(resident_row["observation_horizon"])
            & crossovers["mode"].eq("compiled-gpu-host-visible")
        ]
        if host_row.empty:
            comparisons.append(False)
            continue
        resident_value = resident_row["smallest_tested_crossover_n_best_cpu"]
        host_value = host_row.iloc[0]["smallest_tested_crossover_n_best_cpu"]
        resident_value = np.inf if pd.isna(resident_value) else resident_value
        host_value = np.inf if pd.isna(host_value) else host_value
        comparisons.append(bool(host_value >= resident_value))

    return {
        "S1_resident_h64_crosses_by_n128_on_both_gpus": bool(
            len(h64) == 2
            and h64["smallest_tested_crossover_n_best_cpu"].notna().all()
            and h64["smallest_tested_crossover_n_best_cpu"].le(128).all()
        ),
        "S2_l4_resident_h1_crosses_by_n256": bool(
            len(l4_h1) == 1
            and l4_h1["smallest_tested_crossover_n_best_cpu"].notna().all()
            and l4_h1["smallest_tested_crossover_n_best_cpu"].le(256).all()
        ),
        "S3_host_visible_crossover_is_never_smaller": bool(all(comparisons)),
        "S4_h64_proportional_fusion_advantage_gt_1_at_every_population": bool(
            fusion["shape_valid"].all()
            and fusion["gpu_proportional_fusion_advantage"].gt(1).all()
        ),
        "observed_gtx_resident_h1_crossover": None
        if resident[
            resident["hardware"].eq("NVIDIA GeForce GTX 1660 Ti")
            & resident["observation_horizon"].eq(1)
        ]["smallest_tested_crossover_n_best_cpu"].isna().all()
        else float(
            resident[
                resident["hardware"].eq("NVIDIA GeForce GTX 1660 Ti")
                & resident["observation_horizon"].eq(1)
            ]["smallest_tested_crossover_n_best_cpu"].iloc[0]
        ),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.raw_dir.glob("pilot-00[56]-*sub256-*.csv"))
    if len(paths) != 2:
        raise FileNotFoundError(
            "expected exactly pilots 005 and 006 sub-256 ledgers; "
            f"found {[str(path) for path in paths]}"
        )
    raw, sources = load_runs(paths)
    tables = build_tables(
        raw,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_metadata: dict[str, dict[str, object]] = {}
    for filename, frame in tables.items():
        path = args.output_dir / filename
        frame.to_csv(path, index=False)
        output_metadata[filename] = {
            "path": str(path),
            "rows": len(frame),
            "sha256": sha256_file(path),
        }

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "preregistration": "preregistration/pilot-005.md",
        "primary_cpu_reference": "faster median compiled CPU cell among 1 and 8 threads",
        "secondary_cpu_reference": "compiled CPU at 8 threads",
        "crossover_definition": (
            "smallest tested population with GPU median wall time below tuned CPU median"
        ),
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "interval": "independent percentile bootstrap of median ratio",
            "seed": args.seed,
        },
        "hypotheses": evaluate_hypotheses(tables),
        "sources": sources,
        "outputs": output_metadata,
    }
    manifest_path = args.output_dir / "sub256-analysis-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(tables["sub256-run-quality.csv"].to_string(index=False))
    print("\nCrossovers against tuned CPU:")
    print(
        tables["sub256-crossovers.csv"][
            [
                "hardware",
                "observation_horizon",
                "mode",
                "smallest_tested_crossover_n_best_cpu",
                "smallest_tested_ci_supported_crossover_n_best_cpu",
                "smallest_sustained_crossover_n_best_cpu",
                "max_valid_speedup_best_cpu",
            ]
        ].to_string(index=False)
    )
    print(f"\nmanifest={manifest_path}")


if __name__ == "__main__":
    main()
