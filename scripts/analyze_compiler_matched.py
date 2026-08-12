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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def stable_seed(root_seed: int, values: tuple[object, ...]) -> int:
    payload = "|".join(str(value) for value in (root_seed, *values)).encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def load_runs(paths: list[Path]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for csv_path in paths:
        manifest_path = csv_path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        cuda_devices = manifest.get("cuda_devices") or []
        hardware = cuda_devices[0]["name"] if cuda_devices else "no-cuda-device"
        provider = manifest.get("execution_provider", "local")
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
                "git_revision": manifest.get("git_revision"),
            }
        )
    return pd.concat(frames, ignore_index=True), sources


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


def main() -> None:
    args = parse_args()
    paths = sorted(args.raw_dir.glob("pilot-00[34]-*compiler-matched-*.csv"))
    if len(paths) < 2:
        raise FileNotFoundError(
            "expected local pilot 003 and cloud pilot 004 compiler-matched ledgers; "
            f"found {[str(path) for path in paths]}"
        )
    raw, sources = load_runs(paths)
    run_columns = ["source_file", "run_id", "hardware", "provider"]
    raw["correctness_valid"] = raw["correctness_valid"].astype("boolean")

    shape_quality = (
        raw.groupby([*run_columns, *SHAPE_COLUMNS], as_index=False, dropna=False)
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
        valid_timing.groupby([*run_columns, *CELL_COLUMNS], as_index=False, dropna=False)
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
        .merge(shape_quality, on=[*run_columns, *SHAPE_COLUMNS], how="left")
    )
    cells["wall_ms_cv"] = cells["wall_ms_std"] / cells["wall_ms_mean"]

    cpu8 = cells[
        cells["mode"].eq("compiled-cpu") & cells["threads"].eq(8)
    ][
        [
            *run_columns,
            *SHAPE_COLUMNS,
            "wall_ms_median",
            "wall_ms_cv",
            "compile_first_call_ms",
        ]
    ].rename(
        columns={
            "wall_ms_median": "cpu8_wall_ms_median",
            "wall_ms_cv": "cpu8_wall_ms_cv",
            "compile_first_call_ms": "cpu_compile_first_call_ms",
        }
    )
    best_cpu = (
        cells[cells["mode"].eq("compiled-cpu")]
        .sort_values("wall_ms_median")
        .groupby([*run_columns, *SHAPE_COLUMNS], as_index=False, dropna=False)
        .first()[
            [
                *run_columns,
                *SHAPE_COLUMNS,
                "wall_ms_median",
                "wall_ms_cv",
                "threads",
            ]
        ]
        .rename(
            columns={
                "wall_ms_median": "best_cpu_wall_ms_median",
                "wall_ms_cv": "best_cpu_wall_ms_cv",
                "threads": "best_cpu_threads",
            }
        )
    )
    gpu = cells[cells["mode"].isin(["compiled-gpu-resident", "compiled-gpu-host-visible"])].copy()
    speedups = gpu.merge(cpu8, on=[*run_columns, *SHAPE_COLUMNS], how="left").merge(
        best_cpu, on=[*run_columns, *SHAPE_COLUMNS], how="left"
    )
    speedups["speedup_vs_cpu8"] = (
        speedups["cpu8_wall_ms_median"] / speedups["wall_ms_median"]
    )
    speedups["speedup_vs_best_cpu"] = (
        speedups["best_cpu_wall_ms_median"] / speedups["wall_ms_median"]
    )
    speedups["steady_state_saved_ms"] = (
        speedups["cpu8_wall_ms_median"] - speedups["wall_ms_median"]
    )
    speedups["extra_compile_ms_vs_cpu"] = (
        speedups["compile_first_call_ms"] - speedups["cpu_compile_first_call_ms"]
    ).clip(lower=0)
    speedups["break_even_rollouts"] = np.where(
        speedups["steady_state_saved_ms"] > 0,
        speedups["extra_compile_ms_vs_cpu"] / speedups["steady_state_saved_ms"],
        np.nan,
    )
    speedups["break_even_agent_steps"] = (
        speedups["break_even_rollouts"]
        * speedups["agent_count"]
        * speedups["total_steps"].fillna(64)
        if "total_steps" in speedups
        else speedups["break_even_rollouts"] * speedups["agent_count"] * 64
    )
    speedups["steady_state_saved_ms_best_cpu"] = (
        speedups["best_cpu_wall_ms_median"] - speedups["wall_ms_median"]
    )
    speedups["break_even_rollouts_best_cpu"] = np.where(
        speedups["steady_state_saved_ms_best_cpu"] > 0,
        speedups["extra_compile_ms_vs_cpu"]
        / speedups["steady_state_saved_ms_best_cpu"],
        np.nan,
    )
    speedups["stable_under_10pct_cv"] = (
        speedups["wall_ms_cv"].le(0.10) & speedups["best_cpu_wall_ms_cv"].le(0.10)
    )

    raw_lookup = {
        tuple(key if isinstance(key, tuple) else (key,)): group["wall_ms"].to_numpy()
        for key, group in valid_timing.groupby(
            [*run_columns, *CELL_COLUMNS], dropna=False, sort=False
        )
    }
    ci_low: list[float] = []
    ci_high: list[float] = []
    best_ci_low: list[float] = []
    best_ci_high: list[float] = []
    for row in speedups.itertuples(index=False):
        common = tuple(getattr(row, column) for column in [*run_columns, *SHAPE_COLUMNS])
        cpu_key = (*common, "compiled-cpu", 8.0)
        best_cpu_key = (*common, "compiled-cpu", float(row.best_cpu_threads))

        # NaN is not equality-stable as a dictionary key, so locate the GPU
        # array by matching the non-thread fields explicitly.
        gpu_values = valid_timing[
            np.logical_and.reduce(
                [
                    valid_timing[column].eq(getattr(row, column))
                    for column in [*run_columns, *SHAPE_COLUMNS, "mode"]
                ]
            )
        ]["wall_ms"].to_numpy()
        cpu_values = raw_lookup[cpu_key]
        low, high = bootstrap_speedup_interval(
            cpu_values,
            gpu_values,
            replicates=args.bootstrap_replicates,
            seed=stable_seed(args.seed, (*common, row.mode)),
        )
        ci_low.append(low)
        ci_high.append(high)
        best_low, best_high = bootstrap_speedup_interval(
            raw_lookup[best_cpu_key],
            gpu_values,
            replicates=args.bootstrap_replicates,
            seed=stable_seed(args.seed, (*common, row.mode, "best-cpu")),
        )
        best_ci_low.append(best_low)
        best_ci_high.append(best_high)
    speedups["speedup_bootstrap_ci_low"] = ci_low
    speedups["speedup_bootstrap_ci_high"] = ci_high
    speedups["speedup_vs_best_cpu_bootstrap_ci_low"] = best_ci_low
    speedups["speedup_vs_best_cpu_bootstrap_ci_high"] = best_ci_high

    valid_speedups = speedups[speedups["shape_valid"]].copy()
    crossovers = (
        valid_speedups.assign(
            wins_cpu8=lambda frame: frame["speedup_vs_cpu8"] > 1,
            wins_best_cpu=lambda frame: frame["speedup_vs_best_cpu"] > 1,
        )
        .groupby(
            [
                *run_columns,
                "state_width",
                "action_count",
                "observation_horizon",
                "mode",
            ],
            as_index=False,
        )
        .apply(
            lambda group: pd.Series(
                {
                    "smallest_tested_crossover_n_cpu8": (
                        group.loc[group["wins_cpu8"], "agent_count"].min()
                        if group["wins_cpu8"].any()
                        else np.nan
                    ),
                    "smallest_tested_crossover_n_best_cpu": (
                        group.loc[group["wins_best_cpu"], "agent_count"].min()
                        if group["wins_best_cpu"].any()
                        else np.nan
                    ),
                    "max_valid_speedup_cpu8": group["speedup_vs_cpu8"].max(),
                    "max_valid_speedup_best_cpu": group["speedup_vs_best_cpu"].max(),
                    "valid_populations": group["agent_count"].nunique(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    quality = (
        raw.groupby(run_columns, as_index=False)
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
            shape_quality.groupby(run_columns, as_index=False).agg(
                shapes=("shape_valid", "size"),
                invalid_shapes=("shape_valid", lambda values: int((~values).sum())),
            ),
            on=run_columns,
        )
        .merge(
            cells.groupby(run_columns, as_index=False).agg(
                median_cell_cv=("wall_ms_cv", "median"),
                p90_cell_cv=("wall_ms_cv", lambda values: values.quantile(0.90)),
                cells_cv_over_10pct=("wall_ms_cv", lambda values: int((values > 0.10).sum())),
            ),
            on=run_columns,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "compiler-matched-cell-summary.csv": cells,
        "compiler-matched-speedups.csv": speedups,
        "compiler-matched-crossovers.csv": crossovers,
        "compiler-matched-shape-quality.csv": shape_quality,
        "compiler-matched-run-quality.csv": quality,
    }
    output_metadata: dict[str, dict[str, object]] = {}
    for filename, frame in output_frames.items():
        path = args.output_dir / filename
        frame.to_csv(path, index=False)
        output_metadata[filename] = {
            "path": str(path),
            "rows": len(frame),
            "sha256": sha256_file(path),
        }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "preregistration": "preregistration/pilot-003.md",
        "primary_cpu_reference": "compiled CPU at 8 threads",
        "secondary_cpu_reference": "faster median compiled CPU cell among 1 and 8 threads",
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "interval": "independent percentile bootstrap of median ratio",
            "seed": args.seed,
        },
        "sources": sources,
        "outputs": output_metadata,
    }
    manifest_path = args.output_dir / "compiler-matched-analysis-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(quality.to_string(index=False))
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
