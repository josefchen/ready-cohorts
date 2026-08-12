from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_sub256 import build_tables, load_runs, sha256_file

EXPECTED_GPUS = ["T4", "L4", "A10", "L40S", "A100-80GB", "H100!"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--pricing",
        type=Path,
        default=Path("data/external/modal-pricing-2026-08-11.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def source_metadata(paths: list[Path]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for csv_path in paths:
        manifest = json.loads(csv_path.with_suffix(".manifest.json").read_text())
        records.append(
            {
                "source_file": csv_path.name,
                "requested_gpu": manifest["requested_gpu"],
                "placement_replicate": int(manifest["placement_replicate"]),
                "cpu_vendor": (manifest.get("cpu") or {}).get("vendor_id"),
                "cpu_family": (manifest.get("cpu") or {}).get("cpu_family"),
                "cpu_model": (manifest.get("cpu") or {}).get("model"),
                "cpu_logical_count": (manifest.get("cpu") or {}).get(
                    "logical_cpu_count"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def enrich_tables(
    tables: dict[str, pd.DataFrame], metadata: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    enriched: dict[str, pd.DataFrame] = {}
    for filename, frame in tables.items():
        name = filename.replace("sub256-", "hardware-replication-")
        enriched[name] = frame.merge(metadata, on="source_file", how="left")
    return enriched


def placement_costs(
    speedups: pd.DataFrame, pricing: dict[str, object]
) -> pd.DataFrame:
    resident = speedups[
        speedups["mode"].eq("compiled-gpu-resident")
        & speedups["shape_valid"]
    ].copy()
    resident["pricing_gpu"] = resident["requested_gpu"].str.replace(
        "!", "", regex=False
    )
    resident["gpu_price_usd_per_second"] = resident["pricing_gpu"].map(
        pricing["gpu"]
    )
    resident["gpu_cost_usd_per_billion_agent_steps"] = (
        resident["gpu_price_usd_per_second"]
        * resident["ns_per_agent_step_median"]
    )
    resident["wall_to_device_ratio"] = (
        resident["wall_ms_median"] / resident["device_ms_median"]
    )
    resident["pricing_captured_at"] = pricing["captured_at"]
    resident["pricing_source"] = pricing["source"]
    columns = [
        "source_file",
        "run_id",
        "hardware",
        "provider",
        "requested_gpu",
        "placement_replicate",
        "agent_count",
        "state_width",
        "action_count",
        "observation_horizon",
        "wall_ms_median",
        "device_ms_median",
        "wall_to_device_ratio",
        "wall_ms_cv",
        "best_cpu_wall_ms_median",
        "best_cpu_threads",
        "speedup_vs_best_cpu",
        "speedup_vs_best_cpu_bootstrap_ci_low",
        "speedup_vs_best_cpu_bootstrap_ci_high",
        "compile_first_call_ms",
        "ns_per_agent_step_median",
        "gpu_price_usd_per_second",
        "gpu_cost_usd_per_billion_agent_steps",
        "pricing_captured_at",
        "pricing_source",
        "shape_valid",
    ]
    return resident[columns].sort_values(
        ["requested_gpu", "placement_replicate", "observation_horizon", "agent_count"]
    )


def aggregate_placements(costs: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "requested_gpu",
        "agent_count",
        "state_width",
        "action_count",
        "observation_horizon",
    ]
    return (
        costs.groupby(grouping, as_index=False, dropna=False)
        .agg(
            placements=("source_file", "nunique"),
            wall_ms_median_of_placement_medians=("wall_ms_median", "median"),
            wall_ms_placement_min=("wall_ms_median", "min"),
            wall_ms_placement_max=("wall_ms_median", "max"),
            device_ms_median_of_placement_medians=("device_ms_median", "median"),
            wall_to_device_ratio_median=("wall_to_device_ratio", "median"),
            tuned_cpu_wall_ms_median_of_placements=(
                "best_cpu_wall_ms_median",
                "median",
            ),
            speedup_median_of_placements=("speedup_vs_best_cpu", "median"),
            speedup_placement_min=("speedup_vs_best_cpu", "min"),
            speedup_placement_max=("speedup_vs_best_cpu", "max"),
            placements_speedup_gt_one=(
                "speedup_vs_best_cpu",
                lambda values: int(values.gt(1).sum()),
            ),
            compile_ms_median_of_placements=("compile_first_call_ms", "median"),
            ns_per_agent_step_median_of_placements=(
                "ns_per_agent_step_median",
                "median",
            ),
            gpu_price_usd_per_second=("gpu_price_usd_per_second", "first"),
            cost_usd_per_billion_median=(
                "gpu_cost_usd_per_billion_agent_steps",
                "median",
            ),
            cost_usd_per_billion_placement_min=(
                "gpu_cost_usd_per_billion_agent_steps",
                "min",
            ),
            cost_usd_per_billion_placement_max=(
                "gpu_cost_usd_per_billion_agent_steps",
                "max",
            ),
        )
        .assign(
            fraction_placements_speedup_gt_one=lambda frame: (
                frame["placements_speedup_gt_one"] / frame["placements"]
            )
        )
        .sort_values(["observation_horizon", "requested_gpu", "agent_count"])
    )


def hardware_rank(aggregate: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rank = aggregate[
        aggregate["agent_count"].eq(256)
        & aggregate["observation_horizon"].eq(64)
    ].copy()
    rank["price_rank"] = rank["gpu_price_usd_per_second"].rank(
        method="average", ascending=True
    )
    rank["wall_time_rank"] = rank[
        "wall_ms_median_of_placement_medians"
    ].rank(method="average", ascending=True)
    rank["cost_rank"] = rank["cost_usd_per_billion_median"].rank(
        method="average", ascending=True
    )
    correlation = float(rank["price_rank"].corr(rank["wall_time_rank"]))
    return rank.sort_values("wall_time_rank"), correlation


def hypothesis_results(
    placement_cost: pd.DataFrame,
    temporal_fusion: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> dict[str, object]:
    h64_n8 = placement_cost[
        placement_cost["observation_horizon"].eq(64)
        & placement_cost["agent_count"].eq(8)
    ]
    complete = bool(
        len(placement_cost["source_file"].unique()) == 18
        and set(placement_cost["requested_gpu"].unique()) == set(EXPECTED_GPUS)
        and placement_cost.groupby("requested_gpu")["placement_replicate"].nunique().eq(3).all()
    )

    target = placement_cost[
        placement_cost["observation_horizon"].eq(64)
        & placement_cost["agent_count"].eq(256)
    ]
    h100 = target[target["requested_gpu"].eq("H100!")][
        ["placement_replicate", "gpu_cost_usd_per_billion_agent_steps"]
    ].rename(columns={"gpu_cost_usd_per_billion_agent_steps": "h100_cost"})
    h100_price_rows = target[target["requested_gpu"].eq("H100!")][
        "gpu_price_usd_per_second"
    ]
    h100_price = float(h100_price_rows.iloc[0]) if not h100_price_rows.empty else np.nan
    alternatives = target[
        target["gpu_price_usd_per_second"].lt(h100_price)
    ].merge(
        h100, on="placement_replicate", how="inner"
    )
    alternatives["beats_h100_cost"] = (
        alternatives["gpu_cost_usd_per_billion_agent_steps"]
        < alternatives["h100_cost"]
    )
    alternative_all_placements = (
        alternatives.groupby("requested_gpu")["beats_h100_cost"].all()
        if not alternatives.empty
        else pd.Series(dtype=bool)
    )

    _, rank_correlation = hardware_rank(aggregate)

    one_step_256 = placement_cost[
        placement_cost["observation_horizon"].eq(1)
        & placement_cost["agent_count"].eq(256)
        & placement_cost["requested_gpu"].isin(["A100-80GB", "H100!", "L40S"])
    ]
    high_card_failures = (
        one_step_256.assign(fails=lambda frame: frame["speedup_vs_best_cpu"].le(1))
        .groupby("requested_gpu")["fails"]
        .sum()
    )

    return {
        "complete_expected_18_placements": complete,
        "R1_all_h64_n8_placements_speedup_gt_1": bool(
            complete and len(h64_n8) == 18 and h64_n8["speedup_vs_best_cpu"].gt(1).all()
        ),
        "R2_fusion_advantage_gt_1_every_placement_population": bool(
            complete
            and temporal_fusion["shape_valid"].all()
            and temporal_fusion["gpu_proportional_fusion_advantage"].gt(1).all()
        ),
        "R3_cheaper_gpu_beats_h100_cost_in_all_three_placements": bool(
            alternative_all_placements.any()
        ),
        "R3_qualifying_gpus": sorted(
            alternative_all_placements[alternative_all_placements].index.tolist()
        ),
        "R4_price_wall_time_spearman_below_0_8": bool(rank_correlation < 0.8),
        "R4_price_wall_time_spearman": rank_correlation,
        "R5_two_high_cards_fail_one_step_in_two_of_three_placements": bool(
            high_card_failures.ge(2).sum() >= 2
        ),
        "R5_failed_placement_counts": {
            str(gpu): int(count) for gpu, count in high_card_failures.items()
        },
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.raw_dir.glob("pilot-0??-modal-*-rep?-*.csv"))
    if len(paths) != 18:
        raise FileNotFoundError(
            "expected 18 pilots from preregistration/pilot-012.md; "
            f"found {len(paths)}: {[str(path) for path in paths]}"
        )

    raw, sources = load_runs(paths)
    metadata = source_metadata(paths)
    tables = enrich_tables(
        build_tables(
            raw,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
        metadata,
    )
    pricing = json.loads(args.pricing.read_text())
    placement_cost = placement_costs(
        tables["hardware-replication-speedups.csv"], pricing
    )
    aggregate = aggregate_placements(placement_cost)
    rank, _ = hardware_rank(aggregate)
    tables["hardware-replication-placement-cost.csv"] = placement_cost
    tables["hardware-replication-aggregate.csv"] = aggregate
    tables["hardware-replication-rank-n256-h64.csv"] = rank

    hypotheses = hypothesis_results(
        placement_cost,
        tables["hardware-replication-temporal-fusion.csv"],
        aggregate,
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

    source_by_file = {record["csv_path"]: record for record in sources}
    for record in metadata.to_dict(orient="records"):
        matching = source_by_file[str(args.raw_dir / str(record["source_file"]))]
        matching.update(
            {
                "requested_gpu": record["requested_gpu"],
                "placement_replicate": record["placement_replicate"],
            }
        )

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "preregistration": "preregistration/pilot-012.md",
        "primary_unit_of_cross_placement_inference": "fresh container placement median",
        "timing_outlier_policy": "none removed",
        "primary_cpu_reference": "faster colocated median among compiled CPU 1 and 8 threads",
        "pricing": {
            "path": str(args.pricing),
            "sha256": sha256_file(args.pricing),
            "captured_at": pricing["captured_at"],
            "source": pricing["source"],
            "scope": "GPU-only marginal steady-state cost",
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "interval": "within-placement independent percentile bootstrap of median ratio",
            "seed": args.seed,
        },
        "hypotheses": hypotheses,
        "sources": sources,
        "outputs": output_metadata,
    }
    manifest_path = args.output_dir / "hardware-replication-analysis-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(tables["hardware-replication-run-quality.csv"].to_string(index=False))
    print("\nN=256, H=64 hardware rank:")
    print(
        rank[
            [
                "requested_gpu",
                "wall_ms_median_of_placement_medians",
                "speedup_median_of_placements",
                "cost_usd_per_billion_median",
                "price_rank",
                "wall_time_rank",
                "cost_rank",
            ]
        ].to_string(index=False)
    )
    print("\nHypotheses:")
    print(json.dumps(hypotheses, indent=2, sort_keys=True))
    print(f"\nmanifest={manifest_path}")


if __name__ == "__main__":
    main()
