from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_sub256 import (
    build_tables,
    load_runs,
    sha256_file,
)

EXPECTED_MODAL_GPUS = {"L4", "T4", "A10", "L40S", "A100-80GB", "H100!"}


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
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def source_metadata(paths: list[Path]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for csv_path in paths:
        manifest = json.loads(csv_path.with_suffix(".manifest.json").read_text())
        records.append(
            {
                "source_file": csv_path.name,
                "requested_gpu": manifest.get("requested_gpu"),
                "cpu_model_name": (manifest.get("cpu") or {}).get("model_name"),
                "cpu_logical_count": (manifest.get("cpu") or {}).get("logical_cpu_count"),
                "cpu_affinity_count": len(
                    (manifest.get("cpu") or {}).get("process_affinity_cpus") or []
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def add_source_metadata(
    tables: dict[str, pd.DataFrame], metadata: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    enriched: dict[str, pd.DataFrame] = {}
    for filename, frame in tables.items():
        enriched[filename.replace("sub256-", "hardware-sweep-")] = frame.merge(
            metadata, on="source_file", how="left"
        )
    return enriched


def gpu_cost_table(
    speedups: pd.DataFrame, pricing: dict[str, object]
) -> pd.DataFrame:
    rates = pricing["gpu"]
    frame = speedups.copy()
    frame["pricing_gpu"] = frame["requested_gpu"].str.replace("!", "", regex=False)
    frame["gpu_price_usd_per_second"] = frame["pricing_gpu"].map(rates)
    frame["gpu_cost_usd_per_billion_agent_steps"] = (
        frame["gpu_price_usd_per_second"] * frame["ns_per_agent_step_median"]
    )
    frame["gpu_cost_usd_per_million_agent_steps"] = (
        frame["gpu_cost_usd_per_billion_agent_steps"] / 1000
    )
    frame["pricing_captured_at"] = pricing["captured_at"]
    frame["pricing_source"] = pricing["source"]
    columns = [
        "source_file",
        "run_id",
        "hardware",
        "requested_gpu",
        "provider",
        "agent_count",
        "state_width",
        "action_count",
        "observation_horizon",
        "mode",
        "wall_ms_median",
        "ns_per_agent_step_median",
        "gpu_price_usd_per_second",
        "gpu_cost_usd_per_million_agent_steps",
        "gpu_cost_usd_per_billion_agent_steps",
        "pricing_captured_at",
        "pricing_source",
        "shape_valid",
    ]
    return frame[columns].sort_values(
        ["requested_gpu", "observation_horizon", "mode", "agent_count"],
        na_position="last",
    )


def host_crossover_order(crossovers: pd.DataFrame) -> bool:
    comparisons: list[bool] = []
    grouping = ["source_file", "observation_horizon"]
    for _, group in crossovers.groupby(grouping, dropna=False):
        resident = group[group["mode"].eq("compiled-gpu-resident")]
        host = group[group["mode"].eq("compiled-gpu-host-visible")]
        if len(resident) != 1 or len(host) != 1:
            comparisons.append(False)
            continue
        resident_n = resident.iloc[0]["smallest_tested_crossover_n_best_cpu"]
        host_n = host.iloc[0]["smallest_tested_crossover_n_best_cpu"]
        resident_n = np.inf if pd.isna(resident_n) else resident_n
        host_n = np.inf if pd.isna(host_n) else host_n
        comparisons.append(bool(host_n >= resident_n))
    return bool(comparisons and all(comparisons))


def cheaper_card_beats_h100(cost: pd.DataFrame) -> bool:
    valid = cost[
        cost["mode"].eq("compiled-gpu-resident")
        & cost["requested_gpu"].notna()
        & cost["shape_valid"]
    ].copy()
    h100 = valid[valid["requested_gpu"].eq("H100!")][
        [
            "agent_count",
            "observation_horizon",
            "gpu_cost_usd_per_billion_agent_steps",
        ]
    ].rename(
        columns={"gpu_cost_usd_per_billion_agent_steps": "h100_cost"}
    )
    if h100.empty:
        return False
    alternatives = valid[~valid["requested_gpu"].eq("H100!")].merge(
        h100, on=["agent_count", "observation_horizon"], how="inner"
    )
    return bool(
        (
            alternatives["gpu_cost_usd_per_billion_agent_steps"]
            < alternatives["h100_cost"]
        ).any()
    )


def evaluate_hypotheses(
    crossovers: pd.DataFrame,
    fusion: pd.DataFrame,
    cost: pd.DataFrame,
    observed_requested_gpus: set[str],
) -> dict[str, object]:
    modal_resident = crossovers[
        crossovers["requested_gpu"].notna()
        & crossovers["mode"].eq("compiled-gpu-resident")
    ]
    h64 = modal_resident[modal_resident["observation_horizon"].eq(64)]
    h1 = modal_resident[modal_resident["observation_horizon"].eq(1)]
    complete = observed_requested_gpus == EXPECTED_MODAL_GPUS
    return {
        "complete_expected_modal_gpu_set": complete,
        "expected_modal_gpus": sorted(EXPECTED_MODAL_GPUS),
        "observed_modal_gpus": sorted(observed_requested_gpus),
        "G1_all_datacenter_resident_h64_cross_at_n8": bool(
            complete
            and len(h64) == len(EXPECTED_MODAL_GPUS)
            and h64["smallest_tested_crossover_n_best_cpu"].eq(8).all()
        ),
        "G2_all_datacenter_resident_h1_cross_by_n256": bool(
            complete
            and len(h1) == len(EXPECTED_MODAL_GPUS)
            and h1["smallest_tested_crossover_n_best_cpu"].notna().all()
            and h1["smallest_tested_crossover_n_best_cpu"].le(256).all()
        ),
        "G3_h64_proportional_fusion_advantage_gt_1_everywhere": bool(
            fusion["shape_valid"].all()
            and fusion["gpu_proportional_fusion_advantage"].gt(1).all()
        ),
        "G4_cheaper_gpu_beats_h100_cost_in_a_matched_cell": bool(
            complete and cheaper_card_beats_h100(cost)
        ),
        "G5_host_visible_crossover_is_never_smaller": bool(
            complete and host_crossover_order(crossovers)
        ),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.raw_dir.glob("pilot-*-*sub256-*.csv"))
    if len(paths) < 2:
        raise FileNotFoundError("expected at least pilots 005/006 sub-256 ledgers")

    raw, sources = load_runs(paths)
    metadata = source_metadata(paths)
    tables = add_source_metadata(
        build_tables(
            raw,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
        metadata,
    )
    pricing = json.loads(args.pricing.read_text())
    cost = gpu_cost_table(tables["hardware-sweep-speedups.csv"], pricing)
    tables["hardware-sweep-gpu-cost.csv"] = cost

    observed_requested_gpus = set(metadata["requested_gpu"].dropna())
    hypotheses = evaluate_hypotheses(
        tables["hardware-sweep-crossovers.csv"],
        tables["hardware-sweep-temporal-fusion.csv"],
        cost,
        observed_requested_gpus,
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
        "preregistrations": [
            "preregistration/pilot-005.md",
            "preregistration/pilot-007.md",
        ],
        "primary_cpu_reference": "faster median compiled CPU cell among 1 and 8 threads",
        "primary_cross_card_reference": "synchronized GPU wall time at identical workload shape",
        "pricing": {
            "path": str(args.pricing),
            "sha256": sha256_file(args.pricing),
            "captured_at": pricing["captured_at"],
            "source": pricing["source"],
            "scope": "GPU-only marginal steady-state cost",
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "interval": "independent percentile bootstrap of median ratio",
            "seed": args.seed,
        },
        "hypotheses": hypotheses,
        "sources": sources,
        "outputs": output_metadata,
    }
    manifest_path = args.output_dir / "hardware-sweep-analysis-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(tables["hardware-sweep-run-quality.csv"].to_string(index=False))
    print("\nHypotheses:")
    print(json.dumps(hypotheses, indent=2, sort_keys=True))
    print(f"\nmanifest={manifest_path}")


if __name__ == "__main__":
    main()
