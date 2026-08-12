"""Generate manuscript numbers and tables from the canonical processed evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "arxiv" / "generated"
TRACE = ROOT / "data" / "processed" / "trace-exact-packing-summary.csv"
TRACE_REPETITIONS = ROOT / "data" / "processed" / "trace-exact-packing-repetitions.csv"
TRACE_MANIFEST = ROOT / "data" / "processed" / "trace-exact-packing-manifest.json"
TRACE_SOURCE_MANIFEST = ROOT / "data" / "processed" / "exgentic-tau2-source-manifest.json"
RESIDENT = ROOT / "data" / "processed" / "resident-policy-pilot-contrasts.csv"
RESIDENT_CELLS = ROOT / "data" / "processed" / "resident-policy-pilot-cell-summary.csv"
RESIDENT_MANIFEST = ROOT / "data" / "processed" / "resident-policy-pilot-manifest.json"
NATIVE = ROOT / "data" / "processed" / "native-dispatch-pilot-contrasts.csv"
NATIVE_MANIFEST = ROOT / "data" / "processed" / "native-dispatch-pilot-manifest.json"
FIGURES = [
    {
        "path": ROOT / "results" / "figures" / "trace-exact-opportunity-frontier.png",
        "source_table": TRACE,
        "notebook_builder": ROOT / "scripts" / "build_exact_native_notebook.py",
        "notebook": ROOT / "notebooks" / "09_exact_boundary_and_native_calibration.ipynb",
    },
    {
        "path": ROOT / "results" / "figures" / "resident-policy-speedup-by-horizon.png",
        "source_table": RESIDENT,
        "notebook_builder": ROOT / "scripts" / "build_resident_policy_notebook.py",
        "notebook": ROOT / "notebooks" / "10_resident_policy_pilot.ipynb",
    },
    {
        "path": ROOT / "results" / "figures" / "resident-policy-primary-mechanisms.png",
        "source_table": RESIDENT_CELLS,
        "notebook_builder": ROOT / "scripts" / "build_resident_policy_notebook.py",
        "notebook": ROOT / "notebooks" / "10_resident_policy_pilot.ipynb",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}"


def command(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def verify_declared_hash(manifest: dict, path: Path, declared: str) -> None:
    actual = sha256(path)
    require(actual == declared, f"hash mismatch for {path}: {actual} != {declared}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    trace_manifest = json.loads(TRACE_MANIFEST.read_text(encoding="utf-8"))
    trace_source_manifest = json.loads(TRACE_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    resident_manifest = json.loads(RESIDENT_MANIFEST.read_text(encoding="utf-8"))
    native_manifest = json.loads(NATIVE_MANIFEST.read_text(encoding="utf-8"))

    verify_declared_hash(
        trace_manifest,
        TRACE,
        trace_manifest["outputs"]["summary"]["sha256"],
    )
    verify_declared_hash(
        trace_manifest,
        TRACE_REPETITIONS,
        trace_manifest["outputs"]["repetitions"]["sha256"],
    )
    for source in trace_manifest["source_files"].values():
        source_path = ROOT / source["path"]
        verify_declared_hash(trace_manifest, source_path, source["sha256"])
    for source in trace_source_manifest["source_files"]:
        source_path = ROOT / source["path"]
        verify_declared_hash(trace_source_manifest, source_path, source["sha256"])
    for output in trace_source_manifest["outputs"].values():
        output_path = ROOT / output["path"]
        verify_declared_hash(trace_source_manifest, output_path, output["sha256"])
    require(
        trace_source_manifest["source_conversion_revision"]
        == "f7c94012d0bfbf66fe4d6ed627699508bbb555ff",
        "trace conversion revision changed",
    )
    require(
        trace_source_manifest["source_verification"]["commit_resolved_hash_matches"] == 19
        and trace_source_manifest["source_verification"]["commit_resolved_hash_failures"] == 0,
        "trace source verification is incomplete",
    )
    trace_repetition_rows = rows(TRACE_REPETITIONS)
    require(
        len(trace_repetition_rows) == trace_manifest["outputs"]["repetitions"]["rows"],
        "trace repetition row count changed",
    )
    require(
        all(
            float(row["fixed_window_eligible_share"])
            <= float(row["exact_optimal_share"])
            <= float(row["local_upper_share"])
            for row in trace_repetition_rows
        ),
        "trace boundary invariant failed",
    )
    require(all(trace_manifest["hypotheses"].values()), "recorded trace validity gate failed")
    verify_declared_hash(
        resident_manifest,
        RESIDENT,
        resident_manifest["outputs"]["contrasts"]["sha256"],
    )
    verify_declared_hash(
        resident_manifest,
        RESIDENT_CELLS,
        resident_manifest["outputs"]["cell_summary"]["sha256"],
    )
    verify_declared_hash(
        native_manifest,
        NATIVE,
        native_manifest["outputs"]["contrasts"]["sha256"],
    )

    trace_rows = rows(TRACE)
    trace_primary = [
        row
        for row in trace_rows
        if int(row["target_active_sessions"]) == 100_000
        and int(row["deadline_ms"]) == 50
        and row["grouping"] == "route_key"
        and int(row["threshold_k"]) == 256
    ]
    require(len(trace_primary) == 1, "expected one trace primary row")
    trace = trace_primary[0]
    f_share = float(trace["fixed_window_eligible_share_mean"])
    p_share = float(trace["exact_optimal_share_mean"])
    u_share = float(trace["local_upper_share_mean"])
    gap = float(trace["alignment_gap_closure_mean"])
    require(0.0 <= f_share <= p_share <= u_share <= 1.0, "primary bound failed")

    def trace_cell(active: int, deadline: int, threshold: int) -> dict[str, str]:
        matches = [
            row
            for row in trace_rows
            if int(row["target_active_sessions"]) == active
            and int(row["deadline_ms"]) == deadline
            and row["grouping"] == "route_key"
            and int(row["threshold_k"]) == threshold
        ]
        require(len(matches) == 1, f"missing trace cell {(active, deadline, threshold)}")
        return matches[0]

    p_at_100k_50ms = {
        threshold: float(trace_cell(100_000, 50, threshold)["exact_optimal_share_mean"])
        for threshold in (32, 64, 128, 256)
    }
    p_at_10k_50ms = {
        threshold: float(trace_cell(10_000, 50, threshold)["exact_optimal_share_mean"])
        for threshold in (32, 64, 128, 256)
    }
    require(
        all(
            float(row["exact_optimal_share_mean"]) == 0.0
            for row in trace_rows
            if int(row["target_active_sessions"]) <= 10_000
            and row["grouping"] == "route_key"
            and int(row["threshold_k"]) == 256
        ),
        "K=256 low-concurrency zero boundary changed",
    )
    require(
        all(
            float(trace_cell(100_000, deadline, 256)["exact_optimal_share_mean"]) == 0.0
            for deadline in (10, 25)
        ),
        "K=256 short-deadline zero boundary changed",
    )

    resident_rows = rows(RESIDENT)
    resident_cells = rows(RESIDENT_CELLS)
    require(len(resident_rows) == 36, "resident contrast row count changed")
    placements = sorted({row["placement_id"] for row in resident_rows})
    require(len(placements) == 4, "resident placement count changed")
    resident_ratios = [float(row["host_over_resident_ratio_of_medians"]) for row in resident_rows]
    require(min(resident_ratios) > 1.0, "resident mechanism no longer wins every cell")

    primary_resident = [
        row for row in resident_rows if int(row["agents"]) == 256 and int(row["epochs"]) == 32
    ]
    require(len(primary_resident) == 4, "expected four resident primary rows")

    cell_index = {
        (row["placement_id"], int(row["agents"]), int(row["epochs"]), row["mechanism"]): row
        for row in resident_cells
    }
    require(len(cell_index) == len(resident_cells), "duplicate resident cell")
    primary_resident_us: list[float] = []
    primary_host_us: list[float] = []
    primary_floor_ratios: list[float] = []
    primary_saved_us: list[float] = []
    for row in primary_resident:
        key = (row["placement_id"], 256, 32)
        primary_resident_us.append(
            float(cell_index[(*key, "device_resident")]["wall_ns_median"]) / 1000.0
        )
        primary_host_us.append(
            float(cell_index[(*key, "host_roundtrip")]["wall_ns_median"]) / 1000.0
        )
        primary_floor_ratios.append(float(row["resident_over_floor_ratio_of_medians"]))
        primary_saved_us.append(float(row["wall_ns_saved_per_invocation"]) / 1000.0)

    native_rows = rows(NATIVE)
    native_ratios = [
        float(row["cuda_device_graph_over_cuda_host_graph_wall_ratio_of_medians"])
        for row in native_rows
    ]
    require(len(native_rows) == 60, "native contrast row count changed")
    require(
        len({row["placement_id"] for row in native_rows}) == 5, "native placement count changed"
    )
    require(min(native_ratios) > 1.0, "fixed nested graph no longer loses every cell")

    outcomes = resident_manifest["pilot_outcomes"]
    validated_invocations = sum(int(row["validated_invocations"]) for row in resident_cells)
    legal_validated_invocations = sum(
        int(row["validated_invocations"])
        for row in resident_cells
        if row["mechanism"] in {"host_roundtrip", "device_resident"}
    )
    macros = [
        "% Generated by scripts/build_paper_artifacts.py. Do not edit.",
        command("TraceSessions", str(trace_source_manifest["quality"]["session_rows"])),
        command(
            "TraceSpans",
            f"{int(trace_source_manifest['quality']['span_rows']):,}".replace(",", "{,}"),
        ),
        command(
            "TracePrimaryEvents", f"{float(trace['event_count_mean']):,.0f}".replace(",", "{,}")
        ),
        command("TraceFixedPct", pct(f_share)),
        command("TraceExactPct", pct(p_share)),
        command("TraceUpperPct", pct(u_share)),
        command("TraceGapClosurePct", pct(gap)),
        command("TraceFixedToExactPP", f"{100.0 * (p_share - f_share):.2f}"),
        command("TraceExactToUpperPP", f"{100.0 * (u_share - p_share):.2f}"),
        command(
            "TraceExactBatches",
            f"{float(trace['exact_batch_count_mean']):,.1f}".replace(",", "{,}"),
        ),
        command("TracePrimaryExactMinPct", pct(float(trace["exact_optimal_share_min"]))),
        command("TracePrimaryExactMaxPct", pct(float(trace["exact_optimal_share_max"]))),
        command("TracePAtTenKThirtyTwoPct", pct(p_at_10k_50ms[32], 1)),
        command("TracePAtTenKSixtyFourPct", pct(p_at_10k_50ms[64], 1)),
        command("TracePAtHundredKThirtyTwoPct", pct(p_at_100k_50ms[32], 1)),
        command("TracePAtHundredKSixtyFourPct", pct(p_at_100k_50ms[64], 1)),
        command("TracePAtHundredKOneTwentyEightPct", pct(p_at_100k_50ms[128], 1)),
        command("TracePAtHundredKTwoFiftySixPct", pct(p_at_100k_50ms[256], 1)),
        command("TraceReplayCells", str(len(trace_rows))),
        command("TraceReplayRepetitions", str(trace_manifest["outputs"]["repetitions"]["rows"])),
        command("ResidentPlacements", str(len(placements))),
        command("ResidentCells", str(len(resident_rows))),
        command("ResidentRows", f"{int(outcomes['raw_rows']):,}".replace(",", "{,}")),
        command(
            "ResidentInvocations",
            f"{validated_invocations:,}".replace(",", "{,}"),
        ),
        command(
            "ResidentLegalInvocations",
            f"{legal_validated_invocations:,}".replace(",", "{,}"),
        ),
        command("ResidentRatioMin", f"{min(resident_ratios):.2f}"),
        command("ResidentRatioMax", f"{max(resident_ratios):.2f}"),
        command("ResidentPrimaryUsMin", f"{min(primary_resident_us):.0f}"),
        command("ResidentPrimaryUsMax", f"{max(primary_resident_us):.0f}"),
        command("HostPrimaryUsMin", f"{min(primary_host_us):.0f}"),
        command("HostPrimaryUsMax", f"{max(primary_host_us):.0f}"),
        command("PrimarySavedUsMin", f"{min(primary_saved_us):.0f}"),
        command("PrimarySavedUsMax", f"{max(primary_saved_us):.0f}"),
        command("PrimaryFloorRatioMin", f"{min(primary_floor_ratios):.2f}"),
        command("PrimaryFloorRatioMax", f"{max(primary_floor_ratios):.2f}"),
        command("NativePlacements", str(len({row["placement_id"] for row in native_rows}))),
        command("NativeCells", str(len(native_rows))),
        command("NativeRows", f"{native_manifest['design']['raw_rows']:,}".replace(",", "{,}")),
        command("NativeRatioMin", f"{min(native_ratios):.2f}"),
        command("NativeRatioMax", f"{max(native_ratios):.2f}"),
    ]
    (OUT / "results-macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    trace_table = rf"""% Generated by scripts/build_paper_artifacts.py. Do not edit.
\begin{{tabular}}{{@{{}}lr@{{}}}}
  \toprule
  Quantity & Primary value \\
  \midrule
  Fixed-partition share $F$ & {pct(f_share)}\% \\
  Exact offline share $P^\star$ & {pct(p_share)}\% \\
  Local-overlap bound $U$ & {pct(u_share)}\% \\
  Alignment-gap closure & {pct(gap)}\% \\
  Mean generated events & {float(trace["event_count_mean"]):,.0f} \\
  Mean exact batches & {float(trace["exact_batch_count_mean"]):,.1f} \\
  \bottomrule
\end{{tabular}}
""".replace(",", "{,}")
    (OUT / "trace-primary-table.tex").write_text(trace_table, encoding="utf-8")

    provider_order = {"local": 0, "modal": 1, "runpod": 2, "lambda": 3}
    short_gpu = {
        "local": "GTX 1660 Ti",
        "modal": "L4",
        "runpod": "L4",
        "lambda": "H100 SXM5",
    }
    provider_label = {"local": "Local", "modal": "Modal", "runpod": "RunPod", "lambda": "Lambda"}
    resident_lines = [
        "% Generated by scripts/build_paper_artifacts.py. Do not edit.",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"  \toprule",
        r"  Provider & GPU & Ratio & Resident ($\mu$s) & Host ($\mu$s) & Floor ($\mu$s) \\",
        r"  \midrule",
    ]
    for row in sorted(primary_resident, key=lambda item: provider_order[item["provider"]]):
        key = (row["placement_id"], 256, 32)
        resident = float(cell_index[(*key, "device_resident")]["wall_ns_median"]) / 1000.0
        host = float(cell_index[(*key, "host_roundtrip")]["wall_ns_median"]) / 1000.0
        floor = float(cell_index[(*key, "no_decision_lower_bound")]["wall_ns_median"]) / 1000.0
        resident_lines.append(
            f"  {provider_label[row['provider']]} & {short_gpu[row['provider']]} & "
            f"{float(row['host_over_resident_ratio_of_medians']):.2f}$\\times$ & "
            f"{resident:.1f} & {host:.1f} & {floor:.1f} \\\\"
        )
    resident_lines.extend([r"  \bottomrule", r"\end{tabular}"])
    (OUT / "resident-primary-table.tex").write_text(
        "\n".join(resident_lines) + "\n", encoding="utf-8"
    )

    inputs = [
        TRACE,
        TRACE_REPETITIONS,
        TRACE_MANIFEST,
        TRACE_SOURCE_MANIFEST,
        RESIDENT,
        RESIDENT_CELLS,
        RESIDENT_MANIFEST,
        NATIVE,
        NATIVE_MANIFEST,
    ]
    generated = [
        OUT / "results-macros.tex",
        OUT / "trace-primary-table.tex",
        OUT / "resident-primary-table.tex",
    ]
    manifest = {
        "schema_version": "ready-cohort-paper-artifacts-v1",
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in inputs
        ],
        "figures": [
            {
                "path": str(item["path"].relative_to(ROOT)),
                "sha256": sha256(item["path"]),
                "source_table": str(item["source_table"].relative_to(ROOT)),
                "source_table_sha256": sha256(item["source_table"]),
                "notebook_builder": str(item["notebook_builder"].relative_to(ROOT)),
                "notebook_builder_sha256": sha256(item["notebook_builder"]),
                "notebook": str(item["notebook"].relative_to(ROOT)),
                "notebook_sha256": sha256(item["notebook"]),
            }
            for item in FIGURES
        ],
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in generated
        ],
        "sampling_units": {
            "trace": "Monte Carlo seed conditional on one fixed trace panel",
            "resident_performance": "named GPU placement",
            "timing_rows": "technical repetitions",
        },
    }
    (OUT / "paper-data-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
