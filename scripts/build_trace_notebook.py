from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/02_trace_ready_cohort_analysis.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# Fixed-window ready cohorts in public agent traces

## tl;dr

This notebook asks a stricter question than “are there many agents?”: **how
many same-operator events are ready at the same time?** It replays a pinned,
content-free 851-session tau2 trace panel under stationary swarm load and
measures the event share that can form cohorts of at least `K` within a fixed
latency window.

At 100,000 mean active sessions and a candidate crossover `K=256`, pooling all
events makes 85.6% eligible in a 25 ms window, while exact-route grouping makes
0% eligible. Exact routes reach only 30.3% at 50 ms and 47.9% at 100 ms. The
gap is the empirical regularity tax: population-scale parallelism is not the
same as GPU-ready parallelism. The result is exact for the frozen
non-overlapping partition; arbitrary sliding deadlines require the separate
packing bounds in `paper/formalism.md`.
"""
    ),
    markdown(
        """
## Context & Methods

The source panel is the complete tau2 subset of
`Exgentic/agent-llm-traces` at revision
`70036b93a04e61b0ea2706a68b962f4f26774587`. Derived features contain
timestamps, counts, lengths, route labels, and public identifiers, but no
prompt text, tool arguments, or tool results.

Each recorded span completion is a candidate control event. Independent
session arrivals follow a stationary Poisson process. The arrival rate is set
so the expected active session population equals `C`; empirical session
templates retain their relative timing and route sequence. Events are assigned
to fixed microbatch windows under three nested execution models:

1. **pooled** — a hypothetical universal transition can fuse every event;
2. **event class** — final/text, tool call, and error events are distinct;
3. **exact route** — every named tool route is distinct.

Eligibility is event-weighted: an event is eligible at threshold `K` if its
same-group cohort contains at least `K` events. The design, hypotheses,
thresholds, and exclusions were frozen in
`preregistration/trace-replay-001.md` before cohort computation.
"""
    ),
    code(
        """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
PROCESSED_DIR = ROOT / "data/processed"
FIGURE_DIR = ROOT / "results/figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

spans = pd.read_parquet(PROCESSED_DIR / "exgentic-tau2-span-features.parquet")
sessions = pd.read_csv(PROCESSED_DIR / "exgentic-tau2-session-summary.csv")
repetitions = pd.read_csv(PROCESSED_DIR / "trace-ready-cohort-repetitions.csv")
summary = pd.read_csv(PROCESSED_DIR / "trace-ready-cohort-summary.csv")
source_manifest = json.loads(
    (PROCESSED_DIR / "exgentic-tau2-source-manifest.json").read_text()
)
replay_manifest = json.loads(
    (PROCESSED_DIR / "trace-ready-cohort-manifest.json").read_text()
)

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {
    "pooled": "#0B6E75",
    "event_class": "#D97706",
    "route_key": "#7C3AED",
}

len(spans), len(sessions), len(repetitions), len(summary)
"""
    ),
    markdown("## Data quality"),
    code(
        """
replay_runs = repetitions.drop_duplicates(["target_active_sessions", "repetition"])
quality = pd.DataFrame(
    {
        "measure": [
            "sessions",
            "events",
            "duplicate session IDs",
            "invalid JSON fields",
            "failed/nonpositive spans retained",
            "replay repetitions per population",
            "minimum realized/target active ratio",
            "maximum realized/target active ratio",
        ],
        "value": [
            len(sessions),
            len(spans),
            int(sessions["session_id"].duplicated().sum()),
            int(sessions["invalid_json_fields"].sum()),
            int(sessions["failed_status_spans"].sum()),
            int(replay_runs.groupby("target_active_sessions").size().min()),
            replay_runs["active_population_ratio"].min(),
            replay_runs["active_population_ratio"].max(),
        ],
    }
)
quality
"""
    ),
    code(
        """
required = ["session_id", "benchmark", "harness", "ready_offset_s", "route_key"]
assert spans[required].isna().sum().sum() == 0
assert sessions["session_id"].is_unique
assert source_manifest["source_revision"] == "70036b93a04e61b0ea2706a68b962f4f26774587"
assert replay_manifest["quality"]["grouping_hierarchy_checked"] is True
assert set(replay_runs.groupby("target_active_sessions").size()) == {5}

hierarchy = summary.pivot_table(
    index=["target_active_sessions", "window_ms", "threshold_k"],
    columns="grouping",
    values="eligible_share_mean",
)
assert (hierarchy["pooled"] + 1e-12 >= hierarchy["event_class"]).all()
assert (hierarchy["event_class"] + 1e-12 >= hierarchy["route_key"]).all()

active_validation = (
    replay_runs.groupby("target_active_sessions")["active_population_ratio"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)
active_validation
"""
    ),
    markdown(
        """
The low-population replay has wider Monte Carlo variation because only about
ten events per second are observed at `C=100`. The two largest populations,
which drive the crossover analysis, realize their target active population to
within roughly 1.4% in every repetition.
"""
    ),
    markdown("## Workload structure"),
    code(
        """
session_structure = sessions[
    [
        "span_count",
        "session_duration_s",
        "route_count",
        "dominant_route_share",
        "effective_route_count",
        "peak_recorded_span_concurrency",
    ]
].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
session_structure
"""
    ),
    code(
        """
route_frequency = (
    spans.groupby(["event_class", "route_key"], as_index=False)
    .size()
    .sort_values("size", ascending=False)
)
route_frequency["event_share"] = route_frequency["size"] / len(spans)
route_frequency.head(20)
"""
    ),
    markdown(
        """
The route distribution is concentrated but not homogeneous. Final/text events
are the largest exact route, while tool calls split across a long tail of named
operators. That distinction is precisely what pooled synthetic batches hide.
"""
    ),
    markdown("## Results"),
    code(
        """
k = 256
exact = summary[(summary["grouping"] == "route_key") & (summary["threshold_k"] == k)]
heatmap = exact.pivot(
    index="target_active_sessions", columns="window_ms", values="eligible_share_mean"
)

fig, axis = plt.subplots(figsize=(11.2, 4.4))
sns.heatmap(
    heatmap * 100,
    annot=True,
    fmt=".1f",
    cmap="mako",
    vmin=0,
    vmax=100,
    linewidths=0.7,
    linecolor="white",
    cbar_kws={"label": "Eligible events (%)"},
    ax=axis,
)
axis.set_title("Exact-route fixed-window eligibility at K=256")
axis.set_xlabel("Maximum fixed microbatch window (ms)")
axis.set_ylabel("Mean active sessions")
axis.set_yticklabels([f"{int(value):,}" for value in heatmap.index], rotation=0)
fig.tight_layout()
for suffix in ("png", "svg"):
    fig.savefig(
        FIGURE_DIR / f"trace-ready-cohort-exact-k256-heatmap.{suffix}",
        dpi=220 if suffix == "png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
plt.show()
"""
    ),
    code(
        """
frontier = summary[
    (summary["target_active_sessions"] == 100_000)
    & (summary["threshold_k"] == 256)
].copy()

fig, axis = plt.subplots(figsize=(8.8, 5.0))
for grouping, label in [
    ("pooled", "Universal transition (pooled)"),
    ("event_class", "Coarse event class"),
    ("route_key", "Exact route"),
]:
    series = frontier[frontier["grouping"] == grouping].sort_values("window_ms")
    axis.plot(
        series["window_ms"],
        series["eligible_share_mean"] * 100,
        marker="o",
        linewidth=2.2,
        markersize=5,
        color=PALETTE[grouping],
        label=label,
    )
    axis.fill_between(
        series["window_ms"],
        series["eligible_share_min"] * 100,
        series["eligible_share_max"] * 100,
        color=PALETTE[grouping],
        alpha=0.10,
        linewidth=0,
    )
axis.set_xscale("log")
axis.set_xticks(sorted(frontier["window_ms"].unique()))
axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
axis.set_ylim(-2, 102)
axis.set_xlabel("Maximum fixed microbatch window (ms, log scale)")
axis.set_ylabel("Events in cohorts of at least 256 (%)")
axis.set_title("Regularity tax at 100,000 mean active sessions")
axis.legend(frameon=False, loc="lower right")
axis.grid(True, which="major", color="#E5E7EB")
fig.tight_layout()
for suffix in ("png", "svg"):
    fig.savefig(
        FIGURE_DIR / f"trace-ready-cohort-grouping-frontier-k256.{suffix}",
        dpi=220 if suffix == "png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
plt.show()
"""
    ),
    code(
        """
selected = frontier.pivot(
    index="window_ms", columns="grouping", values="eligible_share_mean"
)
selected["pooled_minus_exact"] = selected["pooled"] - selected["route_key"]
selected["coarse_minus_exact"] = selected["event_class"] - selected["route_key"]
(selected * 100).round(1)
"""
    ),
    code(
        """
threshold_frontier = summary[
    (summary["target_active_sessions"] == 100_000)
    & (summary["grouping"] == "route_key")
].pivot(index="threshold_k", columns="window_ms", values="eligible_share_mean")
(threshold_frontier * 100).round(1)
"""
    ),
    code(
        """
milestones = []
for grouping in ["pooled", "event_class", "route_key"]:
    for window_ms in sorted(summary["window_ms"].unique()):
        candidates = summary[
            (summary["grouping"] == grouping)
            & (summary["threshold_k"] == 256)
            & (summary["window_ms"] == window_ms)
            & (summary["eligible_share_mean"] >= 0.50)
        ].sort_values("target_active_sessions")
        milestones.append(
            {
                "grouping": grouping,
                "window_ms": window_ms,
                "smallest_tested_C_at_50pct": (
                    None if candidates.empty else int(candidates.iloc[0]["target_active_sessions"])
                ),
            }
        )
pd.DataFrame(milestones).pivot(
    index="grouping", columns="window_ms", values="smallest_tested_C_at_50pct"
)
"""
    ),
    markdown(
        """
## Takeaways

1. **Readiness, not population, is the binding quantity.** At `K=256`, exact
   routes do not reach 50% eligibility anywhere below the tested combination
   of 100,000 active sessions and a 250 ms window.
2. **Heterogeneity can erase an apparent GPU opportunity.** At 100,000 active
   sessions and 25 ms, a universal pooled kernel sees 85.6% eligible events;
   exact routes see none. At 50 ms the corresponding values are 100.0% and
   30.3%.
3. **Lowering the hardware crossover is extremely valuable.** At 100,000
   active sessions and 10 ms, exact-route eligibility is 48.0% for `K=32`,
   3.6% for `K=64`, and zero for `K>=128`.
4. **Longer windows eventually expose the route-frequency boundary.** At one
   second and `K=4096`, 47.9% of events are eligible—the share dominated by the
   single largest route—while the rest remain fragmented.
5. **This is fixed-window eligibility, not a universal ceiling or end-to-end
   speedup claim.** A sliding-deadline scheduler can form cohorts across a
   frozen boundary. Kernel cost, transfers, compilation, external queueing,
   tool capacity, and task utility also remain to be measured. The
   compiler-matched hardware pilots provide the `K` side of the equation; this
   trace study provides one controlled workload-side policy.
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
