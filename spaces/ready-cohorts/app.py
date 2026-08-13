from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ASSET_DIR = ROOT / "assets"
MANIFEST_PATH = DATA_DIR / "manifest.json"

PAPER_URL = "https://arxiv.org/abs/2608.12123"
DATA_URL = "https://huggingface.co/datasets/josefchen/ready-cohorts/tree/ready-cohorts-arxiv-v1"
CODE_URL = "https://github.com/josefchen/ready-cohorts/tree/main/spaces/ready-cohorts"
TRACE_URL = (
    "https://huggingface.co/datasets/Exgentic/agent-llm-traces/tree/"
    "f7c94012d0bfbf66fe4d6ed627699508bbb555ff"
)

ACCENT = "#3159c7"
ACCENT_DARK = "#24449a"
INK = "#20242b"
MUTED = "#656d7a"
LINE = "#cbd1db"
PLOT_BG = "#f7f8fa"


@dataclass(frozen=True)
class Evidence:
    trace: pd.DataFrame
    resident_cells: pd.DataFrame
    resident_contrasts: pd.DataFrame
    native_contrasts: pd.DataFrame
    dataset_commit: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def load_evidence() -> Evidence:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    loaded: dict[str, pd.DataFrame] = {}

    for item in manifest.get("assets", []):
        path = ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing released asset: {item['path']}")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Hash mismatch for {item['path']}")

    for item in manifest["files"]:
        path = DATA_DIR / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing evidence file: {item['path']}")
        actual_hash = sha256(path)
        if actual_hash != item["sha256"]:
            raise ValueError(f"Hash mismatch for {item['path']}")
        frame = pd.read_csv(path)
        if len(frame) != item["rows"]:
            raise ValueError(f"Unexpected row count for {item['path']}")
        loaded[item["key"]] = frame

    trace = loaded["trace"]
    resident_cells = loaded["resident_cells"]
    resident_contrasts = loaded["resident_contrasts"]
    native_contrasts = loaded["native_contrasts"]

    require_columns(
        trace,
        {
            "target_active_sessions",
            "deadline_ms",
            "grouping",
            "threshold_k",
            "repetitions",
            "event_count_mean",
            "fixed_window_eligible_share_mean",
            "exact_optimal_share_mean",
            "local_upper_share_mean",
            "alignment_gap_closure_mean",
        },
        "trace summary",
    )
    require_columns(
        resident_cells,
        {
            "placement_id",
            "provider",
            "agents",
            "epochs",
            "mechanism",
            "wall_ns_median",
            "validated_invocations",
        },
        "resident cell summary",
    )
    require_columns(
        resident_contrasts,
        {
            "placement_id",
            "provider",
            "agents",
            "epochs",
            "host_over_resident_ratio_of_medians",
            "wall_ns_saved_per_invocation",
        },
        "resident contrasts",
    )
    require_columns(
        native_contrasts,
        {
            "placement_id",
            "provider",
            "agents",
            "steps",
            "cuda_device_graph_over_cuda_host_graph_wall_ratio_of_medians",
        },
        "native contrasts",
    )

    primary = trace[
        (trace["target_active_sessions"] == 100_000)
        & (trace["deadline_ms"] == 50)
        & (trace["grouping"] == "route_key")
        & (trace["threshold_k"] == 256)
    ]
    if len(primary) != 1:
        raise ValueError("The released trace primary cell is not unique")
    expected = {
        "fixed_window_eligible_share_mean": 0.301902,
        "exact_optimal_share_mean": 0.430007,
        "local_upper_share_mean": 0.458487,
    }
    for column, value in expected.items():
        if not math.isclose(float(primary.iloc[0][column]), value, abs_tol=1e-6):
            raise ValueError(f"The released trace primary value changed for {column}")

    if resident_contrasts["placement_id"].nunique() != 4:
        raise ValueError("The resident study must contain four named placements")
    if native_contrasts["placement_id"].nunique() != 5:
        raise ValueError("The negative control must contain five named placements")

    return Evidence(
        trace=trace,
        resident_cells=resident_cells,
        resident_contrasts=resident_contrasts,
        native_contrasts=native_contrasts,
        dataset_commit=manifest["dataset_commit"],
    )


try:
    EVIDENCE = load_evidence()
    EVIDENCE_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised by deployment failures
    EVIDENCE = None
    EVIDENCE_ERROR = f"{type(exc).__name__}: {exc}"


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": MUTED},
    )
    figure.update_layout(
        height=400,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin={"l": 24, "r": 24, "t": 24, "b": 24},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def base_layout(title: str, subtitle: str, height: int = 430) -> dict:
    return {
        "height": height,
        "paper_bgcolor": PLOT_BG,
        "plot_bgcolor": PLOT_BG,
        "font": {"family": "Arial, sans-serif", "size": 13, "color": INK},
        "title": {
            "text": f"{title}<br><sup>{subtitle}</sup>",
            "x": 0,
            "xanchor": "left",
            "y": 0.95,
            "yanchor": "top",
            "font": {"size": 19, "color": INK},
        },
        "margin": {"l": 68, "r": 32, "t": 112, "b": 64},
        "hoverlabel": {
            "bgcolor": INK,
            "bordercolor": INK,
            "font": {"color": "#f7f8fa", "size": 13},
        },
        "legend": {
            "orientation": "h",
            "x": 0,
            "y": 1.01,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 12},
        },
    }


GROUPING_LABELS = {
    "route_key": "Outcome route key",
    "event_class": "Outcome event class",
    "pooled": "Pooled events",
}


def trace_view(
    target_active_sessions: int,
    threshold_k: int,
    grouping: str,
    selected_deadline_ms: int,
) -> tuple[go.Figure, str]:
    if EVIDENCE is None:
        return empty_figure("Released evidence could not be verified."), error_readout()

    target_active_sessions = int(target_active_sessions)
    threshold_k = int(threshold_k)
    selected_deadline_ms = int(selected_deadline_ms)
    subset = EVIDENCE.trace[
        (EVIDENCE.trace["target_active_sessions"] == target_active_sessions)
        & (EVIDENCE.trace["threshold_k"] == threshold_k)
        & (EVIDENCE.trace["grouping"] == grouping)
    ].sort_values("deadline_ms")

    selected = subset[subset["deadline_ms"] == selected_deadline_ms]
    if subset.empty or len(selected) != 1:
        return empty_figure("No released cell matches these controls."), empty_readout()

    repetitions = int(subset["repetitions"].iloc[0])
    events = int(round(float(subset["event_count_mean"].iloc[0])))
    subtitle = (
        f"C={target_active_sessions:,}; K={threshold_k}; "
        f"{GROUPING_LABELS[grouping].lower()}; {repetitions} generated swarms"
    )
    figure = go.Figure()
    series = [
        (
            "Frozen partition F",
            "fixed_window_eligible_share_mean",
            MUTED,
            "dash",
            "square",
        ),
        (
            "Exact optimum P*",
            "exact_optimal_share_mean",
            ACCENT,
            "solid",
            "circle",
        ),
        (
            "Local upper bound U",
            "local_upper_share_mean",
            ACCENT_DARK,
            "dot",
            "triangle-up-open",
        ),
    ]
    for label, column, color, dash, symbol in series:
        figure.add_trace(
            go.Scatter(
                x=subset["deadline_ms"],
                y=subset[column],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": 3, "dash": dash},
                marker={
                    "color": color,
                    "size": 10,
                    "symbol": symbol,
                    "line": {"color": color, "width": 2},
                },
                customdata=subset["deadline_ms"],
                hovertemplate=(
                    "%{fullData.name}<br>Deadline %{customdata} ms"
                    "<br>Eligible share %{y:.2%}<extra></extra>"
                ),
            )
        )

    figure.add_vline(
        x=math.log10(selected_deadline_ms),
        line={"color": INK, "width": 1.5, "dash": "dashdot"},
        annotation_text=f"Selected: {selected_deadline_ms} ms",
        annotation_position="top right",
        annotation_font={"color": INK, "size": 12},
    )
    figure.update_layout(**base_layout("Eligible event share by launch deadline", subtitle))
    figure.update_xaxes(
        title="Maximum launch wait (ms, log scale)",
        type="log",
        tickmode="array",
        tickvals=[10, 25, 50, 100, 250],
        ticktext=["10", "25", "50", "100", "250"],
        showgrid=False,
        linecolor=INK,
        linewidth=1,
        mirror=False,
    )
    figure.update_yaxes(
        title="Eligible event share",
        range=[0, 1.02],
        tickformat=".0%",
        dtick=0.2,
        gridcolor=LINE,
        zeroline=False,
        linecolor=INK,
        linewidth=1,
    )

    row = selected.iloc[0]
    frozen = float(row["fixed_window_eligible_share_mean"])
    exact = float(row["exact_optimal_share_mean"])
    upper = float(row["local_upper_share_mean"])
    closure = float(row["alignment_gap_closure_mean"])
    added_points = 100 * (exact - frozen)
    closure_text = "Not defined" if math.isnan(closure) else f"{closure:.1%}"

    if exact == 0:
        interpretation = (
            "No route-keyed cohort reaches K under this frozen condition. "
            "This is a measured negative regime, not missing data."
        )
    elif grouping == "pooled":
        interpretation = (
            "Pooling ignores route identity. Treat this as a diagnostic supply ceiling, "
            "not a legal fusion claim."
        )
    elif grouping == "event_class":
        interpretation = (
            "Event class is coarser than the outcome route key. It is a diagnostic "
            "conditioning surface, not proof that events can share code."
        )
    else:
        interpretation = (
            "The route key is outcome-derived. Matching keys condition the replay, "
            "but do not prove executable identity or legal fusion."
        )

    readout = f"""
    <div class="readout-grid" aria-label="Selected trace result">
      <div><span>Frozen F</span><strong>{frozen:.2%}</strong><small>origin-aligned windows</small></div>
      <div><span>Exact P*</span><strong>{exact:.2%}</strong><small>offline optimum</small></div>
      <div><span>Upper U</span><strong>{upper:.2%}</strong><small>local bound</small></div>
      <div><span>Gap closed</span><strong>{closure_text}</strong><small>{added_points:.2f} points added</small></div>
    </div>
    <p class="evidence-note">{html.escape(interpretation)} Each displayed cell is a mean across {repetitions} generated swarms with about {events:,} events per swarm.</p>
    """
    return figure, readout


RESIDENT_LABELS = {
    "local": "Local GTX 1660 Ti",
    "modal": "Modal L4",
    "runpod": "RunPod L4",
    "lambda": "Lambda H100 SXM5",
}
RESIDENT_ORDER = ["local", "modal", "runpod", "lambda"]


def resident_view(agents: int, epochs: int) -> tuple[go.Figure, str]:
    if EVIDENCE is None:
        return empty_figure("Released evidence could not be verified."), error_readout()

    agents = int(agents)
    epochs = int(epochs)
    subset = EVIDENCE.resident_contrasts[
        (EVIDENCE.resident_contrasts["agents"] == agents)
        & (EVIDENCE.resident_contrasts["epochs"] == epochs)
    ].copy()
    subset["order"] = subset["provider"].map(
        {provider: index for index, provider in enumerate(RESIDENT_ORDER)}
    )
    subset = subset.sort_values("order")
    if len(subset) != 4:
        return empty_figure("No released cell matches these controls."), empty_readout()

    labels = [RESIDENT_LABELS[provider] for provider in subset["provider"]]
    ratios = subset["host_over_resident_ratio_of_medians"].astype(float).tolist()
    figure = go.Figure()
    for label, ratio in zip(labels, ratios, strict=True):
        figure.add_trace(
            go.Scatter(
                x=[1, ratio],
                y=[label, label],
                mode="lines",
                line={"color": LINE, "width": 3},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=ratios,
            y=labels,
            mode="markers+text",
            marker={"color": ACCENT, "size": 13, "symbol": "circle"},
            text=[f"{ratio:.2f}×" for ratio in ratios],
            textposition="middle right",
            textfont={"color": INK, "size": 13},
            customdata=subset["wall_ns_saved_per_invocation"] / 1000,
            hovertemplate=(
                "%{y}<br>Host / resident %{x:.3f}×"
                "<br>Saved %{customdata:.1f} μs per cohort invocation<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_vline(
        x=1,
        line={"color": INK, "width": 1.5, "dash": "dot"},
        annotation_text="Equal wall time",
        annotation_position="top left",
        annotation_font={"color": INK, "size": 12},
    )
    subtitle = (
        f"N={agents:,}; H={epochs}; placement is the outer unit; "
        "values above 1 favor the resident path"
    )
    figure.update_layout(
        **base_layout("Host round trip / device resident wall-time ratio", subtitle, 390)
    )
    figure.update_layout(
        showlegend=False,
        margin={"l": 154, "r": 54, "t": 78, "b": 58},
    )
    figure.update_xaxes(
        title="Ratio of within-placement row medians",
        range=[0.95, 2.58],
        dtick=0.25,
        gridcolor=LINE,
        zeroline=False,
        linecolor=INK,
        linewidth=1,
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(labels)),
        showgrid=False,
        linecolor=INK,
        linewidth=1,
    )

    rows = []
    for (_, row), label in zip(subset.iterrows(), labels, strict=True):
        ratio = float(row["host_over_resident_ratio_of_medians"])
        saved_us = float(row["wall_ns_saved_per_invocation"]) / 1000
        rows.append(
            f"<div><span>{html.escape(label)}</span>"
            f"<strong>{ratio:.2f}×</strong>"
            f"<small>{saved_us:.1f} μs saved per cohort invocation</small></div>"
        )
    readout = f"""
    <div class="placement-readout" aria-label="Selected resident mechanism result">
      {"".join(rows)}
    </div>
    <p class="evidence-note">All four selected placement-cells favor the resident path descriptively. The comparison bundles a 4-byte copy, synchronization, host branch selection, redispatch, and different graph topology. It is not a CPU baseline or a placement-population p-value.</p>
    """
    return figure, readout


NATIVE_LABELS = {
    "native-dispatch-001-local": "Local GTX 1660 Ti",
    "native-dispatch-001-modal-l4-p1": "Modal L4 A",
    "native-dispatch-001-modal-l4-p2": "Modal L4 B",
    "native-dispatch-001-runpod-l4-p1": "RunPod L4",
    "native-dispatch-001-lambda-h100-p1": "Lambda H100 SXM5",
}
NATIVE_ORDER = list(NATIVE_LABELS)


def native_view(agents: int, steps: int) -> tuple[go.Figure, str]:
    if EVIDENCE is None:
        return empty_figure("Released evidence could not be verified."), error_readout()

    agents = int(agents)
    steps = int(steps)
    subset = EVIDENCE.native_contrasts[
        (EVIDENCE.native_contrasts["agents"] == agents)
        & (EVIDENCE.native_contrasts["steps"] == steps)
    ].copy()
    subset["order"] = subset["placement_id"].map(
        {placement: index for index, placement in enumerate(NATIVE_ORDER)}
    )
    subset = subset.sort_values("order")
    if len(subset) != 5:
        return empty_figure("No released cell matches these controls."), empty_readout()

    labels = [NATIVE_LABELS[placement] for placement in subset["placement_id"]]
    ratio_column = "cuda_device_graph_over_cuda_host_graph_wall_ratio_of_medians"
    ratios = subset[ratio_column].astype(float).tolist()
    figure = go.Figure()
    for label, ratio in zip(labels, ratios, strict=True):
        figure.add_trace(
            go.Scatter(
                x=[1, ratio],
                y=[label, label],
                mode="lines",
                line={"color": LINE, "width": 3},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=ratios,
            y=labels,
            mode="markers+text",
            marker={
                "color": PLOT_BG,
                "size": 13,
                "symbol": "diamond",
                "line": {"color": ACCENT, "width": 3},
            },
            text=[f"{ratio:.2f}×" for ratio in ratios],
            textposition="middle right",
            textfont={"color": INK, "size": 13},
            hovertemplate=("%{y}<br>Nested / host graph %{x:.3f}×<extra></extra>"),
            showlegend=False,
        )
    )
    figure.add_vline(
        x=1,
        line={"color": INK, "width": 1.5, "dash": "dot"},
        annotation_text="Equal wall time",
        annotation_position="top left",
        annotation_font={"color": INK, "size": 12},
    )
    subtitle = (
        f"N={agents:,}; steps={steps}; placement is the outer unit; "
        "values above 1 mean the nested path is slower"
    )
    figure.update_layout(
        **base_layout("Fixed nested device graph / host graph wall-time ratio", subtitle, 410)
    )
    figure.update_layout(
        showlegend=False,
        margin={"l": 154, "r": 54, "t": 78, "b": 58},
    )
    figure.update_xaxes(
        title="Ratio of within-placement wall-time medians",
        range=[0.95, 2.18],
        dtick=0.2,
        gridcolor=LINE,
        zeroline=False,
        linecolor=INK,
        linewidth=1,
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(labels)),
        showgrid=False,
        linecolor=INK,
        linewidth=1,
    )

    rows = []
    for label, ratio in zip(labels, ratios, strict=True):
        rows.append(
            f"<div><span>{html.escape(label)}</span><strong>{ratio:.2f}× slower</strong></div>"
        )
    readout = f"""
    <div class="placement-readout compact" aria-label="Selected negative control result">
      {"".join(rows)}
    </div>
    <p class="evidence-note">The fixed nested device graph is slower in all 60 released placement-cells. Device-side launch without removing a host decision is therefore retained as a negative control, not promoted as the mechanism.</p>
    """
    return figure, readout


def error_readout() -> str:
    detail = html.escape(EVIDENCE_ERROR or "Unknown evidence error")
    return (
        '<div class="state-message error" role="alert">'
        "The bundled evidence failed verification. No result is displayed. "
        f"<code>{detail}</code></div>"
    )


def empty_readout() -> str:
    return (
        '<div class="state-message" role="status">'
        "No released cell matches this selection. Choose a value from each listed control."
        "</div>"
    )


def integrity_html() -> str:
    if EVIDENCE is None:
        return error_readout()
    commit = html.escape(EVIDENCE.dataset_commit)
    return f"""
    <div class="integrity" role="status">
      <strong>Evidence verified</strong>
      <span>Four bundled tables match the released SHA-256 manifest.</span>
      <code>{commit[:12]}</code>
    </div>
    """


NAV_HTML = f"""
<nav class="site-nav" aria-label="Primary">
  <a class="brand" href="#top">Ready Cohorts</a>
  <div class="nav-links">
    <a class="nav-secondary" href="#explore">Explore</a>
    <a href="{PAPER_URL}" target="_blank" rel="noopener noreferrer">Read paper</a>
    <a href="{CODE_URL}" target="_blank" rel="noopener noreferrer">View code</a>
    <a href="{DATA_URL}" target="_blank" rel="noopener noreferrer">Open data</a>
  </div>
</nav>
"""

HERO_HTML = f"""
<div class="hero-copy" id="top">
  <p class="eyebrow">Ready Cohorts</p>
  <h1>When agent control belongs on GPU</h1>
  <p class="hero-sub">Explore LLM-agent cohort supply, launch deadlines, and the cost of host observation.</p>
  <div class="hero-actions">
    <a class="button primary" href="{PAPER_URL}" target="_blank" rel="noopener noreferrer">Read paper</a>
    <a class="button secondary" href="{DATA_URL}" target="_blank" rel="noopener noreferrer">Open data</a>
  </div>
</div>
"""

METRICS_HTML = """
<div class="metric-strip" aria-label="Released evidence summary">
  <div><span>Trace surface</span><strong>180</strong><small>summary cells</small></div>
  <div><span>Trace panel</span><strong>851</strong><small>sessions</small></div>
  <div><span>Mechanism study</span><strong>4</strong><small>named placements</small></div>
  <div><span>Negative control</span><strong>5</strong><small>named placements</small></div>
</div>
"""

GATES_HTML = """
<section class="gates" id="explore">
  <div class="section-heading">
    <h2>Two gates determine the opportunity.</h2>
    <p>The paper measures workload supply and decision placement separately.</p>
  </div>
  <div class="gate-grid">
    <article class="gate supply">
      <span>Workload gate</span>
      <h3>Can enough route-keyed events become ready before the launch deadline?</h3>
      <p>F measures fixed windows. P* is the exact offline optimum. U is a local upper bound.</p>
    </article>
    <article class="gate placement">
      <span>Runtime gate</span>
      <h3>Can the computed decision remain on device?</h3>
      <p>The mechanism study removes one bundled host observation and redispatch epoch.</p>
    </article>
  </div>
  <p class="join-warning"><strong>Not joined yet.</strong> The current artifact does not claim an online service, CPU displacement, or end-to-end agent speedup.</p>
</section>
"""

TRACE_HEADING = """
<section class="section-heading" id="trace">
  <h2>First, can the workload form a cohort?</h2>
  <p>Sweep active sessions, threshold K, route conditioning, and launch deadline across the frozen replay surface.</p>
</section>
"""

MECHANISM_HEADING = """
<section class="section-heading" id="mechanism">
  <h2>Then, where does the decision live?</h2>
  <p>Compare the resident decision path with the matched host observation bundle on four named placements.</p>
</section>
<div class="path-compare" aria-label="Compared execution paths">
  <div><span>Device path</span><strong>predicate → selector → route</strong></div>
  <div><span>Host path</span><strong>predicate → 4 B copy → synchronize → select → route</strong></div>
</div>
"""

NEGATIVE_HEADING = """
<section class="section-heading" id="negative">
  <h2>A device launch is not enough.</h2>
  <p>The rejected fixed nested graph tests device launch without removing the host decision.</p>
</section>
"""

SCOPE_HTML = f"""
<section class="scope" id="scope">
  <div class="section-heading">
    <h2>Read the claim at the right strength.</h2>
    <p>The release separates formal results, trace computations, placement observations, and open systems questions.</p>
  </div>
  <dl class="scope-grid">
    <div><dt>Proved</dt><dd>P* is exact under zero service time, unlimited capacity, equal relative deadlines, and the stated route model.</dd></div>
    <div><dt>Computed</dt><dd>At C=100,000, K=256, and 50 ms, F=30.19%, P*=43.00%, and U=45.85% on one pinned panel.</dd></div>
    <div><dt>Observed</dt><dd>The resident path wins all 36 named-placement cells, with row-median ratios from 1.19× to 2.39×.</dd></div>
    <div class="open"><dt>Still open</dt><dd>Online achieved share A, CPU core-seconds, raw endpoint P99, end-to-end utility, and shared-inference interference.</dd></div>
  </dl>
  <div class="source-links">
    <a href="{PAPER_URL}" target="_blank" rel="noopener noreferrer">Read paper</a>
    <a href="{CODE_URL}" target="_blank" rel="noopener noreferrer">View code</a>
    <a href="{DATA_URL}" target="_blank" rel="noopener noreferrer">Open data</a>
    <a href="{TRACE_URL}" target="_blank" rel="noopener noreferrer">Pinned trace</a>
  </div>
</section>
"""

FOOTER_HTML = """
<footer class="site-footer">
  <p><strong>Josef Chen</strong><br>Independent Researcher</p>
  <p>Interactive companion to <em>Ready Cohorts: Bounding GPU Opportunity and Avoiding Host Round Trips in LLM-Agent Control</em>.</p>
</footer>
"""

HEAD_META = """
<meta name="description" content="Interactive evidence explorer for Ready Cohorts, a paper on GPU execution of deterministic LLM-agent control.">
<meta property="og:title" content="Ready Cohorts Explorer">
<meta property="og:description" content="Explore cohort supply, launch deadlines, and the cost of host observation.">
<meta property="og:type" content="website">
<meta property="og:image" content="https://huggingface.co/datasets/josefchen/ready-cohorts/resolve/ready-cohorts-arxiv-v1/figures/trace-exact-opportunity-frontier.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://huggingface.co/datasets/josefchen/ready-cohorts/resolve/ready-cohorts-arxiv-v1/figures/trace-exact-opportunity-frontier.png">
"""
HEAD = f"<style>{(ROOT / 'styles.css').read_text(encoding='utf-8')}</style>{HEAD_META}"


initial_trace_plot, initial_trace_readout = trace_view(100_000, 256, "route_key", 50)
initial_resident_plot, initial_resident_readout = resident_view(256, 32)
initial_native_plot, initial_native_readout = native_view(256, 64)


with gr.Blocks(title="Ready Cohorts Explorer", fill_width=True) as demo:
    gr.HTML(NAV_HTML, elem_classes="nav-shell")
    with gr.Row(elem_classes="hero-shell", equal_height=True):
        with gr.Column(scale=5, min_width=320):
            gr.HTML(HERO_HTML)
        with gr.Column(scale=7, min_width=360, elem_classes="hero-visual"):
            gr.Image(
                value=str(ASSET_DIR / "trace-exact-opportunity-frontier.png"),
                label="Released trace opportunity figure",
                show_label=False,
                interactive=False,
                container=False,
                buttons=[],
                elem_classes="hero-image",
            )
    gr.HTML(METRICS_HTML)
    gr.HTML(integrity_html())
    gr.HTML(GATES_HTML)

    gr.HTML(TRACE_HEADING)
    with gr.Row(elem_classes="explorer-row"):
        with gr.Column(scale=3, min_width=248, elem_classes="control-column"):
            trace_sessions = gr.Radio(
                choices=[("1,000", 1_000), ("10,000", 10_000), ("100,000", 100_000)],
                value=100_000,
                label="Target active sessions C",
                info="Stationary replay scale",
                interactive=EVIDENCE is not None,
            )
            trace_k = gr.Radio(
                choices=[32, 64, 128, 256],
                value=256,
                label="Safe cohort threshold K",
                info="Declared profitable suffix threshold",
                interactive=EVIDENCE is not None,
            )
            trace_grouping = gr.Dropdown(
                choices=[
                    ("Outcome route key", "route_key"),
                    ("Outcome event class", "event_class"),
                    ("Pooled events", "pooled"),
                ],
                value="route_key",
                label="Conditioning proxy",
                info="Route key is the paper primary",
                interactive=EVIDENCE is not None,
                filterable=False,
            )
            trace_deadline = gr.Radio(
                choices=[
                    ("10 ms", 10),
                    ("25 ms", 25),
                    ("50 ms", 50),
                    ("100 ms", 100),
                    ("250 ms", 250),
                ],
                value=50,
                label="Read out deadline",
                info="The plot always shows the full deadline sweep",
                interactive=EVIDENCE is not None,
            )
        with gr.Column(scale=9, min_width=420):
            trace_plot = gr.Plot(
                value=initial_trace_plot,
                show_label=False,
                container=False,
                elem_classes="evidence-plot",
            )
            trace_readout = gr.HTML(initial_trace_readout)

    gr.HTML(MECHANISM_HEADING)
    with gr.Row(elem_classes="explorer-row mechanism-row"):
        with gr.Column(scale=3, min_width=248, elem_classes="control-column"):
            resident_agents = gr.Radio(
                choices=[("256", 256), ("2,048", 2_048), ("16,384", 16_384)],
                value=256,
                label="Agents N",
                info="Regular cohort size in the mechanism study",
                interactive=EVIDENCE is not None,
            )
            resident_epochs = gr.Radio(
                choices=[("2", 2), ("8", 8), ("32", 32)],
                value=32,
                label="Decision epochs H",
                info="Observation-free global decisions",
                interactive=EVIDENCE is not None,
            )
        with gr.Column(scale=9, min_width=420):
            resident_plot = gr.Plot(
                value=initial_resident_plot,
                show_label=False,
                container=False,
                elem_classes="evidence-plot",
            )
            resident_readout = gr.HTML(initial_resident_readout)

    gr.HTML(NEGATIVE_HEADING)
    with gr.Row(elem_classes="explorer-row negative-row"):
        with gr.Column(scale=3, min_width=248, elem_classes="control-column"):
            native_agents = gr.Radio(
                choices=[("32", 32), ("256", 256), ("2,048", 2_048), ("16,384", 16_384)],
                value=256,
                label="Agents N",
                info="Cohort size in the negative control",
                interactive=EVIDENCE is not None,
            )
            native_steps = gr.Radio(
                choices=[("1", 1), ("8", 8), ("64", 64)],
                value=64,
                label="Steps",
                info="Fixed nested graph depth",
                interactive=EVIDENCE is not None,
            )
        with gr.Column(scale=9, min_width=420):
            native_plot = gr.Plot(
                value=initial_native_plot,
                show_label=False,
                container=False,
                elem_classes="evidence-plot",
            )
            native_readout = gr.HTML(initial_native_readout)

    gr.HTML(SCOPE_HTML)
    gr.HTML(FOOTER_HTML)

    for control in [trace_sessions, trace_k, trace_grouping, trace_deadline]:
        control.change(
            fn=trace_view,
            inputs=[trace_sessions, trace_k, trace_grouping, trace_deadline],
            outputs=[trace_plot, trace_readout],
            show_progress="hidden",
            trigger_mode="always_last",
        )

    for control in [resident_agents, resident_epochs]:
        control.change(
            fn=resident_view,
            inputs=[resident_agents, resident_epochs],
            outputs=[resident_plot, resident_readout],
            show_progress="hidden",
            trigger_mode="always_last",
        )

    for control in [native_agents, native_steps]:
        control.change(
            fn=native_view,
            inputs=[native_agents, native_steps],
            outputs=[native_plot, native_readout],
            show_progress="hidden",
            trigger_mode="always_last",
        )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=12, max_size=64).launch(
        theme=gr.themes.Base(),
        head=HEAD,
        allowed_paths=[str(ASSET_DIR)],
    )
