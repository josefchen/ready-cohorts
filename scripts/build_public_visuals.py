from __future__ import annotations

import csv
import hashlib
import html
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "spaces" / "ready-cohorts"
DATA = SPACE / "data"
ASSETS = SPACE / "assets"

INK = "#17202D"
MUTED = "#5B6675"
LINE = "#C9D1DC"
PAGE = "#F2F4F7"
SURFACE = "#FFFFFF"
BLUE = "#2457A7"
BLUE_DARK = "#173C78"
BLUE_SOFT = "#E8EEF9"
GOLD = "#A86C12"
GOLD_SOFT = "#F7EEDC"


@dataclass(frozen=True)
class PublicFacts:
    frozen_share: float
    exact_share: float
    upper_share: float
    closure: float
    resident_ratio_min: float
    resident_ratio_max: float
    resident_cells: int
    resident_placements: int
    native_cells: int
    native_placements: int


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_facts() -> PublicFacts:
    trace = read_rows(DATA / "trace-summary.csv")
    primary = [
        row
        for row in trace
        if int(row["target_active_sessions"]) == 100_000
        and int(row["deadline_ms"]) == 50
        and row["grouping"] == "route_key"
        and int(row["threshold_k"]) == 256
    ]
    if len(primary) != 1:
        raise ValueError("The public trace primary cell is not unique")

    resident = read_rows(DATA / "resident-contrasts.csv")
    resident_ratios = [float(row["host_over_resident_ratio_of_medians"]) for row in resident]
    if not resident_ratios or min(resident_ratios) <= 1:
        raise ValueError("The public resident contrast surface changed direction")

    native = read_rows(DATA / "native-contrasts.csv")
    native_ratios = [
        float(row["cuda_device_graph_over_cuda_host_graph_wall_ratio_of_medians"]) for row in native
    ]
    if not native_ratios or min(native_ratios) <= 1:
        raise ValueError("The public negative-control surface changed direction")

    row = primary[0]
    return PublicFacts(
        frozen_share=float(row["fixed_window_eligible_share_mean"]),
        exact_share=float(row["exact_optimal_share_mean"]),
        upper_share=float(row["local_upper_share_mean"]),
        closure=float(row["alignment_gap_closure_mean"]),
        resident_ratio_min=min(resident_ratios),
        resident_ratio_max=max(resident_ratios),
        resident_cells=len(resident),
        resident_placements=len({row["placement_id"] for row in resident}),
        native_cells=len(native),
        native_placements=len({row["placement_id"] for row in native}),
    )


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: int,
    y: int,
    value: object,
    *,
    size: int = 24,
    weight: int = 500,
    color: str = INK,
    anchor: str = "start",
    family: str = "Inter, Aptos, Helvetica Neue, Arial, sans-serif",
    letter_spacing: str | None = None,
    opacity: float = 1,
) -> str:
    spacing = "" if letter_spacing is None else f' letter-spacing="{esc(letter_spacing)}"'
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"'
        f'{spacing} opacity="{opacity}">{esc(value)}</text>'
    )


def multiline(
    x: int,
    y: int,
    lines: list[str],
    *,
    size: int,
    line_height: int,
    weight: int = 500,
    color: str = INK,
    anchor: str = "start",
) -> str:
    spans = "".join(
        f'<tspan x="{x}" dy="{0 if index == 0 else line_height}">{esc(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<text x="{x}" y="{y}" fill="{color}" '
        'font-family="Inter, Aptos, Helvetica Neue, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{spans}</text>'
    )


def rect(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    fill: str = SURFACE,
    stroke: str = LINE,
    radius: int = 18,
    stroke_width: int = 2,
    dash: str | None = None,
) -> str:
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"'
        f"{dash_attr}/>"
    )


def line(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    color: str = LINE,
    width: int = 2,
    dash: str | None = None,
    marker: bool = False,
) -> str:
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}{marker_attr}/>'
    )


def pill(
    x: int,
    y: int,
    width: int,
    label: str,
    *,
    fill: str = BLUE_SOFT,
    color: str = BLUE_DARK,
    stroke: str = "none",
) -> str:
    return "".join(
        [
            (
                f'<rect x="{x}" y="{y}" width="{width}" height="34" rx="17" '
                f'fill="{fill}" stroke="{stroke}"/>'
            ),
            text(x + width // 2, y + 23, label, size=14, weight=700, color=color, anchor="middle"),
        ]
    )


def cohort_glyph(x: int, y: int, *, scale: float = 1) -> str:
    circles = []
    for dx, dy in [(0, 0), (28, -17), (28, 17), (56, 0)]:
        circles.append(
            f'<circle cx="{x + dx * scale:.1f}" cy="{y + dy * scale:.1f}" '
            f'r="{6 * scale:.1f}" fill="{BLUE if dx < 56 else GOLD}"/>'
        )
    circles.append(
        f'<path d="M {x + 7 * scale:.1f} {y:.1f} H {x + 48 * scale:.1f}" '
        f'stroke="{LINE}" stroke-width="{2 * scale:.1f}" fill="none"/>'
    )
    return "".join(circles)


def svg_document(width: int, height: int, title_value: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">
<title id="title">{esc(title_value)}</title>
<desc id="description">Source-backed visual companion to Ready Cohorts, arXiv:2608.12123.</desc>
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/>
  </marker>
</defs>
{body}
</svg>
"""


def architecture_svg(facts: PublicFacts) -> str:
    gap_points = 100 * (facts.exact_share - facts.frozen_share)
    parts = [f'<rect width="1600" height="1000" fill="{PAGE}"/>']
    parts.extend(
        [
            cohort_glyph(1452, 64, scale=1.2),
            text(
                72,
                68,
                "READY COHORTS / SYSTEM MAP",
                size=15,
                weight=760,
                color=BLUE,
                letter_spacing="2.2",
            ),
            text(72, 130, "A GPU control plane has two gates.", size=46, weight=720, color=INK),
            text(
                72,
                176,
                "This paper measures them separately. The finite online runtime remains the next experiment.",
                size=21,
                color=MUTED,
            ),
            line(72, 208, 1528, 208, color=INK, width=2),
        ]
    )

    # Agent-event input column.
    parts.extend(
        [
            text(72, 270, "AGENT EVENTS", size=14, weight=760, color=MUTED, letter_spacing="1.8"),
            text(72, 303, "Completions become", size=20, weight=650),
            text(72, 329, "control transitions.", size=20, weight=650),
        ]
    )
    event_rows = [
        (370, "tool result", "route: search"),
        (434, "model output", "route: final"),
        (498, "timeout", "route: error"),
        (562, "tool result", "route: lookup"),
    ]
    for index, (y, kind, route) in enumerate(event_rows, start=1):
        parts.extend(
            [
                rect(72, y, 210, 50, fill=SURFACE, radius=10, stroke_width=1),
                f'<circle cx="94" cy="{y + 25}" r="7" fill="{BLUE if index < 4 else GOLD}"/>',
                text(112, y + 21, kind, size=14, weight=700),
                text(112, y + 39, route, size=12, color=MUTED),
            ]
        )
    parts.extend(
        [
            text(72, 650, "release time tᵢ", size=14, color=MUTED),
            text(72, 675, "deadline tᵢ + δ", size=14, color=MUTED),
            line(283, 480, 329, 480, color=MUTED, width=2, marker=True),
        ]
    )

    # Gate 1.
    parts.extend(
        [
            rect(340, 246, 512, 242, fill=SURFACE, stroke=BLUE, radius=18),
            pill(366, 270, 222, "01  WORKLOAD GATE · COMPUTED"),
            text(366, 330, "Ready before the deadline?", size=27, weight=700),
            rect(366, 356, 132, 52, fill=PAGE, radius=10, stroke_width=1),
            text(432, 378, "route-key", size=13, weight=700, anchor="middle"),
            text(432, 396, "queues", size=13, color=MUTED, anchor="middle"),
            line(500, 382, 536, 382, color=MUTED, marker=True),
            rect(546, 356, 132, 52, fill=BLUE_SOFT, stroke=BLUE, radius=10, stroke_width=1),
            text(612, 378, "exact packer", size=13, weight=700, color=BLUE_DARK, anchor="middle"),
            text(612, 396, "P*", size=13, color=BLUE_DARK, anchor="middle"),
            line(680, 382, 716, 382, color=MUTED, marker=True),
            rect(726, 356, 100, 52, fill=PAGE, radius=10, stroke_width=1),
            text(776, 378, "cohort", size=13, weight=700, anchor="middle"),
            text(776, 396, "size ≥ K", size=13, color=MUTED, anchor="middle"),
            text(366, 440, f"F  {facts.frozen_share:.2%}", size=20, weight=700, color=MUTED),
            text(507, 440, "→", size=20, weight=700, color=GOLD),
            text(545, 440, f"P*  {facts.exact_share:.2%}", size=20, weight=760, color=BLUE),
            text(701, 440, f"≤ U  {facts.upper_share:.2%}", size=20, weight=700, color=BLUE_DARK),
            text(
                366,
                469,
                f"Primary replay cell · +{gap_points:.2f} points · {facts.closure:.1%} of alignment gap closed",
                size=13,
                color=MUTED,
            ),
        ]
    )

    # Gate 2.
    parts.extend(
        [
            rect(340, 522, 512, 242, fill=SURFACE, stroke=GOLD, radius=18),
            pill(366, 546, 230, "02  PLACEMENT GATE · OBSERVED", fill=GOLD_SOFT, color=GOLD),
            text(366, 606, "Can the decision stay on device?", size=27, weight=700),
        ]
    )
    x_positions = [366, 482, 598, 714]
    labels = [
        ("resident", "state"),
        ("predicate", "GPU"),
        ("selector", "GPU"),
        ("route graph", "GPU"),
    ]
    for index, (x, (top, bottom)) in enumerate(zip(x_positions, labels, strict=True)):
        parts.extend(
            [
                rect(
                    x,
                    632,
                    102,
                    54,
                    fill=GOLD_SOFT if index in {1, 2} else PAGE,
                    stroke=GOLD if index in {1, 2} else LINE,
                    radius=10,
                    stroke_width=1,
                ),
                text(x + 51, 655, top, size=12, weight=700, color=INK, anchor="middle"),
                text(x + 51, 674, bottom, size=11, color=MUTED, anchor="middle"),
            ]
        )
        if index < len(x_positions) - 1:
            parts.append(
                line(x + 104, 659, x_positions[index + 1] - 8, 659, color=MUTED, marker=True)
            )
    parts.extend(
        [
            text(
                366,
                720,
                f"Host / resident  {facts.resident_ratio_min:.2f}–{facts.resident_ratio_max:.2f}×",
                size=20,
                weight=760,
                color=GOLD,
            ),
            text(
                366,
                748,
                f"{facts.resident_cells}/{facts.resident_cells} cells favor resident · fixed nested launch slower in {facts.native_cells}/{facts.native_cells}",
                size=13,
                color=MUTED,
            ),
        ]
    )

    # Proposed joined runtime.
    parts.extend(
        [
            line(854, 366, 914, 366, color=MUTED, dash="8 7", marker=True),
            line(854, 642, 914, 642, color=MUTED, dash="8 7", marker=True),
            rect(926, 246, 602, 518, fill=SURFACE, stroke=MUTED, radius=18, dash="9 8"),
            pill(
                952, 270, 250, "03  JOINED RUNTIME · PROPOSED", fill=PAGE, color=MUTED, stroke=LINE
            ),
            text(952, 330, "Deadline-aware GPU route service", size=29, weight=700),
            text(
                952,
                360,
                "The architecture implied by both gates, not a measured deployment.",
                size=15,
                color=MUTED,
            ),
            rect(952, 397, 132, 66, fill=PAGE, radius=10, stroke_width=1),
            text(1018, 423, "typed ingress", size=13, weight=700, anchor="middle"),
            text(1018, 444, "ordered events", size=12, color=MUTED, anchor="middle"),
            line(1086, 430, 1121, 430, color=MUTED, marker=True),
            rect(1131, 397, 142, 66, fill=BLUE_SOFT, stroke=BLUE, radius=10, stroke_width=1),
            text(
                1202, 423, "deadline queues", size=13, weight=700, color=BLUE_DARK, anchor="middle"
            ),
            text(1202, 444, "GPU state", size=12, color=BLUE_DARK, anchor="middle"),
            line(1275, 430, 1310, 430, color=MUTED, marker=True),
            rect(1320, 397, 178, 66, fill=GOLD_SOFT, stroke=GOLD, radius=10, stroke_width=1),
            text(1409, 423, "effect descriptors", size=13, weight=700, anchor="middle"),
            text(1409, 444, "ordered mailbox", size=12, color=MUTED, anchor="middle"),
            line(1409, 465, 1409, 502, color=MUTED, marker=True),
            rect(1198, 514, 300, 72, fill=PAGE, stroke=INK, radius=10, stroke_width=1),
            text(1348, 542, "CPU / DPU authority plane", size=15, weight=760, anchor="middle"),
            text(
                1348,
                566,
                "validate · commit · recover · audit",
                size=13,
                color=MUTED,
                anchor="middle",
            ),
        ]
    )
    outputs = [(994, "LLM"), (1122, "tool"), (1250, "sandbox")]
    for x, label in outputs:
        parts.extend(
            [
                line(1348, 587, x + 48, 623, color=MUTED, width=1),
                rect(x, 624, 96, 44, fill=PAGE, radius=22, stroke_width=1),
                text(x + 48, 652, label, size=13, weight=700, anchor="middle"),
            ]
        )
    parts.extend(
        [
            rect(952, 694, 546, 46, fill=PAGE, stroke=LINE, radius=8, stroke_width=1, dash="5 5"),
            text(
                1225,
                723,
                "Still unmeasured: A · CPU time · raw P99 · utility · interference",
                size=14,
                weight=650,
                color=MUTED,
                anchor="middle",
            ),
            line(72, 820, 1528, 820, color=INK, width=2),
            pill(72, 850, 126, "COMPUTED", fill=BLUE_SOFT, color=BLUE_DARK),
            text(212, 873, "pinned trace replay", size=14, color=MUTED),
            pill(418, 850, 126, "OBSERVED", fill=GOLD_SOFT, color=GOLD),
            text(558, 873, "named GPU placements", size=14, color=MUTED),
            pill(820, 850, 126, "PROPOSED", fill=PAGE, color=MUTED, stroke=LINE),
            text(960, 873, "finite online system", size=14, color=MUTED),
            text(72, 948, "arXiv:2608.12123", size=16, weight=760, color=BLUE),
            text(
                1528,
                948,
                "Josef Chen · Independent Researcher",
                size=16,
                weight=650,
                color=INK,
                anchor="end",
            ),
        ]
    )
    return svg_document(1600, 1000, "Ready Cohorts system map", "".join(parts))


def social_svg(facts: PublicFacts) -> str:
    parts = [f'<rect width="1600" height="900" fill="{PAGE}"/>']
    parts.extend(
        [
            f'<rect width="1600" height="12" fill="{BLUE}"/>',
            cohort_glyph(1455, 70, scale=1.35),
            text(
                74,
                76,
                "NEW PAPER · arXiv:2608.12123",
                size=17,
                weight=760,
                color=BLUE,
                letter_spacing="1.8",
            ),
            multiline(
                74,
                166,
                ["When should agent", "control move to GPU?"],
                size=61,
                line_height=68,
                weight=730,
            ),
            multiline(
                74,
                328,
                [
                    "Ready Cohorts maps the boundary between",
                    "workload supply and device-resident control.",
                ],
                size=24,
                line_height=34,
                color=MUTED,
            ),
            rect(925, 128, 588, 340, fill=SURFACE, stroke=LINE, radius=22),
            text(
                959, 173, "THE ARCHITECTURE", size=14, weight=760, color=MUTED, letter_spacing="1.7"
            ),
            rect(959, 210, 156, 82, fill=BLUE_SOFT, stroke=BLUE, radius=14),
            text(1037, 242, "ready events", size=15, weight=760, color=BLUE_DARK, anchor="middle"),
            text(1037, 267, "route + deadline", size=13, color=BLUE_DARK, anchor="middle"),
            line(1118, 251, 1162, 251, color=MUTED, marker=True),
            rect(1172, 210, 156, 82, fill=GOLD_SOFT, stroke=GOLD, radius=14),
            text(1250, 242, "GPU decision", size=15, weight=760, anchor="middle"),
            text(1250, 267, "resident state", size=13, color=MUTED, anchor="middle"),
            line(1331, 251, 1375, 251, color=MUTED, marker=True),
            rect(1385, 210, 94, 82, fill=PAGE, stroke=INK, radius=14),
            text(1432, 242, "effect", size=14, weight=760, anchor="middle"),
            text(1432, 267, "mailbox", size=13, color=MUTED, anchor="middle"),
            line(1432, 295, 1432, 330, color=MUTED, marker=True),
            rect(1172, 340, 307, 74, fill=PAGE, stroke=INK, radius=14),
            text(1325, 370, "CPU / DPU authority", size=16, weight=760, anchor="middle"),
            text(
                1325, 394, "external effects stay privileged", size=13, color=MUTED, anchor="middle"
            ),
            text(
                959,
                447,
                "Trace gate and mechanism gate measured separately",
                size=14,
                weight=650,
                color=MUTED,
            ),
            line(74, 500, 1526, 500, color=INK, width=2),
        ]
    )
    metrics = [
        (
            74,
            f"{facts.frozen_share:.2%} → {facts.exact_share:.2%}",
            "FIXED WINDOWS → EXACT PACKING",
        ),
        (
            510,
            f"{facts.resident_ratio_min:.2f}–{facts.resident_ratio_max:.2f}×",
            "HOST / RESIDENT WALL TIME",
        ),
    ]
    # The headline metric row sits directly beneath the question; the lower strip
    # separates the exact gap, direction count, and negative control.
    parts.extend(
        [
            text(
                metrics[0][0],
                451,
                metrics[0][2],
                size=13,
                weight=760,
                color=MUTED,
                letter_spacing="1.0",
            ),
            text(metrics[0][0], 488, metrics[0][1], size=36, weight=760, color=BLUE),
            text(
                metrics[1][0],
                451,
                metrics[1][2],
                size=13,
                weight=760,
                color=MUTED,
                letter_spacing="1.0",
            ),
            text(metrics[1][0], 488, metrics[1][1], size=36, weight=760, color=GOLD),
        ]
    )
    lower_metrics = [
        (
            74,
            f"+{100 * (facts.exact_share - facts.frozen_share):.2f} pp",
            "exact packing vs F",
            "conditional trace computation",
            BLUE,
        ),
        (
            466,
            f"{facts.resident_cells}/{facts.resident_cells}",
            "cells favor resident",
            f"{facts.resident_placements} named placements",
            GOLD,
        ),
        (
            858,
            f"{facts.native_cells}/{facts.native_cells}",
            "nested launch slower",
            f"{facts.native_placements} named placements",
            INK,
        ),
    ]
    for x, value, label, note, color in lower_metrics:
        parts.extend(
            [
                text(x, 593, value, size=44, weight=760, color=color),
                text(x, 628, label, size=18, weight=700),
                text(x, 654, note, size=14, color=MUTED),
            ]
        )
    parts.extend(
        [
            rect(74, 705, 1452, 74, fill=SURFACE, stroke=LINE, radius=12),
            text(
                102,
                736,
                "EVIDENCE BOUNDARY",
                size=13,
                weight=760,
                color=MUTED,
                letter_spacing="1.4",
            ),
            text(
                102,
                763,
                "Not an end-to-end speedup claim. The finite online runtime, CPU use, raw P99, and inference interference remain open.",
                size=17,
                weight=600,
            ),
            line(74, 822, 1526, 822, color=LINE, width=2),
            text(74, 861, "Josef Chen · Independent Researcher", size=17, weight=700),
            text(
                1526,
                861,
                "github.com/josefchen/ready-cohorts",
                size=17,
                weight=650,
                color=BLUE,
                anchor="end",
            ),
        ]
    )
    return svg_document(1600, 900, "Ready Cohorts social launch card", "".join(parts))


def write_if_changed(path: Path, content: str) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current != content:
        path.write_text(content, encoding="utf-8")


def render_png(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    chrome = "/usr/bin/google-chrome"
    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            "--force-device-scale-factor=1",
            f"--window-size={width},{height}",
            f"--screenshot={png_path}",
            svg_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_launch_thread() -> list[int]:
    path = ROOT / "docs" / "x-launch-thread.md"
    source = path.read_text(encoding="utf-8")
    counts: list[int] = []
    for index in range(1, 6):
        match = re.search(
            rf"^## Post {index}\n\n(?P<body>.*?)(?=\n## Post {index + 1}\n|\n## Image description\n)",
            source,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise ValueError(f"X launch post {index} is missing")
        paragraphs = [
            " ".join(paragraph.splitlines())
            for paragraph in match.group("body").strip().split("\n\n")
        ]
        post = "\n\n".join(paragraphs)
        if len(post) > 280:
            raise ValueError(f"X launch post {index} has {len(post)} characters")
        counts.append(len(post))
    return counts


def main() -> None:
    facts = load_facts()
    ASSETS.mkdir(parents=True, exist_ok=True)
    architecture_path = ASSETS / "ready-cohorts-architecture.svg"
    social_svg_path = ASSETS / "ready-cohorts-social-card.svg"
    social_png_path = ASSETS / "ready-cohorts-social-card.png"

    write_if_changed(architecture_path, architecture_svg(facts))
    write_if_changed(social_svg_path, social_svg(facts))
    for svg_path in (architecture_path, social_svg_path):
        ElementTree.parse(svg_path)
    render_png(social_svg_path, social_png_path, 1600, 900)
    thread_counts = validate_launch_thread()

    for path in (architecture_path, social_svg_path, social_png_path):
        print(f"{path.relative_to(ROOT)} {file_sha256(path)}")
    print(f"docs/x-launch-thread.md post_characters={thread_counts}")


if __name__ == "__main__":
    main()
