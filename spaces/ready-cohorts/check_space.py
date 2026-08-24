from __future__ import annotations

import itertools
import re
import struct
from pathlib import Path
from xml.etree import ElementTree

import app

ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_evidence() -> None:
    require(app.EVIDENCE_ERROR is None, f"Evidence load failed: {app.EVIDENCE_ERROR}")
    evidence = app.EVIDENCE
    require(evidence is not None, "Evidence object is missing")
    require(evidence.trace.shape == (180, 14), "Trace summary shape changed")
    require(evidence.resident_cells.shape == (108, 20), "Resident cells shape changed")
    require(
        evidence.resident_contrasts.shape == (36, 20),
        "Resident contrast shape changed",
    )
    require(
        evidence.native_contrasts.shape == (60, 19),
        "Native contrast shape changed",
    )


def check_trace_surface() -> None:
    for sessions, threshold_k, grouping, deadline in itertools.product(
        [1_000, 10_000, 100_000],
        [32, 64, 128, 256],
        ["route_key", "event_class", "pooled"],
        [10, 25, 50, 100, 250],
    ):
        figure, readout = app.trace_view(sessions, threshold_k, grouping, deadline)
        require(len(figure.data) == 3, "Trace callback returned an empty plot")
        require("No released cell" not in readout, "Trace callback returned empty state")
        for trace in figure.data:
            require(len(trace.x) == 5, "Trace plot must show all five deadlines")
            require(len(trace.y) == 5, "Trace plot has an incomplete series")
            require(min(trace.y) >= 0, "Eligible share cannot be negative")
            require(max(trace.y) <= 1, "Eligible share cannot exceed one")
        exact_trace = figure.data[1]
        require(
            exact_trace.error_y.visible is True,
            "Exact trace must retain its descriptive three-swarm range",
        )
        require(len(figure.layout.shapes) == 1, "Trace plot lost its selected-deadline line")
        require(
            figure.layout.shapes[0].x0 == deadline,
            "Selected-deadline line is not in log-axis data coordinates",
        )


def check_mechanism_surfaces() -> None:
    for agents, epochs in itertools.product([256, 2_048, 16_384], [2, 8, 32]):
        figure, readout = app.resident_view(agents, epochs)
        require(len(figure.data) == 5, "Resident plot must have four stems and one marker trace")
        require("All four selected" in readout, "Resident readout lost its sampling caveat")
        ratios = list(figure.data[-1].x)
        require(len(ratios) == 4, "Resident plot must contain four placements")
        require(min(ratios) > 1, "A released resident ratio no longer favors the resident path")

    for agents, steps in itertools.product([32, 256, 2_048, 16_384], [1, 8, 64]):
        figure, readout = app.native_view(agents, steps)
        require(len(figure.data) == 6, "Native plot must have five stems and one marker trace")
        require("all 60 released" in readout, "Negative-control scope statement is missing")
        ratios = list(figure.data[-1].x)
        require(len(ratios) == 5, "Native plot must contain five placements")
        require(min(ratios) > 1, "A released nested path is no longer slower")


def check_visible_copy() -> None:
    text_extensions = {".py", ".css", ".md", ".json", ".txt"}
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in text_extensions:
            content = path.read_text(encoding="utf-8")
            require("\u2014" not in content, f"Em dash found in {path.relative_to(ROOT)}")
            require("\u2013" not in content, f"En dash found in {path.relative_to(ROOT)}")

    hero_sub = re.search(r'<p class="hero-sub">([^<]+)</p>', app.HERO_HTML)
    require(hero_sub is not None, "Hero subtext is missing")
    require(len(hero_sub.group(1).split()) <= 20, "Hero subtext exceeds 20 words")
    require(app.HERO_HTML.count('class="button') == 2, "Hero must contain exactly two actions")
    require("josefchen10@gmail.com" not in app.HERO_HTML, "Private contact is visible")


def check_public_visuals() -> None:
    architecture = ROOT / "assets" / "ready-cohorts-architecture.svg"
    social_svg = ROOT / "assets" / "ready-cohorts-social-card.svg"
    social_png = ROOT / "assets" / "ready-cohorts-social-card.png"
    for path in (architecture, social_svg):
        root = ElementTree.parse(path).getroot()
        require(root.tag.endswith("svg"), f"{path.name} is not valid SVG")
        require(root.attrib.get("role") == "img", f"{path.name} lacks an image role")
        require(
            "<script" not in path.read_text(encoding="utf-8").lower(),
            f"{path.name} contains script",
        )

    with social_png.open("rb") as handle:
        require(handle.read(8) == b"\x89PNG\r\n\x1a\n", "Social image is not PNG")
        length = struct.unpack(">I", handle.read(4))[0]
        require(length == 13 and handle.read(4) == b"IHDR", "Social PNG lacks IHDR")
        width, height = struct.unpack(">II", handle.read(8))
    require((width, height) == (1600, 900), "Social image must be 1600 by 900")

    font_dir = ROOT / "assets" / "fonts"
    font_names = [
        f"geist-sans-latin-{weight}-normal.woff2"
        for weight in (400, 500, 600, 700)
    ] + [
        f"geist-mono-latin-{weight}-normal.woff2"
        for weight in (400, 500, 600)
    ]
    for name in font_names:
        with (font_dir / name).open("rb") as handle:
            require(handle.read(4) == b"wOF2", f"{name} is not a WOFF2 font")
    require((font_dir / "OFL.txt").is_file(), "Bundled Geist fonts lack their license")

    require(
        "Two measured gates. One unmeasured join." in app.ARCHITECTURE_HTML,
        "Architecture evidence boundary is missing",
    )
    require(
        "Measured separately." in app.GATES_HTML,
        "The two study surfaces are no longer separated",
    )
    space_card = (ROOT / "README.md").read_text(encoding="utf-8")
    require(
        f"thumbnail: {app.SOCIAL_IMAGE_URL}" in space_card,
        "Space metadata does not use the source-backed social card",
    )


def check_links_and_secrets() -> None:
    require(
        app.PAPER_URL == "https://arxiv.org/abs/2608.12123",
        "Paper link is not the canonical arXiv record",
    )
    require(app.DATA_URL.endswith("ready-cohorts-arxiv-v1"), "Dataset link is not release-pinned")
    require(
        app.CODE_URL.endswith("tree/main/spaces/ready-cohorts"),
        "Code link does not target the Space source",
    )
    require(
        app.TRACE_URL.endswith("f7c94012d0bfbf66fe4d6ed627699508bbb555ff"),
        "Trace link is not commit-pinned",
    )
    require(
        app.ARCHITECTURE_URL.endswith("assets/ready-cohorts-architecture.svg"),
        "Architecture link does not target the public Space asset",
    )
    require(
        app.SOCIAL_IMAGE_URL.endswith("assets/ready-cohorts-social-card.png"),
        "Social preview does not target the public Space asset",
    )
    require(app.SOCIAL_IMAGE_URL in app.HEAD_META, "Social preview metadata is stale")

    secret_patterns = [
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    ]
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix == ".png":
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in secret_patterns:
            require(
                pattern.search(content) is None,
                f"Secret-like value found in {path.relative_to(ROOT)}",
            )


def main() -> None:
    check_evidence()
    check_trace_surface()
    check_mechanism_surfaces()
    check_visible_copy()
    check_public_visuals()
    check_links_and_secrets()
    print("Ready Cohorts Space checks passed")


if __name__ == "__main__":
    main()
