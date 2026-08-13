from __future__ import annotations

import itertools
import re
from pathlib import Path

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
    check_links_and_secrets()
    print("Ready Cohorts Space checks passed")


if __name__ == "__main__":
    main()
