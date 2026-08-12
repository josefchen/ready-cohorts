"""Mechanical release checks for the Ready-Cohort working manuscript."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "arxiv"
SECTIONS = PAPER / "sections"
CLAIMS = ROOT / "paper" / "governance" / "claim-evidence-map.csv"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def manuscript_files() -> list[Path]:
    return sorted([PAPER / "main.tex", PAPER / "references.bib", *SECTIONS.glob("*.tex")])


def bibliography_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))


def citation_keys(text: str) -> set[str]:
    found: set[str] = set()
    for group in re.findall(r"\\cite\w*\s*\{([^}]+)\}", text):
        found.update(key.strip() for key in group.split(",") if key.strip())
    return found


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_paper_artifacts.py")],
        cwd=ROOT,
        check=True,
    )

    files = manuscript_files()
    for path in files:
        if not path.is_file():
            fail(f"missing manuscript file: {path}")
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    lowered = combined.lower()

    expected_title = (
        "Ready Cohorts: Bounding GPU Opportunity and Avoiding Host Round Trips in LLM-Agent Control"
    )
    brief = (ROOT / "paper" / "brief.yaml").read_text(encoding="utf-8")
    if expected_title not in brief:
        fail("brief title differs from the frozen working title")
    main_text = (PAPER / "main.tex").read_text(encoding="utf-8")
    if "\\textbf{Ready Cohorts:}" not in main_text:
        fail("main title differs from the frozen working title")
    for phrase in ("Josef Chen", "Independent Researcher"):
        if phrase not in main_text:
            fail(f"author metadata is missing: {phrase}")
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", combined):
        fail("public manuscript source contains an email address")
    if "pending author confirmation" in main_text.lower():
        fail("author metadata still contains a pending placeholder")

    for character, label in (("\u2013", "en dash"), ("\u2014", "em dash")):
        if character in combined:
            fail(f"manuscript contains a Unicode {label}")

    forbidden = {
        "linear after route-wise sorting": "frozen evaluator uses binary boundary search",
        "linear after sorting": "frozen evaluator uses binary boundary search",
        "novel algorithm": "algorithm priority is not supported",
        "first exact algorithm": "algorithm priority is not supported",
        "first gpu agent": "GPU-agent priority is not supported",
        "50 ms completion slo": "50 ms is a launch deadline",
        "3,240 independent": "rows are technical repetitions",
        "21,974,573 independent": "invocations are technical repetitions",
        "end-to-end speedup": "no end-to-end result exists",
        "revolutionary": "promotional language",
        "transformative": "promotional language",
        "unprecedented": "promotional language",
        "paradigm shift": "promotional language",
        "these studies are conditional and descriptive": "rebuttal language leaked into the abstract",
        "delve": "template language",
        "tapestry": "template language",
    }
    for phrase, reason in forbidden.items():
        if phrase in lowered:
            fail(f"forbidden phrase {phrase!r}: {reason}")

    abstract = (SECTIONS / "00_abstract.tex").read_text(encoding="utf-8")
    abstract_flat = re.sub(r"\s+", " ", abstract)
    required_abstract = [
        "one pinned \\TraceSessions-session public trace panel",
        "four named GPU placements",
        "both admissible mechanisms",
        "conditioning proxy",
        "A joined finite online runtime is required",
        "two measurable gates for GPU agent control",
    ]
    for phrase in required_abstract:
        if phrase not in abstract_flat:
            fail(f"abstract is missing required scope phrase: {phrase}")
    exact_section = (SECTIONS / "04_exact_packing.tex").read_text(encoding="utf-8")
    for phrase in ("O(NR+\\sum_r n_r\\log n_r)", "quadratic in the worst case"):
        if phrase not in exact_section:
            fail(f"exact-packing section is missing frozen-complexity qualifier: {phrase}")
    abstract_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract)
    if not 180 <= len(abstract_words) <= 300:
        fail(f"abstract length is {len(abstract_words)} words; expected 180..300")

    bib_text = (PAPER / "references.bib").read_text(encoding="utf-8")
    known_citations = bibliography_keys(bib_text)
    used_citations = citation_keys(combined)
    missing_citations = sorted(used_citations - known_citations)
    if missing_citations:
        fail(f"missing bibliography keys: {missing_citations}")

    with CLAIMS.open(newline="", encoding="utf-8") as handle:
        claim_rows = list(csv.DictReader(handle))
    claim_ids = {row["claim_id"] for row in claim_rows}
    if len(claim_ids) != len(claim_rows):
        fail("claim-evidence map contains duplicate claim IDs")
    used_claim_ids = set(re.findall(r"\\claimid\{([^}]+)\}", combined))
    unknown_claim_ids = sorted(used_claim_ids - claim_ids)
    if unknown_claim_ids:
        fail(f"manuscript uses unknown claim IDs: {unknown_claim_ids}")
    for row in claim_rows:
        evidence = ROOT / row["evidence_path"]
        if not evidence.exists():
            fail(f"claim {row['claim_id']} has missing evidence path: {evidence}")
        if not row["locator"] or not row["unit"] or not row["scope"]:
            fail(f"claim {row['claim_id']} lacks locator, unit, or scope")

    manifest_path = PAPER / "generated" / "paper-data-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in [*manifest["inputs"], *manifest["outputs"]]:
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            fail(f"paper-data manifest hash mismatch: {path}")
    for item in manifest["figures"]:
        checks = (
            (ROOT / item["path"], item["sha256"]),
            (ROOT / item["source_table"], item["source_table_sha256"]),
            (ROOT / item["notebook_builder"], item["notebook_builder_sha256"]),
            (ROOT / item["notebook"], item["notebook_sha256"]),
        )
        for path, declared in checks:
            if not path.is_file() or sha256(path) != declared:
                fail(f"figure provenance mismatch: {path}")

    secret_patterns = {
        "OpenAI-style key": r"\bsk-[A-Za-z0-9_-]{20,}\b",
        "Hugging Face token": r"\bhf_[A-Za-z0-9]{20,}\b",
        "AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
        "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    }
    for label, pattern in secret_patterns.items():
        if re.search(pattern, combined):
            fail(f"possible {label} in manuscript sources")

    pdf = PAPER / "main.pdf"
    log = PAPER / "main.log"
    if not pdf.is_file() or pdf.stat().st_size < 10_000:
        fail("compiled main.pdf is missing or implausibly small")
    if not log.is_file():
        fail("main.log is missing")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    for marker in ("undefined citations", "There were undefined references"):
        if marker in log_text:
            fail(f"LaTeX log contains: {marker}")

    print(
        "paper checks passed: "
        f"{len(files)} manuscript files, {len(used_citations)} citations, "
        f"{len(used_claim_ids)} used claim IDs, PDF SHA-256 {sha256(pdf)}"
    )


if __name__ == "__main__":
    main()
