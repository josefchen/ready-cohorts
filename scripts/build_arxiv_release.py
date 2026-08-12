"""Build the deterministic, minimal arXiv source archive for Ready Cohorts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "arxiv"
RELEASE_DIR = ROOT / "release" / "arxiv"
RELEASE_ID = "ready-cohorts-arxiv-v1"
ARCHIVE = RELEASE_DIR / f"{RELEASE_ID}.tar.gz"
MANIFEST = RELEASE_DIR / f"{RELEASE_ID}-manifest.json"
METADATA = RELEASE_DIR / f"{RELEASE_ID}-submission-metadata.txt"
SOURCE_DATE_EPOCH = 1_786_492_800  # 2026-08-12T00:00:00Z
MAX_ARXIV_BYTES = 50 * 1024 * 1024

SECTION_NAMES = (
    "00_abstract.tex",
    "01_introduction.tex",
    "02_scope.tex",
    "03_model.tex",
    "04_exact_packing.tex",
    "05_trace_method.tex",
    "06_trace_results.tex",
    "07_resident_method.tex",
    "08_resident_results.tex",
    "09_joint_interpretation.tex",
    "10_related_work.tex",
    "11_limitations.tex",
    "12_reproducibility.tex",
    "13_conclusion.tex",
    "appendix.tex",
)
GENERATED_NAMES = (
    "results-macros.tex",
    "trace-primary-table.tex",
    "resident-primary-table.tex",
)
FIGURE_NAMES = (
    "trace-exact-opportunity-frontier.png",
    "resident-policy-speedup-by-horizon.png",
    "resident-policy-primary-mechanisms.png",
)
VALID_COMPONENT = re.compile(r"^[A-Za-z0-9_+.,=-]+$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not read page count from {path}")
    return int(match.group(1))


def source_entries() -> list[tuple[Path, str, str | None]]:
    entries: list[tuple[Path, str, str | None]] = [
        (PAPER / "main.tex", "main.tex", "rewrite_graphicspath"),
        (PAPER / "main.bbl", "main.bbl", None),
    ]
    entries.extend((PAPER / "sections" / name, f"sections/{name}", None) for name in SECTION_NAMES)
    entries.extend(
        (PAPER / "generated" / name, f"generated/{name}", None) for name in GENERATED_NAMES
    )
    entries.extend(
        (ROOT / "results" / "figures" / name, f"figures/{name}", None) for name in FIGURE_NAMES
    )
    return entries


def packaged_bytes(path: Path, transform: str | None) -> bytes:
    if not path.is_file():
        raise RuntimeError(f"missing arXiv input: {path}")
    data = path.read_bytes()
    if transform is None:
        return data
    if transform != "rewrite_graphicspath":
        raise RuntimeError(f"unknown source transform: {transform}")
    text = data.decode("utf-8")
    old = r"\graphicspath{{../../results/figures/}}"
    new = r"\graphicspath{{figures/}}"
    if text.count(old) != 1:
        raise RuntimeError("main.tex does not contain exactly one frozen graphic path")
    return text.replace(old, new).encode("utf-8")


def validate_archive_path(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe archive path: {name}")
    for component in path.parts:
        if component.startswith(".") or VALID_COMPONENT.fullmatch(component) is None:
            raise RuntimeError(f"arXiv-incompatible archive path: {name}")


def write_archive(files: list[tuple[str, bytes]]) -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + ".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=9,
            mtime=SOURCE_DATE_EPOCH,
        ) as compressed,
        tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.USTAR_FORMAT,
        ) as archive,
    ):
        for name, data in sorted(files):
            validate_archive_path(name)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = SOURCE_DATE_EPOCH
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            archive.addfile(info, io.BytesIO(data))
    os.replace(temporary, ARCHIVE)
    if ARCHIVE.stat().st_size > MAX_ARXIV_BYTES:
        raise RuntimeError(
            f"archive exceeds arXiv's 50 MiB normal limit: {ARCHIVE.stat().st_size} bytes"
        )


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    canonical_pdf = PAPER / "main.pdf"
    if not canonical_pdf.is_file():
        raise RuntimeError("build paper/arxiv/main.pdf before packaging")
    if not METADATA.is_file():
        raise RuntimeError(f"missing submission handoff metadata: {METADATA}")

    archive_files: list[tuple[str, bytes]] = []
    records: list[dict[str, object]] = []
    for source, archive_path, transform in source_entries():
        validate_archive_path(archive_path)
        data = packaged_bytes(source, transform)
        archive_files.append((archive_path, data))
        record: dict[str, object] = {
            "archive_path": archive_path,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "source_path": relative(source),
            "source_sha256": sha256_file(source),
        }
        if transform is not None:
            record["transformation"] = transform
        records.append(record)

    write_archive(archive_files)
    payload: dict[str, object] = {
        "schema_version": "ready-cohorts-arxiv-release-v1",
        "release_id": RELEASE_ID,
        "release_date": "2026-08-12",
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "top_level_tex": "main.tex",
        "processor": "pdflatex",
        "arxiv_tex_live_target": "2025",
        "archive": {
            "path": relative(ARCHIVE),
            "bytes": ARCHIVE.stat().st_size,
            "sha256": sha256_file(ARCHIVE),
        },
        "expected_pdf": {
            "path": relative(canonical_pdf),
            "bytes": canonical_pdf.stat().st_size,
            "pages": pdf_pages(canonical_pdf),
            "sha256": sha256_file(canonical_pdf),
        },
        "submission_metadata": {
            "path": relative(METADATA),
            "bytes": METADATA.stat().st_size,
            "sha256": sha256_file(METADATA),
        },
        "archive_files": sorted(records, key=lambda item: str(item["archive_path"])),
        "omitted_from_upload": [
            "PDF output and LaTeX auxiliary files",
            "references.bib because the matching main.bbl is supplied",
            "paper-data manifests, raw data, notebooks, and provider receipts",
            "private submitter contact metadata",
        ],
    }
    atomic_json(MANIFEST, payload)
    print(
        f"built {relative(ARCHIVE)}: {len(records)} files, "
        f"{ARCHIVE.stat().st_size} bytes, SHA-256 {sha256_file(ARCHIVE)}"
    )


if __name__ == "__main__":
    main()
