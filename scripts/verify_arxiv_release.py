"""Verify and compile the frozen Ready Cohorts arXiv archive in isolation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release" / "arxiv"
RELEASE_ID = "ready-cohorts-arxiv-v1"
MANIFEST = RELEASE_DIR / f"{RELEASE_ID}-manifest.json"
OUTPUT_PDF = RELEASE_DIR / f"{RELEASE_ID}.pdf"
VERIFICATION = RELEASE_DIR / f"{RELEASE_ID}-verification.json"
OFFICIAL_PREFLIGHT = RELEASE_DIR / f"{RELEASE_ID}-official-preflight.json"
TEXLIVE_2025_CHECK = RELEASE_DIR / f"{RELEASE_ID}-texlive-2025.json"
CHECKSUMS = RELEASE_DIR / "SHA256SUMS"
README = RELEASE_DIR / "README.md"
SOURCE_DATE_EPOCH = "1786492800"
VALID_COMPONENT = re.compile(r"^[A-Za-z0-9_+.,=-]+$")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET_PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "artifact bearer field": re.compile(r"\bartifact_token\b", re.IGNORECASE),
    "API secret field": re.compile(r"\b(?:api_key|api_secret)\b", re.IGNORECASE),
}
MACHINE_PATHS = re.compile(
    r"(?:/home/|/Users/|[A-Za-z]:\\\\|file://|localhost|(?:^|[/\\])\.\.(?:[/\\]|$))"
)
SENSITIVE_ENV = re.compile(
    r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AWS|BEDROCK|MODAL|RUNPOD|LAMBDA|OPENROUTER|HUGGINGFACE|WANDB)",
    re.IGNORECASE,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    atomic_write(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def validate_member_name(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe archive member: {name}")
    for component in path.parts:
        if component.startswith(".") or VALID_COMPONENT.fullmatch(component) is None:
            raise RuntimeError(f"arXiv-incompatible archive member: {name}")


def extract_regular_files(archive_path: Path, destination: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            validate_member_name(member.name)
            if not member.isfile():
                raise RuntimeError(f"archive member is not a regular file: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"could not read archive member: {member.name}")
            data = handle.read()
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[member.name] = data
    return files


def scan_public_sources(files: dict[str, bytes]) -> None:
    for name, data in files.items():
        if Path(name).suffix.lower() not in {".tex", ".bbl"}:
            continue
        text = data.decode("utf-8")
        if EMAIL.search(text):
            raise RuntimeError(f"public source contains an email address: {name}")
        if MACHINE_PATHS.search(text):
            raise RuntimeError(f"public source contains a machine-local path: {name}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                raise RuntimeError(f"possible {label} in public source: {name}")


def safe_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if SENSITIVE_ENV.search(key) is None
    }
    environment["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    environment["FORCE_SOURCE_DATE"] = "1"
    environment["openin_any"] = "p"
    environment["openout_any"] = "p"
    return environment


def run(command: list[str], cwd: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{output[-8000:]}"
        )
    return output


def parse_pdfinfo(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def verify_fonts(pdf: Path, cwd: Path, environment: dict[str, str]) -> int:
    output = run(["pdffonts", str(pdf)], cwd, environment)
    lines = [line for line in output.splitlines()[2:] if line.strip()]
    if not lines:
        raise RuntimeError("pdffonts found no fonts")
    for line in lines:
        fields = line.split()
        # The final five columns are emb/sub/uni/object/ID. The font type can
        # itself contain a space (for example, "Type 1"), so parsing from the
        # front is not stable.
        if len(fields) < 5 or fields[-5].lower() != "yes":
            raise RuntimeError(f"PDF contains an unembedded font: {line}")
    return len(lines)


def write_checksums(paths: list[Path]) -> None:
    lines = [f"{sha256_file(path)}  {path.name}" for path in sorted(paths)]
    atomic_write(CHECKSUMS, ("\n".join(lines) + "\n").encode("ascii"))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    archive_path = ROOT / manifest["archive"]["path"]
    declared_archive_sha = manifest["archive"]["sha256"]
    if sha256_file(archive_path) != declared_archive_sha:
        raise RuntimeError("archive hash does not match release manifest")
    if archive_path.stat().st_size != manifest["archive"]["bytes"]:
        raise RuntimeError("archive size does not match release manifest")
    official_preflight: dict[str, object] | None = None
    if OFFICIAL_PREFLIGHT.is_file():
        official_preflight = json.loads(OFFICIAL_PREFLIGHT.read_text(encoding="utf-8"))
        if official_preflight.get("archive_sha256") != declared_archive_sha:
            raise RuntimeError("official arXiv preflight is bound to a different archive")
        if official_preflight.get("status") != "success":
            raise RuntimeError("official arXiv preflight did not report success")
        if official_preflight.get("all_archive_files_used") is not True:
            raise RuntimeError("official arXiv preflight found unused archive files")
        if official_preflight.get("top_level_issue_count") != 0:
            raise RuntimeError("official arXiv preflight found a top-level issue")
        if official_preflight.get("tex_file_issue_count") != 0:
            raise RuntimeError("official arXiv preflight found a TeX-file issue")
    texlive_2025 = json.loads(TEXLIVE_2025_CHECK.read_text(encoding="utf-8"))
    if texlive_2025.get("archive_sha256") != declared_archive_sha:
        raise RuntimeError("TeX Live 2025 check is bound to a different archive")
    if texlive_2025.get("status") != "success":
        raise RuntimeError("TeX Live 2025 check did not report success")
    texlive_compile = texlive_2025.get("compile", {})
    if not isinstance(texlive_compile, dict):
        raise TypeError("TeX Live 2025 compile record is malformed")
    if texlive_compile.get("pages") != manifest["expected_pdf"]["pages"]:
        raise RuntimeError("TeX Live 2025 page count differs from the canonical build")
    if texlive_compile.get("extracted_text_exact_match") is not True:
        raise RuntimeError("TeX Live 2025 extracted text does not match the canonical build")
    if texlive_compile.get("network_enabled") is not False:
        raise RuntimeError("TeX Live 2025 check was not network-isolated")
    if texlive_compile.get("shell_escape") is not False:
        raise RuntimeError("TeX Live 2025 check enabled shell escape")

    declared_files = {item["archive_path"]: item for item in manifest["archive_files"]}
    environment = safe_environment()
    with tempfile.TemporaryDirectory(prefix="ready-cohorts-arxiv-verify.") as name:
        build_dir = Path(name)
        files = extract_regular_files(archive_path, build_dir)
        if set(files) != set(declared_files):
            missing = sorted(set(declared_files) - set(files))
            extra = sorted(set(files) - set(declared_files))
            raise RuntimeError(f"archive inventory mismatch; missing={missing}, extra={extra}")
        for archive_name, data in files.items():
            record = declared_files[archive_name]
            if len(data) != record["bytes"] or sha256_bytes(data) != record["sha256"]:
                raise RuntimeError(f"archive member hash mismatch: {archive_name}")
        scan_public_sources(files)

        compiler_outputs = []
        command = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-no-shell-escape",
            "-recorder",
            "main.tex",
        ]
        for _ in range(3):
            compiler_outputs.append(run(command, build_dir, environment))

        log = (build_dir / "main.log").read_text(encoding="utf-8", errors="replace")
        fatal_markers = (
            "LaTeX Warning: There were undefined references",
            "Package natbib Warning: There were undefined citations",
            "Undefined control sequence",
            "Emergency stop",
            "Fatal error",
            "Overfull \\hbox",
        )
        for marker in fatal_markers:
            if marker in log:
                raise RuntimeError(f"LaTeX release log contains: {marker}")

        recorder = (build_dir / "main.fls").read_text(encoding="utf-8", errors="replace")
        forbidden_roots = (str(ROOT), "/home/", "/Users/")
        for line in recorder.splitlines():
            if not line.startswith("INPUT "):
                continue
            source = line.removeprefix("INPUT ")
            if any(root in source for root in forbidden_roots):
                raise RuntimeError(f"isolated compile read a machine-local input: {source}")

        pdf = build_dir / "main.pdf"
        expected_pdf = manifest["expected_pdf"]
        actual_pdf_sha = sha256_file(pdf)
        if actual_pdf_sha != expected_pdf["sha256"]:
            raise RuntimeError(
                "isolated PDF differs from the canonical PDF: "
                f"expected {expected_pdf['sha256']}, got {actual_pdf_sha}"
            )
        info_text = run(["pdfinfo", str(pdf)], build_dir, environment)
        info = parse_pdfinfo(info_text)
        if int(info.get("Pages", "0")) != expected_pdf["pages"]:
            raise RuntimeError("isolated PDF page count differs from release manifest")
        if info.get("Title") != (
            "Ready Cohorts: Bounding GPU Opportunity and Avoiding Host Round Trips "
            "in LLM-Agent Control"
        ):
            raise RuntimeError("PDF title metadata is wrong")
        if info.get("Author") != "Josef Chen":
            raise RuntimeError("PDF author metadata is wrong")
        if info.get("JavaScript", "no").lower() != "no":
            raise RuntimeError("PDF contains JavaScript")
        text_output = run(["pdftotext", str(pdf), "-"], build_dir, environment)
        if "Josef Chen" not in text_output or "Independent Researcher" not in text_output:
            raise RuntimeError("public author name or affiliation is missing from rendered PDF")
        if EMAIL.search(text_output):
            raise RuntimeError("rendered PDF contains an email address")
        font_count = verify_fonts(pdf, build_dir, environment)

        atomic_write(OUTPUT_PDF, pdf.read_bytes())
        version = run(["pdflatex", "--version"], build_dir, environment).splitlines()[0]
        verification: dict[str, object] = {
            "schema_version": "ready-cohorts-arxiv-verification-v1",
            "release_id": RELEASE_ID,
            "archive_path": relative(archive_path),
            "archive_sha256": declared_archive_sha,
            "archive_files_verified": len(files),
            "public_source_scan": "passed",
            "isolated_compile": {
                "network_required": False,
                "shell_escape": False,
                "runs": len(compiler_outputs),
                "compiler": version,
                "local_tex_live": "2023",
                "arxiv_target": "TeX Live 2025 / pdflatex",
                "tex_live_2025_static_compatibility": "cleveref dependency removed",
            },
            "tex_live_2025_compile": {
                "path": relative(TEXLIVE_2025_CHECK),
                "sha256": sha256_file(TEXLIVE_2025_CHECK),
                "container_digest": texlive_2025["container"]["digest"],
                "compiler": texlive_compile["compiler"],
                "pages": texlive_compile["pages"],
                "pdf_sha256": texlive_compile["pdf_sha256"],
                "extracted_text_exact_match": True,
                "status": "success",
            },
            "pdf": {
                "path": relative(OUTPUT_PDF),
                "bytes": OUTPUT_PDF.stat().st_size,
                "pages": int(info["Pages"]),
                "sha256": actual_pdf_sha,
                "fonts_embedded": font_count,
                "javascript": False,
                "public_email": False,
            },
            "log_gates": {
                "undefined_references": 0,
                "undefined_citations": 0,
                "overfull_boxes": 0,
            },
        }
        if official_preflight is not None:
            verification["official_arxiv_preflight"] = {
                "path": relative(OFFICIAL_PREFLIGHT),
                "sha256": sha256_file(OFFICIAL_PREFLIGHT),
                "tool_commit": official_preflight["tool_commit"],
                "status": "success",
                "issues": 0,
                "all_archive_files_used": True,
            }
        atomic_json(VERIFICATION, verification)

    checksum_paths = [archive_path, MANIFEST, OUTPUT_PDF, VERIFICATION]
    metadata_path = ROOT / manifest["submission_metadata"]["path"]
    if sha256_file(metadata_path) != manifest["submission_metadata"]["sha256"]:
        raise RuntimeError("submission metadata changed after archive build")
    checksum_paths.append(metadata_path)
    if README.is_file():
        checksum_paths.append(README)
    if OFFICIAL_PREFLIGHT.is_file():
        checksum_paths.append(OFFICIAL_PREFLIGHT)
    checksum_paths.append(TEXLIVE_2025_CHECK)
    write_checksums(checksum_paths)
    print(
        f"verified {relative(archive_path)}; isolated PDF SHA-256 "
        f"{sha256_file(OUTPUT_PDF)}; {manifest['expected_pdf']['pages']} pages"
    )


if __name__ == "__main__":
    main()
