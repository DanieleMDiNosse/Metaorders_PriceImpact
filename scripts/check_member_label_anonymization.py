from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moimpact.workflows.paper.figures import _paper_figure_candidates, _paper_figure_paths

DEFAULT_MAP_PATH = ROOT / "config_ymls" / "member_label_map.json"
DEFAULT_PAPER_DIR = ROOT / "paper"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check paper-visible assets for raw mapped member IDs.")
    parser.add_argument("--map-path", type=Path, default=DEFAULT_MAP_PATH)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    return parser.parse_args(argv)


def _raw_id_pattern(map_path: Path) -> re.Pattern[str]:
    if not map_path.is_file():
        raise FileNotFoundError(
            f"Confidential member-label map not found: {map_path}. "
            "Provide the local crosswalk with --map-path."
        )
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    raw_ids = [re.escape(str(key)) for key in mapping]
    if not raw_ids:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(?:" + "|".join(raw_ids) + r")\b")


def _text_failures(path: Path, text: str, pattern: re.Pattern[str], paper_dir: Path) -> list[str]:
    failures: list[str] = []
    for match in pattern.finditer(text):
        line_no = text.count("\n", 0, match.start()) + 1
        failures.append(f"{path.relative_to(paper_dir)}:{line_no}: raw member ID [REDACTED]")
    return failures


def _pdf_text(path: Path) -> str:
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext is required to check paper-visible PDFs")
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _active_figure_assets(paper_dir: Path) -> list[Path]:
    assets: list[Path] = []
    for figure in _paper_figure_paths(paper_dir / "main.tex"):
        candidates = _paper_figure_candidates(paper_dir, figure)
        existing = next((candidate for candidate in candidates if candidate.is_file()), None)
        if existing is None:
            raise FileNotFoundError(f"Active paper figure not found: {figure}")
        assets.append(existing)
    return assets


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paper_dir = args.paper_dir.resolve()
    try:
        pattern = _raw_id_pattern(args.map_path.resolve())
        figure_assets = _active_figure_assets(paper_dir)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(exc)
        return 2

    text_paths = [
        paper_dir / "main.tex",
        *sorted((paper_dir / "tables").glob("*.tex")),
    ]
    pdf_paths = [
        path
        for path in [paper_dir / "main.pdf", *figure_assets]
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    raster_count = sum(
        path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in figure_assets
    )
    failures: list[str] = []

    for path in text_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        failures.extend(_text_failures(path, text, pattern, paper_dir))

    try:
        for path in pdf_paths:
            failures.extend(_text_failures(path, _pdf_text(path), pattern, paper_dir))
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(exc)
        return 2

    if failures:
        print("\n".join(failures))
        return 1
    print(
        "No raw mapped member IDs found in "
        f"{len(text_paths)} paper-visible text files and {len(pdf_paths)} PDFs "
        f"({raster_count} active raster figures require visual review)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
