#!/usr/bin/env python3
"""
Lightweight smoke checks for the repository's main scripts.

This script is intentionally conservative: it does not require proprietary data
and does not run the heavy research pipelines. It focuses on import safety and
CLI plumbing that should remain stable across refactors.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import safety: these should not patch global print or start pipelines.
    __import__("metaorder_statistics")
    __import__("crowding_analysis")
    __import__("metaorder_computation")

    # CLI smoke: ensure help text works (no data required).
    for script in ("crowding_vs_part_rate.py", "metaorder_clustering.py"):
        subprocess.run([sys.executable, script, "--help"], check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
