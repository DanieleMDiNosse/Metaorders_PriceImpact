#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] 'conda' command not found in PATH." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

python scripts/generate_paper_figures.py "$@"
