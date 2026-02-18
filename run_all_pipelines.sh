#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

METAORDER_COMP_CFG="config_ymls/metaorder_computation.yml"
METAORDER_STATS_CFG="config_ymls/metaorder_statistics.yml"
CROWDING_CFG="config_ymls/crowding_analysis.yml"

if [[ ! -f "$METAORDER_COMP_CFG" ]]; then
  echo "[error] Missing config: $METAORDER_COMP_CFG" >&2
  exit 1
fi
if [[ ! -f "$METAORDER_STATS_CFG" ]]; then
  echo "[error] Missing config: $METAORDER_STATS_CFG" >&2
  exit 1
fi
if [[ ! -f "$CROWDING_CFG" ]]; then
  echo "[error] Missing config: $CROWDING_CFG" >&2
  exit 1
fi

tmp_comp_cfg="$(mktemp)"
tmp_stats_cfg="$(mktemp)"
tmp_crowding_cfg="$(mktemp)"
cp "$METAORDER_COMP_CFG" "$tmp_comp_cfg"
cp "$METAORDER_STATS_CFG" "$tmp_stats_cfg"
cp "$CROWDING_CFG" "$tmp_crowding_cfg"

restore_configs() {
  cp "$tmp_comp_cfg" "$METAORDER_COMP_CFG"
  cp "$tmp_stats_cfg" "$METAORDER_STATS_CFG"
  cp "$tmp_crowding_cfg" "$CROWDING_CFG"
  rm -f "$tmp_comp_cfg" "$tmp_stats_cfg" "$tmp_crowding_cfg"
}
trap restore_configs EXIT

set_yaml_bool() {
  local file_path="$1"
  local key="$2"
  local value="$3"
  if ! grep -qE "^${key}:" "$file_path"; then
    echo "[error] Key '${key}' not found in ${file_path}" >&2
    exit 1
  fi
  local tmp_file
  tmp_file="$(mktemp)"
  awk -v key="$key" -v value="$value" '
    BEGIN { updated = 0 }
    $0 ~ ("^" key ":[[:space:]]*(true|false)([[:space:]]*#.*)?$") {
      sub(/:[[:space:]]*(true|false)/, ": " value)
      updated = 1
    }
    { print }
    END {
      if (updated == 0) {
        exit 42
      }
    }
  ' "$file_path" > "$tmp_file" || {
    rm -f "$tmp_file"
    echo "[error] Failed to update '${key}' in ${file_path}" >&2
    exit 1
  }
  mv "$tmp_file" "$file_path"
}

echo "[info] Activating conda environment: main"
if ! command -v conda >/dev/null 2>&1; then
  echo "[error] 'conda' command not found in PATH." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

echo "[run] metaorder_computation.py (PROPRIETARY=true)"
set_yaml_bool "$METAORDER_COMP_CFG" "PROPRIETARY" "true"
python metaorder_computation.py

echo "[run] metaorder_computation.py (PROPRIETARY=false)"
set_yaml_bool "$METAORDER_COMP_CFG" "PROPRIETARY" "false"
python metaorder_computation.py

echo "[run] metaorder_statistics.py (METAORDER_STATS_PROPRIETARY=true)"
set_yaml_bool "$METAORDER_STATS_CFG" "METAORDER_STATS_PROPRIETARY" "true"
python metaorder_statistics.py

echo "[run] metaorder_statistics.py (METAORDER_STATS_PROPRIETARY=false)"
set_yaml_bool "$METAORDER_STATS_CFG" "METAORDER_STATS_PROPRIETARY" "false"
python metaorder_statistics.py

echo "[run] crowding_analysis.py"
python crowding_analysis.py

echo "[run] member_statistics.py"
python member_statistics.py

echo "[done] Pipeline completed. Original config files restored."
