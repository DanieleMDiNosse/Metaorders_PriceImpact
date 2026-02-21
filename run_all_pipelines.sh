#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_all_pipelines.sh
#
# Run the repository's end-to-end analysis pipeline with the standard
# proprietary/client split and member-nationality slices.
#
# Parallelism:
# - Independent (PROPRIETARY, MEMBER_NATIONALITY) slices are run in parallel as
#   separate `python` processes (no shared state). Control concurrency via
#   `MAX_JOBS` (default: 4).
#
# Safety:
# - Never edits `config_ymls/*.yml` in place. Each run uses a temporary YAML copy
#   passed via environment variables:
#     - METAORDER_COMP_CONFIG for `scripts/metaorder_computation.py`
#     - METAORDER_STATS_CONFIG for `scripts/metaorder_statistics.py`
# - Runs the CSV->parquet transform step (RUN_INTRO) once, serially, before
#   launching parallel jobs, because it writes to `data/parquet/`.
# -----------------------------------------------------------------------------
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

yaml_get_scalar() {
  local file_path="$1"
  local key="$2"
  awk -v key="$key" '
    $0 ~ ("^" key ":[[:space:]]*") {
      sub("^" key ":[[:space:]]*", "", $0)
      sub("[[:space:]]+#.*$", "", $0)
      gsub("^[[:space:]]+|[[:space:]]+$", "", $0)
      gsub("^\"|\"$", "", $0)
      gsub("^\047|\047$", "", $0)
      print $0
      exit
    }
  ' "$file_path"
}

DATASET_NAME="$(yaml_get_scalar "$METAORDER_COMP_CFG" "DATASET_NAME")"
LEVEL="$(yaml_get_scalar "$METAORDER_COMP_CFG" "LEVEL")"
DATASET_NAME="${DATASET_NAME:-dataset}"
LEVEL="${LEVEL:-member}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_LOG_DIR="out_files/${DATASET_NAME}/logs/pipeline_${RUN_TAG}"
mkdir -p "$PIPELINE_LOG_DIR"
echo "[info] Pipeline logs: ${PIPELINE_LOG_DIR}"

MIN_FREE_GB="${MIN_FREE_GB:-5}"
if [[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]]; then
  min_free_kb=$((MIN_FREE_GB * 1024 * 1024))
  free_kb="$(df -Pk "$SCRIPT_DIR" | awk 'NR==2 {print $4}')"
  if [[ "$free_kb" =~ ^[0-9]+$ ]] && (( free_kb < min_free_kb )); then
    echo "[warn] Low disk space: $(df -h "$SCRIPT_DIR" | awk 'NR==2 {print $4}') available (< ${MIN_FREE_GB}GiB)." >&2
  fi
fi

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

set_yaml_scalar() {
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
    $0 ~ ("^" key ":[[:space:]]*") {
      comment = ""
      if (match($0, /[[:space:]]+#.*/)) {
        comment = substr($0, RSTART)
      }
      print key ": " value comment
      updated = 1
      next
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

MAX_JOBS="${MAX_JOBS:-4}"
if ! [[ "$MAX_JOBS" =~ ^[0-9]+$ ]] || (( MAX_JOBS < 1 )); then
  echo "[error] MAX_JOBS must be a positive integer (got: ${MAX_JOBS})" >&2
  exit 1
fi
echo "[info] Parallelism: MAX_JOBS=${MAX_JOBS}"

declare -A JOB_LABEL
declare -A JOB_LOG
RUNNING_JOBS=0

wait_one_job() {
  local finished_pid=""
  if wait -n -p finished_pid; then
    echo "[info] Finished (pid=${finished_pid}): ${JOB_LABEL[$finished_pid]:-unknown} (log: ${JOB_LOG[$finished_pid]:-n/a})"
    unset JOB_LABEL["$finished_pid"]
    unset JOB_LOG["$finished_pid"]
    RUNNING_JOBS=$((RUNNING_JOBS - 1))
    return 0
  else
    local status=$?
    echo "[error] Job failed (pid=${finished_pid})" >&2
    if [[ -n "${JOB_LABEL[$finished_pid]:-}" ]]; then
      echo "[error]   label: ${JOB_LABEL[$finished_pid]}" >&2
    fi
    if [[ -n "${JOB_LOG[$finished_pid]:-}" ]]; then
      echo "[error]   log:   ${JOB_LOG[$finished_pid]}" >&2
    fi
    exit "$status"
  fi
}

start_job() {
  local label="$1"
  local log_path="$2"
  shift 2

  while (( RUNNING_JOBS >= MAX_JOBS )); do
    wait_one_job
  done

  echo "[info] Starting: ${label}"
  echo "[info]   log: ${log_path}"
  "$@" >"$log_path" 2>&1 &
  local pid=$!
  JOB_LABEL["$pid"]="$label"
  JOB_LOG["$pid"]="$log_path"
  # Avoid `((var++))` under `set -e` (it returns status 1 when the old value is 0).
  RUNNING_JOBS=$((RUNNING_JOBS + 1))
}

wait_all_jobs() {
  while (( RUNNING_JOBS > 0 )); do
    wait_one_job
  done
}

echo "[info] Activating conda environment: main"
if ! command -v conda >/dev/null 2>&1; then
  echo "[error] 'conda' command not found in PATH." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

run_intro_transforms() (
  set -euo pipefail
  local tmpdir cfg
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT
  cfg="$tmpdir/metaorder_computation.yml"
  cp "$METAORDER_COMP_CFG" "$cfg"

  set_yaml_bool "$cfg" "RUN_INTRO" "true"
  set_yaml_bool "$cfg" "RUN_METAORDER_COMPUTATION" "false"
  set_yaml_bool "$cfg" "RUN_SQL_FITS" "false"
  set_yaml_bool "$cfg" "RUN_WLS" "false"
  set_yaml_bool "$cfg" "RUN_IMPACT_PATH_PLOT" "false"
  set_yaml_bool "$cfg" "RUN_SIGNATURE_PLOTS" "false"
  set_yaml_bool "$cfg" "SPLIT_BY_SIDE" "false"

  METAORDER_COMP_CONFIG="$cfg" python scripts/metaorder_computation.py
)

metaorder_slice_job() (
  set -euo pipefail
  local proprietary="$1"
  local member_nationality="$2"
  local tmpdir cfg
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT
  cfg="$tmpdir/metaorder_computation.yml"
  cp "$METAORDER_COMP_CFG" "$cfg"

  set_yaml_bool "$cfg" "PROPRIETARY" "$proprietary"
  set_yaml_scalar "$cfg" "MEMBER_NATIONALITY" "$member_nationality"
  set_yaml_bool "$cfg" "RUN_SIGNATURE_PLOTS" "false"

  # Full run (uses precomputed parquet transforms from intro step).
  set_yaml_bool "$cfg" "SPLIT_BY_SIDE" "false"
  set_yaml_bool "$cfg" "RUN_INTRO" "false"
  set_yaml_bool "$cfg" "RUN_METAORDER_COMPUTATION" "true"
  set_yaml_bool "$cfg" "RUN_SQL_FITS" "true"
  set_yaml_bool "$cfg" "RUN_WLS" "true"
  set_yaml_bool "$cfg" "RUN_IMPACT_PATH_PLOT" "true"
  METAORDER_COMP_CONFIG="$cfg" python scripts/metaorder_computation.py

  # WLS-only pass to produce the buy/sell decomposition figures.
  set_yaml_bool "$cfg" "SPLIT_BY_SIDE" "true"
  set_yaml_bool "$cfg" "RUN_METAORDER_COMPUTATION" "false"
  set_yaml_bool "$cfg" "RUN_SQL_FITS" "false"
  set_yaml_bool "$cfg" "RUN_WLS" "true"
  set_yaml_bool "$cfg" "RUN_IMPACT_PATH_PLOT" "true"
  METAORDER_COMP_CONFIG="$cfg" python scripts/metaorder_computation.py
)

metaorder_stats_slice_job() (
  set -euo pipefail
  local proprietary="$1"
  local member_nationality="$2"
  local tmpdir cfg
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT
  cfg="$tmpdir/metaorder_statistics.yml"
  cp "$METAORDER_STATS_CFG" "$cfg"

  set_yaml_bool "$cfg" "METAORDER_STATS_PROPRIETARY" "$proprietary"
  set_yaml_scalar "$cfg" "MEMBER_NATIONALITY" "$member_nationality"

  METAORDER_STATS_CONFIG="$cfg" python scripts/metaorder_statistics.py
)

echo "[phase] Intro transforms (serial)"
run_intro_transforms >"${PIPELINE_LOG_DIR}/intro_transforms.log" 2>&1
echo "[info] Intro transforms completed (log: ${PIPELINE_LOG_DIR}/intro_transforms.log)"

echo "[phase] scripts/metaorder_computation.py slices (parallel)"
for member_nationality in all it foreign; do
  for proprietary in true false; do
    label="metaorder_computation (LEVEL=${LEVEL}, PROPRIETARY=${proprietary}, MEMBER_NATIONALITY=${member_nationality})"
    log_path="${PIPELINE_LOG_DIR}/metaorder_computation_${LEVEL}_prop_${proprietary}_nat_${member_nationality}.log"
    start_job "$label" "$log_path" metaorder_slice_job "$proprietary" "$member_nationality"
  done
done
wait_all_jobs

echo "[phase] scripts/metaorder_statistics.py slices (parallel)"
for member_nationality in all it foreign; do
  for proprietary in true false; do
    label="metaorder_statistics (PROPRIETARY=${proprietary}, MEMBER_NATIONALITY=${member_nationality})"
    log_path="${PIPELINE_LOG_DIR}/metaorder_statistics_prop_${proprietary}_nat_${member_nationality}.log"
    start_job "$label" "$log_path" metaorder_stats_slice_job "$proprietary" "$member_nationality"
  done
done
wait_all_jobs

echo "[phase] scripts/crowding_analysis.py (serial)"
python scripts/crowding_analysis.py >"${PIPELINE_LOG_DIR}/crowding_analysis.log" 2>&1
echo "[info] crowding_analysis completed (log: ${PIPELINE_LOG_DIR}/crowding_analysis.log)"

echo "[phase] scripts/member_statistics.py (serial)"
python scripts/member_statistics.py >"${PIPELINE_LOG_DIR}/member_statistics.log" 2>&1
echo "[info] member_statistics completed (log: ${PIPELINE_LOG_DIR}/member_statistics.log)"

echo "[done] Pipeline completed (logs: ${PIPELINE_LOG_DIR})."
