#!/usr/bin/env python3
"""
Event-study of metaorder starts around high-participation anchors.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import datetime as dt
import json
import multiprocessing as mp
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Ensure repository-root imports work when running from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.config import format_path_template, load_yaml_mapping, resolve_repo_path
from moimpact.plot_style import (
    apply_shared_plotly_style,
    load_plot_style,
)
from moimpact.plotting import (
    COLOR_CLIENT,
    COLOR_NEUTRAL,
    COLOR_PROPRIETARY,
    PlotOutputDirs,
    make_plot_output_dirs,
    save_plotly_figure as _save_plotly_figure,
)


COL_ISIN = "ISIN"
COL_DATE = "Date"
COL_PERIOD = "Period"
COL_DIR = "Direction"
COL_ETA = "Participation Rate"
COL_MEMBER = "Member"
COL_CLIENT = "Client"
COL_START_TS = "StartTimestamp"
MINUTE_NS = np.int64(60 * 1_000_000_000)
VARIANT_ALL_OTHERS = "all_others"
VARIANT_EXCLUDE_SAME_ACTOR = "exclude_same_actor"


@dataclass(frozen=True)
class ResolvedPaths:
    """Resolved input/output paths and identifiers for the workflow."""

    dataset_name: str
    prop_path: Path
    client_path: Path
    out_dir: Path
    img_dir: Path
    config_path: Path


@dataclass(frozen=True)
class BinSpec:
    """Event-time bin specification used by the study."""

    left_minutes: np.ndarray
    right_minutes: np.ndarray
    left_ns: np.ndarray
    right_ns: np.ndarray
    centers_minutes: np.ndarray
    labels: list[str]
    pre_count: int


@dataclass(frozen=True)
class BootstrapPayload:
    """Precomputed stratum-level arrays for bootstrap inference."""

    date_codes: np.ndarray
    n_dates: int
    n_treated: np.ndarray
    n_control: np.ndarray
    treated_sums: np.ndarray
    control_sums: np.ndarray
    treated_valid: np.ndarray
    control_valid: np.ndarray


@dataclass(frozen=True)
class PermutationPayload:
    """Precomputed within-stratum arrays for permutation inference."""

    values_zero_by_stratum: tuple[np.ndarray, ...]
    finite_mask_by_stratum: tuple[np.ndarray, ...]
    treated_mask_by_stratum: tuple[np.ndarray, ...]
    n_treated_by_stratum: np.ndarray
    n_metrics: int


_BOOTSTRAP_WORKER_PAYLOAD: Optional[BootstrapPayload] = None
_PERMUTATION_WORKER_PAYLOAD: Optional[PermutationPayload] = None


class _NullTqdm:
    """Fallback progress-bar object used when tqdm is unavailable or disabled."""

    def __enter__(self) -> "_NullTqdm":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        return None

    def close(self) -> None:
        return None


def _make_tqdm(
    *,
    total: Optional[int],
    desc: str,
    disable: bool,
    leave: bool = True,
    unit: str = "it",
):
    if disable or tqdm is None:
        return _NullTqdm()
    return tqdm(total=total, desc=desc, leave=leave, unit=unit, dynamic_ncols=True)


def save_plotly_figure(fig, *args, **kwargs):
    """
    Summary
    -------
    Save a Plotly figure after clearing the top-level title.

    Parameters
    ----------
    fig
        Plotly figure object.
    *args, **kwargs
        Forwarded to `moimpact.plotting.save_plotly_figure`.

    Returns
    -------
    tuple[Optional[Path], Optional[Path]]
        Output HTML and PNG paths.

    Notes
    -----
    This keeps figure exports consistent with the other research workflows.
    """
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)


def _load_yaml_defaults(config_path: Path) -> dict[str, Any]:
    return load_yaml_mapping(config_path)


def _parse_time_string(raw_value: object, *, label: str) -> time:
    parsed = pd.to_datetime(str(raw_value))
    if pd.isna(parsed):
        raise ValueError(f"Invalid time string for {label}: {raw_value!r}")
    return parsed.time()


def _period_start_ns(period_value: Any) -> Optional[int]:
    if period_value is None:
        return None
    if isinstance(period_value, float) and np.isnan(period_value):
        return None
    if isinstance(period_value, (list, tuple, np.ndarray)):
        if len(period_value) == 0:
            return None
        try:
            return int(period_value[0])
        except Exception:
            return None
    if isinstance(period_value, (np.integer, int)):
        return int(period_value)
    if isinstance(period_value, pd.Timestamp):
        return int(period_value.value)
    if isinstance(period_value, str):
        s = period_value.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = [part.strip() for part in s[1:-1].split(",") if part.strip()]
            if not inner:
                return None
            try:
                return int(inner[0])
            except Exception:
                return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _ensure_date_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if COL_DATE in df.columns:
        out = df.copy()
        out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce").dt.normalize()
        return out
    if COL_PERIOD not in df.columns:
        raise KeyError(f"[{label}] Missing {COL_DATE} and {COL_PERIOD}; cannot infer trading date.")

    out = df.copy()
    start_ns = out[COL_PERIOD].apply(_period_start_ns)
    out[COL_DATE] = pd.to_datetime(start_ns, errors="coerce").dt.normalize()
    if out[COL_DATE].isna().any():
        examples = out.loc[out[COL_DATE].isna(), COL_PERIOD].head(3).tolist()
        raise ValueError(f"[{label}] Failed to infer Date from Period. Examples: {examples}")
    return out


def _ensure_start_timestamp(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if COL_START_TS in df.columns:
        out = df.copy()
        out[COL_START_TS] = pd.to_datetime(out[COL_START_TS], errors="coerce")
        return out
    if COL_PERIOD not in df.columns:
        raise KeyError(f"[{label}] Missing {COL_START_TS} and {COL_PERIOD}; cannot infer start timestamp.")
    out = df.copy()
    start_ns = out[COL_PERIOD].apply(_period_start_ns)
    out[COL_START_TS] = pd.to_datetime(start_ns, errors="coerce")
    if out[COL_START_TS].isna().any():
        examples = out.loc[out[COL_START_TS].isna(), COL_PERIOD].head(3).tolist()
        raise ValueError(f"[{label}] Failed to infer StartTimestamp from Period. Examples: {examples}")
    return out


def _read_parquet_with_fallback(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=list(columns))
    except Exception:
        return pd.read_parquet(path)


def _try_git_hash() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compute_bin_spec(window_minutes: int, bin_minutes: int) -> BinSpec:
    if window_minutes <= 0:
        raise ValueError("EVENT_WINDOW_MINUTES must be positive.")
    if bin_minutes <= 0:
        raise ValueError("BIN_MINUTES must be positive.")
    if window_minutes % bin_minutes != 0:
        raise ValueError("EVENT_WINDOW_MINUTES must be divisible by BIN_MINUTES.")

    n_side = window_minutes // bin_minutes
    pre_left = np.arange(-window_minutes, 0, bin_minutes, dtype=float)
    pre_right = pre_left + float(bin_minutes)
    post_left = np.arange(0, window_minutes, bin_minutes, dtype=float)
    post_right = post_left + float(bin_minutes)
    left_minutes = np.concatenate([pre_left, post_left])
    right_minutes = np.concatenate([pre_right, post_right])
    labels = [
        f"[{int(left)},{int(right)})" if right <= 0 else f"({int(left)},{int(right)}]"
        for left, right in zip(left_minutes, right_minutes)
    ]
    centers = 0.5 * (left_minutes + right_minutes)
    return BinSpec(
        left_minutes=left_minutes,
        right_minutes=right_minutes,
        left_ns=np.asarray(np.rint(left_minutes * MINUTE_NS), dtype=np.int64),
        right_ns=np.asarray(np.rint(right_minutes * MINUTE_NS), dtype=np.int64),
        centers_minutes=centers,
        labels=labels,
        pre_count=int(n_side),
    )


def _resolve_paths(cfg: Mapping[str, Any], args: argparse.Namespace) -> ResolvedPaths:
    dataset_name = str(args.dataset_name or cfg.get("DATASET_NAME", "ftsemib"))
    context = {"DATASET_NAME": dataset_name}

    output_root = str(args.output_file_path or cfg.get("OUTPUT_FILE_PATH", "out_files/{DATASET_NAME}"))
    img_root = str(args.img_output_path or cfg.get("IMG_OUTPUT_PATH", "images/{DATASET_NAME}"))
    analysis_tag = str(args.analysis_tag or cfg.get("ANALYSIS_TAG", "metaorder_start_event_study"))

    output_root = format_path_template(output_root, context)
    img_root = format_path_template(img_root, context)

    default_prop = Path(output_root) / "metaorders_info_sameday_filtered_member_proprietary.parquet"
    default_client = Path(output_root) / "metaorders_info_sameday_filtered_member_non_proprietary.parquet"

    prop_path = resolve_repo_path(_REPO_ROOT, args.prop_path or cfg.get("PROP_PATH") or default_prop)
    client_path = resolve_repo_path(_REPO_ROOT, args.client_path or cfg.get("CLIENT_PATH") or default_client)
    out_dir = resolve_repo_path(_REPO_ROOT, Path(output_root) / analysis_tag)
    img_dir = resolve_repo_path(_REPO_ROOT, Path(img_root) / analysis_tag)

    return ResolvedPaths(
        dataset_name=dataset_name,
        prop_path=prop_path,
        client_path=client_path,
        out_dir=out_dir,
        img_dir=img_dir,
        config_path=resolve_repo_path(_REPO_ROOT, args.config_path),
    )


def _parse_high_eta_quantiles(raw_value: object) -> list[float]:
    values: list[float] = []
    if raw_value is None:
        return values
    if isinstance(raw_value, str):
        candidates: Sequence[object] = [token.strip() for token in raw_value.split(",") if token.strip()]
    elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (bytes, bytearray)):
        candidates = raw_value
    else:
        candidates = [raw_value]
    for candidate in candidates:
        if candidate is None:
            continue
        value = float(candidate)
        if not np.isfinite(value):
            continue
        if not 0.0 < value < 1.0:
            raise ValueError("High-eta quantiles must lie in (0, 1).")
        values.append(value)
    if not values:
        return []
    return [float(item) for item in np.unique(np.asarray(values, dtype=float))]


def _resolve_high_eta_quantiles(cfg: Mapping[str, Any], args: argparse.Namespace) -> list[float]:
    cli_quantiles = _parse_high_eta_quantiles(getattr(args, "high_eta_quantiles", None))
    if cli_quantiles:
        return cli_quantiles
    if args.high_eta_quantile is not None:
        return _parse_high_eta_quantiles([args.high_eta_quantile])
    cfg_quantiles = _parse_high_eta_quantiles(cfg.get("HIGH_ETA_QUANTILES"))
    if cfg_quantiles:
        return cfg_quantiles
    return _parse_high_eta_quantiles([cfg.get("HIGH_ETA_QUANTILE", 0.9)])


def _apply_plot_style_from_cfg(cfg: Mapping[str, Any]) -> None:
    _ = cfg
    try:
        apply_shared_plotly_style(load_plot_style())
    except ImportError:
        pass


def _select_same_actor_col(df: pd.DataFrame, policy: str) -> Optional[str]:
    normalized = str(policy).strip().lower()
    if normalized in {"none", "off"}:
        return None
    if normalized == "member":
        return COL_MEMBER if COL_MEMBER in df.columns else None
    if normalized == "client":
        return COL_CLIENT if COL_CLIENT in df.columns else None
    if normalized != "auto":
        raise ValueError(f"Unknown SAME_ACTOR_KEY policy: {policy!r}")
    if COL_MEMBER in df.columns:
        return COL_MEMBER
    if COL_CLIENT in df.columns:
        return COL_CLIENT
    return None


def _safe_bin_label(label: str) -> str:
    return label.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "_")


def _resolve_event_study_worker_count(requested_jobs: int, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    requested = int(requested_jobs)
    if requested < 0:
        raise ValueError("N_JOBS must be >= 0.")
    if requested == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(n_tasks, min(cpu_count, 4)))
    return max(1, min(n_tasks, requested))


def _resolve_threshold_parallel_mode(requested_jobs: int, n_thresholds: int) -> tuple[str, int, int]:
    if n_thresholds <= 1:
        return "replicates", 1, requested_jobs
    threshold_workers = _resolve_event_study_worker_count(requested_jobs, n_thresholds)
    if threshold_workers <= 1:
        return "replicates", 1, requested_jobs
    return "thresholds", threshold_workers, 1


def _process_pool_context():
    if os.name != "posix":
        return None
    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover - platform-specific fallback
        return None


def _init_bootstrap_worker(payload: BootstrapPayload) -> None:
    global _BOOTSTRAP_WORKER_PAYLOAD
    _BOOTSTRAP_WORKER_PAYLOAD = payload


def _init_permutation_worker(payload: PermutationPayload) -> None:
    global _PERMUTATION_WORKER_PAYLOAD
    _PERMUTATION_WORKER_PAYLOAD = payload


def _compute_exposures_minutes(
    start_ns_sorted: np.ndarray,
    session_open_ns: np.int64,
    session_close_ns: np.int64,
    bin_spec: BinSpec,
) -> np.ndarray:
    abs_left = start_ns_sorted[:, None] + bin_spec.left_ns[None, :]
    abs_right = start_ns_sorted[:, None] + bin_spec.right_ns[None, :]
    overlap_start = np.maximum(abs_left, session_open_ns)
    overlap_end = np.minimum(abs_right, session_close_ns)
    exposure_ns = np.clip(overlap_end - overlap_start, 0, None)
    return exposure_ns.astype(float) / float(MINUTE_NS)


def _assign_event_bin_indices(
    delta_ns: np.ndarray,
    *,
    window_ns: int,
    bin_ns: int,
    pre_count: int,
) -> np.ndarray:
    bin_idx = np.empty(delta_ns.size, dtype=np.int64)
    neg_mask = delta_ns < 0
    if np.any(neg_mask):
        bin_idx[neg_mask] = (delta_ns[neg_mask] + window_ns) // bin_ns
    pos_mask = ~neg_mask
    if np.any(pos_mask):
        bin_idx[pos_mask] = pre_count + ((delta_ns[pos_mask] - 1) // bin_ns)
    return bin_idx


def _materialize_variant_metrics_frame(
    base_df: pd.DataFrame,
    *,
    bin_spec: BinSpec,
    same_rates_sorted: np.ndarray,
    opp_rates_sorted: np.ndarray,
    exact_zero_same_sorted: np.ndarray,
    exact_zero_opp_sorted: np.ndarray,
    truncated_pre_sorted: np.ndarray,
    truncated_post_sorted: np.ndarray,
    inverse: np.ndarray,
) -> pd.DataFrame:
    out = base_df.reset_index(drop=True).copy()
    same_rates = same_rates_sorted[inverse]
    opp_rates = opp_rates_sorted[inverse]
    for idx, label in enumerate(bin_spec.labels):
        safe_label = _safe_bin_label(label)
        out[f"same_rate_{safe_label}"] = same_rates[:, idx]
        out[f"opp_rate_{safe_label}"] = opp_rates[:, idx]
    out["same_exact_zero_count"] = exact_zero_same_sorted[inverse]
    out["opp_exact_zero_count"] = exact_zero_opp_sorted[inverse]
    out["truncated_pre_bins"] = truncated_pre_sorted[inverse]
    out["truncated_post_bins"] = truncated_post_sorted[inverse]
    return out


def _count_event_neighbors_python(
    start_ns_sorted: np.ndarray,
    dir_sorted: np.ndarray,
    actor_codes: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    window_ns: int,
    bin_ns: int,
    pre_count: int,
    n_bins: int,
    use_same_actor_exclusion: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = int(start_ns_sorted.shape[0])
    counts_same_all_sorted = np.zeros((n_rows, n_bins), dtype=float)
    counts_opp_all_sorted = np.zeros((n_rows, n_bins), dtype=float)
    exact_zero_same_all_sorted = np.zeros(n_rows, dtype=float)
    exact_zero_opp_all_sorted = np.zeros(n_rows, dtype=float)
    counts_same_excl_sorted = np.zeros((n_rows, n_bins), dtype=float)
    counts_opp_excl_sorted = np.zeros((n_rows, n_bins), dtype=float)
    exact_zero_same_excl_sorted = np.zeros(n_rows, dtype=float)
    exact_zero_opp_excl_sorted = np.zeros(n_rows, dtype=float)

    for row_idx in range(n_rows):
        left = int(left_idx[row_idx])
        right = int(right_idx[row_idx])
        if right - left <= 1:
            continue

        row_start = int(start_ns_sorted[row_idx])
        row_dir = int(dir_sorted[row_idx])
        row_actor = int(actor_codes[row_idx]) if use_same_actor_exclusion else -1

        for other_idx in range(left, right):
            if other_idx == row_idx:
                continue

            delta_ns = int(start_ns_sorted[other_idx]) - row_start
            same_sign = int(dir_sorted[other_idx]) == row_dir

            excluded = False
            if use_same_actor_exclusion and row_actor >= 0:
                other_actor = int(actor_codes[other_idx])
                excluded = other_actor >= 0 and other_actor == row_actor

            if delta_ns == 0:
                if same_sign:
                    exact_zero_same_all_sorted[row_idx] += 1.0
                    if use_same_actor_exclusion and not excluded:
                        exact_zero_same_excl_sorted[row_idx] += 1.0
                else:
                    exact_zero_opp_all_sorted[row_idx] += 1.0
                    if use_same_actor_exclusion and not excluded:
                        exact_zero_opp_excl_sorted[row_idx] += 1.0
                continue

            if delta_ns < 0:
                bin_idx = (delta_ns + window_ns) // bin_ns
            else:
                bin_idx = pre_count + ((delta_ns - 1) // bin_ns)
            if bin_idx < 0 or bin_idx >= n_bins:
                continue

            if same_sign:
                counts_same_all_sorted[row_idx, bin_idx] += 1.0
                if use_same_actor_exclusion and not excluded:
                    counts_same_excl_sorted[row_idx, bin_idx] += 1.0
            else:
                counts_opp_all_sorted[row_idx, bin_idx] += 1.0
                if use_same_actor_exclusion and not excluded:
                    counts_opp_excl_sorted[row_idx, bin_idx] += 1.0

    return (
        counts_same_all_sorted,
        counts_opp_all_sorted,
        exact_zero_same_all_sorted,
        exact_zero_opp_all_sorted,
        counts_same_excl_sorted,
        counts_opp_excl_sorted,
        exact_zero_same_excl_sorted,
        exact_zero_opp_excl_sorted,
    )


if njit is not None:

    @njit(cache=True)
    def _count_event_neighbors_numba(
        start_ns_sorted: np.ndarray,
        dir_sorted: np.ndarray,
        actor_codes: np.ndarray,
        left_idx: np.ndarray,
        right_idx: np.ndarray,
        window_ns: int,
        bin_ns: int,
        pre_count: int,
        n_bins: int,
        use_same_actor_exclusion: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_rows = start_ns_sorted.shape[0]
        counts_same_all_sorted = np.zeros((n_rows, n_bins), dtype=np.float64)
        counts_opp_all_sorted = np.zeros((n_rows, n_bins), dtype=np.float64)
        exact_zero_same_all_sorted = np.zeros(n_rows, dtype=np.float64)
        exact_zero_opp_all_sorted = np.zeros(n_rows, dtype=np.float64)
        counts_same_excl_sorted = np.zeros((n_rows, n_bins), dtype=np.float64)
        counts_opp_excl_sorted = np.zeros((n_rows, n_bins), dtype=np.float64)
        exact_zero_same_excl_sorted = np.zeros(n_rows, dtype=np.float64)
        exact_zero_opp_excl_sorted = np.zeros(n_rows, dtype=np.float64)

        for row_idx in range(n_rows):
            left = left_idx[row_idx]
            right = right_idx[row_idx]
            if right - left <= 1:
                continue

            row_start = start_ns_sorted[row_idx]
            row_dir = dir_sorted[row_idx]
            row_actor = actor_codes[row_idx] if use_same_actor_exclusion else -1

            for other_idx in range(left, right):
                if other_idx == row_idx:
                    continue

                delta_ns = start_ns_sorted[other_idx] - row_start
                same_sign = dir_sorted[other_idx] == row_dir

                excluded = False
                if use_same_actor_exclusion and row_actor >= 0:
                    other_actor = actor_codes[other_idx]
                    excluded = other_actor >= 0 and other_actor == row_actor

                if delta_ns == 0:
                    if same_sign:
                        exact_zero_same_all_sorted[row_idx] += 1.0
                        if use_same_actor_exclusion and not excluded:
                            exact_zero_same_excl_sorted[row_idx] += 1.0
                    else:
                        exact_zero_opp_all_sorted[row_idx] += 1.0
                        if use_same_actor_exclusion and not excluded:
                            exact_zero_opp_excl_sorted[row_idx] += 1.0
                    continue

                if delta_ns < 0:
                    bin_idx = (delta_ns + window_ns) // bin_ns
                else:
                    bin_idx = pre_count + ((delta_ns - 1) // bin_ns)

                if 0 <= bin_idx < n_bins:
                    if same_sign:
                        counts_same_all_sorted[row_idx, bin_idx] += 1.0
                        if use_same_actor_exclusion and not excluded:
                            counts_same_excl_sorted[row_idx, bin_idx] += 1.0
                    else:
                        counts_opp_all_sorted[row_idx, bin_idx] += 1.0
                        if use_same_actor_exclusion and not excluded:
                            counts_opp_excl_sorted[row_idx, bin_idx] += 1.0

        return (
            counts_same_all_sorted,
            counts_opp_all_sorted,
            exact_zero_same_all_sorted,
            exact_zero_opp_all_sorted,
            counts_same_excl_sorted,
            counts_opp_excl_sorted,
            exact_zero_same_excl_sorted,
            exact_zero_opp_excl_sorted,
        )

else:  # pragma: no cover
    _count_event_neighbors_numba = None


def _compute_anchor_metrics_for_group_variants(
    group_df: pd.DataFrame,
    *,
    bin_spec: BinSpec,
    session_start_time: time,
    session_end_time: time,
    same_actor_col: Optional[str],
    compute_exclude_same_actor: bool,
) -> dict[str, pd.DataFrame]:
    required = [COL_ISIN, COL_DATE, COL_START_TS, COL_DIR, COL_ETA]
    missing = [col for col in required if col not in group_df.columns]
    if missing:
        raise KeyError(f"Group frame missing required columns: {missing}")

    base_cols = [COL_ISIN, COL_DATE, COL_START_TS, COL_ETA]
    base_df = group_df[base_cols].reset_index(drop=True).copy()
    n_rows = len(group_df)
    n_bins = len(bin_spec.labels)

    if n_rows == 0:
        empty_rates = np.empty((0, n_bins), dtype=float)
        empty_counts = np.empty(0, dtype=float)
        empty_inverse = np.empty(0, dtype=int)
        results = {
            VARIANT_ALL_OTHERS: _materialize_variant_metrics_frame(
                base_df,
                bin_spec=bin_spec,
                same_rates_sorted=empty_rates,
                opp_rates_sorted=empty_rates,
                exact_zero_same_sorted=empty_counts,
                exact_zero_opp_sorted=empty_counts,
                truncated_pre_sorted=empty_counts,
                truncated_post_sorted=empty_counts,
                inverse=empty_inverse,
            )
        }
        if compute_exclude_same_actor:
            results[VARIANT_EXCLUDE_SAME_ACTOR] = results[VARIANT_ALL_OTHERS].copy()
        return results

    date_value = pd.Timestamp(group_df[COL_DATE].iloc[0]).normalize()
    session_open_ns = np.int64(pd.Timestamp.combine(date_value.date(), session_start_time).value)
    session_close_ns = np.int64(pd.Timestamp.combine(date_value.date(), session_end_time).value)

    order = np.argsort(group_df[COL_START_TS].to_numpy(dtype="datetime64[ns]").astype(np.int64), kind="mergesort")
    inverse = np.empty(n_rows, dtype=int)
    inverse[order] = np.arange(n_rows)

    sorted_df = group_df.iloc[order].reset_index(drop=True)
    start_ns_sorted = sorted_df[COL_START_TS].to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    dir_sorted = pd.to_numeric(sorted_df[COL_DIR], errors="coerce").to_numpy(dtype=np.int8, copy=False)

    use_same_actor_exclusion = bool(
        compute_exclude_same_actor and same_actor_col is not None and same_actor_col in sorted_df.columns
    )
    actor_codes: Optional[np.ndarray] = None
    if use_same_actor_exclusion:
        actor_codes, _ = pd.factorize(sorted_df[same_actor_col], sort=False, use_na_sentinel=True)
        actor_codes = actor_codes.astype(np.int64, copy=False)

    exposures_sorted = _compute_exposures_minutes(
        start_ns_sorted,
        session_open_ns=session_open_ns,
        session_close_ns=session_close_ns,
        bin_spec=bin_spec,
    )
    bin_width_minutes = float(bin_spec.right_minutes[0] - bin_spec.left_minutes[0])
    pre_slice = slice(0, bin_spec.pre_count)
    post_slice = slice(bin_spec.pre_count, len(bin_spec.labels))
    truncated_pre_sorted = np.sum(exposures_sorted[:, pre_slice] < bin_width_minutes, axis=1)
    truncated_post_sorted = np.sum(exposures_sorted[:, post_slice] < bin_width_minutes, axis=1)

    window_ns = int(-bin_spec.left_ns[0])
    bin_ns = int(bin_spec.right_ns[0] - bin_spec.left_ns[0])
    left_idx = np.searchsorted(start_ns_sorted, start_ns_sorted - window_ns, side="left").astype(np.int64, copy=False)
    right_idx = np.searchsorted(start_ns_sorted, start_ns_sorted + window_ns, side="right").astype(np.int64, copy=False)
    actor_codes_array = (
        np.ascontiguousarray(actor_codes, dtype=np.int64)
        if actor_codes is not None
        else np.full(n_rows, -1, dtype=np.int64)
    )
    count_impl = _count_event_neighbors_numba if _count_event_neighbors_numba is not None else _count_event_neighbors_python
    (
        counts_same_all_sorted,
        counts_opp_all_sorted,
        exact_zero_same_all_sorted,
        exact_zero_opp_all_sorted,
        counts_same_excl_sorted,
        counts_opp_excl_sorted,
        exact_zero_same_excl_sorted,
        exact_zero_opp_excl_sorted,
    ) = count_impl(
        np.ascontiguousarray(start_ns_sorted, dtype=np.int64),
        np.ascontiguousarray(dir_sorted, dtype=np.int8),
        actor_codes_array,
        np.ascontiguousarray(left_idx, dtype=np.int64),
        np.ascontiguousarray(right_idx, dtype=np.int64),
        window_ns,
        bin_ns,
        bin_spec.pre_count,
        n_bins,
        use_same_actor_exclusion,
    )

    nan_template = np.full(exposures_sorted.shape, np.nan, dtype=float)
    same_rates_all_sorted = np.divide(
        counts_same_all_sorted,
        exposures_sorted,
        out=nan_template.copy(),
        where=exposures_sorted > 0,
    )
    opp_rates_all_sorted = np.divide(
        counts_opp_all_sorted,
        exposures_sorted,
        out=nan_template.copy(),
        where=exposures_sorted > 0,
    )

    results = {
        VARIANT_ALL_OTHERS: _materialize_variant_metrics_frame(
            base_df,
            bin_spec=bin_spec,
            same_rates_sorted=same_rates_all_sorted,
            opp_rates_sorted=opp_rates_all_sorted,
            exact_zero_same_sorted=exact_zero_same_all_sorted,
            exact_zero_opp_sorted=exact_zero_opp_all_sorted,
            truncated_pre_sorted=truncated_pre_sorted,
            truncated_post_sorted=truncated_post_sorted,
            inverse=inverse,
        )
    }

    if use_same_actor_exclusion:
        same_rates_excl_sorted = np.divide(
            counts_same_excl_sorted,
            exposures_sorted,
            out=nan_template.copy(),
            where=exposures_sorted > 0,
        )
        opp_rates_excl_sorted = np.divide(
            counts_opp_excl_sorted,
            exposures_sorted,
            out=nan_template.copy(),
            where=exposures_sorted > 0,
        )
        results[VARIANT_EXCLUDE_SAME_ACTOR] = _materialize_variant_metrics_frame(
            base_df,
            bin_spec=bin_spec,
            same_rates_sorted=same_rates_excl_sorted,
            opp_rates_sorted=opp_rates_excl_sorted,
            exact_zero_same_sorted=exact_zero_same_excl_sorted if exact_zero_same_excl_sorted is not None else exact_zero_same_all_sorted,
            exact_zero_opp_sorted=exact_zero_opp_excl_sorted if exact_zero_opp_excl_sorted is not None else exact_zero_opp_all_sorted,
            truncated_pre_sorted=truncated_pre_sorted,
            truncated_post_sorted=truncated_post_sorted,
            inverse=inverse,
        )

    return results


def _compute_group_metrics_task(
    *,
    task_idx: int,
    group_df: pd.DataFrame,
    bin_spec: BinSpec,
    session_start_time: time,
    session_end_time: time,
    same_actor_col: Optional[str],
    compute_exclude_same_actor: bool,
) -> tuple[int, dict[str, pd.DataFrame]]:
    return (
        task_idx,
        _compute_anchor_metrics_for_group_variants(
            group_df,
            bin_spec=bin_spec,
            session_start_time=session_start_time,
            session_end_time=session_end_time,
            same_actor_col=same_actor_col,
            compute_exclude_same_actor=compute_exclude_same_actor,
        ),
    )


def _prepare_group_metrics_variants(
    df: pd.DataFrame,
    *,
    label: str,
    bin_spec: BinSpec,
    session_start_time: time,
    session_end_time: time,
    same_actor_col: Optional[str],
    compute_exclude_same_actor: bool,
    n_jobs: int,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    required = [COL_ISIN, COL_DIR, COL_ETA, COL_PERIOD]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"[{label}] Missing required columns: {missing}")

    out = _ensure_date_column(df, label=label)
    out = _ensure_start_timestamp(out, label=label)
    out[COL_DIR] = pd.to_numeric(out[COL_DIR], errors="coerce")
    out[COL_ETA] = pd.to_numeric(out[COL_ETA], errors="coerce")
    out = out.loc[
        out[COL_DIR].isin([-1, 1])
        & out[COL_ETA].notna()
        & np.isfinite(out[COL_ETA].to_numpy(dtype=float))
        & out[COL_START_TS].notna()
        & out[COL_ISIN].notna()
        & out[COL_DATE].notna()
    ].reset_index(drop=True)

    keep_cols = [COL_ISIN, COL_DATE, COL_START_TS, COL_ETA, COL_DIR]
    if same_actor_col is not None and same_actor_col in out.columns and same_actor_col not in keep_cols:
        keep_cols.append(same_actor_col)
    out = out[keep_cols].copy()

    grouped_items = [
        (task_idx, group_key, grp.reset_index(drop=True))
        for task_idx, (group_key, grp) in enumerate(out.groupby([COL_ISIN, COL_DATE], sort=False, dropna=False))
    ]

    base_empty = out[[COL_ISIN, COL_DATE, COL_START_TS, COL_ETA]].iloc[0:0].copy()
    empty_variant_results = _compute_anchor_metrics_for_group_variants(
        base_empty.assign(**{COL_DIR: np.empty(0, dtype=float)}),
        bin_spec=bin_spec,
        session_start_time=session_start_time,
        session_end_time=session_end_time,
        same_actor_col=same_actor_col,
        compute_exclude_same_actor=compute_exclude_same_actor,
    )
    if not grouped_items:
        return empty_variant_results

    worker_count = _resolve_event_study_worker_count(n_jobs, len(grouped_items))
    if worker_count > 1 and show_progress:
        print(f"[{label}] Parallel event-window groups enabled — workers={worker_count}, groups={len(grouped_items)}")

    ordered_results: list[tuple[int, dict[str, pd.DataFrame]]] = []
    with _make_tqdm(
        total=len(grouped_items),
        desc=progress_desc or f"[{label}] event windows",
        disable=not show_progress,
        leave=False,
        unit="group",
    ) as pbar:
        if worker_count <= 1:
            for task_idx, _, grp in grouped_items:
                ordered_results.append(
                    _compute_group_metrics_task(
                        task_idx=task_idx,
                        group_df=grp,
                        bin_spec=bin_spec,
                        session_start_time=session_start_time,
                        session_end_time=session_end_time,
                        same_actor_col=same_actor_col,
                        compute_exclude_same_actor=compute_exclude_same_actor,
                    )
                )
                pbar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_group = {
                    executor.submit(
                        _compute_group_metrics_task,
                        task_idx=task_idx,
                        group_df=grp,
                        bin_spec=bin_spec,
                        session_start_time=session_start_time,
                        session_end_time=session_end_time,
                        same_actor_col=same_actor_col,
                        compute_exclude_same_actor=compute_exclude_same_actor,
                    ): group_key
                    for task_idx, group_key, grp in grouped_items
                }
                for future in as_completed(future_to_group):
                    group_key = future_to_group[future]
                    try:
                        ordered_results.append(future.result())
                    except Exception as exc:  # pragma: no cover - defensive path
                        raise RuntimeError(f"[{label}] Failed event-window computation for group {group_key!r}") from exc
                    pbar.update(1)

    ordered_results.sort(key=lambda item: item[0])
    variant_chunks: dict[str, list[pd.DataFrame]] = {VARIANT_ALL_OTHERS: []}
    if compute_exclude_same_actor:
        variant_chunks[VARIANT_EXCLUDE_SAME_ACTOR] = []

    for _, result_by_variant in ordered_results:
        variant_chunks[VARIANT_ALL_OTHERS].append(result_by_variant[VARIANT_ALL_OTHERS])
        if compute_exclude_same_actor and VARIANT_EXCLUDE_SAME_ACTOR in result_by_variant:
            variant_chunks[VARIANT_EXCLUDE_SAME_ACTOR].append(result_by_variant[VARIANT_EXCLUDE_SAME_ACTOR])

    final_results = {
        VARIANT_ALL_OTHERS: pd.concat(variant_chunks[VARIANT_ALL_OTHERS], ignore_index=True)
        if variant_chunks[VARIANT_ALL_OTHERS]
        else empty_variant_results[VARIANT_ALL_OTHERS].copy()
    }
    if compute_exclude_same_actor and variant_chunks.get(VARIANT_EXCLUDE_SAME_ACTOR):
        final_results[VARIANT_EXCLUDE_SAME_ACTOR] = pd.concat(
            variant_chunks[VARIANT_EXCLUDE_SAME_ACTOR],
            ignore_index=True,
        )
    return final_results


def _prepare_matching_frame(
    metrics_df: pd.DataFrame,
    *,
    matching_bucket_minutes: int,
) -> pd.DataFrame:
    if matching_bucket_minutes <= 0:
        raise ValueError("MATCHING_BUCKET_MINUTES must be positive.")
    out = metrics_df.copy()
    minute_of_day = out[COL_START_TS].dt.hour * 60 + out[COL_START_TS].dt.minute
    bucket = (minute_of_day // int(matching_bucket_minutes)).astype(int)
    out["clock_bucket_match"] = bucket
    out["stratum_key"] = (
        out[COL_ISIN].astype(str)
        + "|"
        + out[COL_DATE].dt.strftime("%Y-%m-%d")
        + "|"
        + bucket.astype(str)
    )
    return out


def _annotate_high_eta_and_strata(
    prepared_matching_df: pd.DataFrame,
    *,
    high_eta_quantile: float,
) -> tuple[pd.DataFrame, float]:
    if not 0.0 < high_eta_quantile < 1.0:
        raise ValueError("HIGH_ETA_QUANTILE must be in (0, 1).")
    if "stratum_key" not in prepared_matching_df.columns:
        raise KeyError("Prepared matching frame must include 'stratum_key'.")
    out = prepared_matching_df.copy()
    eta_values = pd.to_numeric(out[COL_ETA], errors="coerce").to_numpy(dtype=float)
    eta_threshold = float(np.nanquantile(eta_values, high_eta_quantile))
    out["high_eta"] = eta_values >= eta_threshold
    return out, eta_threshold


def _rate_columns(bin_spec: BinSpec, prefix: str) -> list[str]:
    cols: list[str] = []
    for label in bin_spec.labels:
        safe_label = _safe_bin_label(label)
        cols.append(f"{prefix}_{safe_label}")
    return cols


def _build_bootstrap_payload(
    metrics_df: pd.DataFrame,
    *,
    curve_metric_cols: Sequence[str],
) -> BootstrapPayload:
    n_metrics = len(curve_metric_cols)
    if metrics_df.empty:
        empty_vec = np.empty(0, dtype=float)
        empty_mat = np.empty((0, n_metrics), dtype=float)
        return BootstrapPayload(
            date_codes=np.empty(0, dtype=np.int64),
            n_dates=0,
            n_treated=empty_vec,
            n_control=empty_vec,
            treated_sums=empty_mat,
            control_sums=empty_mat,
            treated_valid=empty_mat,
            control_valid=empty_mat,
        )

    all_metric_cols = list(curve_metric_cols)
    tmp = metrics_df[[COL_DATE, "stratum_key", "high_eta"] + all_metric_cols].copy()
    tmp["high_eta"] = tmp["high_eta"].astype(int)

    group_cols = [COL_DATE, "stratum_key"]
    grp = tmp.groupby(group_cols, sort=False, dropna=False)
    summary = grp.size().rename("n_total").to_frame()
    summary["n_treated"] = grp["high_eta"].sum()
    summary["n_control"] = summary["n_total"] - summary["n_treated"]

    summary_index = summary.index
    treated_frame = tmp.loc[tmp["high_eta"].eq(1), group_cols + all_metric_cols]
    control_frame = tmp.loc[tmp["high_eta"].eq(0), group_cols + all_metric_cols]

    treated_grp = treated_frame.groupby(group_cols, sort=False, dropna=False)
    control_grp = control_frame.groupby(group_cols, sort=False, dropna=False)

    treated_sums = treated_grp[all_metric_cols].sum(min_count=1).reindex(summary_index).fillna(0.0)
    control_sums = control_grp[all_metric_cols].sum(min_count=1).reindex(summary_index).fillna(0.0)
    treated_counts = treated_grp[all_metric_cols].count().reindex(summary_index).fillna(0.0)
    control_counts = control_grp[all_metric_cols].count().reindex(summary_index).fillna(0.0)

    date_index = summary_index.get_level_values(COL_DATE)
    unique_dates = pd.Index(pd.unique(date_index))
    date_codes = unique_dates.get_indexer(date_index)
    return BootstrapPayload(
        date_codes=np.ascontiguousarray(date_codes, dtype=np.int64),
        n_dates=int(len(unique_dates)),
        n_treated=np.ascontiguousarray(summary["n_treated"].to_numpy(dtype=float)),
        n_control=np.ascontiguousarray(summary["n_control"].to_numpy(dtype=float)),
        treated_sums=np.ascontiguousarray(treated_sums.to_numpy(dtype=float)),
        control_sums=np.ascontiguousarray(control_sums.to_numpy(dtype=float)),
        treated_valid=np.ascontiguousarray(treated_counts.to_numpy(dtype=float)),
        control_valid=np.ascontiguousarray(control_counts.to_numpy(dtype=float)),
    )


def _weighted_stat_from_bootstrap_payload(
    payload: BootstrapPayload,
    *,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_metrics = int(payload.treated_sums.shape[1]) if payload.treated_sums.ndim == 2 else 0
    nan_vec = np.full(n_metrics, np.nan, dtype=float)
    if payload.n_treated.size == 0 or n_metrics == 0:
        return nan_vec, nan_vec, nan_vec, 0.0

    valid_rows = (payload.n_treated > 0) & (payload.n_control > 0)
    if not np.any(valid_rows):
        return nan_vec, nan_vec, nan_vec, 0.0

    if weights is None:
        row_weight = np.ones(int(np.sum(valid_rows)), dtype=float)
    else:
        row_weight = np.asarray(weights, dtype=float)[valid_rows]

    n_treated = payload.n_treated[valid_rows]
    total_treated = float(np.sum(row_weight * n_treated))
    if total_treated <= 0:
        return nan_vec, nan_vec, nan_vec, 0.0

    treated_sums = payload.treated_sums[valid_rows]
    control_sums = payload.control_sums[valid_rows]
    treated_valid = payload.treated_valid[valid_rows]
    control_valid = payload.control_valid[valid_rows]

    metric_valid = (treated_valid > 0) & (control_valid > 0)
    metric_row_weight = row_weight[:, None] * metric_valid
    metric_treated_weight = metric_row_weight * treated_valid
    total_metric_treated = np.sum(metric_treated_weight, axis=0, dtype=float)

    treated_mean = np.full(n_metrics, np.nan, dtype=float)
    control_mean = np.full(n_metrics, np.nan, dtype=float)
    excess_mean = np.full(n_metrics, np.nan, dtype=float)
    if not np.any(total_metric_treated > 0):
        return treated_mean, control_mean, excess_mean, total_treated

    control_mean_by_stratum = np.divide(
        control_sums,
        control_valid,
        out=np.zeros_like(control_sums, dtype=float),
        where=control_valid > 0,
    )
    treated_num = np.sum(metric_row_weight * treated_sums, axis=0, dtype=float)
    control_num = np.sum(metric_treated_weight * control_mean_by_stratum, axis=0, dtype=float)
    valid_metric = total_metric_treated > 0
    treated_mean[valid_metric] = treated_num[valid_metric] / total_metric_treated[valid_metric]
    control_mean[valid_metric] = control_num[valid_metric] / total_metric_treated[valid_metric]
    excess_mean[valid_metric] = treated_mean[valid_metric] - control_mean[valid_metric]
    return treated_mean, control_mean, excess_mean, total_treated


def _metric_support_from_bootstrap_payload(
    payload: BootstrapPayload,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin-level effective treated/control counts and valid-stratum counts."""
    n_metrics = int(payload.treated_sums.shape[1]) if payload.treated_sums.ndim == 2 else 0
    zeros = np.zeros(n_metrics, dtype=float)
    if payload.n_treated.size == 0 or n_metrics == 0:
        return zeros.copy(), zeros.copy(), zeros.copy()

    valid_rows = (payload.n_treated > 0) & (payload.n_control > 0)
    if not np.any(valid_rows):
        return zeros.copy(), zeros.copy(), zeros.copy()

    treated_valid = payload.treated_valid[valid_rows]
    control_valid = payload.control_valid[valid_rows]
    metric_valid = (treated_valid > 0) & (control_valid > 0)
    return (
        np.sum(treated_valid * metric_valid, axis=0, dtype=float),
        np.sum(control_valid * metric_valid, axis=0, dtype=float),
        np.sum(metric_valid, axis=0, dtype=float),
    )


def _split_replicate_batches(total_runs: int, worker_count: int, *, batches_per_worker: int = 8) -> list[int]:
    if total_runs <= 0:
        return []
    n_batches = min(total_runs, max(1, worker_count * batches_per_worker))
    base = total_runs // n_batches
    remainder = total_runs % n_batches
    return [base + (1 if idx < remainder else 0) for idx in range(n_batches) if base + (1 if idx < remainder else 0) > 0]


def _run_parallel_replicate_batches(
    *,
    total_runs: int,
    n_cols: int,
    requested_jobs: int,
    seed: int,
    show_progress: bool,
    progress_desc: Optional[str],
    phase_label: str,
    serial_worker_fn,
    process_worker_fn=None,
    process_initializer=None,
    process_initargs: tuple[Any, ...] = (),
) -> np.ndarray:
    if total_runs <= 0:
        return np.empty((0, n_cols), dtype=float)

    worker_count = _resolve_event_study_worker_count(requested_jobs, total_runs)
    base_rng = np.random.default_rng(seed)
    replicate_seeds = base_rng.integers(0, np.iinfo(np.uint64).max, size=total_runs, dtype=np.uint64)
    if worker_count <= 1 or total_runs == 1:
        return serial_worker_fn(replicate_seeds)

    batch_sizes = _split_replicate_batches(total_runs, worker_count)
    seed_batches: list[np.ndarray] = []
    cursor = 0
    for batch_size in batch_sizes:
        seed_batches.append(replicate_seeds[cursor : cursor + batch_size].copy())
        cursor += batch_size

    process_ctx = _process_pool_context() if process_worker_fn is not None else None
    use_process_pool = process_worker_fn is not None
    if use_process_pool and process_ctx is None:
        use_process_pool = False
        if show_progress:
            print(f"[{phase_label}] Process-based batches unavailable on this platform; falling back to threads.")
    if show_progress:
        executor_label = "process" if use_process_pool else "thread"
        print(
            f"[{phase_label}] Parallel replicate batches enabled — workers={worker_count}, "
            f"batches={len(batch_sizes)}, executor={executor_label}"
        )

    ordered_results: list[Optional[np.ndarray]] = [None] * len(batch_sizes)
    with _make_tqdm(
        total=total_runs,
        desc=progress_desc or phase_label,
        disable=not show_progress,
        leave=False,
        unit="rep",
    ) as pbar:
        if use_process_pool:
            executor_kwargs: dict[str, Any] = {"max_workers": worker_count}
            if process_ctx is not None:
                executor_kwargs["mp_context"] = process_ctx
            if process_initializer is not None:
                executor_kwargs["initializer"] = process_initializer
                executor_kwargs["initargs"] = process_initargs
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                future_to_idx = {
                    executor.submit(process_worker_fn, seed_batch): batch_idx
                    for batch_idx, seed_batch in enumerate(seed_batches)
                }
                for future in as_completed(future_to_idx):
                    batch_idx = future_to_idx[future]
                    ordered_results[batch_idx] = future.result()
                    pbar.update(batch_sizes[batch_idx])
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_idx = {
                    executor.submit(serial_worker_fn, seed_batch): batch_idx
                    for batch_idx, seed_batch in enumerate(seed_batches)
                }
                for future in as_completed(future_to_idx):
                    batch_idx = future_to_idx[future]
                    ordered_results[batch_idx] = future.result()
                    pbar.update(batch_sizes[batch_idx])

    materialized = [result for result in ordered_results if result is not None]
    if not materialized:
        return np.empty((0, n_cols), dtype=float)
    return np.vstack(materialized)


def _bootstrap_batch(
    payload: BootstrapPayload,
    *,
    replicate_seeds: np.ndarray,
) -> np.ndarray:
    n_runs = int(len(replicate_seeds))
    n_metrics = int(payload.treated_sums.shape[1]) if payload.treated_sums.ndim == 2 else 0
    if n_runs <= 0 or payload.n_dates <= 0 or n_metrics == 0:
        return np.empty((0, n_metrics), dtype=float)

    reps = np.full((n_runs, n_metrics), np.nan, dtype=float)
    for run_idx, rep_seed in enumerate(replicate_seeds):
        rng = np.random.default_rng(int(rep_seed))
        sampled = rng.integers(0, payload.n_dates, size=payload.n_dates)
        date_weights = np.bincount(sampled, minlength=payload.n_dates).astype(float, copy=False)
        row_weights = date_weights[payload.date_codes]
        _, _, excess, _ = _weighted_stat_from_bootstrap_payload(payload, weights=row_weights)
        reps[run_idx, :] = excess
    return reps


def _bootstrap_batch_from_worker_payload(replicate_seeds: np.ndarray) -> np.ndarray:
    if _BOOTSTRAP_WORKER_PAYLOAD is None:  # pragma: no cover - defensive path
        raise RuntimeError("Bootstrap worker payload is not initialized.")
    return _bootstrap_batch(_BOOTSTRAP_WORKER_PAYLOAD, replicate_seeds=replicate_seeds)


def _bootstrap_stats_by_date(
    payload: BootstrapPayload,
    *,
    n_runs: int,
    seed: int,
    n_jobs: int,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    n_metrics = int(payload.treated_sums.shape[1]) if payload.treated_sums.ndim == 2 else 0
    return _run_parallel_replicate_batches(
        total_runs=n_runs,
        n_cols=n_metrics,
        requested_jobs=n_jobs,
        seed=seed,
        show_progress=show_progress,
        progress_desc=progress_desc,
        phase_label="bootstrap",
        serial_worker_fn=lambda seed_batch: _bootstrap_batch(payload, replicate_seeds=seed_batch),
        process_worker_fn=_bootstrap_batch_from_worker_payload,
        process_initializer=_init_bootstrap_worker,
        process_initargs=(payload,),
    )


def _ci_from_reps(reps: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if reps.size == 0:
        n_cols = int(reps.shape[1]) if reps.ndim == 2 else 0
        return np.full(n_cols, np.nan, dtype=float), np.full(n_cols, np.nan, dtype=float)
    lo = np.nanquantile(reps, alpha / 2.0, axis=0)
    hi = np.nanquantile(reps, 1.0 - alpha / 2.0, axis=0)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _build_permutation_payload(
    metrics_df: pd.DataFrame,
    *,
    metric_cols: Sequence[str],
) -> PermutationPayload:
    n_metrics = len(metric_cols)
    if metrics_df.empty:
        return PermutationPayload(
            values_zero_by_stratum=tuple(),
            finite_mask_by_stratum=tuple(),
            treated_mask_by_stratum=tuple(),
            n_treated_by_stratum=np.empty(0, dtype=np.int64),
            n_metrics=n_metrics,
        )

    values_zero_by_stratum: list[np.ndarray] = []
    finite_mask_by_stratum: list[np.ndarray] = []
    treated_mask_by_stratum: list[np.ndarray] = []
    n_treated_by_stratum: list[int] = []
    metric_names = list(metric_cols)

    for _, frame in metrics_df.groupby("stratum_key", sort=False):
        values = np.ascontiguousarray(frame[metric_names].to_numpy(dtype=float))
        treated_mask = np.ascontiguousarray(frame["high_eta"].to_numpy(dtype=bool))
        n_total = int(values.shape[0])
        n_treated = int(np.sum(treated_mask))
        if n_treated <= 0 or n_treated >= n_total:
            continue
        finite_mask = np.ascontiguousarray(np.isfinite(values))
        values_zero = np.ascontiguousarray(np.where(finite_mask, values, 0.0))
        values_zero_by_stratum.append(values_zero)
        finite_mask_by_stratum.append(finite_mask)
        treated_mask_by_stratum.append(treated_mask)
        n_treated_by_stratum.append(n_treated)

    return PermutationPayload(
        values_zero_by_stratum=tuple(values_zero_by_stratum),
        finite_mask_by_stratum=tuple(finite_mask_by_stratum),
        treated_mask_by_stratum=tuple(treated_mask_by_stratum),
        n_treated_by_stratum=np.asarray(n_treated_by_stratum, dtype=np.int64),
        n_metrics=n_metrics,
    )


def _accumulate_effect_from_treated_mask(
    values_zero: np.ndarray,
    finite_mask: np.ndarray,
    treated_mask: np.ndarray,
    *,
    diff_sum: np.ndarray,
    total_treated_valid: np.ndarray,
) -> None:
    treated_mask_2d = treated_mask[:, None]
    control_mask_2d = ~treated_mask_2d
    n_treated_valid = np.sum(finite_mask & treated_mask_2d, axis=0, dtype=float)
    n_control_valid = np.sum(finite_mask & control_mask_2d, axis=0, dtype=float)
    metric_valid = (n_treated_valid > 0) & (n_control_valid > 0)
    if not np.any(metric_valid):
        return

    treated_sum = np.sum(values_zero * treated_mask_2d, axis=0, dtype=float)
    control_sum = np.sum(values_zero * control_mask_2d, axis=0, dtype=float)
    control_mean = np.divide(control_sum, n_control_valid, out=np.zeros_like(control_sum), where=n_control_valid > 0)
    diff_sum[metric_valid] += treated_sum[metric_valid] - n_treated_valid[metric_valid] * control_mean[metric_valid]
    total_treated_valid[metric_valid] += n_treated_valid[metric_valid]


def _summary_effect_from_permutation_payload(
    payload: PermutationPayload,
    *,
    treated_masks: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    out = np.full(payload.n_metrics, np.nan, dtype=float)
    if payload.n_metrics <= 0 or not payload.values_zero_by_stratum:
        return out

    total_treated_valid = np.zeros(payload.n_metrics, dtype=float)
    diff_sum = np.zeros(payload.n_metrics, dtype=float)
    for stratum_idx, values_zero in enumerate(payload.values_zero_by_stratum):
        treated_mask = (
            payload.treated_mask_by_stratum[stratum_idx]
            if treated_masks is None
            else np.asarray(treated_masks[stratum_idx], dtype=bool)
        )
        _accumulate_effect_from_treated_mask(
            values_zero,
            payload.finite_mask_by_stratum[stratum_idx],
            treated_mask,
            diff_sum=diff_sum,
            total_treated_valid=total_treated_valid,
        )

    valid = total_treated_valid > 0
    if np.any(valid):
        out[valid] = diff_sum[valid] / total_treated_valid[valid]
    return out


def _permutation_batch(
    payload: PermutationPayload,
    *,
    replicate_seeds: np.ndarray,
) -> np.ndarray:
    n_runs = int(len(replicate_seeds))
    if n_runs <= 0 or payload.n_metrics <= 0 or not payload.values_zero_by_stratum:
        return np.empty((0, payload.n_metrics), dtype=float)

    reps = np.full((n_runs, payload.n_metrics), np.nan, dtype=float)
    for run_idx, rep_seed in enumerate(replicate_seeds):
        rng = np.random.default_rng(int(rep_seed))
        total_treated_valid = np.zeros(payload.n_metrics, dtype=float)
        diff_sum = np.zeros(payload.n_metrics, dtype=float)
        for stratum_idx, values_zero in enumerate(payload.values_zero_by_stratum):
            n_total = int(values_zero.shape[0])
            n_treated = int(payload.n_treated_by_stratum[stratum_idx])
            treated_mask = np.zeros(n_total, dtype=bool)
            treated_idx = rng.choice(n_total, size=n_treated, replace=False)
            treated_mask[treated_idx] = True
            _accumulate_effect_from_treated_mask(
                values_zero,
                payload.finite_mask_by_stratum[stratum_idx],
                treated_mask,
                diff_sum=diff_sum,
                total_treated_valid=total_treated_valid,
            )
        valid = total_treated_valid > 0
        if np.any(valid):
            reps[run_idx, valid] = diff_sum[valid] / total_treated_valid[valid]
    return reps


def _permutation_batch_from_worker_payload(replicate_seeds: np.ndarray) -> np.ndarray:
    if _PERMUTATION_WORKER_PAYLOAD is None:  # pragma: no cover - defensive path
        raise RuntimeError("Permutation worker payload is not initialized.")
    return _permutation_batch(_PERMUTATION_WORKER_PAYLOAD, replicate_seeds=replicate_seeds)


def _permutation_effect_stats(
    payload: PermutationPayload,
    *,
    n_runs: int,
    seed: int,
    n_jobs: int,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    observed = _summary_effect_from_permutation_payload(payload)
    permuted = _run_parallel_replicate_batches(
        total_runs=n_runs,
        n_cols=payload.n_metrics,
        requested_jobs=n_jobs,
        seed=seed,
        show_progress=show_progress,
        progress_desc=progress_desc,
        phase_label="permutation",
        serial_worker_fn=lambda seed_batch: _permutation_batch(payload, replicate_seeds=seed_batch),
        process_worker_fn=_permutation_batch_from_worker_payload,
        process_initializer=_init_permutation_worker,
        process_initargs=(payload,),
    )
    return observed, permuted


def _raw_p_value(observed: float, draws: np.ndarray, *, alternative: str) -> float:
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or not np.isfinite(observed):
        return float("nan")
    if alternative == "greater":
        return float((1.0 + np.sum(finite >= observed)) / (finite.size + 1.0))
    if alternative == "less":
        return float((1.0 + np.sum(finite <= observed)) / (finite.size + 1.0))
    return float((1.0 + np.sum(np.abs(finite) >= abs(observed))) / (finite.size + 1.0))


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return out
    order = finite_idx[np.argsort(p[finite_idx])]
    m = len(order)
    running = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        adjusted = min((m / (m - rank + 1)) * p[idx], 1.0)
        running = min(running, adjusted)
        out[idx] = running
    return out


def _build_curve_rows(
    *,
    group_name: str,
    variant: str,
    high_eta_quantile: float,
    eta_threshold: float,
    n_high_eta: int,
    bin_spec: BinSpec,
    curve_metric_cols: Sequence[str],
    observed_treated_curve: np.ndarray,
    observed_control_curve: np.ndarray,
    observed_excess_curve: np.ndarray,
    curve_ci_lo: np.ndarray,
    curve_ci_hi: np.ndarray,
    perm_observed: np.ndarray,
    perm_draws: np.ndarray,
    total_treated_matched: int,
    treated_effective_counts: np.ndarray,
    control_effective_counts: np.ndarray,
    valid_strata_counts: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, metric_name in enumerate(curve_metric_cols):
        sign_relation = "same_sign" if metric_name.startswith("same_rate_") else "opposite_sign"
        bin_idx = idx % len(bin_spec.labels)
        window_side = "pre" if float(bin_spec.right_minutes[bin_idx]) <= 0.0 else "post"
        raw_p = (
            _raw_p_value(perm_observed[idx], perm_draws[:, idx], alternative="two-sided")
            if perm_draws.size
            else float("nan")
        )
        rows.append(
            {
                "group": group_name,
                "variant": variant,
                "high_eta_quantile": float(high_eta_quantile),
                "eta_threshold": float(eta_threshold),
                "n_high_eta": int(n_high_eta),
                "metric_name": metric_name,
                "sign_relation": sign_relation,
                "window_side": window_side,
                "test_alternative": "two-sided",
                "bin_index": idx if sign_relation == "same_sign" else idx - len(bin_spec.labels),
                "bin_label": bin_spec.labels[bin_idx],
                "bin_left_min": float(bin_spec.left_minutes[bin_idx]),
                "bin_right_min": float(bin_spec.right_minutes[bin_idx]),
                "bin_center_min": float(bin_spec.centers_minutes[bin_idx]),
                "treated_rate": observed_treated_curve[idx],
                "control_rate": observed_control_curve[idx],
                "excess_rate": observed_excess_curve[idx],
                "ci_excess_lo": curve_ci_lo[idx],
                "ci_excess_hi": curve_ci_hi[idx],
                "p_raw": raw_p,
                "n_treated_matched": total_treated_matched,
                "n_treated_effective": treated_effective_counts[idx],
                "n_control_effective": control_effective_counts[idx],
                "n_valid_strata": valid_strata_counts[idx],
            }
        )
    return pd.DataFrame(rows)


def _build_diagnostics_rows(
    *,
    metrics_df: pd.DataFrame,
    group_name: str,
    variant: str,
    high_eta_quantile: float,
    eta_threshold: float,
    same_actor_col: Optional[str],
    total_treated_matched: int,
) -> pd.DataFrame:
    high_eta = metrics_df["high_eta"].to_numpy(dtype=bool)
    rows = [
        {
            "group": group_name,
            "variant": variant,
            "high_eta_quantile": float(high_eta_quantile),
            "eta_threshold": float(eta_threshold),
            "metric": "high_eta_quantile",
            "value": float(high_eta_quantile),
        },
        {
            "group": group_name,
            "variant": variant,
            "high_eta_quantile": float(high_eta_quantile),
            "eta_threshold": float(eta_threshold),
            "metric": "eta_threshold",
            "value": eta_threshold,
        },
        {
            "group": group_name,
            "variant": variant,
            "high_eta_quantile": float(high_eta_quantile),
            "eta_threshold": float(eta_threshold),
            "metric": "n_anchors",
            "value": float(len(metrics_df)),
        },
        {
            "group": group_name,
            "variant": variant,
            "high_eta_quantile": float(high_eta_quantile),
            "eta_threshold": float(eta_threshold),
            "metric": "n_high_eta",
            "value": float(np.sum(high_eta)),
        },
        {
            "group": group_name,
            "variant": variant,
            "high_eta_quantile": float(high_eta_quantile),
            "eta_threshold": float(eta_threshold),
            "metric": "mean_same_exact_zero_count_high_eta",
            "value": float(np.nanmean(metrics_df.loc[high_eta, "same_exact_zero_count"].to_numpy(dtype=float))),
        },
        {
            "group": group_name,
            "variant": variant,
            "high_eta_quantile": float(high_eta_quantile),
            "eta_threshold": float(eta_threshold),
            "metric": "mean_opp_exact_zero_count_high_eta",
            "value": float(np.nanmean(metrics_df.loc[high_eta, "opp_exact_zero_count"].to_numpy(dtype=float))),
        },
        {
            "group": group_name,
            "variant": variant,
            "high_eta_quantile": float(high_eta_quantile),
            "eta_threshold": float(eta_threshold),
            "metric": "mean_truncated_pre_bins_high_eta",
            "value": float(np.nanmean(metrics_df.loc[high_eta, "truncated_pre_bins"].to_numpy(dtype=float))),
        },
        {
            "group": group_name,
            "variant": variant,
            "high_eta_quantile": float(high_eta_quantile),
            "eta_threshold": float(eta_threshold),
            "metric": "mean_truncated_post_bins_high_eta",
            "value": float(np.nanmean(metrics_df.loc[high_eta, "truncated_post_bins"].to_numpy(dtype=float))),
        },
        {
            "group": group_name,
            "variant": variant,
            "high_eta_quantile": float(high_eta_quantile),
            "eta_threshold": float(eta_threshold),
            "metric": "n_treated_matched",
            "value": float(total_treated_matched),
        },
        {
            "group": group_name,
            "variant": variant,
            "high_eta_quantile": float(high_eta_quantile),
            "eta_threshold": float(eta_threshold),
            "metric": "same_actor_col",
            "value": same_actor_col or "",
        },
    ]
    return pd.DataFrame(rows)


def _plot_group_variant_curves(
    curve_df: pd.DataFrame,
    *,
    group_name: str,
    variant: str,
    dirs: PlotOutputDirs,
    alpha: Optional[float] = None,
) -> None:
    if curve_df.empty:
        return

    group_label = "Proprietary" if group_name == "prop" else "Client"
    group_color = COLOR_PROPRIETARY if group_name == "prop" else COLOR_CLIENT
    control_color = COLOR_NEUTRAL
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Same-sign starts", "Opposite-sign starts"],
        shared_yaxes=True,
    )

    for col_idx, sign_relation in enumerate(["same_sign", "opposite_sign"], start=1):
        sub = curve_df.loc[curve_df["sign_relation"].eq(sign_relation)].sort_values("bin_center_min")
        x = sub["bin_center_min"].to_numpy(dtype=float)
        y_treated = sub["treated_rate"].to_numpy(dtype=float)
        y_control = sub["control_rate"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_treated,
                mode="lines+markers",
                line=dict(color=group_color, width=2),
                name=f"{group_label} high-eta",
                showlegend=(col_idx == 1),
                hovertemplate="tau=%{x:.1f} min<br>rate=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_control,
                mode="lines+markers",
                line=dict(color=control_color, width=2, dash="dash"),
                name="Matched controls",
                showlegend=(col_idx == 1),
                hovertemplate="tau=%{x:.1f} min<br>rate=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )
        if alpha is not None and "p_adjusted" in sub.columns:
            sig_mask = sub["p_adjusted"].lt(alpha).fillna(False).to_numpy(dtype=bool)
            if np.any(sig_mask):
                y_panel = np.concatenate([y_treated, y_control])
                y_panel = y_panel[np.isfinite(y_panel)]
                if y_panel.size:
                    y_top = float(np.max(y_panel))
                    y_bottom = float(np.min(y_panel))
                    y_span = y_top - y_bottom
                    pad = 0.08 * y_span if y_span > 0 else max(0.01, 0.08 * max(abs(y_top), 1.0))
                    y_sig = np.full(np.sum(sig_mask), y_top + pad, dtype=float)
                    fig.add_trace(
                        go.Scatter(
                            x=x[sig_mask],
                            y=y_sig,
                            mode="markers",
                            marker=dict(symbol="star", size=11, color="#111827"),
                            name=f"Adj. perm. p < {alpha:.2g}",
                            showlegend=(col_idx == 1),
                            hovertemplate="tau=%{x:.1f} min<br>adj. perm. p < alpha<extra></extra>",
                        ),
                        row=1,
                        col=col_idx,
                    )
        fig.add_vline(x=0.0, line=dict(color="#9CA3AF", dash="dot"), row=1, col=col_idx)
        fig.update_xaxes(title_text="Event time (minutes)", row=1, col=col_idx)
    fig.update_yaxes(title_text="Start intensity (per minute)", row=1, col=1)
    fig.update_layout(
        title=f"{group_label} start event-study ({variant})",
        width=1100,
        height=480,
    )
    stem = f"event_curve_{group_name}_{variant}"
    save_plotly_figure(fig, stem=stem, dirs=dirs, write_html=True, write_png=True)


def _build_threshold_heatmap(
    curve_df: pd.DataFrame,
    *,
    value_col: str,
) -> tuple[list[str], np.ndarray, np.ndarray, list[list[str]]]:
    ordered = curve_df.sort_values(["high_eta_quantile", "bin_center_min"]).copy()
    bin_table = ordered[["bin_center_min", "bin_label"]].drop_duplicates().sort_values("bin_center_min")
    x_labels = bin_table["bin_label"].astype(str).tolist()
    quantiles = np.sort(ordered["high_eta_quantile"].dropna().unique().astype(float))
    z = np.full((len(quantiles), len(x_labels)), np.nan, dtype=float)
    hover_text = [["" for _ in x_labels] for _ in quantiles]
    lookup = {
        (float(np.round(row["high_eta_quantile"], 10)), str(row["bin_label"])): row
        for _, row in ordered.iterrows()
    }
    for row_idx, quantile in enumerate(quantiles):
        quantile_key = float(np.round(quantile, 10))
        for col_idx, bin_label in enumerate(x_labels):
            row = lookup.get((quantile_key, bin_label))
            if row is None:
                continue
            value = row.get(value_col)
            z[row_idx, col_idx] = float(value) if pd.notna(value) else np.nan
            hover_parts = [
                f"quantile={quantile:.2f}",
                f"eta threshold={float(row['eta_threshold']):.4g}",
                f"bin={bin_label}",
                f"excess={float(row['excess_rate']):.4g}",
            ]
            if pd.notna(row.get("p_adjusted")):
                hover_parts.append(f"adj. p={float(row['p_adjusted']):.4g}")
            hover_text[row_idx][col_idx] += (
                f"<br>treated eff={float(row['n_treated_effective']):.0f}"
                f"<br>control eff={float(row['n_control_effective']):.0f}"
                f"<br>valid strata={float(row['n_valid_strata']):.0f}"
            )
            hover_text[row_idx][col_idx] = "<br>".join(hover_parts) + hover_text[row_idx][col_idx]
            if value_col == "p_adjusted" and pd.notna(value):
                hover_text[row_idx][col_idx] += f"<br>adj. p value={float(value):.4g}"
            elif value_col == "n_valid_strata" and pd.notna(value):
                hover_text[row_idx][col_idx] += f"<br>support={float(value):.0f}"
            elif pd.notna(value):
                hover_text[row_idx][col_idx] += f"<br>heatmap value={float(value):.4g}"
    return x_labels, quantiles, z, hover_text


def _plot_threshold_sweep_heatmap(
    curve_df: pd.DataFrame,
    *,
    group_name: str,
    variant: str,
    sign_relation: str,
    value_col: str,
    value_label: str,
    stem_prefix: str,
    colorscale: str,
    dirs: PlotOutputDirs,
    alpha: Optional[float] = None,
    zmid: Optional[float] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    add_significance_overlay: bool = False,
) -> None:
    sub = curve_df.loc[curve_df["sign_relation"].eq(sign_relation)].copy()
    if sub.empty or sub["high_eta_quantile"].nunique() <= 1:
        return
    x_labels, quantiles, z, hover_text = _build_threshold_heatmap(sub, value_col=value_col)
    if z.size == 0 or not np.isfinite(z).any():
        return
    heatmap_kwargs: dict[str, Any] = {
        "z": z,
        "x": x_labels,
        "y": quantiles,
        "text": hover_text,
        "hovertemplate": "%{text}<extra></extra>",
        "colorscale": colorscale,
        "colorbar": dict(title=value_label),
    }
    if zmid is not None:
        heatmap_kwargs["zmid"] = zmid
    if zmin is not None:
        heatmap_kwargs["zmin"] = zmin
    if zmax is not None:
        heatmap_kwargs["zmax"] = zmax
    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))
    if add_significance_overlay and alpha is not None and "p_adjusted" in sub.columns:
        sig = sub.loc[sub["p_adjusted"].lt(alpha).fillna(False)].copy()
        if not sig.empty:
            fig.add_trace(
                go.Scatter(
                    x=sig["bin_label"],
                    y=sig["high_eta_quantile"],
                    mode="markers",
                    marker=dict(symbol="star", size=10, color="#111827"),
                    name=f"Adj. p < {alpha:.2g}",
                    hovertemplate="bin=%{x}<br>quantile=%{y:.2f}<br>adj. p < alpha<extra></extra>",
                )
            )
    group_label = "Proprietary" if group_name == "prop" else "Client"
    sign_label = "Same-sign" if sign_relation == "same_sign" else "Opposite-sign"
    fig.update_layout(
        title=f"{group_label} threshold sweep ({variant}, {sign_label.lower()}, {value_label.lower()})",
        xaxis_title="Event-time bin",
        yaxis_title="High-eta quantile",
        width=900,
        height=520,
    )
    fig.update_xaxes(type="category", tickangle=-35)
    stem = f"{stem_prefix}_{group_name}_{variant}_{sign_relation}"
    save_plotly_figure(fig, stem=stem, dirs=dirs, write_html=True, write_png=True)


def _plot_group_variant_threshold_sweep(
    curve_df: pd.DataFrame,
    *,
    group_name: str,
    variant: str,
    dirs: PlotOutputDirs,
    alpha: Optional[float] = None,
) -> None:
    if curve_df.empty or curve_df["high_eta_quantile"].nunique() <= 1:
        return
    for sign_relation in ("same_sign", "opposite_sign"):
        sub = curve_df.loc[curve_df["sign_relation"].eq(sign_relation)].copy()
        if sub.empty:
            continue
        finite_effect = sub["excess_rate"].to_numpy(dtype=float)
        finite_effect = finite_effect[np.isfinite(finite_effect)]
        effect_bound = float(np.nanmax(np.abs(finite_effect))) if finite_effect.size else 0.0
        if effect_bound <= 0.0:
            effect_bound = 1e-9
        _plot_threshold_sweep_heatmap(
            curve_df,
            group_name=group_name,
            variant=variant,
            sign_relation=sign_relation,
            value_col="excess_rate",
            value_label="Excess rate",
            stem_prefix="event_heatmap_effect",
            colorscale="RdBu",
            dirs=dirs,
            alpha=alpha,
            zmid=0.0,
            zmin=-effect_bound,
            zmax=effect_bound,
            add_significance_overlay=True,
        )
        _plot_threshold_sweep_heatmap(
            curve_df,
            group_name=group_name,
            variant=variant,
            sign_relation=sign_relation,
            value_col="p_adjusted",
            value_label="Adj. p",
            stem_prefix="event_heatmap_padj",
            colorscale="Viridis_r",
            dirs=dirs,
            zmin=0.0,
            zmax=1.0,
        )
        finite_support = sub["n_valid_strata"].to_numpy(dtype=float)
        finite_support = finite_support[np.isfinite(finite_support)]
        support_max = float(np.nanmax(finite_support)) if finite_support.size else None
        _plot_threshold_sweep_heatmap(
            curve_df,
            group_name=group_name,
            variant=variant,
            sign_relation=sign_relation,
            value_col="n_valid_strata",
            value_label="Valid strata",
            stem_prefix="event_heatmap_support",
            colorscale="Blues",
            dirs=dirs,
            zmin=0.0,
            zmax=support_max,
        )


def _run_group_variant_at_threshold(
    prepared_matching_df: pd.DataFrame,
    *,
    group_name: str,
    variant: str,
    bin_spec: BinSpec,
    high_eta_quantile: float,
    alpha: float,
    bootstrap_runs: int,
    permutation_runs: int,
    seed: int,
    n_jobs: int,
    same_actor_col: Optional[str],
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_annotated, eta_threshold = _annotate_high_eta_and_strata(
        prepared_matching_df,
        high_eta_quantile=high_eta_quantile,
    )
    n_high_eta = int(np.sum(metrics_annotated["high_eta"].to_numpy(dtype=bool)))

    same_curve_cols = _rate_columns(bin_spec, "same_rate")
    opp_curve_cols = _rate_columns(bin_spec, "opp_rate")
    curve_metric_cols = same_curve_cols + opp_curve_cols

    bootstrap_payload = _build_bootstrap_payload(
        metrics_annotated,
        curve_metric_cols=curve_metric_cols,
    )
    permutation_payload = _build_permutation_payload(
        metrics_annotated,
        metric_cols=curve_metric_cols,
    )

    treated_curve, control_curve, excess_curve, total_treated_matched = _weighted_stat_from_bootstrap_payload(
        bootstrap_payload,
    )
    treated_effective_counts, control_effective_counts, valid_strata_counts = _metric_support_from_bootstrap_payload(
        bootstrap_payload,
    )

    curve_reps = _bootstrap_stats_by_date(
        bootstrap_payload,
        n_runs=bootstrap_runs,
        seed=seed,
        n_jobs=n_jobs,
        show_progress=show_progress,
        progress_desc=f"[{group_name}:{variant}] bootstrap",
    )
    curve_ci_lo, curve_ci_hi = _ci_from_reps(curve_reps, alpha)

    curve_perm_observed, curve_perm_draws = _permutation_effect_stats(
        permutation_payload,
        n_runs=permutation_runs,
        seed=seed + 101,
        n_jobs=n_jobs,
        show_progress=show_progress,
        progress_desc=f"[{group_name}:{variant}] permutation",
    )
    curve_df = _build_curve_rows(
        group_name=group_name,
        variant=variant,
        high_eta_quantile=high_eta_quantile,
        eta_threshold=eta_threshold,
        n_high_eta=n_high_eta,
        bin_spec=bin_spec,
        curve_metric_cols=curve_metric_cols,
        observed_treated_curve=treated_curve,
        observed_control_curve=control_curve,
        observed_excess_curve=excess_curve,
        curve_ci_lo=curve_ci_lo,
        curve_ci_hi=curve_ci_hi,
        perm_observed=curve_perm_observed,
        perm_draws=curve_perm_draws,
        total_treated_matched=int(total_treated_matched),
        treated_effective_counts=treated_effective_counts,
        control_effective_counts=control_effective_counts,
        valid_strata_counts=valid_strata_counts,
    )
    curve_df = _apply_curve_pvalue_adjustments(curve_df)
    diagnostics_df = _build_diagnostics_rows(
        metrics_df=metrics_annotated,
        group_name=group_name,
        variant=variant,
        high_eta_quantile=high_eta_quantile,
        eta_threshold=eta_threshold,
        same_actor_col=same_actor_col,
        total_treated_matched=int(total_treated_matched),
    )
    return curve_df, diagnostics_df


def _run_group_variant(
    metrics_df: pd.DataFrame,
    *,
    group_name: str,
    variant: str,
    bin_spec: BinSpec,
    high_eta_quantile: float,
    matching_bucket_minutes: int,
    alpha: float,
    bootstrap_runs: int,
    permutation_runs: int,
    seed: int,
    n_jobs: int,
    same_actor_col: Optional[str],
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_matching_df = _prepare_matching_frame(
        metrics_df,
        matching_bucket_minutes=matching_bucket_minutes,
    )
    return _run_group_variant_at_threshold(
        prepared_matching_df,
        group_name=group_name,
        variant=variant,
        bin_spec=bin_spec,
        high_eta_quantile=high_eta_quantile,
        alpha=alpha,
        bootstrap_runs=bootstrap_runs,
        permutation_runs=permutation_runs,
        seed=seed,
        n_jobs=n_jobs,
        same_actor_col=same_actor_col,
        show_progress=show_progress,
    )


def _run_group_variant_threshold_grid(
    metrics_df: pd.DataFrame,
    *,
    group_name: str,
    variant: str,
    bin_spec: BinSpec,
    high_eta_quantiles: Sequence[float],
    matching_bucket_minutes: int,
    alpha: float,
    bootstrap_runs: int,
    permutation_runs: int,
    seed: int,
    n_jobs: int,
    same_actor_col: Optional[str],
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quantiles = [float(q) for q in high_eta_quantiles]
    if not quantiles:
        raise ValueError("At least one high-eta quantile is required.")
    prepared_matching_df = _prepare_matching_frame(
        metrics_df,
        matching_bucket_minutes=matching_bucket_minutes,
    )
    if len(quantiles) == 1:
        return _run_group_variant_at_threshold(
            prepared_matching_df,
            group_name=group_name,
            variant=variant,
            bin_spec=bin_spec,
            high_eta_quantile=quantiles[0],
            alpha=alpha,
            bootstrap_runs=bootstrap_runs,
            permutation_runs=permutation_runs,
            seed=seed,
            n_jobs=n_jobs,
            same_actor_col=same_actor_col,
            show_progress=show_progress,
        )

    parallel_mode, threshold_workers, inner_n_jobs = _resolve_threshold_parallel_mode(n_jobs, len(quantiles))
    if show_progress:
        print(
            f"[{group_name}:{variant}] Threshold sweep enabled — quantiles={len(quantiles)}, "
            f"parallel_mode={parallel_mode}, workers={threshold_workers}"
        )

    ordered_curves: list[Optional[pd.DataFrame]] = [None] * len(quantiles)
    ordered_diagnostics: list[Optional[pd.DataFrame]] = [None] * len(quantiles)
    with _make_tqdm(
        total=len(quantiles),
        desc=f"[{group_name}:{variant}] thresholds",
        disable=not show_progress,
        leave=False,
        unit="thr",
    ) as pbar:
        if parallel_mode == "thresholds" and threshold_workers > 1:
            with ThreadPoolExecutor(max_workers=threshold_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        _run_group_variant_at_threshold,
                        prepared_matching_df,
                        group_name=group_name,
                        variant=variant,
                        bin_spec=bin_spec,
                        high_eta_quantile=quantile,
                        alpha=alpha,
                        bootstrap_runs=bootstrap_runs,
                        permutation_runs=permutation_runs,
                        seed=seed + 10_000 * idx,
                        n_jobs=inner_n_jobs,
                        same_actor_col=same_actor_col,
                        show_progress=False,
                    ): idx
                    for idx, quantile in enumerate(quantiles)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    curve_df, diagnostics_df = future.result()
                    ordered_curves[idx] = curve_df
                    ordered_diagnostics[idx] = diagnostics_df
                    pbar.update(1)
        else:
            for idx, quantile in enumerate(quantiles):
                curve_df, diagnostics_df = _run_group_variant_at_threshold(
                    prepared_matching_df,
                    group_name=group_name,
                    variant=variant,
                    bin_spec=bin_spec,
                    high_eta_quantile=quantile,
                    alpha=alpha,
                    bootstrap_runs=bootstrap_runs,
                    permutation_runs=permutation_runs,
                    seed=seed + 10_000 * idx,
                    n_jobs=inner_n_jobs,
                    same_actor_col=same_actor_col,
                    show_progress=show_progress,
                )
                ordered_curves[idx] = curve_df
                ordered_diagnostics[idx] = diagnostics_df
                pbar.update(1)

    curve_chunks = [chunk for chunk in ordered_curves if chunk is not None]
    diagnostics_chunks = [chunk for chunk in ordered_diagnostics if chunk is not None]
    if not curve_chunks or not diagnostics_chunks:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(curve_chunks, ignore_index=True), pd.concat(diagnostics_chunks, ignore_index=True)


def _apply_curve_pvalue_adjustments(curve_df: pd.DataFrame) -> pd.DataFrame:
    out = curve_df.copy()
    out["p_adjusted"] = np.nan
    out["p_adjustment_method"] = ""
    if out.empty or "p_raw" not in out.columns:
        return out

    group_cols = ["group", "variant", "sign_relation"]
    if "high_eta_quantile" in out.columns:
        group_cols.insert(2, "high_eta_quantile")
    grouped = out.groupby(group_cols, sort=False, dropna=False)
    for _, idx in grouped.groups.items():
        indexer = list(idx)
        adjusted = _bh_adjust(out.loc[indexer, "p_raw"].to_numpy(dtype=float))
        out.loc[indexer, "p_adjusted"] = adjusted
        out.loc[indexer, "p_adjustment_method"] = "bh_curve_by_sign"
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Summary
    -------
    Build the CLI parser for the metaorder start event-study workflow.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI parser.

    Notes
    -----
    Defaults are read from `config_ymls/metaorder_start_event_study.yml`.

    Examples
    --------
    >>> parser = build_arg_parser()
    >>> isinstance(parser, argparse.ArgumentParser)
    True
    """
    parser = argparse.ArgumentParser(description="Event-study of metaorder starts around high-participation anchors.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="config_ymls/metaorder_start_event_study.yml",
        help="YAML config path. Default: config_ymls/metaorder_start_event_study.yml.",
    )
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name. Default: YAML DATASET_NAME.")
    parser.add_argument("--prop-path", type=str, default=None, help="Override proprietary filtered parquet path.")
    parser.add_argument("--client-path", type=str, default=None, help="Override client filtered parquet path.")
    parser.add_argument("--output-file-path", type=str, default=None, help="Override output root.")
    parser.add_argument("--img-output-path", type=str, default=None, help="Override image root.")
    parser.add_argument("--analysis-tag", type=str, default=None, help="Output subfolder name.")
    parser.add_argument("--event-window-minutes", type=int, default=None, help="Symmetric event-study horizon in minutes.")
    parser.add_argument("--bin-minutes", type=int, default=None, help="Bin width in minutes.")
    parser.add_argument("--matching-bucket-minutes", type=int, default=None, help="Clock-time matching bucket width.")
    parser.add_argument("--high-eta-quantile", type=float, default=None, help="High-eta quantile within group.")
    parser.add_argument(
        "--high-eta-quantiles",
        type=str,
        default=None,
        help="Comma-separated high-eta quantile grid (for threshold sweeps). Overrides --high-eta-quantile.",
    )
    parser.add_argument("--bootstrap-runs", type=int, default=None, help="Date-cluster bootstrap replicates.")
    parser.add_argument("--permutation-runs", type=int, default=None, help="Within-stratum permutation replicates.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel worker count for event-window and resampling phases. Use 0 for auto.",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Confidence level alpha.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--same-actor-key", type=str, default=None, help='Actor key for robustness: auto|member|client|none.')
    parser.add_argument(
        "--run-same-actor-robustness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to rerun excluding same-actor neighbors.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write Plotly HTML/PNG figures.",
    )
    parser.add_argument(
        "--write-parquet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write parquet copies of large result tables.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show tqdm progress bars. Default: enabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved configuration without writing outputs.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Summary
    -------
    Run the start-intensity event-study workflow end-to-end.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        CLI arguments. When None, uses `sys.argv[1:]`.

    Returns
    -------
    int
        Process exit code (`0` on success).

    Notes
    -----
    The workflow:
    1. loads the filtered proprietary and client metaorder tables,
    2. computes per-anchor event-time start rates on the same ISIN and day,
    3. compares high-eta anchors to matched controls within `(ISIN, Date, clock bucket)`,
    4. reports bootstrap confidence intervals and within-stratum permutation p-values for each event bin,
    5. writes curve tables, diagnostics, figures, and a reproducibility manifest.

    Examples
    --------
    >>> # main([\"--dry-run\"])  # doctest: +SKIP
    """
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = _load_yaml_defaults(resolve_repo_path(_REPO_ROOT, args.config_path))
    _apply_plot_style_from_cfg(cfg)

    paths = _resolve_paths(cfg, args)

    event_window_minutes = int(args.event_window_minutes if args.event_window_minutes is not None else cfg.get("EVENT_WINDOW_MINUTES", 20))
    bin_minutes = int(args.bin_minutes if args.bin_minutes is not None else cfg.get("BIN_MINUTES", 5))
    matching_bucket_minutes = int(
        args.matching_bucket_minutes if args.matching_bucket_minutes is not None else cfg.get("MATCHING_BUCKET_MINUTES", 30)
    )
    high_eta_quantiles = _resolve_high_eta_quantiles(cfg, args)
    threshold_sweep_mode = len(high_eta_quantiles) > 1
    high_eta_quantile = float(high_eta_quantiles[0])
    bootstrap_runs = int(args.bootstrap_runs if args.bootstrap_runs is not None else cfg.get("BOOTSTRAP_RUNS", 1000))
    permutation_runs = int(
        args.permutation_runs if args.permutation_runs is not None else cfg.get("PERMUTATION_RUNS", 1000)
    )
    n_jobs = int(args.n_jobs if args.n_jobs is not None else cfg.get("N_JOBS", 0))
    alpha = float(args.alpha if args.alpha is not None else cfg.get("ALPHA", 0.05))
    seed = int(args.seed if args.seed is not None else cfg.get("SEED", 0))
    same_actor_key = str(args.same_actor_key or cfg.get("SAME_ACTOR_KEY", "auto"))
    run_same_actor_robustness = bool(
        args.run_same_actor_robustness
        if args.run_same_actor_robustness is not None
        else cfg.get("RUN_SAME_ACTOR_ROBUSTNESS", True)
    )
    plots_enabled = bool(args.plots if args.plots is not None else cfg.get("PLOTS", True))
    write_parquet = bool(args.write_parquet if args.write_parquet is not None else cfg.get("WRITE_PARQUET", False))
    show_progress = bool(args.progress if args.progress is not None else cfg.get("SHOW_PROGRESS", True))

    trading_hours_raw = cfg.get("TRADING_HOURS", ["09:30:00", "17:30:00"])
    if not isinstance(trading_hours_raw, Sequence) or len(trading_hours_raw) != 2:
        raise ValueError("TRADING_HOURS must be a 2-element sequence of start/end strings.")
    session_start_time = _parse_time_string(trading_hours_raw[0], label="TRADING_HOURS[0]")
    session_end_time = _parse_time_string(trading_hours_raw[1], label="TRADING_HOURS[1]")
    if session_start_time >= session_end_time:
        raise ValueError("TRADING_HOURS must satisfy start < end.")

    if matching_bucket_minutes <= 0 or 60 % np.gcd(60, matching_bucket_minutes) < 0:
        raise ValueError("MATCHING_BUCKET_MINUTES must be positive.")
    bin_spec = _compute_bin_spec(event_window_minutes, bin_minutes)
    threshold_parallel_mode, threshold_workers, threshold_inner_jobs = _resolve_threshold_parallel_mode(
        n_jobs,
        len(high_eta_quantiles),
    )

    manifest = {
        "run_timestamp": dt.datetime.now().isoformat(),
        "git_hash": _try_git_hash(),
        "dataset_name": paths.dataset_name,
        "prop_path": str(paths.prop_path),
        "client_path": str(paths.client_path),
        "out_dir": str(paths.out_dir),
        "img_dir": str(paths.img_dir),
        "config_path": str(paths.config_path),
        "event_window_minutes": event_window_minutes,
        "bin_minutes": bin_minutes,
        "matching_bucket_minutes": matching_bucket_minutes,
        "high_eta_quantile": high_eta_quantile,
        "high_eta_quantiles": [float(q) for q in high_eta_quantiles],
        "threshold_mode": "sweep" if threshold_sweep_mode else "single",
        "bootstrap_runs": bootstrap_runs,
        "permutation_runs": permutation_runs,
        "n_jobs": n_jobs,
        "threshold_parallel_mode": threshold_parallel_mode,
        "threshold_workers": threshold_workers,
        "threshold_inner_n_jobs": threshold_inner_jobs,
        "alpha": alpha,
        "seed": seed,
        "same_actor_key": same_actor_key,
        "run_same_actor_robustness": run_same_actor_robustness,
        "plots_enabled": plots_enabled,
        "write_parquet": write_parquet,
        "show_progress": show_progress,
        "trading_hours": [str(trading_hours_raw[0]), str(trading_hours_raw[1])],
        "numba_enabled": bool(njit is not None),
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    img_dirs = make_plot_output_dirs(paths.img_dir, use_subdirs=True)
    img_dirs.base_dir.mkdir(parents=True, exist_ok=True)

    if plots_enabled:
        img_dirs.html_dir.mkdir(parents=True, exist_ok=True)
        img_dirs.png_dir.mkdir(parents=True, exist_ok=True)
    _write_json(paths.out_dir / "run_manifest.json", manifest)

    load_cols = [COL_ISIN, COL_DATE, COL_PERIOD, COL_DIR, COL_ETA, COL_MEMBER, COL_CLIENT]
    estimated_total_steps = 1 + 2 * 2 + 1
    if run_same_actor_robustness and str(same_actor_key).strip().lower() not in {"none", "off"}:
        estimated_total_steps += 2

    with _make_tqdm(
        total=estimated_total_steps,
        desc="Event-study overall",
        disable=not show_progress,
        leave=True,
        unit="phase",
    ) as overall_pbar:
        overall_pbar.set_postfix_str("load inputs")
        prop_raw = _read_parquet_with_fallback(paths.prop_path, columns=load_cols)
        client_raw = _read_parquet_with_fallback(paths.client_path, columns=load_cols)
        overall_pbar.update(1)

        all_curves: list[pd.DataFrame] = []
        all_diagnostics: list[pd.DataFrame] = []

        group_specs = [
            ("prop", prop_raw, COLOR_PROPRIETARY),
            ("client", client_raw, COLOR_CLIENT),
        ]

        for group_idx, (group_name, raw_df, _) in enumerate(group_specs):
            same_actor_col = _select_same_actor_col(raw_df, same_actor_key)
            compute_exclude_same_actor = bool(run_same_actor_robustness and same_actor_col is not None)
            overall_pbar.set_postfix_str(f"{group_name}: build anchor windows")
            metrics_by_variant = _prepare_group_metrics_variants(
                raw_df,
                label=group_name,
                bin_spec=bin_spec,
                session_start_time=session_start_time,
                session_end_time=session_end_time,
                same_actor_col=same_actor_col,
                compute_exclude_same_actor=compute_exclude_same_actor,
                n_jobs=n_jobs,
                show_progress=show_progress,
                progress_desc=f"[{group_name}] event windows",
            )
            overall_pbar.update(1)

            overall_pbar.set_postfix_str(f"{group_name}: matched analysis")
            curve_df, diagnostics_df = _run_group_variant_threshold_grid(
                metrics_by_variant[VARIANT_ALL_OTHERS],
                group_name=group_name,
                variant=VARIANT_ALL_OTHERS,
                bin_spec=bin_spec,
                high_eta_quantiles=high_eta_quantiles,
                matching_bucket_minutes=matching_bucket_minutes,
                alpha=alpha,
                bootstrap_runs=bootstrap_runs,
                permutation_runs=permutation_runs,
                seed=seed + group_idx * 1000,
                n_jobs=n_jobs,
                same_actor_col=same_actor_col,
                show_progress=show_progress,
            )
            all_curves.append(curve_df)
            all_diagnostics.append(diagnostics_df)
            if plots_enabled:
                if threshold_sweep_mode:
                    _plot_group_variant_threshold_sweep(
                        curve_df,
                        group_name=group_name,
                        variant=VARIANT_ALL_OTHERS,
                        dirs=img_dirs,
                        alpha=alpha,
                    )
                else:
                    _plot_group_variant_curves(
                        curve_df,
                        group_name=group_name,
                        variant=VARIANT_ALL_OTHERS,
                        dirs=img_dirs,
                        alpha=alpha,
                    )
            overall_pbar.update(1)

            if compute_exclude_same_actor and VARIANT_EXCLUDE_SAME_ACTOR in metrics_by_variant:
                overall_pbar.set_postfix_str(f"{group_name}: robustness analysis")
                curve_excl, diagnostics_excl = _run_group_variant_threshold_grid(
                    metrics_by_variant[VARIANT_EXCLUDE_SAME_ACTOR],
                    group_name=group_name,
                    variant=VARIANT_EXCLUDE_SAME_ACTOR,
                    bin_spec=bin_spec,
                    high_eta_quantiles=high_eta_quantiles,
                    matching_bucket_minutes=matching_bucket_minutes,
                    alpha=alpha,
                    bootstrap_runs=bootstrap_runs,
                    permutation_runs=permutation_runs,
                    seed=seed + 500 + group_idx * 1000,
                    n_jobs=n_jobs,
                    same_actor_col=same_actor_col,
                    show_progress=show_progress,
                )
                all_curves.append(curve_excl)
                all_diagnostics.append(diagnostics_excl)
                if plots_enabled:
                    if threshold_sweep_mode:
                        _plot_group_variant_threshold_sweep(
                            curve_excl,
                            group_name=group_name,
                            variant=VARIANT_EXCLUDE_SAME_ACTOR,
                            dirs=img_dirs,
                            alpha=alpha,
                        )
                    else:
                        _plot_group_variant_curves(
                            curve_excl,
                            group_name=group_name,
                            variant=VARIANT_EXCLUDE_SAME_ACTOR,
                            dirs=img_dirs,
                            alpha=alpha,
                        )
                overall_pbar.update(1)

        overall_pbar.set_postfix_str("write outputs")
        curves_all = pd.concat(all_curves, ignore_index=True)
        diagnostics_all = pd.concat(all_diagnostics, ignore_index=True)
        curves_stem = "event_study_curves_threshold_sweep" if threshold_sweep_mode else "event_study_curves"
        diagnostics_stem = (
            "event_study_diagnostics_threshold_sweep" if threshold_sweep_mode else "event_study_diagnostics"
        )
        curves_all.to_csv(paths.out_dir / f"{curves_stem}.csv", index=False)
        diagnostics_all.to_csv(paths.out_dir / f"{diagnostics_stem}.csv", index=False)
        if write_parquet:
            curves_all.to_parquet(paths.out_dir / f"{curves_stem}.parquet", index=False)
            diagnostics_all.to_parquet(paths.out_dir / f"{diagnostics_stem}.parquet", index=False)
        overall_pbar.update(1)

    print(f"[event-study] Wrote tables to {paths.out_dir}")
    if plots_enabled:
        print(f"[event-study] Wrote figures to {img_dirs.base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
