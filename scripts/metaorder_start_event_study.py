#!/usr/bin/env python3
"""
Event-study of metaorder starts around high-participation anchors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
    THEME_BG_COLOR,
    THEME_COLORWAY,
    THEME_FONT_FAMILY,
    THEME_GRID_COLOR,
    apply_plotly_style,
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
COL_BUCKET = "clock_bucket_30m"
MINUTE_NS = np.int64(60 * 1_000_000_000)

SUMMARY_METRIC_SPECS = (
    ("same_pre_mean_rate", "same_sign", "pre", "greater"),
    ("same_post_mean_rate", "same_sign", "post", "greater"),
    ("opp_pre_mean_rate", "opposite_sign", "pre", "two-sided"),
    ("opp_post_mean_rate", "opposite_sign", "post", "two-sided"),
)


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


def _apply_plot_style_from_cfg(cfg: Mapping[str, Any]) -> None:
    tick_font_size = int(cfg.get("TICK_FONT_SIZE", 12))
    label_font_size = int(cfg.get("LABEL_FONT_SIZE", 14))
    title_font_size = int(cfg.get("TITLE_FONT_SIZE", 15))
    legend_font_size = int(cfg.get("LEGEND_FONT_SIZE", 12))
    try:
        apply_plotly_style(
            tick_font_size=tick_font_size,
            label_font_size=label_font_size,
            title_font_size=title_font_size,
            legend_font_size=legend_font_size,
            theme_colorway=THEME_COLORWAY,
            theme_grid_color=THEME_GRID_COLOR,
            theme_bg_color=THEME_BG_COLOR,
            theme_font_family=THEME_FONT_FAMILY,
        )
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


def _compute_anchor_metrics_for_group(
    group_df: pd.DataFrame,
    *,
    bin_spec: BinSpec,
    session_start_time: time,
    session_end_time: time,
    same_actor_col: Optional[str],
    exclude_same_actor: bool,
) -> pd.DataFrame:
    n_rows = len(group_df)
    if n_rows == 0:
        return group_df.copy()

    date_value = pd.Timestamp(group_df[COL_DATE].iloc[0]).normalize()
    session_open_ns = np.int64(pd.Timestamp.combine(date_value.date(), session_start_time).value)
    session_close_ns = np.int64(pd.Timestamp.combine(date_value.date(), session_end_time).value)

    order = np.argsort(group_df[COL_START_TS].to_numpy(dtype="datetime64[ns]").astype(np.int64), kind="mergesort")
    inverse = np.empty(n_rows, dtype=int)
    inverse[order] = np.arange(n_rows)

    sorted_df = group_df.iloc[order].reset_index(drop=True)
    start_ns_sorted = sorted_df[COL_START_TS].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    dir_sorted = pd.to_numeric(sorted_df[COL_DIR], errors="coerce").to_numpy(dtype=float)

    tau_ns = start_ns_sorted[None, :] - start_ns_sorted[:, None]
    offdiag = ~np.eye(n_rows, dtype=bool)
    same_sign = dir_sorted[None, :] == dir_sorted[:, None]
    opposite_sign = ~same_sign
    valid = offdiag.copy()

    if exclude_same_actor and same_actor_col is not None and same_actor_col in sorted_df.columns:
        actor = sorted_df[same_actor_col].to_numpy(dtype=object)
        actor_valid = pd.notna(actor)
        same_actor = (actor[None, :] == actor[:, None]) & actor_valid[None, :] & actor_valid[:, None]
        valid &= ~same_actor

    exact_zero = tau_ns == 0
    exact_zero_same = np.sum(valid & exact_zero & same_sign, axis=1).astype(float)
    exact_zero_opp = np.sum(valid & exact_zero & opposite_sign, axis=1).astype(float)

    exposures_sorted = _compute_exposures_minutes(
        start_ns_sorted,
        session_open_ns=session_open_ns,
        session_close_ns=session_close_ns,
        bin_spec=bin_spec,
    )
    counts_same_sorted = np.zeros((n_rows, len(bin_spec.labels)), dtype=float)
    counts_opp_sorted = np.zeros((n_rows, len(bin_spec.labels)), dtype=float)

    for idx, (left_ns, right_ns) in enumerate(zip(bin_spec.left_ns, bin_spec.right_ns)):
        if right_ns <= 0:
            in_bin = (tau_ns >= left_ns) & (tau_ns < right_ns)
        else:
            in_bin = (tau_ns > left_ns) & (tau_ns <= right_ns)
        counts_same_sorted[:, idx] = np.sum(valid & same_sign & in_bin, axis=1, dtype=float)
        counts_opp_sorted[:, idx] = np.sum(valid & opposite_sign & in_bin, axis=1, dtype=float)

    nan_template = np.full(exposures_sorted.shape, np.nan, dtype=float)
    same_rates_sorted = np.divide(
        counts_same_sorted,
        exposures_sorted,
        out=nan_template.copy(),
        where=exposures_sorted > 0,
    )
    opp_rates_sorted = np.divide(
        counts_opp_sorted,
        exposures_sorted,
        out=nan_template.copy(),
        where=exposures_sorted > 0,
    )

    counts_same = counts_same_sorted[inverse]
    counts_opp = counts_opp_sorted[inverse]
    exposures = exposures_sorted[inverse]
    same_rates = same_rates_sorted[inverse]
    opp_rates = opp_rates_sorted[inverse]
    exact_zero_same = exact_zero_same[inverse]
    exact_zero_opp = exact_zero_opp[inverse]

    out = group_df.reset_index(drop=True).copy()
    pre_slice = slice(0, bin_spec.pre_count)
    post_slice = slice(bin_spec.pre_count, len(bin_spec.labels))

    for idx, label in enumerate(bin_spec.labels):
        safe_label = label.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "_")
        out[f"same_count_{safe_label}"] = counts_same[:, idx]
        out[f"opp_count_{safe_label}"] = counts_opp[:, idx]
        out[f"exposure_{safe_label}"] = exposures[:, idx]
        out[f"same_rate_{safe_label}"] = same_rates[:, idx]
        out[f"opp_rate_{safe_label}"] = opp_rates[:, idx]

    out["same_pre_mean_rate"] = np.nanmean(same_rates[:, pre_slice], axis=1)
    out["same_post_mean_rate"] = np.nanmean(same_rates[:, post_slice], axis=1)
    out["opp_pre_mean_rate"] = np.nanmean(opp_rates[:, pre_slice], axis=1)
    out["opp_post_mean_rate"] = np.nanmean(opp_rates[:, post_slice], axis=1)
    out["same_exact_zero_count"] = exact_zero_same
    out["opp_exact_zero_count"] = exact_zero_opp
    out["truncated_pre_bins"] = np.sum(exposures[:, pre_slice] < (bin_spec.right_minutes[0] - bin_spec.left_minutes[0]), axis=1)
    out["truncated_post_bins"] = np.sum(exposures[:, post_slice] < (bin_spec.right_minutes[0] - bin_spec.left_minutes[0]), axis=1)
    return out


def _prepare_group_metrics(
    df: pd.DataFrame,
    *,
    label: str,
    bin_spec: BinSpec,
    session_start_time: time,
    session_end_time: time,
    same_actor_col: Optional[str],
    exclude_same_actor: bool,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> pd.DataFrame:
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

    minute_of_day = out[COL_START_TS].dt.hour * 60 + out[COL_START_TS].dt.minute
    out[COL_BUCKET] = (minute_of_day // 30).astype(int)

    chunks: list[pd.DataFrame] = []
    grouped = out.groupby([COL_ISIN, COL_DATE], sort=False, dropna=False)
    with _make_tqdm(
        total=int(grouped.ngroups),
        desc=progress_desc or f"[{label}] event windows",
        disable=not show_progress,
        leave=False,
        unit="group",
    ) as pbar:
        for _, grp in grouped:
            chunks.append(
                _compute_anchor_metrics_for_group(
                    grp,
                    bin_spec=bin_spec,
                    session_start_time=session_start_time,
                    session_end_time=session_end_time,
                    same_actor_col=same_actor_col,
                    exclude_same_actor=exclude_same_actor,
                )
            )
            pbar.update(1)
    if not chunks:
        return out.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)


def _annotate_high_eta_and_strata(
    metrics_df: pd.DataFrame,
    *,
    high_eta_quantile: float,
    matching_bucket_minutes: int,
) -> tuple[pd.DataFrame, float]:
    if not 0.0 < high_eta_quantile < 1.0:
        raise ValueError("HIGH_ETA_QUANTILE must be in (0, 1).")
    if matching_bucket_minutes <= 0:
        raise ValueError("MATCHING_BUCKET_MINUTES must be positive.")

    out = metrics_df.copy()
    eta_threshold = float(np.nanquantile(out[COL_ETA].to_numpy(dtype=float), high_eta_quantile))
    out["high_eta"] = out[COL_ETA].to_numpy(dtype=float) >= eta_threshold

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
    return out, eta_threshold


def _rate_columns(bin_spec: BinSpec, prefix: str) -> list[str]:
    cols: list[str] = []
    for label in bin_spec.labels:
        safe_label = label.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "_")
        cols.append(f"{prefix}_{safe_label}")
    return cols


def _build_stratum_summaries(
    metrics_df: pd.DataFrame,
    *,
    curve_metric_cols: Sequence[str],
    summary_metric_cols: Sequence[str],
) -> pd.DataFrame:
    all_metric_cols = list(curve_metric_cols) + list(summary_metric_cols)
    tmp = metrics_df[[COL_DATE, "stratum_key", "high_eta"] + all_metric_cols].copy()
    tmp["high_eta"] = tmp["high_eta"].astype(int)

    group_cols = [COL_DATE, "stratum_key"]
    grp = tmp.groupby(group_cols, sort=False, dropna=False)
    summary = grp.size().rename("n_total").to_frame()
    summary["n_treated"] = grp["high_eta"].sum()
    summary["n_control"] = summary["n_total"] - summary["n_treated"]

    total_sums = grp[all_metric_cols].sum(min_count=1)
    treated_sums = grp.apply(
        lambda frame: frame.loc[frame["high_eta"].eq(1), all_metric_cols].sum(min_count=1)
    )
    if isinstance(treated_sums.index, pd.MultiIndex):
        treated_sums.index = total_sums.index
    treated_sums = treated_sums.fillna(0.0)
    control_sums = total_sums - treated_sums

    for col in all_metric_cols:
        summary[f"sum_treated__{col}"] = treated_sums[col].to_numpy(dtype=float)
        summary[f"sum_control__{col}"] = control_sums[col].to_numpy(dtype=float)

    return summary.reset_index()


def _weighted_stat_from_strata(
    strata: pd.DataFrame,
    *,
    metric_cols: Sequence[str],
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if strata.empty:
        n_metrics = len(metric_cols)
        nan_vec = np.full(n_metrics, np.nan, dtype=float)
        return nan_vec, nan_vec, nan_vec, 0.0

    valid = strata["n_treated"].to_numpy(dtype=float) > 0
    valid &= strata["n_control"].to_numpy(dtype=float) > 0
    if not np.any(valid):
        n_metrics = len(metric_cols)
        nan_vec = np.full(n_metrics, np.nan, dtype=float)
        return nan_vec, nan_vec, nan_vec, 0.0

    strata_valid = strata.loc[valid].reset_index(drop=True)
    if weights is None:
        row_weight = np.ones(len(strata_valid), dtype=float)
    else:
        row_weight = np.asarray(weights, dtype=float)[valid]

    n_treated = strata_valid["n_treated"].to_numpy(dtype=float)
    n_control = strata_valid["n_control"].to_numpy(dtype=float)
    treated_weight = row_weight * n_treated
    total_treated = float(np.sum(treated_weight))
    if total_treated <= 0:
        n_metrics = len(metric_cols)
        nan_vec = np.full(n_metrics, np.nan, dtype=float)
        return nan_vec, nan_vec, nan_vec, 0.0

    treated_vals = []
    control_vals = []
    excess_vals = []
    for col in metric_cols:
        treated_sum = strata_valid[f"sum_treated__{col}"].to_numpy(dtype=float)
        control_sum = strata_valid[f"sum_control__{col}"].to_numpy(dtype=float)
        treated_mean = float(np.sum(row_weight * treated_sum) / total_treated)
        control_mean = float(np.sum(treated_weight * (control_sum / n_control)) / total_treated)
        treated_vals.append(treated_mean)
        control_vals.append(control_mean)
        excess_vals.append(treated_mean - control_mean)
    return (
        np.asarray(treated_vals, dtype=float),
        np.asarray(control_vals, dtype=float),
        np.asarray(excess_vals, dtype=float),
        total_treated,
    )


def _bootstrap_stats_by_date(
    strata: pd.DataFrame,
    *,
    curve_metric_cols: Sequence[str],
    summary_metric_cols: Sequence[str],
    n_runs: int,
    seed: int,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_runs <= 0 or strata.empty:
        return (
            np.empty((0, len(curve_metric_cols)), dtype=float),
            np.empty((0, len(summary_metric_cols)), dtype=float),
        )

    unique_dates = pd.Index(pd.unique(strata[COL_DATE]))
    if unique_dates.empty:
        return (
            np.empty((0, len(curve_metric_cols)), dtype=float),
            np.empty((0, len(summary_metric_cols)), dtype=float),
        )

    date_codes = unique_dates.get_indexer(strata[COL_DATE])
    rng = np.random.default_rng(seed)
    curve_reps = np.full((n_runs, len(curve_metric_cols)), np.nan, dtype=float)
    summary_reps = np.full((n_runs, len(summary_metric_cols)), np.nan, dtype=float)

    with _make_tqdm(
        total=n_runs,
        desc=progress_desc or "Bootstrap",
        disable=not show_progress,
        leave=False,
        unit="rep",
    ) as pbar:
        for run_idx in range(n_runs):
            sampled = rng.integers(0, len(unique_dates), size=len(unique_dates))
            date_weights = np.bincount(sampled, minlength=len(unique_dates)).astype(float)
            row_weights = date_weights[date_codes]
            _, _, curve_excess, _ = _weighted_stat_from_strata(
                strata,
                metric_cols=curve_metric_cols,
                weights=row_weights,
            )
            _, _, summary_excess, _ = _weighted_stat_from_strata(
                strata,
                metric_cols=summary_metric_cols,
                weights=row_weights,
            )
            curve_reps[run_idx, :] = curve_excess
            summary_reps[run_idx, :] = summary_excess
            pbar.update(1)
    return curve_reps, summary_reps


def _ci_from_reps(reps: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if reps.size == 0:
        n_cols = int(reps.shape[1]) if reps.ndim == 2 else 0
        return np.full(n_cols, np.nan, dtype=float), np.full(n_cols, np.nan, dtype=float)
    lo = np.nanquantile(reps, alpha / 2.0, axis=0)
    hi = np.nanquantile(reps, 1.0 - alpha / 2.0, axis=0)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _permutation_summary_stats(
    metrics_df: pd.DataFrame,
    *,
    summary_metric_cols: Sequence[str],
    n_runs: int,
    seed: int,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.full(len(summary_metric_cols), np.nan, dtype=float)
    if metrics_df.empty:
        return observed, np.empty((0, len(summary_metric_cols)), dtype=float)

    observed_metrics, _ = _stratum_metric_arrays(metrics_df, summary_metric_cols)
    observed = _summary_effect_from_metric_arrays(observed_metrics)
    if n_runs <= 0:
        return observed, np.empty((0, len(summary_metric_cols)), dtype=float)

    rng = np.random.default_rng(seed)
    permuted = np.full((n_runs, len(summary_metric_cols)), np.nan, dtype=float)
    with _make_tqdm(
        total=n_runs,
        desc=progress_desc or "Permutation",
        disable=not show_progress,
        leave=False,
        unit="rep",
    ) as pbar:
        for run_idx in range(n_runs):
            permuted_metrics, _ = _stratum_metric_arrays(metrics_df, summary_metric_cols, rng=rng)
            permuted[run_idx, :] = _summary_effect_from_metric_arrays(permuted_metrics)
            pbar.update(1)
    return observed, permuted


def _stratum_metric_arrays(
    metrics_df: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[tuple[np.ndarray, int]], int]:
    arrays: list[tuple[np.ndarray, int]] = []
    treated_total = 0
    for _, frame in metrics_df.groupby("stratum_key", sort=False):
        values = frame[list(metric_cols)].to_numpy(dtype=float)
        n_total = len(values)
        n_treated = int(frame["high_eta"].sum())
        if n_treated <= 0 or n_treated >= n_total:
            continue
        if rng is None:
            treated_mask = frame["high_eta"].to_numpy(dtype=bool)
        else:
            treated_mask = np.zeros(n_total, dtype=bool)
            treated_idx = rng.choice(n_total, size=n_treated, replace=False)
            treated_mask[treated_idx] = True
        arrays.append((values[treated_mask], n_treated))
        arrays.append((values[~treated_mask], -1))
        treated_total += n_treated
    return arrays, treated_total


def _summary_effect_from_metric_arrays(stratum_arrays: list[tuple[np.ndarray, int]]) -> np.ndarray:
    if not stratum_arrays:
        return np.full(len(SUMMARY_METRIC_SPECS), np.nan, dtype=float)

    total_treated = 0
    diff_sum: Optional[np.ndarray] = None
    idx = 0
    while idx < len(stratum_arrays):
        treated_values, n_treated = stratum_arrays[idx]
        control_values, _ = stratum_arrays[idx + 1]
        idx += 2
        if treated_values.size == 0 or control_values.size == 0:
            continue
        treated_sum = np.sum(treated_values, axis=0)
        control_mean = np.mean(control_values, axis=0)
        contribution = treated_sum - int(n_treated) * control_mean
        diff_sum = contribution if diff_sum is None else diff_sum + contribution
        total_treated += int(n_treated)

    if diff_sum is None or total_treated <= 0:
        return np.full(len(SUMMARY_METRIC_SPECS), np.nan, dtype=float)
    return diff_sum / float(total_treated)


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


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return out
    order = finite_idx[np.argsort(p[finite_idx])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min((m - rank) * p[idx], 1.0)
        running = max(running, adjusted)
        out[idx] = running
    return out


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


def _build_summary_rows(
    *,
    group_name: str,
    variant: str,
    eta_threshold: float,
    strata: pd.DataFrame,
    summary_metric_cols: Sequence[str],
    summary_observed_treated: np.ndarray,
    summary_observed_control: np.ndarray,
    summary_observed_excess: np.ndarray,
    summary_ci_lo: np.ndarray,
    summary_ci_hi: np.ndarray,
    perm_observed: np.ndarray,
    perm_draws: np.ndarray,
    total_treated_all: int,
    total_treated_matched: int,
    total_control_matched: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dropped_treated = total_treated_all - total_treated_matched
    for idx, (metric_name, sign_relation, window_side, alternative) in enumerate(SUMMARY_METRIC_SPECS):
        raw_p = _raw_p_value(perm_observed[idx], perm_draws[:, idx], alternative=alternative) if perm_draws.size else float("nan")
        rows.append(
            {
                "group": group_name,
                "variant": variant,
                "metric_name": metric_name,
                "sign_relation": sign_relation,
                "window_side": window_side,
                "test_alternative": alternative,
                "eta_threshold": eta_threshold,
                "treated_rate": summary_observed_treated[idx],
                "control_rate": summary_observed_control[idx],
                "excess_rate": summary_observed_excess[idx],
                "ci_excess_lo": summary_ci_lo[idx],
                "ci_excess_hi": summary_ci_hi[idx],
                "p_raw": raw_p,
                "n_treated_total": total_treated_all,
                "n_treated_matched": total_treated_matched,
                "n_control_in_valid_strata": total_control_matched,
                "n_treated_dropped_no_control": dropped_treated,
                "share_treated_dropped_no_control": (
                    float(dropped_treated) / float(total_treated_all) if total_treated_all > 0 else float("nan")
                ),
                "n_valid_strata": int(np.sum((strata["n_treated"] > 0) & (strata["n_control"] > 0))),
            }
        )
    return pd.DataFrame(rows)


def _build_curve_rows(
    *,
    group_name: str,
    variant: str,
    bin_spec: BinSpec,
    curve_metric_cols: Sequence[str],
    observed_treated_curve: np.ndarray,
    observed_control_curve: np.ndarray,
    observed_excess_curve: np.ndarray,
    curve_ci_lo: np.ndarray,
    curve_ci_hi: np.ndarray,
    total_treated_matched: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, metric_name in enumerate(curve_metric_cols):
        sign_relation = "same_sign" if metric_name.startswith("same_rate_") else "opposite_sign"
        rows.append(
            {
                "group": group_name,
                "variant": variant,
                "metric_name": metric_name,
                "sign_relation": sign_relation,
                "bin_index": idx if sign_relation == "same_sign" else idx - len(bin_spec.labels),
                "bin_label": bin_spec.labels[idx % len(bin_spec.labels)],
                "bin_left_min": float(bin_spec.left_minutes[idx % len(bin_spec.labels)]),
                "bin_right_min": float(bin_spec.right_minutes[idx % len(bin_spec.labels)]),
                "bin_center_min": float(bin_spec.centers_minutes[idx % len(bin_spec.labels)]),
                "treated_rate": observed_treated_curve[idx],
                "control_rate": observed_control_curve[idx],
                "excess_rate": observed_excess_curve[idx],
                "ci_excess_lo": curve_ci_lo[idx],
                "ci_excess_hi": curve_ci_hi[idx],
                "n_treated_matched": total_treated_matched,
            }
        )
    return pd.DataFrame(rows)


def _build_diagnostics_rows(
    *,
    metrics_df: pd.DataFrame,
    group_name: str,
    variant: str,
    eta_threshold: float,
    same_actor_col: Optional[str],
    total_treated_matched: int,
) -> pd.DataFrame:
    high_eta = metrics_df["high_eta"].to_numpy(dtype=bool)
    rows = [
        {
            "group": group_name,
            "variant": variant,
            "metric": "eta_threshold",
            "value": eta_threshold,
        },
        {
            "group": group_name,
            "variant": variant,
            "metric": "n_anchors",
            "value": float(len(metrics_df)),
        },
        {
            "group": group_name,
            "variant": variant,
            "metric": "n_high_eta",
            "value": float(np.sum(high_eta)),
        },
        {
            "group": group_name,
            "variant": variant,
            "metric": "mean_same_exact_zero_count_high_eta",
            "value": float(np.nanmean(metrics_df.loc[high_eta, "same_exact_zero_count"].to_numpy(dtype=float))),
        },
        {
            "group": group_name,
            "variant": variant,
            "metric": "mean_opp_exact_zero_count_high_eta",
            "value": float(np.nanmean(metrics_df.loc[high_eta, "opp_exact_zero_count"].to_numpy(dtype=float))),
        },
        {
            "group": group_name,
            "variant": variant,
            "metric": "mean_truncated_pre_bins_high_eta",
            "value": float(np.nanmean(metrics_df.loc[high_eta, "truncated_pre_bins"].to_numpy(dtype=float))),
        },
        {
            "group": group_name,
            "variant": variant,
            "metric": "mean_truncated_post_bins_high_eta",
            "value": float(np.nanmean(metrics_df.loc[high_eta, "truncated_post_bins"].to_numpy(dtype=float))),
        },
        {
            "group": group_name,
            "variant": variant,
            "metric": "n_treated_matched",
            "value": float(total_treated_matched),
        },
        {
            "group": group_name,
            "variant": variant,
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
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sub["treated_rate"].to_numpy(dtype=float),
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
                y=sub["control_rate"].to_numpy(dtype=float),
                mode="lines+markers",
                line=dict(color=control_color, width=2, dash="dash"),
                name="Matched controls",
                showlegend=(col_idx == 1),
                hovertemplate="tau=%{x:.1f} min<br>rate=%{y:.4f}<extra></extra>",
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
    same_actor_col: Optional[str],
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_annotated, eta_threshold = _annotate_high_eta_and_strata(
        metrics_df,
        high_eta_quantile=high_eta_quantile,
        matching_bucket_minutes=matching_bucket_minutes,
    )

    same_curve_cols = _rate_columns(bin_spec, "same_rate")
    opp_curve_cols = _rate_columns(bin_spec, "opp_rate")
    curve_metric_cols = same_curve_cols + opp_curve_cols
    summary_metric_cols = [name for name, _, _, _ in SUMMARY_METRIC_SPECS]

    strata = _build_stratum_summaries(
        metrics_annotated,
        curve_metric_cols=curve_metric_cols,
        summary_metric_cols=summary_metric_cols,
    )

    treated_curve, control_curve, excess_curve, total_treated_matched = _weighted_stat_from_strata(
        strata,
        metric_cols=curve_metric_cols,
    )
    treated_summary, control_summary, excess_summary, _ = _weighted_stat_from_strata(
        strata,
        metric_cols=summary_metric_cols,
    )
    total_treated_all = int(metrics_annotated["high_eta"].sum())
    total_control_matched = int(
        strata.loc[(strata["n_treated"] > 0) & (strata["n_control"] > 0), "n_control"].sum()
    )

    curve_reps, summary_reps = _bootstrap_stats_by_date(
        strata,
        curve_metric_cols=curve_metric_cols,
        summary_metric_cols=summary_metric_cols,
        n_runs=bootstrap_runs,
        seed=seed,
        show_progress=show_progress,
        progress_desc=f"[{group_name}:{variant}] bootstrap",
    )
    curve_ci_lo, curve_ci_hi = _ci_from_reps(curve_reps, alpha)
    summary_ci_lo, summary_ci_hi = _ci_from_reps(summary_reps, alpha)

    perm_observed, perm_draws = _permutation_summary_stats(
        metrics_annotated,
        summary_metric_cols=summary_metric_cols,
        n_runs=permutation_runs,
        seed=seed + 101,
        show_progress=show_progress,
        progress_desc=f"[{group_name}:{variant}] permutation",
    )

    summary_df = _build_summary_rows(
        group_name=group_name,
        variant=variant,
        eta_threshold=eta_threshold,
        strata=strata,
        summary_metric_cols=summary_metric_cols,
        summary_observed_treated=treated_summary,
        summary_observed_control=control_summary,
        summary_observed_excess=excess_summary,
        summary_ci_lo=summary_ci_lo,
        summary_ci_hi=summary_ci_hi,
        perm_observed=perm_observed,
        perm_draws=perm_draws,
        total_treated_all=total_treated_all,
        total_treated_matched=int(total_treated_matched),
        total_control_matched=total_control_matched,
    )
    curve_df = _build_curve_rows(
        group_name=group_name,
        variant=variant,
        bin_spec=bin_spec,
        curve_metric_cols=curve_metric_cols,
        observed_treated_curve=treated_curve,
        observed_control_curve=control_curve,
        observed_excess_curve=excess_curve,
        curve_ci_lo=curve_ci_lo,
        curve_ci_hi=curve_ci_hi,
        total_treated_matched=int(total_treated_matched),
    )
    diagnostics_df = _build_diagnostics_rows(
        metrics_df=metrics_annotated,
        group_name=group_name,
        variant=variant,
        eta_threshold=eta_threshold,
        same_actor_col=same_actor_col,
        total_treated_matched=int(total_treated_matched),
    )
    return summary_df, curve_df, diagnostics_df


def _apply_pvalue_adjustments(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["p_adjusted"] = np.nan
    out["p_adjustment_method"] = ""

    primary_mask = (
        out["variant"].eq("all_others")
        & out["sign_relation"].eq("same_sign")
        & out["window_side"].isin(["pre", "post"])
    )
    if primary_mask.any():
        adjusted = _holm_adjust(out.loc[primary_mask, "p_raw"].to_numpy(dtype=float))
        out.loc[primary_mask, "p_adjusted"] = adjusted
        out.loc[primary_mask, "p_adjustment_method"] = "holm_primary"

    secondary_mask = out["variant"].eq("all_others") & out["sign_relation"].eq("opposite_sign")
    if secondary_mask.any():
        adjusted = _bh_adjust(out.loc[secondary_mask, "p_raw"].to_numpy(dtype=float))
        out.loc[secondary_mask, "p_adjusted"] = adjusted
        out.loc[secondary_mask, "p_adjustment_method"] = "bh_secondary"
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
    parser.add_argument("--bootstrap-runs", type=int, default=None, help="Date-cluster bootstrap replicates.")
    parser.add_argument("--permutation-runs", type=int, default=None, help="Within-stratum permutation replicates.")
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
    4. reports bootstrap confidence intervals and within-stratum permutation p-values,
    5. writes tables, figures, and a reproducibility manifest.

    Examples
    --------
    >>> # main([\"--dry-run\"])  # doctest: +SKIP
    """
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = _load_yaml_defaults(resolve_repo_path(_REPO_ROOT, args.config_path))
    _apply_plot_style_from_cfg(cfg)

    paths = _resolve_paths(cfg, args)

    event_window_minutes = int(args.event_window_minutes if args.event_window_minutes is not None else cfg.get("EVENT_WINDOW_MINUTES", 30))
    bin_minutes = int(args.bin_minutes if args.bin_minutes is not None else cfg.get("BIN_MINUTES", 5))
    matching_bucket_minutes = int(
        args.matching_bucket_minutes if args.matching_bucket_minutes is not None else cfg.get("MATCHING_BUCKET_MINUTES", 30)
    )
    high_eta_quantile = float(
        args.high_eta_quantile if args.high_eta_quantile is not None else cfg.get("HIGH_ETA_QUANTILE", 0.9)
    )
    bootstrap_runs = int(args.bootstrap_runs if args.bootstrap_runs is not None else cfg.get("BOOTSTRAP_RUNS", 1000))
    permutation_runs = int(
        args.permutation_runs if args.permutation_runs is not None else cfg.get("PERMUTATION_RUNS", 1000)
    )
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
        "bootstrap_runs": bootstrap_runs,
        "permutation_runs": permutation_runs,
        "alpha": alpha,
        "seed": seed,
        "same_actor_key": same_actor_key,
        "run_same_actor_robustness": run_same_actor_robustness,
        "plots_enabled": plots_enabled,
        "write_parquet": write_parquet,
        "show_progress": show_progress,
        "trading_hours": [str(trading_hours_raw[0]), str(trading_hours_raw[1])],
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
        estimated_total_steps += 2 * 2

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

        all_summaries: list[pd.DataFrame] = []
        all_curves: list[pd.DataFrame] = []
        all_diagnostics: list[pd.DataFrame] = []

        group_specs = [
            ("prop", prop_raw, COLOR_PROPRIETARY),
            ("client", client_raw, COLOR_CLIENT),
        ]

        for group_idx, (group_name, raw_df, _) in enumerate(group_specs):
            same_actor_col = _select_same_actor_col(raw_df, same_actor_key)
            overall_pbar.set_postfix_str(f"{group_name}: build anchor windows")
            metrics_all = _prepare_group_metrics(
                raw_df,
                label=group_name,
                bin_spec=bin_spec,
                session_start_time=session_start_time,
                session_end_time=session_end_time,
                same_actor_col=same_actor_col,
                exclude_same_actor=False,
                show_progress=show_progress,
                progress_desc=f"[{group_name}:all_others] event windows",
            )
            overall_pbar.update(1)

            overall_pbar.set_postfix_str(f"{group_name}: matched analysis")
            summary_df, curve_df, diagnostics_df = _run_group_variant(
                metrics_all,
                group_name=group_name,
                variant="all_others",
                bin_spec=bin_spec,
                high_eta_quantile=high_eta_quantile,
                matching_bucket_minutes=matching_bucket_minutes,
                alpha=alpha,
                bootstrap_runs=bootstrap_runs,
                permutation_runs=permutation_runs,
                seed=seed + group_idx * 1000,
                same_actor_col=same_actor_col,
                show_progress=show_progress,
            )
            all_summaries.append(summary_df)
            all_curves.append(curve_df)
            all_diagnostics.append(diagnostics_df)
            if plots_enabled:
                _plot_group_variant_curves(curve_df, group_name=group_name, variant="all_others", dirs=img_dirs)
            overall_pbar.update(1)

            if run_same_actor_robustness and same_actor_col is not None:
                overall_pbar.set_postfix_str(f"{group_name}: exclude same actor")
                metrics_excl = _prepare_group_metrics(
                    raw_df,
                    label=f"{group_name}_exclude_same_actor",
                    bin_spec=bin_spec,
                    session_start_time=session_start_time,
                    session_end_time=session_end_time,
                    same_actor_col=same_actor_col,
                    exclude_same_actor=True,
                    show_progress=show_progress,
                    progress_desc=f"[{group_name}:exclude_same_actor] event windows",
                )
                overall_pbar.update(1)

                overall_pbar.set_postfix_str(f"{group_name}: robustness analysis")
                summary_excl, curve_excl, diagnostics_excl = _run_group_variant(
                    metrics_excl,
                    group_name=group_name,
                    variant="exclude_same_actor",
                    bin_spec=bin_spec,
                    high_eta_quantile=high_eta_quantile,
                    matching_bucket_minutes=matching_bucket_minutes,
                    alpha=alpha,
                    bootstrap_runs=bootstrap_runs,
                    permutation_runs=permutation_runs,
                    seed=seed + 500 + group_idx * 1000,
                    same_actor_col=same_actor_col,
                    show_progress=show_progress,
                )
                all_summaries.append(summary_excl)
                all_curves.append(curve_excl)
                all_diagnostics.append(diagnostics_excl)
                if plots_enabled:
                    _plot_group_variant_curves(curve_excl, group_name=group_name, variant="exclude_same_actor", dirs=img_dirs)
                overall_pbar.update(1)

        overall_pbar.set_postfix_str("write outputs")
        summary_all = _apply_pvalue_adjustments(pd.concat(all_summaries, ignore_index=True))
        curves_all = pd.concat(all_curves, ignore_index=True)
        diagnostics_all = pd.concat(all_diagnostics, ignore_index=True)

        summary_all.to_csv(paths.out_dir / "event_study_summary.csv", index=False)
        curves_all.to_csv(paths.out_dir / "event_study_curves.csv", index=False)
        diagnostics_all.to_csv(paths.out_dir / "event_study_diagnostics.csv", index=False)

        robustness_summary = summary_all.loc[summary_all["variant"].eq("exclude_same_actor")].copy()
        if not robustness_summary.empty:
            robustness_summary.to_csv(paths.out_dir / "robustness_same_actor_exclusion_summary.csv", index=False)

        if write_parquet:
            summary_all.to_parquet(paths.out_dir / "event_study_summary.parquet", index=False)
            curves_all.to_parquet(paths.out_dir / "event_study_curves.parquet", index=False)
            diagnostics_all.to_parquet(paths.out_dir / "event_study_diagnostics.parquet", index=False)
            if not robustness_summary.empty:
                robustness_summary.to_parquet(paths.out_dir / "robustness_same_actor_exclusion_summary.parquet", index=False)
        overall_pbar.update(1)

    print(f"[event-study] Wrote tables to {paths.out_dir}")
    if plots_enabled:
        print(f"[event-study] Wrote figures to {img_dirs.base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
