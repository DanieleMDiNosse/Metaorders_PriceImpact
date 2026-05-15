#!/usr/bin/env python3
"""
Member-level active-overlap crowding for proprietary versus client flow.

This workflow repeats the member prop-client correlation from the crowding
analysis, but replaces the member-day client imbalance with a more local
active-overlap client imbalance. For every proprietary target metaorder, the
client environment is restricted to metaorders executed by the same member and
active during the target's execution interval.

How to run
----------
1) Activate the repository conda environment.
2) Run from the repository root:

    python scripts/run_analysis.py crowding member-overlap

Outputs are written under:

- `out_files/{DATASET_NAME}/member_active_overlap_crowding/`
- `images/{DATASET_NAME}/member_active_overlap_crowding/{html,png}/`
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.config import format_path_template, load_yaml_mapping, resolve_repo_path
from moimpact.paper_figure_styles import apply_plotly_paper_figure_style, plotly_size_from_paper_style
from moimpact.plot_style import apply_shared_plotly_style, load_plot_style
from moimpact.plotting import (
    COLOR_CLIENT,
    COLOR_NEUTRAL,
    COLOR_PROPRIETARY,
    ensure_plot_dirs,
    make_plot_output_dirs,
    save_plotly_figure,
)
from moimpact.stats.active_overlap import (
    BUCKET_ALL_ACTIVE,
    BUCKET_PREEXISTING,
    BUCKET_STARTS_DURING,
    VALID_LEAD_LAG_BUCKETS,
    VALID_SCOPES,
    compute_member_active_overlap_features,
)
from moimpact.stats.correlation import corr_with_cluster_bootstrap_ci_and_permutation_p


CONFIG_ENV_VAR = "MEMBER_ACTIVE_OVERLAP_CROWDING_CONFIG"
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config_ymls" / "member_active_overlap_crowding.yml"

COL_SCOPE = "scope"
COL_BUCKET = "lead_lag_bucket"
COL_DATE = "Date"
COL_MEMBER = "Member"
COL_DIRECTION = "target_direction"
COL_IMBALANCE = "active_client_imbalance"

PLOT_STYLE = load_plot_style()
try:
    apply_shared_plotly_style(PLOT_STYLE)
except Exception:
    pass


@dataclass(frozen=True)
class ResolvedPaths:
    """Concrete paths used by one workflow run."""

    dataset_name: str
    input_path: Path
    out_dir: Path
    img_dir: Path
    config_path: Path


@dataclass(frozen=True)
class RuntimeOptions:
    """Validated runtime options for member active-overlap crowding."""

    scopes: tuple[str, ...]
    lead_lag_buckets: tuple[str, ...]
    bootstrap_runs: int
    alpha: float
    random_state: int
    member_window_days: int
    min_obs_per_member: int
    min_obs_per_member_window: int
    comovement_scope: str
    comovement_lead_lag_bucket: str
    comovement_top_n_members: int
    comovement_window_days: int
    overlap_batch_size: int
    n_jobs: int
    plots: bool
    write_target_parquet: bool
    write_html: bool
    write_png: bool


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Summary
    -------
    Run member-level active-overlap crowding analysis.

    Parameters
    ----------
    argv : Optional[Sequence[str]], default=None
        Optional command-line arguments. When `None`, arguments are read from
        `sys.argv`.

    Returns
    -------
    int
        Process-style return code. Zero indicates success.

    Notes
    -----
    The script reads a combined overlap-feature table and never overwrites the
    canonical proprietary/client metaorder parquet files.

    Examples
    --------
    >>> # From the repository root:
    >>> # python scripts/run_analysis.py crowding member-overlap
    """
    args = _parse_args(argv)
    cfg_path = _resolve_config_path(args.config_path)
    cfg = load_yaml_mapping(cfg_path)
    paths = _resolve_paths(args, cfg, config_path=cfg_path)
    options = _resolve_options(args, cfg)

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dirs = make_plot_output_dirs(paths.img_dir)
    if options.plots:
        ensure_plot_dirs(plot_dirs)

    print("[Member active overlap] Run started.")
    print(f"[Member active overlap] Input: {paths.input_path}")
    print(f"[Member active overlap] Output: {paths.out_dir}")

    metaorders = _load_input_table(paths.input_path)
    features = compute_member_active_overlap_features(
        metaorders,
        scopes=options.scopes,
        lead_lag_buckets=options.lead_lag_buckets,
        batch_size=options.overlap_batch_size,
        n_jobs=options.n_jobs,
    )

    target_path: Optional[Path] = None
    if options.write_target_parquet:
        target_path = paths.out_dir / "active_member_overlap_targets.parquet"
        features.to_parquet(target_path, index=False)
        print(f"[Member active overlap] Wrote target features: {target_path}")

    global_correlations = _build_global_correlations(features, options)
    per_member = _build_per_member_correlations(features, options)
    member_window = _build_member_window_correlations(features, options)
    comovement = _build_member_comovement_series(features, per_member, options)
    sample_counts = _build_sample_counts(features)

    _write_table(global_correlations, paths.out_dir / "global_correlations.csv")
    _write_table(per_member, paths.out_dir / "per_member_correlations.csv")
    per_member.to_parquet(paths.out_dir / "per_member_correlations.parquet", index=False)
    _write_table(member_window, paths.out_dir / "member_window_correlations.csv")
    member_window.to_parquet(paths.out_dir / "member_window_correlations.parquet", index=False)
    _write_table(comovement, paths.out_dir / "member_comovement_series.csv")
    comovement.to_parquet(paths.out_dir / "member_comovement_series.parquet", index=False)
    _write_table(sample_counts, paths.out_dir / "sample_counts.csv")

    if options.plots:
        _plot_global_lead_lag(global_correlations, plot_dirs, options)
        _plot_per_member(per_member, plot_dirs, options)
        _plot_member_comovement_series(comovement, plot_dirs, options)
        _plot_member_window_heatmaps(member_window, plot_dirs, options)

    manifest_path = paths.out_dir / "run_manifest.json"
    manifest = _build_manifest(paths, options, target_path=target_path, sample_counts=sample_counts)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[Member active overlap] Wrote manifest: {manifest_path}")
    print("[Member active overlap] Run completed.")
    return 0


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute member-level prop-client active-overlap crowding correlations."
    )
    parser.add_argument("--config-path", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name for path templates.")
    parser.add_argument("--input-path", type=str, default=None, help="Combined overlap_features parquet.")
    parser.add_argument("--output-file-path", type=str, default=None, help="Base output path.")
    parser.add_argument("--img-output-path", type=str, default=None, help="Base image output path.")
    parser.add_argument("--analysis-tag", type=str, default=None, help="Analysis folder name.")
    parser.add_argument("--scopes", type=str, default=None, help="Comma-separated scopes.")
    parser.add_argument("--lead-lag-buckets", type=str, default=None, help="Comma-separated lead-lag buckets.")
    parser.add_argument("--bootstrap-runs", type=int, default=None, help="Date-cluster bootstrap runs.")
    parser.add_argument("--alpha", type=float, default=None, help="Confidence level alpha.")
    parser.add_argument("--random-state", type=int, default=None, help="Random seed.")
    parser.add_argument("--member-window-days", type=int, default=None, help="Non-overlapping heatmap window length.")
    parser.add_argument("--min-obs-per-member", type=int, default=None, help="Minimum valid rows per member.")
    parser.add_argument(
        "--min-obs-per-member-window",
        type=int,
        default=None,
        help="Minimum valid rows per member-window heatmap cell.",
    )
    parser.add_argument("--comovement-scope", type=str, default=None, help="Scope for the member co-movement figure.")
    parser.add_argument(
        "--comovement-lead-lag-bucket",
        type=str,
        default=None,
        help="Lead-lag bucket for the member co-movement figure.",
    )
    parser.add_argument(
        "--comovement-top-n-members",
        type=int,
        default=None,
        help="Number of top positive global-correlation members in the co-movement figure.",
    )
    parser.add_argument(
        "--comovement-window-days",
        type=int,
        default=None,
        help="Non-overlapping trading-date window length for the co-movement figure.",
    )
    parser.add_argument("--overlap-batch-size", type=int, default=None, help="Overlap matrix target batch size.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Process workers. Zero means auto capped at 4.")
    parser.add_argument("--no-plots", action="store_true", help="Skip Plotly figure exports.")
    parser.add_argument("--no-write-target-parquet", action="store_true", help="Skip target-level parquet export.")
    parser.add_argument("--no-write-html", action="store_true", help="Skip HTML figure exports.")
    parser.add_argument("--no-write-png", action="store_true", help="Skip PNG figure exports.")
    return parser.parse_args(argv)


def _resolve_config_path(raw_path: Optional[str]) -> Path:
    if raw_path is None:
        raw_path = os.environ.get(CONFIG_ENV_VAR)
    return resolve_repo_path(_REPO_ROOT, raw_path or DEFAULT_CONFIG_PATH)


def _resolve_paths(args: argparse.Namespace, cfg: dict[str, Any], *, config_path: Path) -> ResolvedPaths:
    dataset_name = str(args.dataset_name or cfg.get("DATASET_NAME", "ftsemib"))
    analysis_tag = str(args.analysis_tag or cfg.get("ANALYSIS_TAG", "member_active_overlap_crowding"))
    context = {"DATASET_NAME": dataset_name, "ANALYSIS_TAG": analysis_tag}

    output_template = str(args.output_file_path or cfg.get("OUTPUT_FILE_PATH", "out_files/{DATASET_NAME}"))
    img_template = str(args.img_output_path or cfg.get("IMG_OUTPUT_PATH", "images/{DATASET_NAME}"))
    output_base = resolve_repo_path(_REPO_ROOT, format_path_template(output_template, context))
    img_base = resolve_repo_path(_REPO_ROOT, format_path_template(img_template, context))

    input_cfg = args.input_path if args.input_path is not None else cfg.get("INPUT_PATH")
    if input_cfg is None:
        input_path = output_base / "crowding_overlap_analysis" / "overlap_features.parquet"
    else:
        input_path = resolve_repo_path(_REPO_ROOT, format_path_template(str(input_cfg), context))

    return ResolvedPaths(
        dataset_name=dataset_name,
        input_path=input_path,
        out_dir=output_base / analysis_tag,
        img_dir=img_base / analysis_tag,
        config_path=config_path,
    )


def _resolve_options(args: argparse.Namespace, cfg: dict[str, Any]) -> RuntimeOptions:
    scopes_raw = args.scopes if args.scopes is not None else cfg.get("SCOPES", list(VALID_SCOPES))
    buckets_raw = (
        args.lead_lag_buckets
        if args.lead_lag_buckets is not None
        else cfg.get("LEAD_LAG_BUCKETS", list(VALID_LEAD_LAG_BUCKETS))
    )
    scopes = tuple(_as_string_list(scopes_raw))
    buckets = tuple(_as_string_list(buckets_raw))

    bootstrap_runs = int(args.bootstrap_runs if args.bootstrap_runs is not None else cfg.get("BOOTSTRAP_RUNS", 1000))
    alpha = float(args.alpha if args.alpha is not None else cfg.get("ALPHA", 0.05))
    random_state = int(args.random_state if args.random_state is not None else cfg.get("RANDOM_STATE", 0))
    member_window_days = int(
        args.member_window_days if args.member_window_days is not None else cfg.get("MEMBER_WINDOW_DAYS", 3)
    )
    min_obs_per_member = int(
        args.min_obs_per_member if args.min_obs_per_member is not None else cfg.get("MIN_OBS_PER_MEMBER", 30)
    )
    min_obs_per_member_window = int(
        args.min_obs_per_member_window
        if args.min_obs_per_member_window is not None
        else cfg.get("MIN_OBS_PER_MEMBER_WINDOW", 5)
    )
    comovement_scope = str(args.comovement_scope or cfg.get("COMOVEMENT_SCOPE", "same_isin"))
    comovement_lead_lag_bucket = str(
        args.comovement_lead_lag_bucket or cfg.get("COMOVEMENT_LEAD_LAG_BUCKET", BUCKET_ALL_ACTIVE)
    )
    comovement_top_n_members = int(
        args.comovement_top_n_members
        if args.comovement_top_n_members is not None
        else cfg.get("COMOVEMENT_TOP_N_MEMBERS", 2)
    )
    comovement_window_days = int(
        args.comovement_window_days
        if args.comovement_window_days is not None
        else cfg.get("COMOVEMENT_WINDOW_DAYS", 5)
    )
    overlap_batch_size = int(
        args.overlap_batch_size if args.overlap_batch_size is not None else cfg.get("OVERLAP_BATCH_SIZE", 2048)
    )
    raw_n_jobs = int(args.n_jobs if args.n_jobs is not None else cfg.get("N_JOBS", 0))
    n_jobs = _resolve_n_jobs(raw_n_jobs)
    plots = (not args.no_plots) and _as_bool(cfg.get("PLOTS", True))
    write_target_parquet = (not args.no_write_target_parquet) and _as_bool(cfg.get("WRITE_TARGET_PARQUET", True))
    write_html = (not args.no_write_html) and _as_bool(cfg.get("WRITE_HTML", True))
    write_png = (not args.no_write_png) and _as_bool(cfg.get("WRITE_PNG", True))

    if bootstrap_runs < 0:
        raise ValueError("BOOTSTRAP_RUNS must be non-negative.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("ALPHA must be between 0 and 1.")
    if member_window_days <= 0:
        raise ValueError("MEMBER_WINDOW_DAYS must be positive.")
    if min_obs_per_member <= 0:
        raise ValueError("MIN_OBS_PER_MEMBER must be positive.")
    if min_obs_per_member_window <= 0:
        raise ValueError("MIN_OBS_PER_MEMBER_WINDOW must be positive.")
    if comovement_scope not in VALID_SCOPES:
        raise ValueError(f"COMOVEMENT_SCOPE must be one of {sorted(VALID_SCOPES)}.")
    if comovement_lead_lag_bucket not in VALID_LEAD_LAG_BUCKETS:
        raise ValueError(f"COMOVEMENT_LEAD_LAG_BUCKET must be one of {sorted(VALID_LEAD_LAG_BUCKETS)}.")
    if comovement_top_n_members <= 0:
        raise ValueError("COMOVEMENT_TOP_N_MEMBERS must be positive.")
    if comovement_window_days <= 0:
        raise ValueError("COMOVEMENT_WINDOW_DAYS must be positive.")
    if overlap_batch_size <= 0:
        raise ValueError("OVERLAP_BATCH_SIZE must be positive.")

    invalid_scopes = [scope for scope in scopes if scope not in VALID_SCOPES]
    invalid_buckets = [bucket for bucket in buckets if bucket not in VALID_LEAD_LAG_BUCKETS]
    if invalid_scopes:
        raise ValueError(f"Invalid SCOPES values: {invalid_scopes}. Valid: {list(VALID_SCOPES)}")
    if invalid_buckets:
        raise ValueError(f"Invalid LEAD_LAG_BUCKETS values: {invalid_buckets}. Valid: {list(VALID_LEAD_LAG_BUCKETS)}")

    return RuntimeOptions(
        scopes=scopes,
        lead_lag_buckets=buckets,
        bootstrap_runs=bootstrap_runs,
        alpha=alpha,
        random_state=random_state,
        member_window_days=member_window_days,
        min_obs_per_member=min_obs_per_member,
        min_obs_per_member_window=min_obs_per_member_window,
        comovement_scope=comovement_scope,
        comovement_lead_lag_bucket=comovement_lead_lag_bucket,
        comovement_top_n_members=comovement_top_n_members,
        comovement_window_days=comovement_window_days,
        overlap_batch_size=overlap_batch_size,
        n_jobs=n_jobs,
        plots=plots,
        write_target_parquet=write_target_parquet,
        write_html=write_html,
        write_png=write_png,
    )


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [piece.strip() for piece in value.split(",") if piece.strip()]
    if isinstance(value, (list, tuple)):
        return [str(piece).strip() for piece in value if str(piece).strip()]
    raise TypeError(f"Expected a comma-separated string or list, got {type(value).__name__}.")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _resolve_n_jobs(raw_n_jobs: int) -> int:
    if raw_n_jobs > 0:
        return int(raw_n_jobs)
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, int(cpu_count)))


def _load_input_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing active-overlap input table: {path}. "
            "Run scripts/run_analysis.py crowding overlap first or set INPUT_PATH."
        )
    required = ["group", "Member", "ISIN", "Date", "StartTimestamp", "EndTimestamp", "Q", "Direction"]
    try:
        return pd.read_parquet(path, columns=required)
    except Exception:
        frame = pd.read_parquet(path)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"Input table is missing required columns: {missing}") from None
        return frame[required].copy()


def _build_global_correlations(features: pd.DataFrame, options: RuntimeOptions) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scope, bucket), group in features.groupby([COL_SCOPE, COL_BUCKET], sort=False, dropna=False):
        summary = _correlation_row(
            group,
            scope=str(scope),
            lead_lag_bucket=str(bucket),
            level="global",
            member=None,
            window=None,
            alpha=options.alpha,
            n_bootstrap=options.bootstrap_runs,
            random_state=_stable_seed(options.random_state, "global", scope, bucket),
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def _build_per_member_correlations(features: pd.DataFrame, options: RuntimeOptions) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = features.groupby([COL_SCOPE, COL_BUCKET, COL_MEMBER], sort=True, dropna=False)
    for (scope, bucket, member), group in grouped:
        row = _correlation_row(
            group,
            scope=str(scope),
            lead_lag_bucket=str(bucket),
            level="member",
            member=member,
            window=None,
            alpha=options.alpha,
            n_bootstrap=options.bootstrap_runs,
            random_state=_stable_seed(options.random_state, "member", scope, bucket, member),
        )
        row["passes_min_obs"] = bool(row["n_valid"] >= options.min_obs_per_member)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_member_window_correlations(features: pd.DataFrame, options: RuntimeOptions) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    working = features.copy()
    working[COL_DATE] = pd.to_datetime(working[COL_DATE], errors="coerce").dt.normalize()
    window_map = _build_window_map(working[COL_DATE], window_days=options.member_window_days)
    working["Window"] = working[COL_DATE].dt.date.map(window_map)
    working = working.dropna(subset=["Window"])

    rows: list[dict[str, object]] = []
    grouped = working.groupby([COL_SCOPE, COL_BUCKET, COL_MEMBER, "Window"], sort=True, dropna=False)
    for (scope, bucket, member, window), group in grouped:
        row = _correlation_row(
            group,
            scope=str(scope),
            lead_lag_bucket=str(bucket),
            level="member_window",
            member=member,
            window=str(window),
            alpha=options.alpha,
            n_bootstrap=0,
            random_state=_stable_seed(options.random_state, "member_window", scope, bucket, member, window),
        )
        row["passes_min_obs"] = bool(row["n_valid"] >= options.min_obs_per_member_window)
        if not row["passes_min_obs"]:
            row["r"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _build_member_comovement_series(
    features: pd.DataFrame,
    per_member: pd.DataFrame,
    options: RuntimeOptions,
) -> pd.DataFrame:
    selected_members = _select_comovement_members(per_member, options)
    if features.empty or not selected_members:
        return pd.DataFrame(
            columns=[
                "scope",
                "lead_lag_bucket",
                "Member",
                "Window",
                "window_start",
                "window_end",
                "window_mid",
                "n_valid",
                "n_dates",
                "prop_target_imbalance",
                "active_client_imbalance",
                "mean_signed_alignment",
                "member_global_r",
                "member_global_lo",
                "member_global_hi",
                "window_series_corr",
            ]
        )

    working = features[
        (features[COL_SCOPE].astype(str) == options.comovement_scope)
        & (features[COL_BUCKET].astype(str) == options.comovement_lead_lag_bucket)
        & (features[COL_MEMBER].astype(str).isin([str(member) for member in selected_members]))
    ].copy()
    if working.empty:
        return pd.DataFrame()

    for col in [COL_DIRECTION, COL_IMBALANCE, "active_client_alignment"]:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working[COL_DATE] = pd.to_datetime(working[COL_DATE], errors="coerce").dt.normalize()
    working = working.replace([np.inf, -np.inf], np.nan).dropna(subset=[COL_DIRECTION, COL_IMBALANCE, COL_DATE])
    if working.empty:
        return pd.DataFrame()

    window_meta = _build_window_metadata(working[COL_DATE], window_days=options.comovement_window_days)
    if window_meta.empty:
        return pd.DataFrame()
    window_map = dict(zip(window_meta["window_date"], window_meta["Window"]))
    working["Window"] = working[COL_DATE].dt.date.map(window_map)
    working = working.dropna(subset=["Window"])
    if working.empty:
        return pd.DataFrame()

    member_stats = per_member[
        (per_member["scope"].astype(str) == options.comovement_scope)
        & (per_member["lead_lag_bucket"].astype(str) == options.comovement_lead_lag_bucket)
    ].copy()
    member_stats["__member_key__"] = member_stats[COL_MEMBER].astype(str)
    member_stats = member_stats.set_index("__member_key__")[["r", "lo", "hi"]].apply(pd.to_numeric, errors="coerce")

    grouped = working.groupby([COL_MEMBER, "Window"], sort=True, dropna=False)
    rows: list[dict[str, object]] = []
    for (member, window), group in grouped:
        window_rows = window_meta[window_meta["Window"] == str(window)]
        if window_rows.empty:
            continue
        window_info = window_rows.iloc[0]
        directions = group[COL_DIRECTION].to_numpy(dtype=float)
        client_imb = group[COL_IMBALANCE].to_numpy(dtype=float)
        align = group["active_client_alignment"].to_numpy(dtype=float)
        member_key = str(member)
        global_row = member_stats.loc[member_key] if member_key in member_stats.index else None
        rows.append(
            {
                "scope": options.comovement_scope,
                "lead_lag_bucket": options.comovement_lead_lag_bucket,
                "Member": member,
                "Window": str(window),
                "window_start": window_info["window_start"],
                "window_end": window_info["window_end"],
                "window_mid": window_info["window_mid"],
                "n_valid": int(len(group)),
                "n_dates": int(group[COL_DATE].nunique()),
                # Equal-weighted target averages preserve the population used
                # by Corr(target_direction, active_client_imbalance).
                "prop_target_imbalance": float(np.nanmean(directions)),
                "active_client_imbalance": float(np.nanmean(client_imb)),
                "mean_signed_alignment": float(np.nanmean(align)),
                "member_global_r": float(global_row["r"]) if global_row is not None else np.nan,
                "member_global_lo": float(global_row["lo"]) if global_row is not None else np.nan,
                "member_global_hi": float(global_row["hi"]) if global_row is not None else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["Member", "window_start"]).reset_index(drop=True)
    out["window_series_corr"] = np.nan
    for member, idx in out.groupby("Member", sort=False).groups.items():
        sub = out.loc[idx, ["prop_target_imbalance", "active_client_imbalance"]].dropna()
        if len(sub) >= 3:
            out.loc[idx, "window_series_corr"] = _safe_corr(
                sub["prop_target_imbalance"].to_numpy(dtype=float),
                sub["active_client_imbalance"].to_numpy(dtype=float),
            )
    return out


def _select_comovement_members(per_member: pd.DataFrame, options: RuntimeOptions) -> list[object]:
    if per_member.empty:
        return []
    candidates = per_member[
        (per_member["scope"].astype(str) == options.comovement_scope)
        & (per_member["lead_lag_bucket"].astype(str) == options.comovement_lead_lag_bucket)
        & (per_member["passes_min_obs"].astype(bool))
    ].copy()
    if candidates.empty:
        return []
    candidates["r"] = pd.to_numeric(candidates["r"], errors="coerce")
    candidates = candidates[np.isfinite(candidates["r"].to_numpy(dtype=float))].copy()
    if candidates.empty:
        return []
    positive = candidates[candidates["r"] > 0.0].sort_values("r", ascending=False)
    if len(positive) >= options.comovement_top_n_members:
        selected = positive.head(options.comovement_top_n_members)
    else:
        remainder = candidates.loc[~candidates.index.isin(positive.index)].sort_values("r", ascending=False)
        selected = pd.concat([positive, remainder], axis=0).head(options.comovement_top_n_members)
    return selected[COL_MEMBER].tolist()


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    x_valid = x[finite]
    y_valid = y[finite]
    if np.nanstd(x_valid) <= 1e-12 or np.nanstd(y_valid) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _correlation_row(
    frame: pd.DataFrame,
    *,
    scope: str,
    lead_lag_bucket: str,
    level: str,
    member: object,
    window: Optional[str],
    alpha: float,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, object]:
    valid = frame[[COL_DIRECTION, COL_IMBALANCE, COL_DATE, COL_MEMBER]].copy()
    valid[COL_DIRECTION] = pd.to_numeric(valid[COL_DIRECTION], errors="coerce")
    valid[COL_IMBALANCE] = pd.to_numeric(valid[COL_IMBALANCE], errors="coerce")
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=[COL_DIRECTION, COL_IMBALANCE, COL_DATE])
    if len(valid) >= 3:
        r, lo, hi, p, n_obs, n_clusters = corr_with_cluster_bootstrap_ci_and_permutation_p(
            valid[COL_DIRECTION],
            valid[COL_IMBALANCE],
            valid[COL_DATE],
            alpha=alpha,
            n_bootstrap=int(n_bootstrap),
            n_permutations=0,
            random_state=random_state,
        )
    else:
        r, lo, hi, p, n_obs, n_clusters = np.nan, np.nan, np.nan, np.nan, len(valid), valid[COL_DATE].nunique()

    return {
        "level": level,
        "scope": scope,
        "lead_lag_bucket": lead_lag_bucket,
        "Member": member,
        "Window": window,
        "r": float(r) if np.isfinite(r) else np.nan,
        "lo": float(lo) if np.isfinite(lo) else np.nan,
        "hi": float(hi) if np.isfinite(hi) else np.nan,
        "p": float(p) if np.isfinite(p) else np.nan,
        "n_valid": int(n_obs),
        "n_targets": int(len(frame)),
        "n_dates": int(n_clusters),
        "n_members_valid": int(valid[COL_MEMBER].nunique(dropna=True)),
    }


def _build_window_map(dates: pd.Series, *, window_days: int) -> dict[dt.date, str]:
    window_meta = _build_window_metadata(dates, window_days=window_days)
    return dict(zip(window_meta["window_date"], window_meta["Window"]))


def _build_window_metadata(dates: pd.Series, *, window_days: int) -> pd.DataFrame:
    unique_dates = sorted(pd.to_datetime(dates, errors="coerce").dt.date.dropna().unique())
    rows: list[dict[str, object]] = []
    for idx in range(0, len(unique_dates), int(window_days)):
        chunk = unique_dates[idx : idx + int(window_days)]
        if not chunk:
            continue
        label = f"{chunk[0]}_to_{chunk[-1]}"
        start_ts = pd.Timestamp(chunk[0])
        end_ts = pd.Timestamp(chunk[-1])
        mid_ts = start_ts + (end_ts - start_ts) / 2
        for date_value in chunk:
            rows.append(
                {
                    "window_date": date_value,
                    "Window": label,
                    "window_start": start_ts.date().isoformat(),
                    "window_end": end_ts.date().isoformat(),
                    "window_mid": mid_ts.date().isoformat(),
                }
            )
    return pd.DataFrame(rows)



def _build_sample_counts(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["scope", "lead_lag_bucket", "n_targets", "n_valid_overlap", "n_members_valid"])
    valid_mask = np.isfinite(pd.to_numeric(features[COL_IMBALANCE], errors="coerce").to_numpy(dtype=float))
    work = features.assign(__valid_overlap__=valid_mask)
    counts = (
        work.groupby([COL_SCOPE, COL_BUCKET], sort=False, dropna=False)
        .agg(
            n_targets=("target_row_id", "size"),
            n_valid_overlap=("__valid_overlap__", "sum"),
            n_members_valid=(COL_MEMBER, lambda s: s[work.loc[s.index, "__valid_overlap__"].to_numpy(dtype=bool)].nunique()),
        )
        .reset_index()
    )
    counts["n_valid_overlap"] = counts["n_valid_overlap"].astype(int)
    return counts


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Member active overlap] Wrote table: {path}")


def _plot_global_lead_lag(global_correlations: pd.DataFrame, plot_dirs, options: RuntimeOptions) -> None:
    if global_correlations.empty:
        return
    label_map = _bucket_label_map()
    fig = go.Figure()
    colors = {"same_isin": COLOR_PROPRIETARY, "all_isin": COLOR_CLIENT}
    for scope, group in global_correlations.groupby("scope", sort=False):
        group = group.copy()
        group["bucket_label"] = group["lead_lag_bucket"].map(label_map).fillna(group["lead_lag_bucket"])
        error_plus = group["hi"] - group["r"]
        error_minus = group["r"] - group["lo"]
        fig.add_trace(
            go.Bar(
                x=group["bucket_label"],
                y=group["r"],
                error_y=dict(type="data", array=error_plus, arrayminus=error_minus, visible=True),
                name=str(scope),
                marker_color=colors.get(str(scope), COLOR_NEUTRAL),
                customdata=np.column_stack([group["n_valid"], group["n_dates"]]),
                hovertemplate="scope=%{fullData.name}<br>bucket=%{x}<br>r=%{y:.3f}<br>n=%{customdata[0]}<br>dates=%{customdata[1]}<extra></extra>",
            )
        )
    fig.add_hline(y=0.0, line=dict(color=COLOR_NEUTRAL, width=1, dash="dot"))
    fig.update_layout(
        title="Global prop-client active-overlap correlations",
        xaxis_title="Lead-lag bucket",
        yaxis_title="Corr(prop direction, active client imbalance)",
        barmode="group",
        legend_title="Scope",
    )
    save_plotly_figure(
        fig,
        stem="global_lead_lag_correlations",
        dirs=plot_dirs,
        write_html=options.write_html,
        write_png=options.write_png,
        strict_png=False,
    )


def _plot_per_member(per_member: pd.DataFrame, plot_dirs, options: RuntimeOptions) -> None:
    if per_member.empty:
        return
    for (scope, bucket), group in per_member.groupby(["scope", "lead_lag_bucket"], sort=False):
        group = group.copy()
        group["r"] = pd.to_numeric(group["r"], errors="coerce")
        plot_df = group[(group["passes_min_obs"]) & (np.isfinite(group["r"].to_numpy(dtype=float)))].copy()
        if plot_df.empty:
            continue
        plot_df = plot_df.sort_values("r").reset_index(drop=True)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=plot_df["Member"].astype(str),
                    y=plot_df["r"],
                    marker_color=COLOR_PROPRIETARY,
                    customdata=np.column_stack([plot_df["n_valid"], plot_df["n_dates"]]),
                    hovertemplate="Member=%{x}<br>r=%{y:.3f}<br>n=%{customdata[0]}<br>dates=%{customdata[1]}<extra></extra>",
                )
            ]
        )
        fig.add_hline(y=0.0, line=dict(color=COLOR_NEUTRAL, width=1, dash="dot"))
        fig.update_layout(
            title=f"Per-member active-overlap correlation ({scope}, {bucket})",
            xaxis_title="Member",
            yaxis_title="Corr(prop direction, active client imbalance)",
            xaxis=dict(tickangle=90),
        )
        save_plotly_figure(
            fig,
            stem=f"per_member_correlations_{scope}_{bucket}",
            dirs=plot_dirs,
            write_html=options.write_html,
            write_png=options.write_png,
            strict_png=False,
        )


def _plot_member_comovement_series(comovement: pd.DataFrame, plot_dirs, options: RuntimeOptions) -> None:
    if comovement.empty:
        return

    plot_df = comovement.copy()
    plot_df["window_mid"] = pd.to_datetime(plot_df["window_mid"], errors="coerce")
    plot_df["prop_target_imbalance"] = pd.to_numeric(plot_df["prop_target_imbalance"], errors="coerce")
    plot_df["active_client_imbalance"] = pd.to_numeric(plot_df["active_client_imbalance"], errors="coerce")
    plot_df["n_valid"] = pd.to_numeric(plot_df["n_valid"], errors="coerce").fillna(0).astype(int)
    plot_df = plot_df.dropna(subset=["window_mid", "prop_target_imbalance", "active_client_imbalance"])
    if plot_df.empty:
        return

    member_order = (
        plot_df.groupby(plot_df["Member"].astype(str), sort=False)["member_global_r"]
        .first()
        .sort_values(ascending=False)
    )
    members = member_order.index.tolist()
    fig = make_subplots(
        rows=len(members),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10 if len(members) > 1 else 0.05,
        subplot_titles=[_comovement_subplot_title(plot_df, member) for member in members],
    )

    for row_idx, member in enumerate(members, start=1):
        member_df = plot_df[plot_df["Member"].astype(str) == member].sort_values("window_mid").copy()
        if member_df.empty:
            continue
        custom = np.column_stack(
            [
                member_df["Window"].astype(str),
                member_df["n_valid"].astype(int),
                member_df["mean_signed_alignment"].round(4),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=member_df["window_mid"],
                y=member_df["prop_target_imbalance"],
                mode="lines+markers",
                name="Proprietary target imbalance",
                legendgroup="prop",
                showlegend=row_idx == 1,
                marker=dict(size=7, color=COLOR_PROPRIETARY),
                line=dict(color=COLOR_PROPRIETARY, width=2.2),
                customdata=custom,
                hovertemplate=(
                    "Member="
                    + member
                    + "<br>Window=%{customdata[0]}<br>n=%{customdata[1]}"
                    + "<br>prop imbalance=%{y:.3f}<br>mean alignment=%{customdata[2]:.3f}<extra></extra>"
                ),
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=member_df["window_mid"],
                y=member_df["active_client_imbalance"],
                mode="lines+markers",
                name="Active client imbalance",
                legendgroup="client",
                showlegend=row_idx == 1,
                marker=dict(size=7, color=COLOR_CLIENT),
                line=dict(color=COLOR_CLIENT, width=2.2, dash="dash"),
                customdata=custom,
                hovertemplate=(
                    "Member="
                    + member
                    + "<br>Window=%{customdata[0]}<br>n=%{customdata[1]}"
                    + "<br>client imbalance=%{y:.3f}<br>mean alignment=%{customdata[2]:.3f}<extra></extra>"
                ),
            ),
            row=row_idx,
            col=1,
        )
        fig.add_hline(y=0.0, line=dict(color=COLOR_NEUTRAL, width=1, dash="dot"), row=row_idx, col=1)
        fig.update_yaxes(range=[-1.05, 1.05], title_text="Imbalance", row=row_idx, col=1)

    fig.update_layout(
        title=dict(
            text=(
                "Prop-client active-imbalance co-movement"
                f"<br><sup>{options.comovement_scope}, {options.comovement_lead_lag_bucket}; "
                f"{options.comovement_window_days}-trading-day windows</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        width=1200,
        height=max(540, 300 * len(members)),
        font=dict(size=14),
        title_font=dict(size=18),
        legend=dict(
            title_text="",
            orientation="v",
            x=1.01,
            xanchor="left",
            y=1.0,
            yanchor="top",
            font=dict(size=13),
        ),
        margin=dict(l=70, r=260, t=120, b=70),
    )
    fig.update_annotations(font_size=14)
    fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=13))
    fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=13))
    fig.update_xaxes(title_text="Window midpoint", row=len(members), col=1)
    stem = f"member_comovement_{options.comovement_scope}_{options.comovement_lead_lag_bucket}"
    style = apply_plotly_paper_figure_style(
        fig,
        stem,
        default_tick_font_size=12,
        default_label_font_size=13,
        default_title_font_size=18,
        default_legend_font_size=13,
        default_annotation_font_size=14,
        default_line_width=2.2,
        default_reference_line_width=1,
    )
    save_plotly_figure(
        fig,
        stem=stem,
        dirs=plot_dirs,
        write_html=options.write_html,
        write_png=options.write_png,
        strict_png=False,
        **plotly_size_from_paper_style(style),
    )


def _comovement_subplot_title(plot_df: pd.DataFrame, member: str) -> str:
    member_df = plot_df[plot_df["Member"].astype(str) == member]
    if member_df.empty:
        return f"Member {member}"
    global_r = pd.to_numeric(member_df["member_global_r"], errors="coerce").dropna()
    n_total = int(pd.to_numeric(member_df["n_valid"], errors="coerce").sum())
    pieces = [f"Member {member} (n={n_total})"]
    if not global_r.empty:
        pieces.append(f"target r={float(global_r.iloc[0]):.3f}")
    # The window-level correlation is descriptive and can be unstable with
    # short windows; keep it in the exported table rather than the plot title.
    return " | ".join(pieces)


def _plot_member_window_heatmaps(member_window: pd.DataFrame, plot_dirs, options: RuntimeOptions) -> None:
    if member_window.empty:
        return
    for (scope, bucket), group in member_window.groupby(["scope", "lead_lag_bucket"], sort=False):
        plot_df = group.copy()
        if plot_df["r"].notna().sum() == 0:
            continue
        pivot = plot_df.pivot(index="Window", columns="Member", values="r")
        pivot = pivot.sort_index()
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.to_numpy(dtype=float),
                x=[str(x) for x in pivot.columns],
                y=[str(y) for y in pivot.index],
                colorscale="RdBu",
                zmin=-1.0,
                zmax=1.0,
                zmid=0.0,
                colorbar=dict(title="Correlation"),
                hovertemplate="Window=%{y}<br>Member=%{x}<br>r=%{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Member-window active-overlap heatmap ({scope}, {bucket})",
            xaxis_title="Member",
            yaxis_title="Non-overlapping window",
        )
        save_plotly_figure(
            fig,
            stem=f"member_window_heatmap_{scope}_{bucket}",
            dirs=plot_dirs,
            write_html=options.write_html,
            write_png=options.write_png,
            strict_png=False,
        )


def _bucket_label_map() -> dict[str, str]:
    return {
        BUCKET_ALL_ACTIVE: "All active",
        BUCKET_PREEXISTING: "Already active",
        BUCKET_STARTS_DURING: "Starts during prop",
    }


def _build_manifest(
    paths: ResolvedPaths,
    options: RuntimeOptions,
    *,
    target_path: Optional[Path],
    sample_counts: pd.DataFrame,
) -> dict[str, object]:
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_name": paths.dataset_name,
        "config_path": str(paths.config_path),
        "input_path": str(paths.input_path),
        "output_dir": str(paths.out_dir),
        "image_dir": str(paths.img_dir),
        "target_features_path": str(target_path) if target_path is not None else None,
        "options": {
            "scopes": list(options.scopes),
            "lead_lag_buckets": list(options.lead_lag_buckets),
            "bootstrap_runs": options.bootstrap_runs,
            "alpha": options.alpha,
            "random_state": options.random_state,
            "member_window_days": options.member_window_days,
            "min_obs_per_member": options.min_obs_per_member,
            "min_obs_per_member_window": options.min_obs_per_member_window,
            "comovement_scope": options.comovement_scope,
            "comovement_lead_lag_bucket": options.comovement_lead_lag_bucket,
            "comovement_top_n_members": options.comovement_top_n_members,
            "comovement_window_days": options.comovement_window_days,
            "overlap_batch_size": options.overlap_batch_size,
            "n_jobs": options.n_jobs,
        },
        "sample_counts": sample_counts.to_dict(orient="records"),
    }


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    offset = sum((idx + 1) * ord(char) for idx, char in enumerate(text))
    return int((int(base_seed) + offset) % (2**32 - 1))


if __name__ == "__main__":
    raise SystemExit(main())
