#!/usr/bin/env python3
"""
Active-overlap crowding analysis for proprietary and client metaorders.

This script computes leave-one-out overlap features from true execution
interval intersections, then compares proprietary and client metaorders through
distribution plots, intraday profiles, impact regressions with Date-cluster
standard errors, and binned log-WLS regressions aligned with the standard
power-law impact fit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure repository-root imports work when running from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.config import format_path_template, load_yaml_mapping, resolve_repo_path
from moimpact.metaorder_distribution_samples import (
    parse_member_nationality,
    with_member_nationality_tag,
)
from moimpact.plot_style import apply_shared_plotly_style, load_plot_style
from moimpact.plotting import (
    COLOR_CLIENT,
    COLOR_NEUTRAL,
    COLOR_PROPRIETARY,
    make_plot_output_dirs,
    save_plotly_figure as _save_plotly_figure,
)


CONFIG_ENV_VAR = "CROWDING_OVERLAP_ANALYSIS_CONFIG"
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config_ymls" / "crowding_overlap_analysis.yml"

COL_ISIN = "ISIN"
COL_DATE = "Date"
COL_PERIOD = "Period"
COL_DIR = "Direction"
COL_Q = "Q"
COL_QV = "Q/V"
COL_ETA = "Participation Rate"
COL_VTV = "Vt/V"
COL_PRICE_CHANGE = "Price Change"
COL_DAILY_VOL = "Daily Vol"
COL_IMPACT = "Impact"
COL_START_TS = "StartTimestamp"
COL_END_TS = "EndTimestamp"
COL_GROUP = "group"
COL_PROPRIETARY = "proprietary"
COL_BIN_ID = "start_bin_id"
COL_BIN_LABEL = "start_bin_label"

GROUP_PROPRIETARY = "proprietary"
GROUP_CLIENT = "client"
GROUP_LABELS = {GROUP_PROPRIETARY: "Proprietary", GROUP_CLIENT: "Client"}
GROUP_COLORS = {GROUP_PROPRIETARY: COLOR_PROPRIETARY, GROUP_CLIENT: COLOR_CLIENT}

NS_PER_MINUTE = 60_000_000_000.0
FEATURE_SUFFIXES = ("all", "prop_env", "client_env")
FEATURE_BASES = (
    "overlap_any_count",
    "overlap_count_tw",
    "overlap_gross_q_tw",
    "overlap_gross_q_tw_over_Q",
    "overlap_same_q_tw",
    "overlap_opp_q_tw",
    "overlap_net_signed_q_tw",
    "overlap_net_signed_q_tw_over_Q",
    "overlap_active_imbalance_tw",
)
OVERLAP_FEATURE_COLUMNS = [f"{base}_{suffix}" for suffix in FEATURE_SUFFIXES for base in FEATURE_BASES]
OPTIONAL_TARGET_COLUMNS = (COL_QV, COL_ETA, COL_VTV)


@dataclass(frozen=True)
class ResolvedPaths:
    """Resolved input and output paths for the overlap workflow."""

    dataset_name: str
    prop_path: Path
    client_path: Path
    out_dir: Path
    img_dir: Path
    config_path: Path


@dataclass(frozen=True)
class RunOptions:
    """Validated runtime options for the overlap workflow."""

    level: str
    member_nationality: Optional[str]
    trading_hours: tuple[dt.time, dt.time]
    start_bin_minutes: int
    overlap_batch_size: int
    n_jobs: int
    run_regressions: bool
    min_regression_n: int
    run_wls_regressions: bool
    wls_n_logbins: int
    wls_min_cell_n: int
    plots: bool
    write_parquet: bool


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Summary
    -------
    Run the active-overlap analysis end-to-end.

    Parameters
    ----------
    argv : Optional[Sequence[str]], default=None
        Command-line arguments excluding the executable name. When ``None``,
        arguments are read from ``sys.argv[1:]``.

    Returns
    -------
    int
        Process-style return code. Returns 0 on success.

    Notes
    -----
    The overlap environment is restricted to metaorders with the same ``ISIN``
    and ``Date``. For positive-duration target intervals, weights are
    ``overlap_ns(i, j) / duration_ns(i)``. For zero-duration targets, the
    point-in-time rule gives weight 1 when ``start_j <= start_i <= end_j``.
    ``N_JOBS=0`` uses a conservative auto setting capped at four process
    workers; set ``--n-jobs 1`` for deterministic single-process debugging.

    Examples
    --------
    >>> isinstance(main(["--dry-run"]), int)
    True
    """
    args = _parse_args(argv)
    config_path = _resolve_config_path(args.config_path)
    cfg = load_yaml_mapping(config_path)
    paths = _resolve_paths(cfg, args, config_path=config_path)
    options = _resolve_options(cfg, args)

    if args.dry_run:
        _print_dry_run(paths, options)
        return 0

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    if options.plots:
        paths.img_dir.mkdir(parents=True, exist_ok=True)

    prop = _load_group_table(paths.prop_path, group=GROUP_PROPRIETARY)
    client = _load_group_table(paths.client_path, group=GROUP_CLIENT)
    combined = pd.concat([prop, client], ignore_index=True, sort=False)
    bin_frame = _build_bin_frame(trading_hours=options.trading_hours, bin_minutes=options.start_bin_minutes)
    prepared = _prepare_metaorders(
        combined,
        bin_frame=bin_frame,
        trading_hours=options.trading_hours,
    )

    features = compute_overlap_features(
        prepared,
        batch_size=options.overlap_batch_size,
        n_jobs=options.n_jobs,
    )

    overlap_features_path: Optional[Path] = None
    if options.write_parquet:
        overlap_features_path = paths.out_dir / "overlap_features.parquet"
        features.to_parquet(overlap_features_path, index=False)

    feature_summary = _build_feature_summary(features)
    feature_summary_path = paths.out_dir / "overlap_feature_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)

    intraday_summary = _build_intraday_summary(features)
    intraday_summary_path = paths.out_dir / "overlap_intraday_summary.csv"
    intraday_summary.to_csv(intraday_summary_path, index=False)

    regression_table = _run_impact_regressions(
        features,
        run_regressions=options.run_regressions,
        min_regression_n=options.min_regression_n,
    )
    regression_path = paths.out_dir / "overlap_impact_regressions.csv"
    regression_table.to_csv(regression_path, index=False)

    wls_cells, wls_regression_table = _run_binned_wls_impact_regressions(
        features,
        run_regressions=options.run_wls_regressions,
        n_logbins=options.wls_n_logbins,
        min_cell_n=options.wls_min_cell_n,
    )
    wls_cells_path = paths.out_dir / "overlap_impact_wls_cells.csv"
    wls_regression_path = paths.out_dir / "overlap_impact_wls_regressions.csv"
    wls_cells.to_csv(wls_cells_path, index=False)
    wls_regression_table.to_csv(wls_regression_path, index=False)

    figure_outputs: dict[str, Optional[Path]] = {}
    if options.plots:
        apply_shared_plotly_style(load_plot_style())
        plot_dirs = make_plot_output_dirs(paths.img_dir, use_subdirs=True)

        distribution_fig = _build_distribution_figure(features)
        distribution_html, distribution_png = _save_plotly_figure(
            distribution_fig,
            stem="overlap_feature_distributions",
            dirs=plot_dirs,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        figure_outputs["distribution_html"] = distribution_html
        figure_outputs["distribution_png"] = distribution_png

        intraday_fig = _build_intraday_profile_figure(intraday_summary)
        intraday_html, intraday_png = _save_plotly_figure(
            intraday_fig,
            stem="overlap_intraday_profile",
            dirs=plot_dirs,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        figure_outputs["intraday_html"] = intraday_html
        figure_outputs["intraday_png"] = intraday_png

        coefficient_fig = _build_regression_coefficient_figure(regression_table)
        coefficient_html, coefficient_png = _save_plotly_figure(
            coefficient_fig,
            stem="overlap_impact_regression_proprietary_coefficient",
            dirs=plot_dirs,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        figure_outputs["regression_coefficient_html"] = coefficient_html
        figure_outputs["regression_coefficient_png"] = coefficient_png

        wls_coefficient_fig = _build_wls_regression_coefficient_figure(wls_regression_table)
        wls_coefficient_html, wls_coefficient_png = _save_plotly_figure(
            wls_coefficient_fig,
            stem="overlap_impact_wls_proprietary_coefficient",
            dirs=plot_dirs,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        figure_outputs["wls_regression_coefficient_html"] = wls_coefficient_html
        figure_outputs["wls_regression_coefficient_png"] = wls_coefficient_png

    manifest_path = _write_manifest(
        paths=paths,
        options=options,
        config=cfg,
        diagnostics=_build_diagnostics(features, regression_table, wls_regression_table),
        outputs={
            "overlap_features": overlap_features_path,
            "feature_summary": feature_summary_path,
            "intraday_summary": intraday_summary_path,
            "impact_regressions": regression_path,
            "impact_wls_cells": wls_cells_path,
            "impact_wls_regressions": wls_regression_path,
            **figure_outputs,
        },
    )

    print(f"[Crowding overlap] Rows processed: {len(features):,}")
    if overlap_features_path is not None:
        print(f"[Crowding overlap] Wrote overlap features: {overlap_features_path}")
    print(f"[Crowding overlap] Wrote feature summary: {feature_summary_path}")
    print(f"[Crowding overlap] Wrote intraday summary: {intraday_summary_path}")
    print(f"[Crowding overlap] Wrote impact regressions: {regression_path}")
    print(f"[Crowding overlap] Wrote binned WLS cells: {wls_cells_path}")
    print(f"[Crowding overlap] Wrote binned WLS regressions: {wls_regression_path}")
    if options.plots:
        print(f"[Crowding overlap] Wrote figures under: {paths.img_dir}")
    print(f"[Crowding overlap] Wrote manifest: {manifest_path}")
    return 0


def compute_overlap_features(df: pd.DataFrame, *, batch_size: int, n_jobs: int) -> pd.DataFrame:
    """
    Summary
    -------
    Compute active-overlap features for a prepared metaorder table.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared metaorder table containing ``ISIN``, ``Date``, ``Q``,
        ``Direction``, ``group``, ``_row_id``, ``_start_ns``, and ``_end_ns``.
    batch_size : int
        Number of target metaorders processed at a time within each
        ``(ISIN, Date)`` block.
    n_jobs : int
        Number of process workers. Use 1 for serial execution.

    Returns
    -------
    pd.DataFrame
        Target columns plus all overlap feature columns.

    Notes
    -----
    Work is parallelized over independent ``(ISIN, Date)`` blocks. Workers are
    top-level functions, so the ProcessPoolExecutor path remains compatible with
    multiprocessing pickling rules.

    Examples
    --------
    >>> demo = pd.DataFrame({
    ...     "ISIN": ["X"], "Date": [pd.Timestamp("2024-01-01")], "Q": [1.0],
    ...     "Direction": [1.0], "group": ["proprietary"], "_row_id": [0],
    ...     "_start_ns": [1], "_end_ns": [2],
    ... })
    >>> out = compute_overlap_features(demo, batch_size=16, n_jobs=1)
    >>> float(out["overlap_count_tw_all"].iat[0])
    0.0
    """
    _validate_required_columns(
        df,
        [COL_ISIN, COL_DATE, COL_Q, COL_DIR, COL_GROUP, "_row_id", "_start_ns", "_end_ns"],
        label="overlap_features",
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive after option resolution.")

    target_cols = _target_output_columns(df)
    base = df[target_cols].copy()
    items = [(group.copy(), int(batch_size)) for _, group in df.groupby([COL_ISIN, COL_DATE], sort=False, dropna=False)]

    if n_jobs > 1 and len(items) > 1:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as executor:
            parts = list(executor.map(_compute_overlap_features_worker, items))
    else:
        parts = [_compute_overlap_features_worker(item) for item in items]

    if parts:
        feature_frame = pd.concat(parts, ignore_index=True).sort_values("_row_id").reset_index(drop=True)
    else:
        feature_frame = pd.DataFrame(columns=["_row_id", *OVERLAP_FEATURE_COLUMNS])

    out = base.merge(feature_frame, on="_row_id", how="left", sort=False)
    for column in OVERLAP_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    out = out.sort_values("_row_id").reset_index(drop=True)
    return out.drop(columns=["_row_id"])


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute active-overlap crowding features for proprietary and client metaorders."
    )
    parser.add_argument("--config-path", type=str, default=None, help="Path to the YAML config file.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name used in path templates.")
    parser.add_argument("--level", type=str, default=None, help="Metaorder level, e.g. member or client.")
    parser.add_argument("--member-nationality", type=str, default=None, help="Optional filter suffix: it, foreign, all/null.")
    parser.add_argument("--prop-path", type=str, default=None, help="Explicit proprietary metaorder parquet.")
    parser.add_argument("--client-path", type=str, default=None, help="Explicit client metaorder parquet.")
    parser.add_argument("--output-file-path", type=str, default=None, help="Output root path template.")
    parser.add_argument("--img-output-path", type=str, default=None, help="Image root path template.")
    parser.add_argument("--analysis-tag", type=str, default=None, help="Output subfolder name.")
    parser.add_argument("--trading-hours", type=str, default=None, help="Trading window START,END.")
    parser.add_argument("--start-bin-minutes", type=int, default=None, help="Start-bin width in minutes.")
    parser.add_argument("--overlap-batch-size", type=int, default=None, help="Target batch size for overlap matrices.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Process workers; 0 means auto capped at 4.")
    parser.add_argument("--run-regressions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-regression-n", type=int, default=None, help="Minimum observations required per regression.")
    parser.add_argument("--run-wls-regressions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wls-n-logbins", type=int, default=None, help="Number of log(Q/V) bins for binned WLS.")
    parser.add_argument(
        "--wls-min-cell-n",
        type=int,
        default=None,
        help="Minimum metaorders required per WLS cell; default follows metaorder_computation MIN_COUNT.",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write-parquet", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and paths without reading data.")
    return parser.parse_args(argv)


def _resolve_config_path(raw_path: Optional[str]) -> Path:
    if raw_path is not None:
        path = Path(raw_path).expanduser()
    else:
        override = os.environ.get(CONFIG_ENV_VAR)
        path = Path(override).expanduser() if override else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _resolve_paths(
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    config_path: Path,
) -> ResolvedPaths:
    dataset_name = str(args.dataset_name or cfg.get("DATASET_NAME", "ftsemib"))
    level = str(args.level or cfg.get("LEVEL", "member"))
    member_nationality = parse_member_nationality(
        args.member_nationality if args.member_nationality is not None else cfg.get("MEMBER_NATIONALITY")
    )
    analysis_tag = str(args.analysis_tag or cfg.get("ANALYSIS_TAG", "crowding_overlap_analysis"))

    context = {"DATASET_NAME": dataset_name, "LEVEL": level}
    output_root_raw = str(args.output_file_path or cfg.get("OUTPUT_FILE_PATH", "out_files/{DATASET_NAME}"))
    img_root_raw = str(args.img_output_path or cfg.get("IMG_OUTPUT_PATH", "images/{DATASET_NAME}"))
    output_root = Path(format_path_template(output_root_raw, context))
    img_root = Path(format_path_template(img_root_raw, context))

    default_prop = output_root / with_member_nationality_tag(
        f"metaorders_info_sameday_filtered_{level}_proprietary.parquet",
        member_nationality,
    )
    default_client = output_root / with_member_nationality_tag(
        f"metaorders_info_sameday_filtered_{level}_non_proprietary.parquet",
        member_nationality,
    )

    prop_path = resolve_repo_path(_REPO_ROOT, args.prop_path or cfg.get("PROP_PATH") or default_prop)
    client_path = resolve_repo_path(_REPO_ROOT, args.client_path or cfg.get("CLIENT_PATH") or default_client)
    out_dir = resolve_repo_path(_REPO_ROOT, output_root / analysis_tag)
    img_dir = resolve_repo_path(_REPO_ROOT, img_root / analysis_tag)

    return ResolvedPaths(
        dataset_name=dataset_name,
        prop_path=prop_path,
        client_path=client_path,
        out_dir=out_dir,
        img_dir=img_dir,
        config_path=config_path,
    )


def _resolve_options(cfg: Mapping[str, Any], args: argparse.Namespace) -> RunOptions:
    level = str(args.level or cfg.get("LEVEL", "member"))
    member_nationality = parse_member_nationality(
        args.member_nationality if args.member_nationality is not None else cfg.get("MEMBER_NATIONALITY")
    )
    trading_hours = _parse_trading_hours(
        args.trading_hours if args.trading_hours is not None else cfg.get("TRADING_HOURS", ["09:30:00", "17:30:00"])
    )
    start_bin_minutes = int(
        args.start_bin_minutes if args.start_bin_minutes is not None else cfg.get("START_BIN_MINUTES", 10)
    )
    overlap_batch_size = int(
        args.overlap_batch_size if args.overlap_batch_size is not None else cfg.get("OVERLAP_BATCH_SIZE", 2048)
    )
    raw_n_jobs = int(args.n_jobs if args.n_jobs is not None else cfg.get("N_JOBS", 0))
    n_jobs = _resolve_n_jobs(raw_n_jobs)
    run_regressions = bool(cfg.get("RUN_REGRESSIONS", True) if args.run_regressions is None else args.run_regressions)
    min_regression_n = int(
        args.min_regression_n if args.min_regression_n is not None else cfg.get("MIN_REGRESSION_N", 1000)
    )
    run_wls_regressions = bool(
        cfg.get("RUN_WLS_REGRESSIONS", True) if args.run_wls_regressions is None else args.run_wls_regressions
    )
    impact_cfg = _load_impact_fit_config_defaults()
    wls_n_logbins = int(
        args.wls_n_logbins
        if args.wls_n_logbins is not None
        else cfg.get("WLS_N_LOGBINS", impact_cfg.get("N_LOGBIN", 30))
    )
    raw_wls_min_cell_n = args.wls_min_cell_n if args.wls_min_cell_n is not None else cfg.get("WLS_MIN_CELL_N")
    wls_min_cell_n = int(
        impact_cfg.get("MIN_COUNT", 20) if raw_wls_min_cell_n is None else raw_wls_min_cell_n
    )
    plots = bool(cfg.get("PLOTS", True) if args.plots is None else args.plots)
    write_parquet = bool(cfg.get("WRITE_PARQUET", True) if args.write_parquet is None else args.write_parquet)

    if start_bin_minutes <= 0:
        raise ValueError("START_BIN_MINUTES must be positive.")
    if overlap_batch_size <= 0:
        raise ValueError("OVERLAP_BATCH_SIZE must be positive.")
    if raw_n_jobs < 0:
        raise ValueError("N_JOBS must be >= 0.")
    if min_regression_n <= 0:
        raise ValueError("MIN_REGRESSION_N must be positive.")
    if wls_n_logbins <= 1:
        raise ValueError("WLS_N_LOGBINS must be greater than 1.")
    if wls_min_cell_n <= 1:
        raise ValueError("WLS_MIN_CELL_N must be greater than 1.")

    return RunOptions(
        level=level,
        member_nationality=member_nationality,
        trading_hours=trading_hours,
        start_bin_minutes=start_bin_minutes,
        overlap_batch_size=overlap_batch_size,
        n_jobs=n_jobs,
        run_regressions=run_regressions,
        min_regression_n=min_regression_n,
        run_wls_regressions=run_wls_regressions,
        wls_n_logbins=wls_n_logbins,
        wls_min_cell_n=wls_min_cell_n,
        plots=plots,
        write_parquet=write_parquet,
    )


def _resolve_n_jobs(value: int) -> int:
    if value == 0:
        return max(1, min(4, os.cpu_count() or 1))
    return int(value)


def _load_impact_fit_config_defaults() -> dict[str, object]:
    """
    Summary
    -------
    Load the canonical WLS fit defaults from ``metaorder_computation.yml``.

    Parameters
    ----------
    None.

    Returns
    -------
    dict[str, object]
        Mapping with any available impact-fit defaults, especially
        ``N_LOGBIN`` and ``MIN_COUNT``.

    Notes
    -----
    The active-overlap WLS is meant to mirror the standard power-law fit. When
    the overlap YAML does not explicitly override WLS cell settings, this helper
    keeps the bin count and minimum cell size synchronized with
    ``scripts/run_analysis.py metaorders compute``.

    Examples
    --------
    >>> isinstance(_load_impact_fit_config_defaults(), dict)
    True
    """
    path = _REPO_ROOT / "config_ymls" / "metaorder_computation.yml"
    try:
        return dict(load_yaml_mapping(path))
    except Exception:
        return {}


def _parse_trading_hours(raw_value: object) -> tuple[dt.time, dt.time]:
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (bytes, bytearray)):
        parts = [str(part).strip() for part in raw_value]
    else:
        raise TypeError("TRADING_HOURS must be a two-item sequence or a START,END string.")
    if len(parts) != 2:
        raise ValueError("TRADING_HOURS must contain exactly two values.")
    start = _parse_time_string(parts[0], label="TRADING_HOURS.start")
    end = _parse_time_string(parts[1], label="TRADING_HOURS.end")
    if start >= end:
        raise ValueError("TRADING_HOURS must satisfy start < end.")
    return start, end


def _parse_time_string(raw_value: object, *, label: str) -> dt.time:
    parsed = pd.to_datetime(str(raw_value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid time string for {label}: {raw_value!r}")
    return parsed.time()


def _load_group_table(path: Path, *, group: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {GROUP_LABELS[group]} metaorder table: {path}")
    df = pd.read_parquet(path)
    required = [COL_ISIN, COL_Q, COL_DIR]
    if COL_PERIOD not in df.columns and not {COL_START_TS, COL_END_TS}.issubset(df.columns):
        required.append(COL_PERIOD)
    _validate_required_columns(df, required, label=GROUP_LABELS[group])
    out = df.copy()
    out[COL_GROUP] = group
    out[COL_PROPRIETARY] = 1 if group == GROUP_PROPRIETARY else 0
    return out


def _prepare_metaorders(
    df: pd.DataFrame,
    *,
    bin_frame: pd.DataFrame,
    trading_hours: tuple[dt.time, dt.time],
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["_row_id"] = np.arange(len(out), dtype=np.int64)

    start_ns, end_ns = _extract_interval_ns_columns(out)
    out["_start_ns"] = start_ns
    out["_end_ns"] = end_ns
    if pd.isna(out["_start_ns"]).any() or pd.isna(out["_end_ns"]).any():
        bad = out.loc[pd.isna(out["_start_ns"]) | pd.isna(out["_end_ns"]), [COL_PERIOD]].head(3).to_dict("records")
        raise ValueError(f"Failed to parse start/end timestamps for some metaorders. Examples: {bad}")

    start_int = out["_start_ns"].astype("int64")
    end_int = out["_end_ns"].astype("int64")
    if (end_int < start_int).any():
        raise ValueError("Found metaorders with EndTimestamp earlier than StartTimestamp.")
    out[COL_START_TS] = pd.to_datetime(start_int, errors="coerce")
    out[COL_END_TS] = pd.to_datetime(end_int, errors="coerce")
    out["duration_minutes"] = (end_int.to_numpy(dtype=float) - start_int.to_numpy(dtype=float)) / NS_PER_MINUTE

    if COL_DATE in out.columns:
        out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce").dt.normalize()
        missing_date = out[COL_DATE].isna()
        out.loc[missing_date, COL_DATE] = out.loc[missing_date, COL_START_TS].dt.normalize()
    else:
        out[COL_DATE] = out[COL_START_TS].dt.normalize()
    if out[COL_DATE].isna().any():
        raise ValueError("Failed to infer Date for some metaorders.")

    out[COL_Q] = pd.to_numeric(out[COL_Q], errors="coerce")
    out[COL_DIR] = pd.to_numeric(out[COL_DIR], errors="coerce")
    if out[COL_Q].isna().any() or (~np.isfinite(out[COL_Q].to_numpy(dtype=float))).any():
        raise ValueError("Found non-finite Q values.")
    if (out[COL_Q].to_numpy(dtype=float) < 0.0).any():
        raise ValueError("Found negative Q values.")
    if out[COL_DIR].isna().any() or (~np.isfinite(out[COL_DIR].to_numpy(dtype=float))).any():
        raise ValueError("Found non-finite Direction values.")

    out[COL_IMPACT] = _build_impact_series(out)
    for column in OPTIONAL_TARGET_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[column] = np.nan

    return _attach_start_bin_columns(out, bin_frame=bin_frame, trading_hours=trading_hours)


def _extract_interval_ns_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if COL_START_TS in df.columns and COL_END_TS in df.columns:
        start_ns = df[COL_START_TS].apply(_timestamp_to_ns)
        end_ns = df[COL_END_TS].apply(_timestamp_to_ns)
        missing = start_ns.isna() | end_ns.isna()
        if (not missing.any()) or COL_PERIOD not in df.columns:
            return start_ns, end_ns

        period_parsed = df.loc[missing, COL_PERIOD].apply(_period_to_start_end_ns)
        start_ns.loc[missing] = period_parsed.apply(lambda value: value[0])
        end_ns.loc[missing] = period_parsed.apply(lambda value: value[1])
        return start_ns, end_ns

    if COL_PERIOD not in df.columns:
        raise KeyError("Missing Period or StartTimestamp/EndTimestamp columns.")
    parsed = df[COL_PERIOD].apply(_period_to_start_end_ns)
    start_ns = parsed.apply(lambda value: value[0])
    end_ns = parsed.apply(lambda value: value[1])
    return start_ns, end_ns


def _timestamp_to_ns(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else int(value.value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.datetime64):
        timestamp = pd.Timestamp(value)
        return None if pd.isna(timestamp) else int(timestamp.value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except Exception:
            timestamp = pd.to_datetime(stripped, errors="coerce")
            return None if pd.isna(timestamp) else int(timestamp.value)
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else int(timestamp.value)


def _period_to_start_end_ns(period_value: Any) -> tuple[Optional[int], Optional[int]]:
    if period_value is None:
        return None, None
    if isinstance(period_value, float) and np.isnan(period_value):
        return None, None
    if isinstance(period_value, pd.Timestamp):
        value = int(period_value.value)
        return value, value
    if isinstance(period_value, (np.integer, int)):
        value = int(period_value)
        return value, value
    if isinstance(period_value, np.datetime64):
        timestamp = pd.Timestamp(period_value)
        if pd.isna(timestamp):
            return None, None
        value = int(timestamp.value)
        return value, value
    if isinstance(period_value, (list, tuple, np.ndarray, pd.Series)):
        if len(period_value) == 0:
            return None, None
        values = list(period_value)
        start = _timestamp_to_ns(values[0])
        end = _timestamp_to_ns(values[1]) if len(values) > 1 else start
        return start, end
    if isinstance(period_value, str):
        stripped = period_value.strip()
        if not stripped:
            return None, None
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1].replace(",", " ")
            pieces = [piece.strip() for piece in inner.split() if piece.strip()]
            if not pieces:
                return None, None
            start = _timestamp_to_ns(pieces[0])
            end = _timestamp_to_ns(pieces[1]) if len(pieces) > 1 else start
            return start, end
        value = _timestamp_to_ns(stripped)
        return value, value
    return None, None


def _build_impact_series(df: pd.DataFrame) -> pd.Series:
    if COL_IMPACT in df.columns:
        impact = pd.to_numeric(df[COL_IMPACT], errors="coerce")
    else:
        impact = pd.Series(np.nan, index=df.index, dtype=float)

    if {COL_DIR, COL_PRICE_CHANGE, COL_DAILY_VOL}.issubset(df.columns):
        direction = pd.to_numeric(df[COL_DIR], errors="coerce")
        price_change = pd.to_numeric(df[COL_PRICE_CHANGE], errors="coerce")
        daily_vol = pd.to_numeric(df[COL_DAILY_VOL], errors="coerce")
        fallback = direction * price_change / daily_vol.where(daily_vol != 0.0)
        impact = impact.where(impact.notna(), fallback)
    return pd.to_numeric(impact, errors="coerce")


def _attach_start_bin_columns(
    df: pd.DataFrame,
    *,
    bin_frame: pd.DataFrame,
    trading_hours: tuple[dt.time, dt.time],
) -> pd.DataFrame:
    out = df.copy()
    start_minutes = out[COL_START_TS].apply(_timestamp_minutes_from_midnight)
    open_minutes = _time_minutes_from_midnight(trading_hours[0])
    close_minutes = _time_minutes_from_midnight(trading_hours[1])
    out["start_minutes_from_open"] = start_minutes - open_minutes
    out["inside_trading_hours"] = (
        out[COL_START_TS].notna()
        & (start_minutes >= open_minutes)
        & (start_minutes <= close_minutes)
    )

    edges = np.concatenate(
        [
            bin_frame["bin_start_minutes_from_open"].to_numpy(dtype=float),
            [float(bin_frame["bin_end_minutes_from_open"].iloc[-1])],
        ]
    )
    bin_ids = _assign_offsets_to_bins(out["start_minutes_from_open"].to_numpy(dtype=float), edges=edges)
    bin_ids[~out["inside_trading_hours"].to_numpy(dtype=bool)] = -1
    out[COL_BIN_ID] = pd.Series(bin_ids, index=out.index).where(bin_ids >= 0)
    out = out.merge(
        bin_frame[
            [
                "bin_id",
                "bin_label",
                "bin_start_time",
                "bin_end_time",
                "bin_center_minutes_from_open",
            ]
        ],
        left_on=COL_BIN_ID,
        right_on="bin_id",
        how="left",
        sort=False,
    )
    out = out.rename(
        columns={
            "bin_label": COL_BIN_LABEL,
            "bin_start_time": "start_bin_start_time",
            "bin_end_time": "start_bin_end_time",
            "bin_center_minutes_from_open": "start_bin_center_minutes_from_open",
        }
    )
    return out.drop(columns=["bin_id"])


def _compute_overlap_features_worker(payload: tuple[pd.DataFrame, int]) -> pd.DataFrame:
    group, batch_size = payload
    return _compute_overlap_features_for_group(group, batch_size=int(batch_size))


def _compute_overlap_features_for_group(group: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    n = len(group)
    row_id = group["_row_id"].to_numpy(dtype=np.int64)
    features = {column: np.zeros(n, dtype=float) for column in OVERLAP_FEATURE_COLUMNS}
    for suffix in FEATURE_SUFFIXES:
        features[f"overlap_active_imbalance_tw_{suffix}"][:] = np.nan
        features[f"overlap_gross_q_tw_over_Q_{suffix}"][:] = np.nan
        features[f"overlap_net_signed_q_tw_over_Q_{suffix}"][:] = np.nan

    if n == 0:
        return pd.DataFrame({"_row_id": row_id, **features})

    start_ns = group["_start_ns"].to_numpy(dtype=np.int64)
    end_ns = group["_end_ns"].to_numpy(dtype=np.int64)
    duration_ns = np.maximum(end_ns - start_ns, 0)
    q = group[COL_Q].to_numpy(dtype=float)
    direction = group[COL_DIR].to_numpy(dtype=float)
    direction_sign = np.sign(direction)
    is_prop = (group[COL_GROUP].to_numpy(dtype=object) == GROUP_PROPRIETARY)

    for left in range(0, n, int(batch_size)):
        right = min(left + int(batch_size), n)
        frac = _overlap_fraction_matrix(
            target_start_ns=start_ns[left:right],
            target_end_ns=end_ns[left:right],
            target_duration_ns=duration_ns[left:right],
            other_start_ns=start_ns,
            other_end_ns=end_ns,
        )
        local_rows = np.arange(right - left)
        self_cols = np.arange(left, right)
        frac[local_rows, self_cols] = 0.0

        target_q = q[left:right]
        target_dir = direction_sign[left:right]
        _fill_env_features(
            features,
            suffix="all",
            rows=slice(left, right),
            frac=frac,
            q=q,
            other_dir=direction_sign,
            target_q=target_q,
            target_dir=target_dir,
            env_mask=np.ones(n, dtype=bool),
        )
        _fill_env_features(
            features,
            suffix="prop_env",
            rows=slice(left, right),
            frac=frac,
            q=q,
            other_dir=direction_sign,
            target_q=target_q,
            target_dir=target_dir,
            env_mask=is_prop,
        )
        _fill_env_features(
            features,
            suffix="client_env",
            rows=slice(left, right),
            frac=frac,
            q=q,
            other_dir=direction_sign,
            target_q=target_q,
            target_dir=target_dir,
            env_mask=~is_prop,
        )

    return pd.DataFrame({"_row_id": row_id, **features})


def _overlap_fraction_matrix(
    *,
    target_start_ns: np.ndarray,
    target_end_ns: np.ndarray,
    target_duration_ns: np.ndarray,
    other_start_ns: np.ndarray,
    other_end_ns: np.ndarray,
) -> np.ndarray:
    start_i = np.asarray(target_start_ns, dtype=np.int64)[:, None]
    end_i = np.asarray(target_end_ns, dtype=np.int64)[:, None]
    duration_i = np.asarray(target_duration_ns, dtype=np.int64)
    start_j = np.asarray(other_start_ns, dtype=np.int64)[None, :]
    end_j = np.asarray(other_end_ns, dtype=np.int64)[None, :]

    overlap_ns = np.minimum(end_i, end_j) - np.maximum(start_i, start_j)
    overlap_ns = np.maximum(overlap_ns, 0)
    frac = np.zeros(overlap_ns.shape, dtype=float)

    positive_duration = duration_i > 0
    if np.any(positive_duration):
        frac[positive_duration, :] = (
            overlap_ns[positive_duration, :].astype(float) / duration_i[positive_duration, None].astype(float)
        )

    zero_duration = ~positive_duration
    if np.any(zero_duration):
        point_active = (start_j <= start_i[zero_duration, :]) & (start_i[zero_duration, :] <= end_j)
        frac[zero_duration, :] = point_active.astype(float)

    return frac


def _fill_env_features(
    features: dict[str, np.ndarray],
    *,
    suffix: str,
    rows: slice,
    frac: np.ndarray,
    q: np.ndarray,
    other_dir: np.ndarray,
    target_q: np.ndarray,
    target_dir: np.ndarray,
    env_mask: np.ndarray,
) -> None:
    weights = frac * env_mask.astype(float)[None, :]
    gross = weights @ q
    signed = weights @ (q * other_dir)
    pos_q = weights @ (q * (other_dir > 0.0))
    neg_q = weights @ (q * (other_dir < 0.0))

    same = np.where(target_dir > 0.0, pos_q, np.where(target_dir < 0.0, neg_q, np.nan))
    opp = np.where(target_dir > 0.0, neg_q, np.where(target_dir < 0.0, pos_q, np.nan))
    net_target_signed = target_dir * signed

    features[f"overlap_any_count_{suffix}"][rows] = (weights > 0.0).sum(axis=1).astype(float)
    features[f"overlap_count_tw_{suffix}"][rows] = weights.sum(axis=1)
    features[f"overlap_gross_q_tw_{suffix}"][rows] = gross
    features[f"overlap_same_q_tw_{suffix}"][rows] = same
    features[f"overlap_opp_q_tw_{suffix}"][rows] = opp
    features[f"overlap_net_signed_q_tw_{suffix}"][rows] = net_target_signed

    features[f"overlap_gross_q_tw_over_Q_{suffix}"][rows] = np.divide(
        gross,
        target_q,
        out=np.full_like(gross, np.nan, dtype=float),
        where=target_q > 0.0,
    )
    features[f"overlap_net_signed_q_tw_over_Q_{suffix}"][rows] = np.divide(
        net_target_signed,
        target_q,
        out=np.full_like(net_target_signed, np.nan, dtype=float),
        where=target_q > 0.0,
    )
    features[f"overlap_active_imbalance_tw_{suffix}"][rows] = np.divide(
        net_target_signed,
        gross,
        out=np.full_like(net_target_signed, np.nan, dtype=float),
        where=gross > 0.0,
    )


def _target_output_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        "_row_id",
        COL_GROUP,
        COL_PROPRIETARY,
        COL_ISIN,
        COL_DATE,
        COL_START_TS,
        COL_END_TS,
        "duration_minutes",
        COL_BIN_ID,
        COL_BIN_LABEL,
        COL_DIR,
        COL_Q,
        COL_IMPACT,
        COL_QV,
        COL_ETA,
        COL_VTV,
    ]
    optional = [
        "Member",
        "Client",
        "N Child",
        "start_bin_start_time",
        "start_bin_end_time",
        "start_bin_center_minutes_from_open",
    ]
    return [column for column in [*columns, *optional] if column in df.columns]


def _build_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_frame in features.groupby(COL_GROUP, sort=False, dropna=False):
        for metric in OVERLAP_FEATURE_COLUMNS:
            values = pd.to_numeric(group_frame[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    COL_GROUP: group,
                    "metric": metric,
                    "n": int(finite.size),
                    "missing": int(values.size - finite.size),
                    "mean": float(finite.mean()) if finite.size else np.nan,
                    "std": float(finite.std(ddof=1)) if finite.size > 1 else np.nan,
                    "min": float(finite.min()) if finite.size else np.nan,
                    "p05": float(finite.quantile(0.05)) if finite.size else np.nan,
                    "p25": float(finite.quantile(0.25)) if finite.size else np.nan,
                    "median": float(finite.median()) if finite.size else np.nan,
                    "p75": float(finite.quantile(0.75)) if finite.size else np.nan,
                    "p95": float(finite.quantile(0.95)) if finite.size else np.nan,
                    "max": float(finite.max()) if finite.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_intraday_summary(features: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overlap_count_tw_all",
        "overlap_gross_q_tw_over_Q_all",
        "overlap_active_imbalance_tw_all",
        "overlap_net_signed_q_tw_over_Q_all",
    ]
    valid = features[features[COL_BIN_ID].notna()].copy()
    if valid.empty:
        columns = [COL_GROUP, COL_BIN_ID, COL_BIN_LABEL, "start_bin_center_minutes_from_open", "metric", "n", "mean", "median"]
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    group_cols = [COL_GROUP, COL_BIN_ID, COL_BIN_LABEL, "start_bin_center_minutes_from_open"]
    for key, sub in valid.groupby(group_cols, sort=True, dropna=False):
        group, bin_id, label, center = key
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    COL_GROUP: group,
                    COL_BIN_ID: int(bin_id) if pd.notna(bin_id) else np.nan,
                    COL_BIN_LABEL: label,
                    "start_bin_center_minutes_from_open": float(center) if pd.notna(center) else np.nan,
                    "metric": metric,
                    "n": int(finite.size),
                    "mean": float(finite.mean()) if finite.size else np.nan,
                    "median": float(finite.median()) if finite.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _run_impact_regressions(
    features: pd.DataFrame,
    *,
    run_regressions: bool,
    min_regression_n: int,
) -> pd.DataFrame:
    if not run_regressions:
        return pd.DataFrame([{"model": "all", "term": "", "status": "skipped", "reason": "RUN_REGRESSIONS=false"}])

    df = features.copy()
    df["Date_str"] = pd.to_datetime(df[COL_DATE], errors="coerce").dt.strftime("%Y-%m-%d")
    df["proprietary_flag"] = pd.to_numeric(df[COL_PROPRIETARY], errors="coerce")
    df["log_qv"] = _positive_log(df[COL_QV])
    df["log_participation_rate"] = _positive_log(df[COL_ETA])
    df["log_duration_minutes"] = _positive_log(df["duration_minutes"])
    df["log1p_overlap_count_tw_all"] = _nonnegative_log1p(df["overlap_count_tw_all"])
    df["log1p_overlap_gross_q_tw_over_Q_all"] = _nonnegative_log1p(df["overlap_gross_q_tw_over_Q_all"])

    specs = [
        ("M0", "Impact ~ proprietary_flag", [COL_IMPACT, "proprietary_flag", "Date_str"]),
        (
            "M1",
            "Impact ~ proprietary_flag + log_qv + log_participation_rate + log_duration_minutes"
            " + C(start_bin_id) + C(ISIN) + C(Date_str)",
            [
                COL_IMPACT,
                "proprietary_flag",
                "log_qv",
                "log_participation_rate",
                "log_duration_minutes",
                COL_BIN_ID,
                COL_ISIN,
                "Date_str",
            ],
        ),
        (
            "M2",
            "Impact ~ proprietary_flag + log_qv + log_participation_rate + log_duration_minutes"
            " + C(start_bin_id) + C(ISIN) + C(Date_str)"
            " + log1p_overlap_count_tw_all + log1p_overlap_gross_q_tw_over_Q_all"
            " + overlap_net_signed_q_tw_over_Q_all + overlap_active_imbalance_tw_all",
            [
                COL_IMPACT,
                "proprietary_flag",
                "log_qv",
                "log_participation_rate",
                "log_duration_minutes",
                COL_BIN_ID,
                COL_ISIN,
                "Date_str",
                "log1p_overlap_count_tw_all",
                "log1p_overlap_gross_q_tw_over_Q_all",
                "overlap_net_signed_q_tw_over_Q_all",
                "overlap_active_imbalance_tw_all",
            ],
        ),
    ]

    try:
        import statsmodels.formula.api as smf
    except Exception:
        return _run_impact_regressions_sparse(df, specs=specs, min_regression_n=min_regression_n)

    rows: list[dict[str, object]] = []
    for model_name, formula, required in specs:
        reg_df = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
        n_obs = int(len(reg_df))
        n_clusters = int(reg_df["Date_str"].nunique(dropna=True)) if "Date_str" in reg_df.columns else 0
        if n_obs < int(min_regression_n):
            rows.append(
                {
                    "model": model_name,
                    "term": "",
                    "status": "skipped",
                    "reason": f"n_obs<{int(min_regression_n)}",
                    "nobs": n_obs,
                    "n_clusters": n_clusters,
                }
            )
            continue
        if n_clusters < 2:
            rows.append(
                {
                    "model": model_name,
                    "term": "",
                    "status": "skipped",
                    "reason": "n_clusters<2",
                    "nobs": n_obs,
                    "n_clusters": n_clusters,
                }
            )
            continue

        try:
            result = smf.ols(formula=formula, data=reg_df).fit(
                cov_type="cluster",
                cov_kwds={"groups": reg_df["Date_str"]},
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "term": "",
                    "status": "failed",
                    "reason": str(exc),
                    "nobs": n_obs,
                    "n_clusters": n_clusters,
                }
            )
            continue

        for term in result.params.index:
            coef = float(result.params.loc[term])
            se = float(result.bse.loc[term])
            rows.append(
                {
                    "model": model_name,
                    "term": str(term),
                    "status": "ok",
                    "reason": "",
                    "estimate": coef,
                    "std_error": se,
                    "ci95_lo": coef - 1.96 * se,
                    "ci95_hi": coef + 1.96 * se,
                    "t_value": float(result.tvalues.loc[term]),
                    "p_value": float(result.pvalues.loc[term]),
                    "nobs": int(result.nobs),
                    "n_clusters": n_clusters,
                    "r2": float(result.rsquared),
                    "adj_r2": float(result.rsquared_adj),
                    "formula": formula,
                }
            )
    return pd.DataFrame(rows)


def _run_binned_wls_impact_regressions(
    features: pd.DataFrame,
    *,
    run_regressions: bool,
    n_logbins: int,
    min_cell_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run log-binned WLS regressions aligned with the power-law impact fit."""
    cell_columns = _wls_cell_columns()
    if not run_regressions:
        return pd.DataFrame(columns=cell_columns), pd.DataFrame(
            [{"model": "all", "term": "", "status": "skipped", "reason": "RUN_WLS_REGRESSIONS=false"}]
        )

    df = features.copy()
    df["proprietary_flag"] = pd.to_numeric(df[COL_PROPRIETARY], errors="coerce")
    df["overlap_same_q_tw_over_Q_all"] = _positive_ratio(df["overlap_same_q_tw_all"], df[COL_Q])
    df["overlap_opp_q_tw_over_Q_all"] = _positive_ratio(df["overlap_opp_q_tw_all"], df[COL_Q])

    qv_values = pd.to_numeric(df[COL_QV], errors="coerce").to_numpy(dtype=float)
    try:
        qv_edges = _build_log_qv_edges(qv_values, n_logbins=int(n_logbins))
    except Exception as exc:
        return pd.DataFrame(columns=cell_columns), pd.DataFrame(
            [{"model": "all", "term": "", "status": "skipped", "reason": str(exc)}]
        )

    specs: list[tuple[str, list[str]]] = [
        ("W0", ["proprietary_flag", "log_qv"]),
        ("W1", ["proprietary_flag", "log_qv", "log_mean_participation_rate", "log_mean_duration_minutes"]),
        (
            "W2",
            [
                "proprietary_flag",
                "log_qv",
                "log_mean_participation_rate",
                "log_mean_duration_minutes",
                "log_mean_overlap_count_tw_all",
                "log_mean_overlap_same_q_tw_over_Q_all",
                "log_mean_overlap_opp_q_tw_over_Q_all",
            ],
        ),
    ]

    all_cells: list[pd.DataFrame] = []
    all_rows: list[dict[str, object]] = []
    for model_name, terms in specs:
        cells = _build_binned_wls_cells(
            df,
            model_name=model_name,
            terms=terms,
            qv_edges=qv_edges,
            min_cell_n=int(min_cell_n),
        )
        all_cells.append(cells)
        all_rows.extend(_fit_binned_wls_cells(cells, model_name=model_name, terms=terms))

    cell_table = pd.concat(all_cells, ignore_index=True, sort=False) if all_cells else pd.DataFrame(columns=cell_columns)
    regression_table = pd.DataFrame(all_rows)
    return cell_table, regression_table


def _wls_cell_columns() -> list[str]:
    return [
        "model",
        COL_GROUP,
        "proprietary_flag",
        "qv_bin_id",
        "qv_bin_left",
        "qv_bin_right",
        "center_qv",
        "log_qv",
        "mean_qv",
        "n_metaorders",
        "mean_impact",
        "std_impact",
        "sem_impact",
        "log_mean_impact",
        "weight",
        "mean_participation_rate",
        "log_mean_participation_rate",
        "mean_duration_minutes",
        "log_mean_duration_minutes",
        "mean_overlap_count_tw_all",
        "log_mean_overlap_count_tw_all",
        "mean_overlap_gross_q_tw_over_Q_all",
        "log_mean_overlap_gross_q_tw_over_Q_all",
        "mean_overlap_same_q_tw_over_Q_all",
        "log_mean_overlap_same_q_tw_over_Q_all",
        "mean_overlap_opp_q_tw_over_Q_all",
        "log_mean_overlap_opp_q_tw_over_Q_all",
    ]


def _build_log_qv_edges(qv_values: np.ndarray, *, n_logbins: int) -> np.ndarray:
    return _build_positive_log_edges(qv_values, n_logbins=n_logbins, label="Q/V")


def _build_positive_log_edges(values: np.ndarray | pd.Series, *, n_logbins: int, label: str) -> np.ndarray:
    """Build observed-range log-bin edges for strictly positive values."""
    arr = np.asarray(values, dtype=float)
    clean = arr[np.isfinite(arr) & (arr > 0.0)]
    if clean.size < 2:
        raise ValueError(f"Not enough positive {label} values for binned WLS.")
    value_min = float(np.min(clean))
    value_max = float(np.max(clean))
    if not (np.isfinite(value_min) and np.isfinite(value_max)) or value_max <= value_min:
        raise ValueError(f"Invalid {label} range for binned WLS.")
    edges = np.logspace(np.log10(value_min), np.log10(value_max), int(n_logbins) + 1)
    if np.unique(edges).size != edges.size:
        raise ValueError(f"{label} range is too narrow for the requested WLS_N_LOGBINS.")
    return edges


def _assign_qv_log_bins(qv_values: np.ndarray, *, edges: np.ndarray) -> np.ndarray:
    return _assign_positive_log_bins(qv_values, edges=edges)


def _assign_positive_log_bins(values: np.ndarray | pd.Series, *, edges: np.ndarray) -> np.ndarray:
    """Assign positive values to precomputed log bins, using -1 for invalid rows."""
    arr = np.asarray(values, dtype=float)
    bin_ids = np.searchsorted(edges, arr, side="right") - 1
    bin_ids[np.isclose(arr, edges[-1], rtol=1.0e-12, atol=0.0)] = edges.size - 2
    invalid = (~np.isfinite(arr)) | (arr <= 0.0) | (bin_ids < 0) | (bin_ids >= edges.size - 1)
    bin_ids[invalid] = -1
    return bin_ids.astype(int)


def _build_binned_wls_cells(
    df: pd.DataFrame,
    *,
    model_name: str,
    terms: Sequence[str],
    qv_edges: np.ndarray,
    min_cell_n: int,
) -> pd.DataFrame:
    # Keep the same statistical unit as the canonical impact curves: one cell is
    # a group-by-Q/V-log-bin. Additional controls summarize the composition of
    # that same curve cell; they do not redefine the binning grid.
    term_sources = {
        "log_mean_participation_rate": COL_ETA,
        "log_mean_duration_minutes": "duration_minutes",
        "log_mean_overlap_count_tw_all": "overlap_count_tw_all",
        "log_mean_overlap_gross_q_tw_over_Q_all": "overlap_gross_q_tw_over_Q_all",
        "log_mean_overlap_same_q_tw_over_Q_all": "overlap_same_q_tw_over_Q_all",
        "log_mean_overlap_opp_q_tw_over_Q_all": "overlap_opp_q_tw_over_Q_all",
    }
    term_mean_columns = {
        "log_mean_participation_rate": "mean_participation_rate",
        "log_mean_duration_minutes": "mean_duration_minutes",
        "log_mean_overlap_count_tw_all": "mean_overlap_count_tw_all",
        "log_mean_overlap_gross_q_tw_over_Q_all": "mean_overlap_gross_q_tw_over_Q_all",
        "log_mean_overlap_same_q_tw_over_Q_all": "mean_overlap_same_q_tw_over_Q_all",
        "log_mean_overlap_opp_q_tw_over_Q_all": "mean_overlap_opp_q_tw_over_Q_all",
    }
    row_required = [COL_GROUP, COL_IMPACT, COL_QV, "proprietary_flag"]
    row_required.extend(term_sources[term] for term in terms if term in term_sources)
    row_required = list(dict.fromkeys(row_required))
    _validate_required_columns(df, row_required, label=f"{model_name}_wls_cells")

    work = df[row_required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work[COL_IMPACT] = pd.to_numeric(work[COL_IMPACT], errors="coerce")
    work[COL_QV] = pd.to_numeric(work[COL_QV], errors="coerce")
    work["proprietary_flag"] = pd.to_numeric(work["proprietary_flag"], errors="coerce")
    for source in {term_sources[term] for term in terms if term in term_sources}:
        work[source] = pd.to_numeric(work[source], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work[COL_QV] > 0.0) & np.isfinite(work[COL_IMPACT])]
    if work.empty:
        return pd.DataFrame(columns=_wls_cell_columns())

    qv_bin = _assign_qv_log_bins(work[COL_QV].to_numpy(dtype=float), edges=qv_edges)
    work = work.loc[qv_bin >= 0].copy()
    work["qv_bin_id"] = qv_bin[qv_bin >= 0]
    if work.empty:
        return pd.DataFrame(columns=_wls_cell_columns())

    rows: list[dict[str, object]] = []
    grouped = work.groupby([COL_GROUP, "qv_bin_id"], sort=True, dropna=False)
    for (group, qv_bin_id), sub in grouped:
        count = int(len(sub))
        if count < int(min_cell_n):
            continue
        mean_impact = float(sub[COL_IMPACT].mean())
        std_impact = float(sub[COL_IMPACT].std(ddof=1))
        sem_impact = std_impact / math.sqrt(count) if count > 0 else np.nan
        if not (
            np.isfinite(mean_impact)
            and np.isfinite(sem_impact)
            and mean_impact > 0.0
            and sem_impact > 0.0
        ):
            continue

        bin_left = float(qv_edges[int(qv_bin_id)])
        bin_right = float(qv_edges[int(qv_bin_id) + 1])
        center_qv = math.sqrt(bin_left * bin_right)
        row: dict[str, object] = {
            "model": model_name,
            COL_GROUP: str(group),
            "proprietary_flag": float(sub["proprietary_flag"].mean()),
            "qv_bin_id": int(qv_bin_id),
            "qv_bin_left": bin_left,
            "qv_bin_right": bin_right,
            "center_qv": center_qv,
            "log_qv": math.log(center_qv),
            "mean_qv": float(sub[COL_QV].mean()),
            "n_metaorders": count,
            "mean_impact": mean_impact,
            "std_impact": std_impact,
            "sem_impact": sem_impact,
            "log_mean_impact": math.log(mean_impact),
            "weight": 1.0 / ((sem_impact / mean_impact) ** 2),
        }
        valid_row = True
        for term, source in term_sources.items():
            if source in sub.columns:
                mean_value = float(sub[source].mean())
                row[term_mean_columns[term]] = mean_value
                row[term] = math.log(mean_value) if mean_value > 0.0 and np.isfinite(mean_value) else np.nan
            if term in terms and not np.isfinite(row.get(term, np.nan)):
                valid_row = False
        if not valid_row:
            continue
        rows.append(row)

    return pd.DataFrame(rows, columns=_wls_cell_columns())


def _fit_binned_wls_cells(
    cells: pd.DataFrame,
    *,
    model_name: str,
    terms: Sequence[str],
) -> list[dict[str, object]]:
    formula = "log(mean Impact_cell) ~ " + " + ".join(terms)
    if cells.empty:
        return [_wls_skip_row(model_name, formula=formula, reason="no_retained_cells")]

    required = ["log_mean_impact", "weight", "n_metaorders", *terms]
    work = cells[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = work[(work["weight"] > 0.0) & np.isfinite(work["log_mean_impact"])]
    n_cells = int(len(work))
    param_names = ["const", *terms]
    n_params = len(param_names)
    if n_cells <= n_params:
        return [
            _wls_skip_row(
                model_name,
                formula=formula,
                reason=f"n_cells<={n_params}",
                n_cells=n_cells,
                n_metaorders=int(work["n_metaorders"].sum()) if "n_metaorders" in work else 0,
            )
        ]

    y = work["log_mean_impact"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(n_cells, dtype=float), *[work[term].to_numpy(dtype=float) for term in terms]])
    w = work["weight"].to_numpy(dtype=float)
    sqrt_w = np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
    fitted = design @ coef
    residuals = y - fitted
    dof = max(n_cells - n_params, 1)
    rss = float(np.sum(w * residuals**2))
    s2 = rss / float(dof)
    xtwx = design.T @ (w[:, None] * design)
    try:
        cov = s2 * np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        cov = s2 * np.linalg.pinv(xtwx)
    std_error = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    t_value = np.divide(coef, std_error, out=np.full_like(coef, np.nan, dtype=float), where=std_error > 0.0)
    p_value = _two_sided_t_p_values(t_value, dof=dof)
    r2 = _weighted_r2(y, fitted, w=w)
    adj_r2 = 1.0 - (1.0 - r2) * (n_cells - 1.0) / dof if np.isfinite(r2) else np.nan
    n_metaorders = int(work["n_metaorders"].sum())

    rows: list[dict[str, object]] = []
    for idx, term in enumerate(param_names):
        estimate = float(coef[idx])
        se = float(std_error[idx])
        rows.append(
            {
                "model": model_name,
                "term": term,
                "status": "ok",
                "reason": "",
                "estimate": estimate,
                "std_error": se,
                "ci95_lo": estimate - 1.96 * se if np.isfinite(se) else np.nan,
                "ci95_hi": estimate + 1.96 * se if np.isfinite(se) else np.nan,
                "t_value": float(t_value[idx]),
                "p_value": float(p_value[idx]),
                "n_cells": n_cells,
                "n_metaorders": n_metaorders,
                "r2_log": float(r2),
                "adj_r2_log": float(adj_r2),
                "formula": formula,
                "estimator": "log_binned_wls",
            }
        )
    return rows


def _wls_skip_row(
    model_name: str,
    *,
    formula: str,
    reason: str,
    n_cells: int = 0,
    n_metaorders: int = 0,
) -> dict[str, object]:
    return {
        "model": model_name,
        "term": "",
        "status": "skipped",
        "reason": reason,
        "n_cells": int(n_cells),
        "n_metaorders": int(n_metaorders),
        "formula": formula,
        "estimator": "log_binned_wls",
    }


def _weighted_r2(y: np.ndarray, y_hat: np.ndarray, *, w: np.ndarray) -> float:
    weights = np.asarray(w, dtype=float)
    target = np.asarray(y, dtype=float)
    fitted = np.asarray(y_hat, dtype=float)
    valid = np.isfinite(weights) & (weights > 0.0) & np.isfinite(target) & np.isfinite(fitted)
    if not np.any(valid):
        return np.nan
    target = target[valid]
    fitted = fitted[valid]
    weights = weights[valid]
    weighted_mean = float(np.sum(weights * target) / np.sum(weights))
    ss_res = float(np.sum(weights * (target - fitted) ** 2))
    ss_tot = float(np.sum(weights * (target - weighted_mean) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan


def _two_sided_t_p_values(t_values: np.ndarray, *, dof: int) -> np.ndarray:
    values = np.asarray(t_values, dtype=float)
    try:
        from scipy import stats

        return 2.0 * stats.t.sf(np.abs(values), df=max(int(dof), 1))
    except Exception:
        return np.asarray(
            [math.erfc(abs(float(value)) / math.sqrt(2.0)) if np.isfinite(value) else np.nan for value in values],
            dtype=float,
        )


def _run_impact_regressions_sparse(
    df: pd.DataFrame,
    *,
    specs: Sequence[tuple[str, str, list[str]]],
    min_regression_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    continuous_by_model = {
        "M0": ["proprietary_flag"],
        "M1": ["proprietary_flag", "log_qv", "log_participation_rate", "log_duration_minutes"],
        "M2": [
            "proprietary_flag",
            "log_qv",
            "log_participation_rate",
            "log_duration_minutes",
            "log1p_overlap_count_tw_all",
            "log1p_overlap_gross_q_tw_over_Q_all",
            "overlap_net_signed_q_tw_over_Q_all",
            "overlap_active_imbalance_tw_all",
        ],
    }
    categorical_by_model = {
        "M0": [],
        "M1": [COL_BIN_ID, COL_ISIN, "Date_str"],
        "M2": [COL_BIN_ID, COL_ISIN, "Date_str"],
    }

    try:
        from scipy import sparse, stats
    except Exception:
        return pd.DataFrame(
            [{"model": "all", "term": "", "status": "skipped", "reason": "statsmodels_and_scipy_unavailable"}]
        )

    for model_name, formula, required in specs:
        reg_df = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
        n_obs = int(len(reg_df))
        n_clusters = int(reg_df["Date_str"].nunique(dropna=True)) if "Date_str" in reg_df.columns else 0
        if n_obs < int(min_regression_n):
            rows.append(
                {
                    "model": model_name,
                    "term": "",
                    "status": "skipped",
                    "reason": f"n_obs<{int(min_regression_n)}",
                    "nobs": n_obs,
                    "n_clusters": n_clusters,
                    "formula": formula,
                    "estimator": "scipy_sparse_cluster_ols",
                }
            )
            continue
        if n_clusters < 2:
            rows.append(
                {
                    "model": model_name,
                    "term": "",
                    "status": "skipped",
                    "reason": "n_clusters<2",
                    "nobs": n_obs,
                    "n_clusters": n_clusters,
                    "formula": formula,
                    "estimator": "scipy_sparse_cluster_ols",
                }
            )
            continue

        try:
            fit = _fit_sparse_cluster_ols(
                reg_df,
                y_col=COL_IMPACT,
                continuous_cols=continuous_by_model[model_name],
                categorical_cols=categorical_by_model[model_name],
                cluster_col="Date_str",
                sparse_module=sparse,
                stats_module=stats,
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "term": "",
                    "status": "failed",
                    "reason": str(exc),
                    "nobs": n_obs,
                    "n_clusters": n_clusters,
                    "formula": formula,
                    "estimator": "scipy_sparse_cluster_ols",
                }
            )
            continue

        for idx, term in enumerate(fit["terms"]):
            coef = float(fit["beta"][idx])
            se = float(fit["std_error"][idx])
            rows.append(
                {
                    "model": model_name,
                    "term": str(term),
                    "status": "ok",
                    "reason": "",
                    "estimate": coef,
                    "std_error": se,
                    "ci95_lo": coef - 1.96 * se if np.isfinite(se) else np.nan,
                    "ci95_hi": coef + 1.96 * se if np.isfinite(se) else np.nan,
                    "t_value": float(fit["t_value"][idx]),
                    "p_value": float(fit["p_value"][idx]),
                    "nobs": int(fit["nobs"]),
                    "n_clusters": int(fit["n_clusters"]),
                    "r2": float(fit["r2"]),
                    "adj_r2": float(fit["adj_r2"]),
                    "formula": formula,
                    "estimator": "scipy_sparse_cluster_ols",
                }
            )
    return pd.DataFrame(rows)


def _fit_sparse_cluster_ols(
    df: pd.DataFrame,
    *,
    y_col: str,
    continuous_cols: Sequence[str],
    categorical_cols: Sequence[str],
    cluster_col: str,
    sparse_module: Any,
    stats_module: Any,
) -> dict[str, object]:
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    n_obs = int(y.size)
    blocks: list[Any] = [sparse_module.csr_matrix(np.ones((n_obs, 1), dtype=float))]
    terms: list[str] = ["Intercept"]

    if continuous_cols:
        continuous = df[list(continuous_cols)].to_numpy(dtype=float)
        blocks.append(sparse_module.csr_matrix(continuous))
        terms.extend(list(continuous_cols))

    for col in categorical_cols:
        labels = df[col].astype(str)
        codes, uniques = pd.factorize(labels, sort=True)
        if len(uniques) <= 1:
            continue
        mask = codes > 0
        rows = np.arange(n_obs, dtype=np.int64)[mask]
        cols = (codes[mask] - 1).astype(np.int64)
        data = np.ones(mask.sum(), dtype=float)
        block = sparse_module.csr_matrix((data, (rows, cols)), shape=(n_obs, len(uniques) - 1))
        blocks.append(block)
        terms.extend([f"C({col})[T.{uniques[idx]}]" for idx in range(1, len(uniques))])

    x = sparse_module.hstack(blocks, format="csr")
    xpx = (x.T @ x).toarray()
    xpy = np.asarray(x.T @ y).reshape(-1)
    xpx_inv = np.linalg.pinv(xpx)
    beta = xpx_inv @ xpy
    fitted = np.asarray(x @ beta).reshape(-1)
    resid = y - fitted

    cluster_codes, cluster_uniques = pd.factorize(df[cluster_col].astype(str), sort=True)
    n_clusters = int(len(cluster_uniques))
    p = int(x.shape[1])
    meat = np.zeros((p, p), dtype=float)
    for cluster_id in range(n_clusters):
        mask = cluster_codes == cluster_id
        score = np.asarray(x[mask, :].T @ resid[mask]).reshape(-1)
        meat += np.outer(score, score)

    rank = int(np.linalg.matrix_rank(xpx))
    df_resid = max(n_obs - rank, 1)
    correction = 1.0
    if n_clusters > 1 and n_obs > rank:
        correction = (n_clusters / (n_clusters - 1.0)) * ((n_obs - 1.0) / (n_obs - rank))
    cov = correction * (xpx_inv @ meat @ xpx_inv)
    var = np.diag(cov)
    std_error = np.sqrt(np.where(var >= 0.0, var, np.nan))
    t_value = np.divide(beta, std_error, out=np.full_like(beta, np.nan, dtype=float), where=std_error > 0.0)
    p_value = 2.0 * stats_module.t.sf(np.abs(t_value), df=max(n_clusters - 1, 1))

    ssr = float(np.sum(resid**2))
    centered = y - float(np.mean(y))
    tss = float(np.sum(centered**2))
    r2 = 1.0 - ssr / tss if tss > 0.0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1.0) / df_resid if np.isfinite(r2) else np.nan

    return {
        "terms": terms,
        "beta": beta,
        "std_error": std_error,
        "t_value": t_value,
        "p_value": p_value,
        "nobs": n_obs,
        "n_clusters": n_clusters,
        "r2": r2,
        "adj_r2": adj_r2,
    }


def _positive_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.log(numeric.where(numeric > 0.0))


def _nonnegative_log1p(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.log1p(numeric.where(numeric >= 0.0))


def _positive_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.where(den > 0.0)


def _build_distribution_figure(features: pd.DataFrame) -> go.Figure:
    metrics = [
        ("overlap_count_tw_all", "Time-weighted active count"),
        ("overlap_gross_q_tw_over_Q_all", "Gross overlap / target Q"),
        ("overlap_active_imbalance_tw_all", "Active imbalance"),
        ("overlap_net_signed_q_tw_over_Q_all", "Net signed overlap / target Q"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[label for _, label in metrics], horizontal_spacing=0.10)
    for idx, (metric, label) in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1
        for group in [GROUP_PROPRIETARY, GROUP_CLIENT]:
            values = pd.to_numeric(features.loc[features[COL_GROUP] == group, metric], errors="coerce").to_numpy(dtype=float)
            x, y = _ecdf_points(values, max_points=1200)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=GROUP_LABELS[group],
                    line=dict(color=GROUP_COLORS[group], width=2.5),
                    legendgroup=group,
                    showlegend=(idx == 0),
                    hovertemplate=f"{label}: %{{x:.4g}}<br>ECDF: %{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text=label, row=row, col=col)
        fig.update_yaxes(title_text="ECDF" if col == 1 else None, range=[0.0, 1.0], row=row, col=col)
    fig.update_layout(
        template="moimpact_white",
        width=1100,
        height=780,
        margin=dict(l=70, r=40, t=70, b=70),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.05, yanchor="bottom"),
    )
    return fig


def _ecdf_points(values: np.ndarray, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    finite.sort()
    if finite.size <= max_points:
        x = finite
        y = np.arange(1, finite.size + 1, dtype=float) / float(finite.size)
        return x, y
    probs = np.linspace(0.0, 1.0, int(max_points))
    x = np.quantile(finite, probs)
    return x, probs


def _build_intraday_profile_figure(intraday_summary: pd.DataFrame) -> go.Figure:
    metrics = [
        ("overlap_count_tw_all", "Active count"),
        ("overlap_gross_q_tw_over_Q_all", "Gross overlap / Q"),
        ("overlap_active_imbalance_tw_all", "Active imbalance"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in metrics], horizontal_spacing=0.08)
    for col_idx, (metric, label) in enumerate(metrics, start=1):
        sub_metric = intraday_summary[intraday_summary["metric"] == metric].copy()
        for group in [GROUP_PROPRIETARY, GROUP_CLIENT]:
            sub = sub_metric[sub_metric[COL_GROUP] == group].sort_values(COL_BIN_ID)
            fig.add_trace(
                go.Scatter(
                    x=sub["start_bin_center_minutes_from_open"],
                    y=sub["mean"],
                    mode="lines+markers",
                    name=f"{GROUP_LABELS[group]} mean",
                    legendgroup=group,
                    showlegend=(col_idx == 1),
                    line=dict(color=GROUP_COLORS[group], width=2.5),
                    marker=dict(size=6),
                    customdata=np.column_stack([sub[COL_BIN_LABEL], sub["n"], sub["median"]]) if not sub.empty else None,
                    hovertemplate=(
                        "Bin: %{customdata[0]}<br>"
                        "n: %{customdata[1]:,}<br>"
                        "Mean: %{y:.4g}<br>"
                        "Median: %{customdata[2]:.4g}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["start_bin_center_minutes_from_open"],
                    y=sub["median"],
                    mode="lines",
                    name=f"{GROUP_LABELS[group]} median",
                    legendgroup=f"{group}_median",
                    showlegend=False,
                    line=dict(color=GROUP_COLORS[group], width=1.8, dash="dot"),
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(title_text="Minutes from open", row=1, col=col_idx)
        fig.update_yaxes(title_text=label if col_idx == 1 else None, row=1, col=col_idx)
    fig.update_layout(
        template="moimpact_white",
        width=1150,
        height=520,
        margin=dict(l=70, r=40, t=70, b=70),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08, yanchor="bottom"),
    )
    return fig


def _build_regression_coefficient_figure(regression_table: pd.DataFrame) -> go.Figure:
    required = {"status", "term", "estimate"}
    if required.issubset(regression_table.columns):
        sub = regression_table[
            (regression_table["status"] == "ok")
            & (regression_table["term"] == "proprietary_flag")
            & regression_table["estimate"].notna()
        ].copy()
    else:
        sub = pd.DataFrame()
    fig = go.Figure()
    if not sub.empty:
        sub["model_order"] = sub["model"].map({"M0": 0, "M1": 1, "M2": 2}).fillna(99)
        sub = sub.sort_values("model_order")
        fig.add_trace(
            go.Scatter(
                x=sub["model"],
                y=sub["estimate"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(sub["ci95_hi"] - sub["estimate"]).to_numpy(dtype=float),
                    arrayminus=(sub["estimate"] - sub["ci95_lo"]).to_numpy(dtype=float),
                ),
                mode="markers+lines",
                marker=dict(color=COLOR_PROPRIETARY, size=10),
                line=dict(color=COLOR_PROPRIETARY, width=2.5),
                hovertemplate=(
                    "Model: %{x}<br>"
                    "Coefficient: %{y:.4g}<br>"
                    "95% CI: [%{customdata[0]:.4g}, %{customdata[1]:.4g}]<br>"
                    "n: %{customdata[2]:,}<extra></extra>"
                ),
                customdata=np.column_stack([sub["ci95_lo"], sub["ci95_hi"], sub["nobs"]]),
            )
        )
    fig.add_hline(y=0.0, line_color=COLOR_NEUTRAL, line_dash="dot", line_width=1.2)
    fig.update_layout(
        template="moimpact_white",
        width=800,
        height=520,
        margin=dict(l=80, r=40, t=60, b=70),
        xaxis_title="Regression model",
        yaxis_title="Proprietary coefficient",
        showlegend=False,
    )
    return fig


def _build_wls_regression_coefficient_figure(wls_regression_table: pd.DataFrame) -> go.Figure:
    required = {"status", "term", "estimate"}
    if required.issubset(wls_regression_table.columns):
        sub = wls_regression_table[
            (wls_regression_table["status"] == "ok")
            & (wls_regression_table["term"] == "proprietary_flag")
            & wls_regression_table["estimate"].notna()
        ].copy()
    else:
        sub = pd.DataFrame()
    fig = go.Figure()
    if not sub.empty:
        sub["model_order"] = sub["model"].map({"W0": 0, "W1": 1, "W2": 2}).fillna(99)
        sub = sub.sort_values("model_order")
        multiplicative = np.expm1(sub["estimate"].to_numpy(dtype=float))
        fig.add_trace(
            go.Scatter(
                x=sub["model"],
                y=sub["estimate"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(sub["ci95_hi"] - sub["estimate"]).to_numpy(dtype=float),
                    arrayminus=(sub["estimate"] - sub["ci95_lo"]).to_numpy(dtype=float),
                ),
                mode="markers+lines",
                marker=dict(color=COLOR_PROPRIETARY, size=10),
                line=dict(color=COLOR_PROPRIETARY, width=2.5),
                hovertemplate=(
                    "Model: %{x}<br>"
                    "Log coefficient: %{y:.4g}<br>"
                    "Multiplicative effect: %{customdata[0]:.2%}<br>"
                    "95% CI: [%{customdata[1]:.4g}, %{customdata[2]:.4g}]<br>"
                    "cells: %{customdata[3]:,}<br>"
                    "metaorders: %{customdata[4]:,}<extra></extra>"
                ),
                customdata=np.column_stack(
                    [multiplicative, sub["ci95_lo"], sub["ci95_hi"], sub["n_cells"], sub["n_metaorders"]]
                ),
            )
        )
    fig.add_hline(y=0.0, line_color=COLOR_NEUTRAL, line_dash="dot", line_width=1.2)
    fig.update_layout(
        template="moimpact_white",
        width=800,
        height=520,
        margin=dict(l=80, r=40, t=60, b=70),
        xaxis_title="Binned WLS model",
        yaxis_title="Proprietary coefficient in log mean impact",
        showlegend=False,
    )
    return fig


def _build_diagnostics(
    features: pd.DataFrame,
    regression_table: pd.DataFrame,
    wls_regression_table: pd.DataFrame,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "n_metaorders": int(len(features)),
        "n_dates": int(pd.to_datetime(features[COL_DATE], errors="coerce").nunique(dropna=True)),
        "n_isins": int(features[COL_ISIN].nunique(dropna=True)),
        "group_counts": {str(k): int(v) for k, v in features[COL_GROUP].value_counts(dropna=False).items()},
    }
    prop_coeff = regression_table[
        (regression_table.get("status") == "ok") & (regression_table.get("term") == "proprietary_flag")
    ]
    if not prop_coeff.empty:
        diagnostics["proprietary_coefficients"] = {
            str(row["model"]): float(row["estimate"]) for _, row in prop_coeff.iterrows()
        }
    wls_prop_coeff = wls_regression_table[
        (wls_regression_table.get("status") == "ok") & (wls_regression_table.get("term") == "proprietary_flag")
    ]
    if not wls_prop_coeff.empty:
        diagnostics["wls_proprietary_coefficients"] = {
            str(row["model"]): float(row["estimate"]) for _, row in wls_prop_coeff.iterrows()
        }
    return diagnostics


def _write_manifest(
    *,
    paths: ResolvedPaths,
    options: RunOptions,
    config: Mapping[str, Any],
    diagnostics: Mapping[str, object],
    outputs: Mapping[str, Optional[Path]],
) -> Path:
    manifest = {
        "script": str(Path(__file__).relative_to(_REPO_ROOT)),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "command": sys.argv,
        "dataset_name": paths.dataset_name,
        "config_path": str(paths.config_path),
        "input_paths": {
            "proprietary": str(paths.prop_path),
            "client": str(paths.client_path),
        },
        "output_dir": str(paths.out_dir),
        "img_dir": str(paths.img_dir),
        "options": {
            "level": options.level,
            "member_nationality": options.member_nationality,
            "trading_hours": [item.isoformat() for item in options.trading_hours],
            "start_bin_minutes": options.start_bin_minutes,
            "overlap_batch_size": options.overlap_batch_size,
            "n_jobs": options.n_jobs,
            "run_regressions": options.run_regressions,
            "min_regression_n": options.min_regression_n,
            "run_wls_regressions": options.run_wls_regressions,
            "wls_n_logbins": options.wls_n_logbins,
            "wls_min_cell_n": options.wls_min_cell_n,
            "plots": options.plots,
            "write_parquet": options.write_parquet,
        },
        "outputs": {key: None if value is None else str(value) for key, value in outputs.items()},
        "diagnostics": diagnostics,
        "config": dict(config),
    }
    manifest_path = paths.out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest_path


def _git_commit_hash() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def _validate_required_columns(df: pd.DataFrame, required: Sequence[str], *, label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"[{label}] Missing required columns: {missing}")


def _build_bin_frame(*, trading_hours: tuple[dt.time, dt.time], bin_minutes: int) -> pd.DataFrame:
    open_minutes = _time_minutes_from_midnight(trading_hours[0])
    close_minutes = _time_minutes_from_midnight(trading_hours[1])
    duration = close_minutes - open_minutes
    if duration <= 0.0:
        raise ValueError("trading_hours must describe a positive same-day interval.")
    edges = np.arange(0.0, duration, float(bin_minutes))
    edges = np.concatenate([edges, [duration]])
    edges = np.unique(np.round(edges, decimals=10))
    left = edges[:-1]
    right = edges[1:]
    center = 0.5 * (left + right)
    return pd.DataFrame(
        {
            "bin_id": np.arange(left.size, dtype=int),
            "bin_start_time": [_format_time_from_open(trading_hours[0], value) for value in left],
            "bin_end_time": [_format_time_from_open(trading_hours[0], value) for value in right],
            "bin_label": [
                f"{_format_time_from_open(trading_hours[0], l)}-{_format_time_from_open(trading_hours[0], r)}"
                for l, r in zip(left, right)
            ],
            "bin_start_minutes_from_open": left,
            "bin_end_minutes_from_open": right,
            "bin_center_minutes_from_open": center,
        }
    )


def _assign_offsets_to_bins(offsets: np.ndarray, *, edges: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=float)
    edges = np.asarray(edges, dtype=float)
    bin_ids = np.searchsorted(edges, offsets, side="right") - 1
    close_hits = np.isclose(offsets, edges[-1], rtol=0.0, atol=1.0e-9)
    bin_ids[close_hits] = edges.size - 2
    invalid = (~np.isfinite(offsets)) | (offsets < edges[0]) | (offsets > edges[-1])
    bin_ids[invalid] = -1
    return bin_ids.astype(int)


def _timestamp_minutes_from_midnight(timestamp: pd.Timestamp) -> float:
    if pd.isna(timestamp):
        return np.nan
    return _time_minutes_from_midnight(timestamp.time())


def _time_minutes_from_midnight(value: dt.time) -> float:
    return value.hour * 60.0 + value.minute + value.second / 60.0 + value.microsecond / 60_000_000.0


def _format_time_from_open(open_time: dt.time, minutes_from_open: float) -> str:
    base = dt.datetime.combine(dt.date(2000, 1, 1), open_time)
    timestamp = base + dt.timedelta(minutes=float(minutes_from_open))
    return timestamp.strftime("%H:%M")


def _print_dry_run(paths: ResolvedPaths, options: RunOptions) -> None:
    print("[Crowding overlap] Dry run")
    print(f"  DATASET_NAME={paths.dataset_name}")
    print(f"  LEVEL={options.level}")
    print(f"  MEMBER_NATIONALITY={options.member_nationality or 'all'}")
    print(f"  PROP_PATH={paths.prop_path}")
    print(f"  CLIENT_PATH={paths.client_path}")
    print(f"  OUT_DIR={paths.out_dir}")
    print(f"  IMG_DIR={paths.img_dir}")
    print(f"  TRADING_HOURS={[item.isoformat() for item in options.trading_hours]}")
    print(f"  START_BIN_MINUTES={options.start_bin_minutes}")
    print(f"  OVERLAP_BATCH_SIZE={options.overlap_batch_size}")
    print(f"  N_JOBS={options.n_jobs}")
    print(f"  RUN_REGRESSIONS={options.run_regressions}")
    print(f"  MIN_REGRESSION_N={options.min_regression_n}")
    print(f"  RUN_WLS_REGRESSIONS={options.run_wls_regressions}")
    print(f"  WLS_N_LOGBINS={options.wls_n_logbins}")
    print(f"  WLS_MIN_CELL_N={options.wls_min_cell_n}")
    print(f"  PLOTS={options.plots}")
    print(f"  WRITE_PARQUET={options.write_parquet}")


if __name__ == "__main__":
    raise SystemExit(main())
