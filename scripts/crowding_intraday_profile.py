#!/usr/bin/env python3
"""
Intraday profile of aligned all-vs-all crowding for proprietary and client metaorders.
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
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure repository-root imports work when running from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
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
    PlotOutputDirs,
    make_plot_output_dirs,
    save_plotly_figure as _save_plotly_figure,
)


CONFIG_ENV_VAR = "CROWDING_INTRADAY_PROFILE_CONFIG"
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config_ymls" / "crowding_intraday_profile.yml"

COL_PERIOD = "Period"
COL_ISIN = "ISIN"
COL_DATE = "Date"
COL_DIR = "Direction"
COL_Q = "Q"
COL_GROUP = "group"
COL_GROUP_LABEL = "group_label"
COL_START_TS = "StartTimestamp"
COL_BIN_ID = "start_bin_id"
COL_BIN_LABEL = "bin_label"
COL_ALIGNED = "aligned_crowding_intraday_all"
COL_IMBALANCE = "imbalance_all_others_intraday"
GROUP_PROPRIETARY = "proprietary"
GROUP_CLIENT = "client"
GROUP_LABELS = {GROUP_PROPRIETARY: "Proprietary", GROUP_CLIENT: "Client"}
GROUP_COLORS = {GROUP_PROPRIETARY: COLOR_PROPRIETARY, GROUP_CLIENT: COLOR_CLIENT}


@dataclass(frozen=True)
class ResolvedPaths:
    """Resolved input and output paths for the intraday crowding workflow."""

    dataset_name: str
    prop_path: Path
    client_path: Path
    out_dir: Path
    img_dir: Path
    config_path: Path


@dataclass(frozen=True)
class RunOptions:
    """Validated runtime options for the intraday crowding workflow."""

    level: str
    member_nationality: Optional[str]
    trading_hours: tuple[dt.time, dt.time]
    bin_minutes: int
    min_n_day_bin: int
    alpha: float
    bootstrap_runs: int
    seed: int
    plots: bool
    write_parquet: bool


def save_plotly_figure(fig: go.Figure, *args: Any, **kwargs: Any) -> tuple[Optional[Path], Optional[Path]]:
    """
    Summary
    -------
    Save a Plotly figure after removing the top-level title.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to save.
    *args, **kwargs
        Forwarded to `moimpact.plotting.save_plotly_figure`.

    Returns
    -------
    tuple[Optional[Path], Optional[Path]]
        HTML and PNG output paths for the requested exports.

    Notes
    -----
    Repository figures are exported without top-level titles so panel labels and
    captions stay external and consistent across paper workflows.

    Examples
    --------
    >>> isinstance(make_plot_output_dirs(Path("images/demo")), PlotOutputDirs)
    True
    """
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Summary
    -------
    Build and export the intraday aligned-crowding profile for prop and client flow.

    Parameters
    ----------
    argv : Optional[Sequence[str]], default=None
        Command-line arguments excluding the program name. When `None`, uses
        `sys.argv[1:]`.

    Returns
    -------
    int
        Process-style exit code. Returns 0 on success.

    Notes
    -----
    The aligned crowding is computed within `(ISIN, Date, 10-minute start bin)`
    cells using a leave-one-out all-vs-all imbalance:
    `aligned_i = Direction_i * imbalance_{-i}`.
    The script then aggregates to one row per `(group, Date, bin)` and averages
    those day-level means and medians across dates so each trading day gets
    equal weight.

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

    prop = _load_group_table(paths.prop_path, group=GROUP_PROPRIETARY)
    client = _load_group_table(paths.client_path, group=GROUP_CLIENT)
    bin_frame = _build_bin_frame(trading_hours=options.trading_hours, bin_minutes=options.bin_minutes)

    prepared = pd.concat(
        [
            _attach_start_bin_columns(prop, bin_frame=bin_frame, trading_hours=options.trading_hours),
            _attach_start_bin_columns(client, bin_frame=bin_frame, trading_hours=options.trading_hours),
        ],
        ignore_index=True,
    )
    prepared = prepared[prepared["inside_trading_hours"] & prepared[COL_BIN_ID].notna()].reset_index(drop=True)
    prepared = _compute_intraday_all_others_crowding(prepared)

    day_panel = _build_day_bin_panel(prepared)
    summary = _build_summary_table(
        day_panel,
        bin_frame=bin_frame,
        min_n_day_bin=options.min_n_day_bin,
        alpha=options.alpha,
        bootstrap_runs=options.bootstrap_runs,
        seed=options.seed,
    )

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    day_panel_path = paths.out_dir / f"day_bin_panel_{options.bin_minutes}min.csv"
    summary_path = paths.out_dir / f"intraday_profile_summary_{options.bin_minutes}min.csv"
    day_panel.to_csv(day_panel_path, index=False)
    summary.to_csv(summary_path, index=False)

    day_panel_parquet_path: Optional[Path] = None
    summary_parquet_path: Optional[Path] = None
    if options.write_parquet:
        day_panel_parquet_path = paths.out_dir / f"day_bin_panel_{options.bin_minutes}min.parquet"
        summary_parquet_path = paths.out_dir / f"intraday_profile_summary_{options.bin_minutes}min.parquet"
        day_panel.to_parquet(day_panel_parquet_path, index=False)
        summary.to_parquet(summary_parquet_path, index=False)

    profile_html_path: Optional[Path] = None
    profile_png_path: Optional[Path] = None
    mean_heatmap_html_path: Optional[Path] = None
    mean_heatmap_png_path: Optional[Path] = None
    median_heatmap_html_path: Optional[Path] = None
    median_heatmap_png_path: Optional[Path] = None
    if options.plots:
        apply_shared_plotly_style(load_plot_style())
        plot_dirs = make_plot_output_dirs(paths.img_dir, use_subdirs=True)
        fig = _build_profile_figure(
            summary,
            trading_hours=options.trading_hours,
            bin_minutes=options.bin_minutes,
        )
        stem = f"crowding_intraday_profile_{options.bin_minutes}min"
        profile_html_path, profile_png_path = save_plotly_figure(
            fig,
            stem=stem,
            dirs=plot_dirs,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        mean_heatmap = _build_day_bin_heatmap_figure(
            day_panel,
            metric="mean_aligned_crowding",
            metric_label="Daily mean aligned crowding",
            trading_hours=options.trading_hours,
        )
        mean_heatmap_html_path, mean_heatmap_png_path = save_plotly_figure(
            mean_heatmap,
            stem=f"crowding_intraday_profile_heatmap_mean_{options.bin_minutes}min",
            dirs=plot_dirs,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        median_heatmap = _build_day_bin_heatmap_figure(
            day_panel,
            metric="median_aligned_crowding",
            metric_label="Daily median aligned crowding",
            trading_hours=options.trading_hours,
        )
        median_heatmap_html_path, median_heatmap_png_path = save_plotly_figure(
            median_heatmap,
            stem=f"crowding_intraday_profile_heatmap_median_{options.bin_minutes}min",
            dirs=plot_dirs,
            write_html=True,
            write_png=True,
            strict_png=False,
        )

    manifest_path = _write_manifest(
        paths=paths,
        options=options,
        outputs={
            "day_panel_csv": day_panel_path,
            "summary_csv": summary_path,
            "day_panel_parquet": day_panel_parquet_path,
            "summary_parquet": summary_parquet_path,
            "profile_html": profile_html_path,
            "profile_png": profile_png_path,
            "mean_heatmap_html": mean_heatmap_html_path,
            "mean_heatmap_png": mean_heatmap_png_path,
            "median_heatmap_html": median_heatmap_html_path,
            "median_heatmap_png": median_heatmap_png_path,
        },
        diagnostics=_diagnostics_from_summary(summary),
        config=cfg,
    )

    print(f"[Crowding intraday] Wrote day-bin panel: {day_panel_path}")
    print(f"[Crowding intraday] Wrote summary table: {summary_path}")
    if day_panel_parquet_path is not None:
        print(f"[Crowding intraday] Wrote day-bin parquet: {day_panel_parquet_path}")
    if summary_parquet_path is not None:
        print(f"[Crowding intraday] Wrote summary parquet: {summary_parquet_path}")
    if profile_html_path is not None:
        print(f"[Crowding intraday] Wrote profile HTML figure: {profile_html_path}")
    if profile_png_path is not None:
        print(f"[Crowding intraday] Wrote profile PNG figure: {profile_png_path}")
    if mean_heatmap_html_path is not None:
        print(f"[Crowding intraday] Wrote mean-heatmap HTML figure: {mean_heatmap_html_path}")
    if mean_heatmap_png_path is not None:
        print(f"[Crowding intraday] Wrote mean-heatmap PNG figure: {mean_heatmap_png_path}")
    if median_heatmap_html_path is not None:
        print(f"[Crowding intraday] Wrote median-heatmap HTML figure: {median_heatmap_html_path}")
    if median_heatmap_png_path is not None:
        print(f"[Crowding intraday] Wrote median-heatmap PNG figure: {median_heatmap_png_path}")
    print(f"[Crowding intraday] Wrote manifest: {manifest_path}")
    return 0


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute an intraday profile of aligned all-vs-all crowding using "
            "10-minute start bins and day-equal weighting."
        )
    )
    parser.add_argument("--config-path", type=str, default=None, help="Path to the YAML config file.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name used in path templates.")
    parser.add_argument("--level", type=str, default=None, help="Metaorder level, e.g. member or client.")
    parser.add_argument(
        "--member-nationality",
        type=str,
        default=None,
        help="Optional member-nationality suffix: it, foreign, all/null.",
    )
    parser.add_argument("--output-file-path", type=str, default=None, help="Output root path template.")
    parser.add_argument("--img-output-path", type=str, default=None, help="Image root path template.")
    parser.add_argument("--analysis-tag", type=str, default=None, help="Output subfolder name.")
    parser.add_argument("--prop-path", type=str, default=None, help="Explicit proprietary metaorder parquet.")
    parser.add_argument("--client-path", type=str, default=None, help="Explicit client metaorder parquet.")
    parser.add_argument("--bin-minutes", type=int, default=None, help="Intraday start-bin width in minutes.")
    parser.add_argument(
        "--trading-hours",
        type=str,
        default=None,
        help="Trading window as START,END, e.g. 09:30:00,17:30:00.",
    )
    parser.add_argument("--min-n-day-bin", type=int, default=None, help="Minimum valid metaorders per day-bin.")
    parser.add_argument("--alpha", type=float, default=None, help="Significance level for bootstrap CIs.")
    parser.add_argument("--bootstrap-runs", type=int, default=None, help="Date-cluster bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for bootstrap reproducibility.")
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
    analysis_tag = str(args.analysis_tag or cfg.get("ANALYSIS_TAG", "crowding_intraday_profile"))

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
    bin_minutes = int(args.bin_minutes if args.bin_minutes is not None else cfg.get("BIN_MINUTES", 10))
    min_n_day_bin = int(args.min_n_day_bin if args.min_n_day_bin is not None else cfg.get("MIN_N_DAY_BIN", 5))
    alpha = float(args.alpha if args.alpha is not None else cfg.get("ALPHA", 0.05))
    bootstrap_runs = int(args.bootstrap_runs if args.bootstrap_runs is not None else cfg.get("BOOTSTRAP_RUNS", 1000))
    seed = int(args.seed if args.seed is not None else cfg.get("SEED", 0))
    plots = bool(cfg.get("PLOTS", True) if args.plots is None else args.plots)
    write_parquet = bool(cfg.get("WRITE_PARQUET", True) if args.write_parquet is None else args.write_parquet)

    if bin_minutes <= 0:
        raise ValueError("BIN_MINUTES must be a positive integer.")
    if min_n_day_bin <= 0:
        raise ValueError("MIN_N_DAY_BIN must be a positive integer.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("ALPHA must lie in (0, 1).")
    if bootstrap_runs < 0:
        raise ValueError("BOOTSTRAP_RUNS must be >= 0.")

    return RunOptions(
        level=level,
        member_nationality=member_nationality,
        trading_hours=trading_hours,
        bin_minutes=bin_minutes,
        min_n_day_bin=min_n_day_bin,
        alpha=alpha,
        bootstrap_runs=bootstrap_runs,
        seed=seed,
        plots=plots,
        write_parquet=write_parquet,
    )


def _parse_trading_hours(raw_value: object) -> tuple[dt.time, dt.time]:
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (bytes, bytearray)):
        parts = [str(part).strip() for part in raw_value]
    else:
        raise TypeError("TRADING_HOURS must be a two-item sequence or a START,END string.")
    if len(parts) != 2:
        raise ValueError("TRADING_HOURS must contain exactly two values: start and end.")
    start = _parse_time_string(parts[0], label="TRADING_HOURS.start")
    end = _parse_time_string(parts[1], label="TRADING_HOURS.end")
    if start >= end:
        raise ValueError("TRADING_HOURS must satisfy start < end within the same day.")
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
    required = [COL_ISIN, COL_Q, COL_DIR, COL_PERIOD]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    out = df.copy()
    out[COL_GROUP] = group
    out[COL_GROUP_LABEL] = GROUP_LABELS[group]
    return out


def _attach_start_bin_columns(
    df: pd.DataFrame,
    *,
    bin_frame: pd.DataFrame,
    trading_hours: tuple[dt.time, dt.time],
) -> pd.DataFrame:
    out = df.copy()
    start_ns = out[COL_PERIOD].apply(lambda value: _period_endpoint_ns(value, 0))
    out[COL_START_TS] = pd.to_datetime(start_ns, errors="coerce")

    if COL_DATE in out.columns:
        out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce").dt.normalize()
        missing_date = out[COL_DATE].isna()
        out.loc[missing_date, COL_DATE] = out.loc[missing_date, COL_START_TS].dt.normalize()
    else:
        out[COL_DATE] = out[COL_START_TS].dt.normalize()

    out[COL_Q] = pd.to_numeric(out[COL_Q], errors="coerce")
    out[COL_DIR] = pd.to_numeric(out[COL_DIR], errors="coerce")
    if out[COL_Q].isna().any():
        raise ValueError("Found non-numeric Q values after parsing.")
    if out[COL_DIR].isna().any():
        raise ValueError("Found non-numeric Direction values after parsing.")

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
        bin_frame[["bin_id", "bin_label", "bin_start_time", "bin_end_time", "bin_center_minutes_from_open"]],
        left_on=COL_BIN_ID,
        right_on="bin_id",
        how="left",
        sort=False,
    )
    return out


def _period_endpoint_ns(period_value: Any, index: int) -> Optional[int]:
    if period_value is None:
        return None
    if isinstance(period_value, float) and np.isnan(period_value):
        return None
    if isinstance(period_value, (list, tuple, np.ndarray)):
        if len(period_value) <= index:
            return None
        try:
            return int(period_value[index])
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
            if len(inner) <= index:
                return None
            try:
                return int(inner[index])
            except Exception:
                return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _compute_intraday_all_others_crowding(df: pd.DataFrame) -> pd.DataFrame:
    required = [COL_ISIN, COL_DATE, COL_BIN_ID, COL_Q, COL_DIR]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for intraday crowding: {missing}")

    out = df.copy()
    out["__Q__"] = out[COL_Q].to_numpy(dtype=float)
    out["__QD__"] = out["__Q__"] * out[COL_DIR].to_numpy(dtype=float)

    grouped = out.groupby([COL_ISIN, COL_DATE, COL_BIN_ID], dropna=False, sort=False)
    total_q = grouped["__Q__"].transform("sum")
    total_qd = grouped["__QD__"].transform("sum")
    denom = total_q - out["__Q__"]
    numer = total_qd - out["__QD__"]

    # The intraday profile should reflect only flow available in the same
    # `(ISIN, Date, start-bin)` neighborhood, not future starts from later bins.
    out[COL_IMBALANCE] = np.where(denom > 0.0, numer / denom, np.nan)
    out[COL_ALIGNED] = out[COL_DIR].to_numpy(dtype=float) * out[COL_IMBALANCE].to_numpy(dtype=float)
    return out.drop(columns=["__Q__", "__QD__"])


def _build_day_bin_panel(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[np.isfinite(df[COL_ALIGNED])].copy()
    summary = (
        valid.groupby(
            [
                COL_GROUP,
                COL_GROUP_LABEL,
                COL_DATE,
                COL_BIN_ID,
                COL_BIN_LABEL,
                "bin_start_time",
                "bin_end_time",
                "bin_center_minutes_from_open",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            n_valid_metaorders=(COL_ALIGNED, "size"),
            mean_aligned_crowding=(COL_ALIGNED, "mean"),
            median_aligned_crowding=(COL_ALIGNED, "median"),
        )
        .reset_index()
    )

    counts_all = (
        df.groupby(
            [
                COL_GROUP,
                COL_GROUP_LABEL,
                COL_DATE,
                COL_BIN_ID,
                COL_BIN_LABEL,
                "bin_start_time",
                "bin_end_time",
                "bin_center_minutes_from_open",
            ],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("n_metaorders")
        .reset_index()
    )
    out = counts_all.merge(
        summary,
        on=[
            COL_GROUP,
            COL_GROUP_LABEL,
            COL_DATE,
            COL_BIN_ID,
            COL_BIN_LABEL,
            "bin_start_time",
            "bin_end_time",
            "bin_center_minutes_from_open",
        ],
        how="left",
        sort=True,
    )
    out["n_valid_metaorders"] = out["n_valid_metaorders"].fillna(0).astype(int)
    return out


def _build_summary_table(
    day_panel: pd.DataFrame,
    *,
    bin_frame: pd.DataFrame,
    min_n_day_bin: int,
    alpha: float,
    bootstrap_runs: int,
    seed: int,
) -> pd.DataFrame:
    eligible = day_panel[
        (day_panel["n_valid_metaorders"] >= int(min_n_day_bin))
        & np.isfinite(day_panel["mean_aligned_crowding"])
        & np.isfinite(day_panel["median_aligned_crowding"])
    ].copy()

    rows: list[dict[str, object]] = []
    for group in [GROUP_PROPRIETARY, GROUP_CLIENT]:
        group_panel = eligible[eligible[COL_GROUP] == group]
        for bin_id in bin_frame["bin_id"].tolist():
            meta = bin_frame.loc[bin_frame["bin_id"] == bin_id].iloc[0]
            sub = group_panel[group_panel[COL_BIN_ID] == bin_id].sort_values(COL_DATE)
            daily_mean = sub["mean_aligned_crowding"].to_numpy(dtype=float)
            daily_median = sub["median_aligned_crowding"].to_numpy(dtype=float)
            date_codes = sub[COL_DATE].to_numpy()

            mean_ci_lo, mean_ci_hi = _date_cluster_mean_ci(
                daily_mean,
                date_codes,
                alpha=alpha,
                bootstrap_runs=bootstrap_runs,
                seed=seed + int(bin_id),
            )
            median_ci_lo, median_ci_hi = _date_cluster_mean_ci(
                daily_median,
                date_codes,
                alpha=alpha,
                bootstrap_runs=bootstrap_runs,
                seed=seed + 10_000 + int(bin_id),
            )

            rows.append(
                {
                    COL_GROUP: group,
                    COL_GROUP_LABEL: GROUP_LABELS[group],
                    "bin_id": int(bin_id),
                    COL_BIN_LABEL: str(meta["bin_label"]),
                    "bin_start_time": str(meta["bin_start_time"]),
                    "bin_end_time": str(meta["bin_end_time"]),
                    "bin_center_minutes_from_open": float(meta["bin_center_minutes_from_open"]),
                    "n_days": int(len(sub)),
                    "n_metaorders_total": int(sub["n_metaorders"].sum()) if not sub.empty else 0,
                    "n_valid_metaorders_total": int(sub["n_valid_metaorders"].sum()) if not sub.empty else 0,
                    "mean_daily_n_metaorders": float(sub["n_metaorders"].mean()) if not sub.empty else np.nan,
                    "mean_daily_n_valid_metaorders": (
                        float(sub["n_valid_metaorders"].mean()) if not sub.empty else np.nan
                    ),
                    # The estimand is the arithmetic mean of the day-level means
                    # and medians, so each day contributes one vote per bin.
                    "avg_daily_mean_aligned_crowding": float(np.mean(daily_mean)) if daily_mean.size else np.nan,
                    "avg_daily_median_aligned_crowding": float(np.mean(daily_median)) if daily_median.size else np.nan,
                    "ci_avg_daily_mean_lo": mean_ci_lo,
                    "ci_avg_daily_mean_hi": mean_ci_hi,
                    "ci_avg_daily_median_lo": median_ci_lo,
                    "ci_avg_daily_median_hi": median_ci_hi,
                    "min_n_day_bin": int(min_n_day_bin),
                }
            )
    return pd.DataFrame(rows)


def _date_cluster_mean_ci(
    values: np.ndarray,
    dates: np.ndarray,
    *,
    alpha: float,
    bootstrap_runs: int,
    seed: int,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    date_arr = np.asarray(dates)
    mask = np.isfinite(data)
    data = data[mask]
    date_arr = date_arr[mask]
    if data.size == 0:
        return np.nan, np.nan
    unique_dates = pd.Index(date_arr).unique().to_numpy()
    if bootstrap_runs <= 0 or unique_dates.size < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    draws = np.empty(int(bootstrap_runs), dtype=float)
    for rep in range(int(bootstrap_runs)):
        sampled_dates = rng.choice(unique_dates, size=unique_dates.size, replace=True)
        sampled_values: list[np.ndarray] = []
        for sampled_date in sampled_dates:
            sampled_values.append(data[date_arr == sampled_date])
        draws[rep] = float(np.mean(np.concatenate(sampled_values)))

    valid = draws[np.isfinite(draws)]
    if valid.size == 0:
        return np.nan, np.nan
    return float(np.quantile(valid, alpha / 2.0)), float(np.quantile(valid, 1.0 - alpha / 2.0))


def _build_profile_figure(
    summary: pd.DataFrame,
    *,
    trading_hours: tuple[dt.time, dt.time],
    bin_minutes: int,
) -> go.Figure:
    _ensure_plotly_style()
    style = load_plot_style()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[GROUP_LABELS[GROUP_PROPRIETARY], GROUP_LABELS[GROUP_CLIENT]],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    for col_idx, group in enumerate([GROUP_PROPRIETARY, GROUP_CLIENT], start=1):
        sub = summary[summary[COL_GROUP] == group].sort_values("bin_id")
        x = sub["bin_center_minutes_from_open"].to_numpy(dtype=float)
        hover = np.column_stack(
            [
                sub[COL_BIN_LABEL].to_numpy(dtype=object),
                sub["n_days"].to_numpy(dtype=object),
                sub["mean_daily_n_valid_metaorders"].to_numpy(dtype=object),
                sub["avg_daily_mean_aligned_crowding"].to_numpy(dtype=object),
                sub["avg_daily_median_aligned_crowding"].to_numpy(dtype=object),
            ]
        )

        _add_ci_band(
            fig,
            x=x,
            lo=sub["ci_avg_daily_mean_lo"].to_numpy(dtype=float),
            hi=sub["ci_avg_daily_mean_hi"].to_numpy(dtype=float),
            color=_rgba(GROUP_COLORS[group], 0.18),
            row=1,
            col=col_idx,
        )
        _add_ci_band(
            fig,
            x=x,
            lo=sub["ci_avg_daily_median_lo"].to_numpy(dtype=float),
            hi=sub["ci_avg_daily_median_hi"].to_numpy(dtype=float),
            color="rgba(107, 114, 128, 0.14)",
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sub["avg_daily_mean_aligned_crowding"],
                mode="lines+markers",
                name="Avg daily mean",
                line=dict(color=GROUP_COLORS[group], width=3),
                marker=dict(size=7),
                customdata=hover,
                hovertemplate=(
                    "Bin: %{customdata[0]}<br>"
                    "Informative days: %{customdata[1]:,}<br>"
                    "Mean daily valid n: %{customdata[2]:.2f}<br>"
                    "Avg daily mean: %{customdata[3]:.3f}<br>"
                    "Avg daily median: %{customdata[4]:.3f}<extra></extra>"
                ),
                showlegend=(col_idx == 1),
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sub["avg_daily_median_aligned_crowding"],
                mode="lines+markers",
                name="Avg daily median",
                line=dict(color=COLOR_NEUTRAL, width=2.5, dash="dash"),
                marker=dict(size=6),
                customdata=hover,
                hovertemplate=(
                    "Bin: %{customdata[0]}<br>"
                    "Informative days: %{customdata[1]:,}<br>"
                    "Mean daily valid n: %{customdata[2]:.2f}<br>"
                    "Avg daily mean: %{customdata[3]:.3f}<br>"
                    "Avg daily median: %{customdata[4]:.3f}<extra></extra>"
                ),
                showlegend=(col_idx == 1),
            ),
            row=1,
            col=col_idx,
        )

    tick_vals, tick_text = _hourly_ticks(trading_hours=trading_hours)
    max_x = _time_minutes_from_midnight(trading_hours[1]) - _time_minutes_from_midnight(trading_hours[0])
    for col_idx in [1, 2]:
        fig.update_xaxes(
            title_text="Start time",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[0.0, max_x],
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            title_text="Aligned crowding" if col_idx == 1 else None,
            range=[-1.0, 1.0],
            row=1,
            col=col_idx,
        )
        fig.add_hline(y=0.0, line_color=COLOR_NEUTRAL, line_width=1.2, line_dash="dot", row=1, col=col_idx)

    fig.update_layout(
        template="moimpact_white",
        width=style.export_width,
        height=style.export_height,
        margin=dict(l=80, r=40, t=70, b=80),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08, yanchor="bottom"),
    )
    fig.update_annotations(font_size=style.title_font_size)
    return fig


def _build_day_bin_heatmap_figure(
    day_panel: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    trading_hours: tuple[dt.time, dt.time],
) -> go.Figure:
    _ensure_plotly_style()
    style = load_plot_style()
    all_dates = sorted(pd.to_datetime(day_panel[COL_DATE], errors="coerce").dropna().dt.normalize().unique())
    y_values = [pd.Timestamp(date).strftime("%Y-%m-%d") for date in all_dates]
    bin_meta = (
        day_panel[[COL_BIN_ID, "bin_center_minutes_from_open"]]
        .drop_duplicates()
        .sort_values(COL_BIN_ID)
    )
    bin_ids = bin_meta[COL_BIN_ID].to_numpy(dtype=int)
    x_values = bin_meta["bin_center_minutes_from_open"].to_numpy(dtype=float)
    tick_vals, tick_text = _hourly_ticks(trading_hours=trading_hours)
    max_x = _time_minutes_from_midnight(trading_hours[1]) - _time_minutes_from_midnight(trading_hours[0])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[GROUP_LABELS[GROUP_PROPRIETARY], GROUP_LABELS[GROUP_CLIENT]],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    for col_idx, group in enumerate([GROUP_PROPRIETARY, GROUP_CLIENT], start=1):
        sub = day_panel[day_panel[COL_GROUP] == group].copy()
        sub[COL_DATE] = pd.to_datetime(sub[COL_DATE], errors="coerce").dt.normalize()
        value_map = (
            sub.pivot_table(
                index=COL_DATE,
                columns=COL_BIN_ID,
                values=metric,
                aggfunc="first",
                sort=True,
            )
            .reindex(index=all_dates, columns=bin_ids)
        )
        n_map = (
            sub.pivot_table(
                index=COL_DATE,
                columns=COL_BIN_ID,
                values="n_valid_metaorders",
                aggfunc="first",
                sort=True,
            )
            .reindex(index=all_dates, columns=bin_ids)
        )
        total_n_map = (
            sub.pivot_table(
                index=COL_DATE,
                columns=COL_BIN_ID,
                values="n_metaorders",
                aggfunc="first",
                sort=True,
            )
            .reindex(index=all_dates, columns=bin_ids)
        )
        custom = np.stack(
            [
                n_map.to_numpy(dtype=float),
                total_n_map.to_numpy(dtype=float),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Heatmap(
                x=x_values,
                y=y_values,
                z=value_map.to_numpy(dtype=float),
                colorscale="RdBu",
                reversescale=True,
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title=metric_label) if col_idx == 2 else None,
                customdata=custom,
                hovertemplate=(
                    "Date: %{y}<br>"
                    "Bin center: %{x:.1f} min from open<br>"
                    f"{metric_label}: %{{z:.3f}}<br>"
                    "Valid n: %{customdata[0]:.0f}<br>"
                    "Total n: %{customdata[1]:.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )

    for col_idx in [1, 2]:
        fig.update_xaxes(
            title_text="Start time",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[0.0, max_x],
            row=1,
            col=col_idx,
        )
    fig.update_yaxes(
        title_text="Date",
        autorange="reversed",
        row=1,
        col=1,
    )
    fig.update_layout(
        template="moimpact_white",
        width=style.export_width,
        height=max(style.export_height, min(2400, 14 * max(len(y_values), 1))),
        margin=dict(l=90, r=50, t=70, b=80),
    )
    fig.update_annotations(font_size=style.title_font_size)
    return fig


def _ensure_plotly_style() -> None:
    try:
        apply_shared_plotly_style(load_plot_style())
    except Exception:
        return None


def _add_ci_band(
    fig: go.Figure,
    *,
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    color: str,
    row: int,
    col: int,
) -> None:
    mask = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(mask):
        return
    x_plot = x[mask]
    lo_plot = lo[mask]
    hi_plot = hi[mask]
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_plot, x_plot[::-1]]),
            y=np.concatenate([hi_plot, lo_plot[::-1]]),
            fill="toself",
            fillcolor=color,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _build_bin_frame(
    *,
    trading_hours: tuple[dt.time, dt.time],
    bin_minutes: int,
) -> pd.DataFrame:
    open_minutes = _time_minutes_from_midnight(trading_hours[0])
    close_minutes = _time_minutes_from_midnight(trading_hours[1])
    duration = close_minutes - open_minutes
    if duration <= 0:
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
    return (
        value.hour * 60.0
        + value.minute
        + value.second / 60.0
        + value.microsecond / 60_000_000.0
    )


def _format_time_from_open(open_time: dt.time, minutes_from_open: float) -> str:
    base = dt.datetime.combine(dt.date(2000, 1, 1), open_time)
    timestamp = base + dt.timedelta(minutes=float(minutes_from_open))
    return timestamp.strftime("%H:%M")


def _hourly_ticks(*, trading_hours: tuple[dt.time, dt.time]) -> tuple[list[float], list[str]]:
    open_minutes = _time_minutes_from_midnight(trading_hours[0])
    close_minutes = _time_minutes_from_midnight(trading_hours[1])
    first_hour = int(np.ceil(open_minutes / 60.0) * 60)
    ticks_abs = [open_minutes]
    ticks_abs.extend(float(value) for value in range(first_hour, int(close_minutes) + 1, 60))
    if close_minutes not in ticks_abs:
        ticks_abs.append(close_minutes)
    ticks = sorted({float(value) for value in ticks_abs if open_minutes <= value <= close_minutes})
    labels = [_format_time_from_open(dt.time(0, 0), value) for value in ticks]
    return [value - open_minutes for value in ticks], labels


def _rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return f"rgba(107, 114, 128, {alpha:.3f})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def _diagnostics_from_summary(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, dict[str, float]] = {}
    for group, sub in summary.groupby(COL_GROUP, sort=False):
        diagnostics[str(group)] = {
            "n_bins": int(len(sub)),
            "n_bins_with_support": int((sub["n_days"] > 0).sum()),
            "max_n_days": int(sub["n_days"].max()) if not sub.empty else 0,
            "mean_daily_valid_n_across_supported_bins": (
                float(sub.loc[sub["n_days"] > 0, "mean_daily_n_valid_metaorders"].mean())
                if (sub["n_days"] > 0).any()
                else float("nan")
            ),
        }
    return diagnostics


def _write_manifest(
    *,
    paths: ResolvedPaths,
    options: RunOptions,
    outputs: Mapping[str, Optional[Path]],
    diagnostics: Mapping[str, Mapping[str, float]],
    config: Mapping[str, Any],
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
            "bin_minutes": options.bin_minutes,
            "min_n_day_bin": options.min_n_day_bin,
            "alpha": options.alpha,
            "bootstrap_runs": options.bootstrap_runs,
            "seed": options.seed,
            "plots": options.plots,
            "write_parquet": options.write_parquet,
        },
        "outputs": {key: None if value is None else str(value) for key, value in outputs.items()},
        "diagnostics": diagnostics,
        "config": dict(config),
    }
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
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


def _print_dry_run(paths: ResolvedPaths, options: RunOptions) -> None:
    print("[Crowding intraday] Dry run")
    print(f"  DATASET_NAME={paths.dataset_name}")
    print(f"  LEVEL={options.level}")
    print(f"  MEMBER_NATIONALITY={options.member_nationality or 'all'}")
    print(f"  PROP_PATH={paths.prop_path}")
    print(f"  CLIENT_PATH={paths.client_path}")
    print(f"  OUT_DIR={paths.out_dir}")
    print(f"  IMG_DIR={paths.img_dir}")
    print(f"  TRADING_HOURS={[item.isoformat() for item in options.trading_hours]}")
    print(f"  BIN_MINUTES={options.bin_minutes}")
    print(f"  MIN_N_DAY_BIN={options.min_n_day_bin}")
    print(f"  ALPHA={options.alpha}")
    print(f"  BOOTSTRAP_RUNS={options.bootstrap_runs}")
    print(f"  SEED={options.seed}")
    print(f"  PLOTS={options.plots}")
    print(f"  WRITE_PARQUET={options.write_parquet}")


if __name__ == "__main__":
    raise SystemExit(main())
