#!/usr/bin/env python3
"""
Compare intraday start-time distributions for proprietary and client metaorders.
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
    COLOR_PROPRIETARY,
    PlotOutputDirs,
    make_plot_output_dirs,
    save_plotly_figure as _save_plotly_figure,
)


CONFIG_ENV_VAR = "METAORDER_START_TIME_DISTRIBUTION_CONFIG"
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config_ymls" / "metaorder_start_time_distribution.yml"

COL_PERIOD = "Period"
COL_GROUP = "group"
COL_GROUP_LABEL = "group_label"
GROUP_PROPRIETARY = "proprietary"
GROUP_CLIENT = "client"
GROUP_LABELS = {GROUP_PROPRIETARY: "Proprietary", GROUP_CLIENT: "Client"}
GROUP_COLORS = {GROUP_PROPRIETARY: COLOR_PROPRIETARY, GROUP_CLIENT: COLOR_CLIENT}
Y_METRIC_COLUMNS = {
    "share": "share_metaorders",
    "count": "n_metaorders",
    "mean_daily_count": "mean_daily_n_metaorders",
}
Y_METRIC_LABELS = {
    "share": "Share of metaorders",
    "count": "Number of metaorders",
    "mean_daily_count": "Mean daily number of metaorders",
}


@dataclass(frozen=True)
class ResolvedPaths:
    """Resolved paths used by the start-time distribution workflow."""

    dataset_name: str
    prop_path: Path
    client_path: Path
    out_dir: Path
    img_dir: Path
    config_path: Path


@dataclass(frozen=True)
class RunOptions:
    """Validated run options for intraday start-time binning and output."""

    level: str
    member_nationality: Optional[str]
    trading_hours: tuple[dt.time, dt.time]
    bin_minutes: int
    y_metric: str
    plots: bool
    write_parquet: bool


def save_plotly_figure(fig: go.Figure, *args: Any, **kwargs: Any) -> tuple[Optional[Path], Optional[Path]]:
    """
    Summary
    -------
    Save a Plotly figure after clearing the top-level title.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to save.
    *args, **kwargs
        Forwarded to `moimpact.plotting.save_plotly_figure`.

    Returns
    -------
    tuple[Optional[Path], Optional[Path]]
        HTML and PNG output paths, when those exports are requested and succeed.

    Notes
    -----
    The repository's figure scripts avoid top-level titles in exported paper
    figures and rely on captions/filenames instead.

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
    Run the proprietary-vs-client intraday start-time distribution workflow.

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
    The script reads already identified metaorders from filtered parquet tables.
    It does not re-identify metaorders and does not modify the input files.

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

    prop_df = _load_group_table(paths.prop_path, group=GROUP_PROPRIETARY)
    client_df = _load_group_table(paths.client_path, group=GROUP_CLIENT)
    distribution = _build_combined_distribution_table(
        prop_df,
        client_df,
        trading_hours=options.trading_hours,
        bin_minutes=options.bin_minutes,
    )

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = paths.out_dir / f"start_time_distribution_{options.bin_minutes}min.csv"
    distribution.to_csv(csv_path, index=False)

    parquet_path: Optional[Path] = None
    if options.write_parquet:
        parquet_path = paths.out_dir / f"start_time_distribution_{options.bin_minutes}min.parquet"
        distribution.to_parquet(parquet_path, index=False)

    html_path: Optional[Path] = None
    png_path: Optional[Path] = None
    if options.plots:
        apply_shared_plotly_style(load_plot_style())
        plot_dirs = make_plot_output_dirs(paths.img_dir, use_subdirs=True)
        fig = _build_distribution_figure(
            distribution,
            bin_minutes=options.bin_minutes,
            trading_hours=options.trading_hours,
            y_metric=options.y_metric,
        )
        stem = f"metaorder_start_time_distribution_{options.bin_minutes}min_{options.y_metric}"
        html_path, png_path = save_plotly_figure(
            fig,
            stem=stem,
            dirs=plot_dirs,
            write_html=True,
            write_png=True,
            strict_png=False,
        )

    manifest_path = _write_manifest(
        paths=paths,
        options=options,
        config=cfg,
        outputs={
            "csv": csv_path,
            "parquet": parquet_path,
            "html": html_path,
            "png": png_path,
        },
        diagnostics=_diagnostics_from_distribution(distribution),
    )

    print(f"[Start-time distribution] Wrote table: {csv_path}")
    if parquet_path is not None:
        print(f"[Start-time distribution] Wrote parquet: {parquet_path}")
    if html_path is not None:
        print(f"[Start-time distribution] Wrote HTML figure: {html_path}")
    if png_path is not None:
        print(f"[Start-time distribution] Wrote PNG figure: {png_path}")
    print(f"[Start-time distribution] Wrote manifest: {manifest_path}")
    return 0


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate identified metaorders by intraday start-time bins and compare "
            "proprietary vs client distributions."
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
    parser.add_argument("--bin-minutes", type=int, default=None, help="Intraday bin width in minutes.")
    parser.add_argument(
        "--trading-hours",
        type=str,
        default=None,
        help="Trading window as START,END, e.g. 09:30:00,17:30:00.",
    )
    parser.add_argument(
        "--y-metric",
        choices=sorted(Y_METRIC_COLUMNS),
        default=None,
        help="Metric plotted on the y-axis.",
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
    analysis_tag = str(args.analysis_tag or cfg.get("ANALYSIS_TAG", "metaorder_start_time_distribution"))

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

    prop_cfg = cfg.get("PROP_PATH", cfg.get("PROPRIETARY_PATH"))
    client_cfg = cfg.get("CLIENT_PATH")
    prop_path = resolve_repo_path(_REPO_ROOT, args.prop_path or prop_cfg or default_prop)
    client_path = resolve_repo_path(_REPO_ROOT, args.client_path or client_cfg or default_client)
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
    if bin_minutes <= 0:
        raise ValueError("BIN_MINUTES must be a positive integer.")

    y_metric = str(args.y_metric or cfg.get("Y_METRIC", "share")).strip().lower()
    if y_metric not in Y_METRIC_COLUMNS:
        allowed = ", ".join(sorted(Y_METRIC_COLUMNS))
        raise ValueError(f"Y_METRIC must be one of: {allowed}.")

    plots = bool(cfg.get("PLOTS", True) if args.plots is None else args.plots)
    write_parquet = bool(cfg.get("WRITE_PARQUET", True) if args.write_parquet is None else args.write_parquet)
    return RunOptions(
        level=level,
        member_nationality=member_nationality,
        trading_hours=trading_hours,
        bin_minutes=bin_minutes,
        y_metric=y_metric,
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
    if COL_PERIOD not in df.columns:
        raise KeyError(f"Missing required column {COL_PERIOD!r} in {path}")
    out = df.copy()
    out[COL_GROUP] = group
    out[COL_GROUP_LABEL] = GROUP_LABELS[group]
    return out


def _build_combined_distribution_table(
    proprietary: pd.DataFrame,
    client: pd.DataFrame,
    *,
    trading_hours: tuple[dt.time, dt.time],
    bin_minutes: int,
) -> pd.DataFrame:
    bin_frame = _build_bin_frame(trading_hours=trading_hours, bin_minutes=bin_minutes)
    pieces = [
        _build_group_distribution_table(
            proprietary,
            group=GROUP_PROPRIETARY,
            bin_frame=bin_frame,
            trading_hours=trading_hours,
        ),
        _build_group_distribution_table(
            client,
            group=GROUP_CLIENT,
            bin_frame=bin_frame,
            trading_hours=trading_hours,
        ),
    ]
    return pd.concat(pieces, ignore_index=True)


def _build_group_distribution_table(
    df: pd.DataFrame,
    *,
    group: str,
    bin_frame: pd.DataFrame,
    trading_hours: tuple[dt.time, dt.time],
) -> pd.DataFrame:
    starts = _attach_start_bin_columns(df, bin_frame=bin_frame, trading_hours=trading_hours)
    inside = starts[starts["inside_trading_hours"] & starts["start_bin_id"].notna()].copy()
    n_bins = len(bin_frame)
    counts = np.zeros(n_bins, dtype=int)
    if not inside.empty:
        bin_ids = inside["start_bin_id"].to_numpy(dtype=int)
        counts = np.bincount(bin_ids, minlength=n_bins).astype(int)

    total_inside = int(counts.sum())
    n_dates = int(inside["start_date"].nunique()) if total_inside > 0 else 0
    shares = counts / total_inside if total_inside > 0 else np.zeros(n_bins, dtype=float)
    mean_daily = counts / n_dates if n_dates > 0 else np.zeros(n_bins, dtype=float)

    out = bin_frame.copy()
    out[COL_GROUP] = group
    out[COL_GROUP_LABEL] = GROUP_LABELS[group]
    out["n_metaorders"] = counts
    out["share_metaorders"] = shares
    out["mean_daily_n_metaorders"] = mean_daily
    out["n_input_metaorders"] = int(len(df))
    out["n_valid_start_times"] = int(starts["StartTimestamp"].notna().sum())
    out["n_inside_trading_hours"] = total_inside
    out["n_outside_trading_hours"] = int(starts["StartTimestamp"].notna().sum() - total_inside)
    out["n_observed_start_dates"] = n_dates
    return out


def _attach_start_bin_columns(
    df: pd.DataFrame,
    *,
    bin_frame: pd.DataFrame,
    trading_hours: tuple[dt.time, dt.time],
) -> pd.DataFrame:
    if COL_PERIOD not in df.columns:
        raise KeyError(f"Missing required column {COL_PERIOD!r} in the metaorder info table.")

    out = df.copy()
    start_ns = out[COL_PERIOD].apply(lambda value: _period_endpoint_ns(value, 0))
    out["StartTimestamp"] = pd.to_datetime(start_ns, errors="coerce")
    out["start_date"] = out["StartTimestamp"].dt.date
    start_minutes = out["StartTimestamp"].apply(_timestamp_minutes_from_midnight)
    open_minutes = _time_minutes_from_midnight(trading_hours[0])
    close_minutes = _time_minutes_from_midnight(trading_hours[1])
    out["start_minutes_from_midnight"] = start_minutes
    out["start_minutes_from_open"] = start_minutes - open_minutes
    out["inside_trading_hours"] = (
        out["StartTimestamp"].notna()
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
    out["start_bin_id"] = pd.Series(bin_ids, index=out.index).where(bin_ids >= 0)
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


def _build_bin_frame(
    *,
    trading_hours: tuple[dt.time, dt.time],
    bin_minutes: int,
) -> pd.DataFrame:
    if bin_minutes <= 0:
        raise ValueError("bin_minutes must be positive.")
    open_minutes = _time_minutes_from_midnight(trading_hours[0])
    close_minutes = _time_minutes_from_midnight(trading_hours[1])
    duration = close_minutes - open_minutes
    if duration <= 0:
        raise ValueError("trading_hours must describe a positive same-day interval.")

    edges = np.arange(0.0, duration, float(bin_minutes))
    edges = np.concatenate([edges, [duration]])
    edges = np.unique(np.round(edges, decimals=10))
    if edges.size < 2:
        raise ValueError("At least one intraday bin is required.")

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
    # Include starts exactly at the market close in the final interval. The
    # trading-hour filter is inclusive at the right edge, matching the rest of
    # the repository's trade-time filters.
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


def _build_distribution_figure(
    distribution: pd.DataFrame,
    *,
    bin_minutes: int,
    trading_hours: tuple[dt.time, dt.time],
    y_metric: str,
) -> go.Figure:
    style = load_plot_style()
    y_col = Y_METRIC_COLUMNS[y_metric]
    y_label = Y_METRIC_LABELS[y_metric]
    max_y = float(distribution[y_col].max()) if not distribution.empty else 0.0
    y_upper = max(max_y * 1.10, 1.0 if y_metric == "share" else max_y + 1.0)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[GROUP_LABELS[GROUP_PROPRIETARY], GROUP_LABELS[GROUP_CLIENT]],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    for col_idx, group in enumerate([GROUP_PROPRIETARY, GROUP_CLIENT], start=1):
        sub = distribution[distribution[COL_GROUP] == group].sort_values("bin_id")
        custom = np.column_stack(
            [
                sub["bin_label"].to_numpy(dtype=object),
                sub["n_metaorders"].to_numpy(dtype=object),
                sub["share_metaorders"].to_numpy(dtype=object),
                sub["mean_daily_n_metaorders"].to_numpy(dtype=object),
            ]
        )
        fig.add_trace(
            go.Bar(
                x=sub["bin_center_minutes_from_open"],
                y=sub[y_col],
                width=max(float(bin_minutes) * 0.82, 0.5),
                marker_color=GROUP_COLORS[group],
                customdata=custom,
                hovertemplate=(
                    "Bin: %{customdata[0]}<br>"
                    "Count: %{customdata[1]:,}<br>"
                    "Share: %{customdata[2]:.2%}<br>"
                    "Mean daily count: %{customdata[3]:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

    tick_vals, tick_text = _hourly_ticks(trading_hours=trading_hours)
    for col_idx in [1, 2]:
        fig.update_xaxes(
            title_text="Start time",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            range=[0, _time_minutes_from_midnight(trading_hours[1]) - _time_minutes_from_midnight(trading_hours[0])],
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(title_text=y_label if col_idx == 1 else None, range=[0, y_upper], row=1, col=col_idx)

    fig.update_layout(
        template="moimpact_white",
        width=style.export_width,
        height=style.export_height,
        bargap=0.08,
        margin=dict(l=80, r=40, t=70, b=80),
        font=dict(size=style.tick_font_size),
    )
    fig.update_annotations(font_size=style.title_font_size)
    return fig


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


def _diagnostics_from_distribution(distribution: pd.DataFrame) -> dict[str, dict[str, int]]:
    diagnostics: dict[str, dict[str, int]] = {}
    for group, sub in distribution.groupby(COL_GROUP, sort=False):
        if sub.empty:
            continue
        first = sub.iloc[0]
        diagnostics[str(group)] = {
            "n_input_metaorders": int(first["n_input_metaorders"]),
            "n_valid_start_times": int(first["n_valid_start_times"]),
            "n_inside_trading_hours": int(first["n_inside_trading_hours"]),
            "n_outside_trading_hours": int(first["n_outside_trading_hours"]),
            "n_observed_start_dates": int(first["n_observed_start_dates"]),
        }
    return diagnostics


def _write_manifest(
    *,
    paths: ResolvedPaths,
    options: RunOptions,
    config: Mapping[str, Any],
    outputs: Mapping[str, Optional[Path]],
    diagnostics: Mapping[str, Mapping[str, int]],
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
            "y_metric": options.y_metric,
            "plots": options.plots,
            "write_parquet": options.write_parquet,
        },
        "config": dict(config),
        "outputs": {key: None if value is None else str(value) for key, value in outputs.items()},
        "diagnostics": diagnostics,
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
    print("[Start-time distribution] Dry run")
    print(f"  DATASET_NAME={paths.dataset_name}")
    print(f"  LEVEL={options.level}")
    print(f"  MEMBER_NATIONALITY={options.member_nationality or 'all'}")
    print(f"  PROP_PATH={paths.prop_path}")
    print(f"  CLIENT_PATH={paths.client_path}")
    print(f"  OUT_DIR={paths.out_dir}")
    print(f"  IMG_DIR={paths.img_dir}")
    print(f"  TRADING_HOURS={[item.isoformat() for item in options.trading_hours]}")
    print(f"  BIN_MINUTES={options.bin_minutes}")
    print(f"  Y_METRIC={options.y_metric}")
    print(f"  PLOTS={options.plots}")
    print(f"  WRITE_PARQUET={options.write_parquet}")


if __name__ == "__main__":
    raise SystemExit(main())
