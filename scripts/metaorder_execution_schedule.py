#!/usr/bin/env python3
"""
Execution-schedule comparison for proprietary and client metaorders.

The script expects the fit-filtered metaorder tables produced by
`metaorder_computation.py` to contain two packed path columns:

- `child_time_norm`
- `child_volume_fraction`

It builds:

- a cumulative execution-schedule comparison with mean curves, SEM bands, and
  a TWAP benchmark;
- a side-by-side heatmap of the cumulative schedule density.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# Ensure repository-root imports work when running from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.config import (
    cfg_require,
    format_path_template,
    load_yaml_mapping,
    resolve_opt_repo_path,
    resolve_repo_path,
)
from moimpact.logging_utils import PrintTee, setup_file_logger
from moimpact.metaorder_distribution_samples import parse_member_nationality, with_member_nationality_tag
from moimpact.plot_style import (
    THEME_BG_COLOR,
    THEME_COLORWAY,
    THEME_FONT_FAMILY,
    THEME_GRID_COLOR,
    apply_plotly_style,
)
from moimpact.plotting import (
    COLOR_BAND_CLIENT,
    COLOR_BAND_PROPRIETARY,
    COLOR_CLIENT,
    COLOR_NEUTRAL,
    COLOR_PROPRIETARY,
    PlotOutputDirs,
    ensure_plot_dirs,
    make_plot_output_dirs,
    save_plotly_figure as _save_plotly_figure,
)


_CONFIG_ENV_VAR = "METAORDER_EXECUTION_SCHEDULE_CONFIG"
_config_override = os.environ.get(_CONFIG_ENV_VAR)
if _config_override:
    _CONFIG_PATH = Path(_config_override).expanduser()
    if not _CONFIG_PATH.is_absolute():
        _CONFIG_PATH = (_REPO_ROOT / _CONFIG_PATH).resolve()
else:
    _CONFIG_PATH = _REPO_ROOT / "config_ymls" / "metaorder_execution_schedule.yml"
_CFG = load_yaml_mapping(_CONFIG_PATH)


def _cfg_require(key: str) -> Any:
    return cfg_require(_CFG, key, _CONFIG_PATH)


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_repo_path(_REPO_ROOT, value)


def _format_path_template(template: str, context: Mapping[str, str]) -> str:
    return format_path_template(template, context)


def save_plotly_figure(fig, *args, **kwargs):
    """Save a Plotly figure after removing the top-level title."""
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)


def _default_info_path(
    output_root: Path,
    level: str,
    proprietary: bool,
    member_nationality: Optional[str],
) -> Path:
    group_tag = "proprietary" if proprietary else "non_proprietary"
    return output_root / with_member_nationality_tag(
        f"metaorders_info_sameday_filtered_{level}_{group_tag}.parquet",
        member_nationality,
    )


def _unpack_path(blob: Optional[bytes | bytearray | memoryview | list[float] | np.ndarray]) -> Optional[np.ndarray]:
    """Unpack the float32 path blobs written by `metaorder_computation.py`."""
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return np.frombuffer(blob, dtype=np.float32)
    return np.asarray(blob, dtype=np.float32)


@dataclass(frozen=True)
class GroupScheduleSummary:
    group: str
    display_name: str
    n_input_rows: int
    n_valid_metaorders: int
    tau_grid: np.ndarray
    mean_curve: np.ndarray
    sem_curve: np.ndarray
    n_eff_curve: np.ndarray
    heatmap_counts: np.ndarray
    heatmap_density: np.ndarray
    y_bin_edges: np.ndarray
    skipped_reasons: dict[str, int]


TICK_FONT_SIZE = int(_cfg_require("TICK_FONT_SIZE"))
LABEL_FONT_SIZE = int(_cfg_require("LABEL_FONT_SIZE"))
TITLE_FONT_SIZE = int(_cfg_require("TITLE_FONT_SIZE"))
LEGEND_FONT_SIZE = int(_cfg_require("LEGEND_FONT_SIZE"))
apply_plotly_style(
    tick_font_size=TICK_FONT_SIZE,
    label_font_size=LABEL_FONT_SIZE,
    title_font_size=TITLE_FONT_SIZE,
    legend_font_size=LEGEND_FONT_SIZE,
    theme_colorway=THEME_COLORWAY,
    theme_grid_color=THEME_GRID_COLOR,
    theme_bg_color=THEME_BG_COLOR,
    theme_font_family=THEME_FONT_FAMILY,
)

DATASET_NAME = str(_CFG.get("DATASET_NAME") or "ftsemib")
LEVEL = str(_cfg_require("LEVEL"))
MEMBER_NATIONALITY = parse_member_nationality(_CFG.get("MEMBER_NATIONALITY"))
MEMBER_NATIONALITY_TAG = MEMBER_NATIONALITY or "all"
RUN_EXECUTION_SCHEDULE = bool(_cfg_require("RUN_EXECUTION_SCHEDULE"))
N_TIME_GRID = int(_cfg_require("N_TIME_GRID"))
N_HEATMAP_BINS_Y = int(_cfg_require("N_HEATMAP_BINS_Y"))

if N_TIME_GRID < 2:
    raise ValueError("N_TIME_GRID must be at least 2.")
if N_HEATMAP_BINS_Y < 2:
    raise ValueError("N_HEATMAP_BINS_Y must be at least 2.")

_PATH_CONTEXT = {
    "DATASET_NAME": DATASET_NAME,
    "LEVEL": LEVEL,
    "MEMBER_NATIONALITY_TAG": MEMBER_NATIONALITY_TAG,
}
OUTPUT_ROOT = _resolve_repo_path(_format_path_template(str(_cfg_require("OUTPUT_FILE_PATH")), _PATH_CONTEXT))
IMG_ROOT = _resolve_repo_path(_format_path_template(str(_cfg_require("IMG_OUTPUT_PATH")), _PATH_CONTEXT))
ANALYSIS_DIRNAME = f"{LEVEL}_metaorder_execution_schedule"
SUMMARY_OUTPUT_DIR = OUTPUT_ROOT / ANALYSIS_DIRNAME
IMG_DIR = IMG_ROOT / ANALYSIS_DIRNAME
PLOT_OUTPUT_DIRS: PlotOutputDirs = make_plot_output_dirs(IMG_DIR, use_subdirs=True)

PROPRIETARY_PATH = resolve_opt_repo_path(
    _REPO_ROOT,
    _CFG.get("PROPRIETARY_PATH"),
    _default_info_path(OUTPUT_ROOT, LEVEL, proprietary=True, member_nationality=MEMBER_NATIONALITY),
)
CLIENT_PATH = resolve_opt_repo_path(
    _REPO_ROOT,
    _CFG.get("CLIENT_PATH"),
    _default_info_path(OUTPUT_ROOT, LEVEL, proprietary=False, member_nationality=MEMBER_NATIONALITY),
)


def _build_cumulative_curve(
    time_blob: Optional[bytes | bytearray | memoryview | list[float] | np.ndarray],
    volume_blob: Optional[bytes | bytearray | memoryview | list[float] | np.ndarray],
    tau_grid: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Convert one packed schedule into a cumulative-volume-fraction curve."""
    child_time = _unpack_path(time_blob)
    child_volume = _unpack_path(volume_blob)
    if child_time is None or child_volume is None:
        return None, "missing_schedule"

    time_arr = np.asarray(child_time, dtype=float).ravel()
    volume_arr = np.asarray(child_volume, dtype=float).ravel()
    if time_arr.size == 0 or volume_arr.size == 0:
        return None, "empty_schedule"
    if time_arr.size != volume_arr.size:
        return None, "length_mismatch"
    if not np.all(np.isfinite(time_arr)) or not np.all(np.isfinite(volume_arr)):
        return None, "non_finite_values"
    if np.any(volume_arr < 0.0):
        return None, "negative_volume_fraction"

    volume_sum = float(volume_arr.sum())
    if not np.isfinite(volume_sum) or volume_sum <= 0.0:
        return None, "nonpositive_volume_sum"
    volume_arr = volume_arr / volume_sum

    time_arr = np.clip(time_arr, 0.0, 1.0)
    order = np.argsort(time_arr, kind="mergesort")
    time_sorted = time_arr[order]
    volume_sorted = volume_arr[order]

    cumulative_sorted = np.cumsum(volume_sorted)
    if not np.all(np.isfinite(cumulative_sorted)):
        return None, "invalid_cumulative_sum"

    x = np.concatenate(([0.0], time_sorted, [1.0]))
    y = np.concatenate(([0.0], cumulative_sorted, [1.0]))

    y = np.clip(y, 0.0, 1.0)
    keep_last = np.concatenate((x[1:] > x[:-1], [True]))
    x_unique = x[keep_last]
    y_unique = y[keep_last]
    if x_unique.size < 2:
        return None, "insufficient_unique_times"
    if x_unique[0] > 0.0 or x_unique[-1] < 1.0:
        return None, "missing_execution_anchors"

    curve = np.interp(tau_grid, x_unique, y_unique)
    curve = np.asarray(curve, dtype=float)
    if curve.size != tau_grid.size or not np.all(np.isfinite(curve)):
        return None, "invalid_interpolation"
    curve[0] = 0.0
    curve[-1] = 1.0
    return curve, None


def _aggregate_group(
    df: pd.DataFrame,
    *,
    group: str,
    display_name: str,
    tau_grid: np.ndarray,
    n_heatmap_bins_y: int,
) -> GroupScheduleSummary:
    """Aggregate one group's schedule curves into mean/SEM and heatmap tables."""
    n_grid = int(tau_grid.size)
    y_bin_edges = np.linspace(0.0, 1.0, int(n_heatmap_bins_y) + 1)
    sum_curve = np.zeros(n_grid, dtype=float)
    sumsq_curve = np.zeros(n_grid, dtype=float)
    count_curve = np.zeros(n_grid, dtype=float)
    heatmap_counts = np.zeros((n_grid, int(n_heatmap_bins_y)), dtype=float)
    tau_idx = np.arange(n_grid, dtype=int)

    skipped: dict[str, int] = {}
    n_valid_metaorders = 0
    iterator = tqdm(
        df[["child_time_norm", "child_volume_fraction"]].itertuples(index=False, name=None),
        total=len(df),
        desc=f"[Schedule] {display_name}",
        dynamic_ncols=True,
    )
    for time_blob, volume_blob in iterator:
        curve, reason = _build_cumulative_curve(time_blob, volume_blob, tau_grid)
        if curve is None:
            reason_key = str(reason or "invalid_schedule")
            skipped[reason_key] = skipped.get(reason_key, 0) + 1
            continue

        n_valid_metaorders += 1
        sum_curve += curve
        sumsq_curve += np.square(curve)
        count_curve += 1.0

        y_idx = np.searchsorted(y_bin_edges, curve, side="right") - 1
        y_idx = np.clip(y_idx, 0, n_heatmap_bins_y - 1)
        heatmap_counts[tau_idx, y_idx] += 1.0

    if n_valid_metaorders == 0:
        raise ValueError(f"[{display_name}] No valid execution schedules were found.")

    mean_curve = np.divide(
        sum_curve,
        count_curve,
        out=np.full_like(sum_curve, np.nan, dtype=float),
        where=count_curve > 0,
    )
    variance = np.full_like(sum_curve, np.nan, dtype=float)
    valid_var = count_curve > 1
    variance[valid_var] = (
        sumsq_curve[valid_var] - np.square(sum_curve[valid_var]) / count_curve[valid_var]
    ) / (count_curve[valid_var] - 1.0)
    variance[valid_var] = np.maximum(variance[valid_var], 0.0)
    sem_curve = np.full_like(sum_curve, np.nan, dtype=float)
    sem_curve[valid_var] = np.sqrt(variance[valid_var] / count_curve[valid_var])

    heatmap_total = float(heatmap_counts.sum())
    if heatmap_total > 0.0:
        heatmap_density = heatmap_counts / heatmap_total
    else:
        heatmap_density = np.full_like(heatmap_counts, np.nan, dtype=float)

    return GroupScheduleSummary(
        group=group,
        display_name=display_name,
        n_input_rows=int(len(df)),
        n_valid_metaorders=int(n_valid_metaorders),
        tau_grid=np.asarray(tau_grid, dtype=float),
        mean_curve=mean_curve,
        sem_curve=sem_curve,
        n_eff_curve=count_curve.astype(int),
        heatmap_counts=heatmap_counts.astype(int),
        heatmap_density=heatmap_density,
        y_bin_edges=y_bin_edges,
        skipped_reasons={str(k): int(v) for k, v in sorted(skipped.items())},
    )


def _build_curve_table(summary: GroupScheduleSummary) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": summary.group,
            "tau": summary.tau_grid,
            "mean_cum_volume_fraction": summary.mean_curve,
            "sem_cum_volume_fraction": summary.sem_curve,
            "n_eff": summary.n_eff_curve,
            "n_input_rows": int(summary.n_input_rows),
            "n_valid_metaorders": int(summary.n_valid_metaorders),
        }
    )


def _build_heatmap_table(summary: GroupScheduleSummary) -> pd.DataFrame:
    y_left = summary.y_bin_edges[:-1]
    y_right = summary.y_bin_edges[1:]
    y_center = 0.5 * (y_left + y_right)
    tau_mesh, y_mesh = np.meshgrid(summary.tau_grid, y_center, indexing="ij")
    return pd.DataFrame(
        {
            "group": summary.group,
            "tau": tau_mesh.ravel(order="C"),
            "cum_volume_bin_left": np.tile(y_left, summary.tau_grid.size),
            "cum_volume_bin_right": np.tile(y_right, summary.tau_grid.size),
            "cum_volume_bin_center": y_mesh.ravel(order="C"),
            "density": summary.heatmap_density.ravel(order="C"),
            "count": summary.heatmap_counts.ravel(order="C"),
            "n_input_rows": int(summary.n_input_rows),
            "n_valid_metaorders": int(summary.n_valid_metaorders),
        }
    )


def _add_band(
    fig: go.Figure,
    *,
    x: np.ndarray,
    mean_curve: np.ndarray,
    sem_curve: np.ndarray,
    line_color: str,
    band_color: str,
    name: str,
) -> None:
    upper = mean_curve + sem_curve
    lower = mean_curve - sem_curve
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=band_color,
            name=f"{name} ± SEM",
            hovertemplate="tau=%{x:.3f}<br>cum. vol=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean_curve,
            mode="lines",
            line=dict(color=line_color, width=3),
            name=name,
            hovertemplate="tau=%{x:.3f}<br>mean cum. vol=%{y:.3f}<extra></extra>",
        )
    )


def _plot_cumulative_schedule(
    proprietary_summary: GroupScheduleSummary,
    client_summary: GroupScheduleSummary,
) -> go.Figure:
    fig = go.Figure()
    _add_band(
        fig,
        x=proprietary_summary.tau_grid,
        mean_curve=proprietary_summary.mean_curve,
        sem_curve=proprietary_summary.sem_curve,
        line_color=COLOR_PROPRIETARY,
        band_color=COLOR_BAND_PROPRIETARY,
        name="Proprietary",
    )
    _add_band(
        fig,
        x=client_summary.tau_grid,
        mean_curve=client_summary.mean_curve,
        sem_curve=client_summary.sem_curve,
        line_color=COLOR_CLIENT,
        band_color=COLOR_BAND_CLIENT,
        name="Client",
    )
    fig.add_trace(
        go.Scatter(
            x=proprietary_summary.tau_grid,
            y=proprietary_summary.tau_grid,
            mode="lines",
            line=dict(color=COLOR_NEUTRAL, width=2, dash="dash"),
            name="TWAP",
            hovertemplate="tau=%{x:.3f}<br>TWAP=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Normalized execution time",
        yaxis_title="Cumulative volume fraction",
        legend=dict(orientation="h", x=0.0, y=1.12),
        height=700,
        width=1200,
    )
    fig.update_xaxes(range=[0.0, 1.0])
    fig.update_yaxes(range=[0.0, 1.0])
    return fig


def _plot_heatmap(
    proprietary_summary: GroupScheduleSummary,
    client_summary: GroupScheduleSummary,
) -> go.Figure:
    y_center = 0.5 * (proprietary_summary.y_bin_edges[:-1] + proprietary_summary.y_bin_edges[1:])
    zmax = float(
        np.nanmax(
            [
                np.nanmax(proprietary_summary.heatmap_density),
                np.nanmax(client_summary.heatmap_density),
            ]
        )
    )
    if not np.isfinite(zmax) or zmax <= 0.0:
        zmax = 1.0

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=("Proprietary", "Client"),
    )
    for col, summary in enumerate((proprietary_summary, client_summary), start=1):
        fig.add_trace(
            go.Heatmap(
                x=summary.tau_grid,
                y=y_center,
                z=summary.heatmap_density.T,
                coloraxis="coloraxis",
                hovertemplate=(
                    "tau=%{x:.3f}<br>"
                    "cum. vol bin=%{y:.3f}<br>"
                    "density=%{z:.4f}<extra></extra>"
                ),
                showscale=(col == 2),
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=summary.tau_grid,
                y=summary.tau_grid,
                mode="lines",
                line=dict(color=COLOR_NEUTRAL, width=2, dash="dash"),
                name="TWAP" if col == 1 else None,
                showlegend=(col == 1),
                hovertemplate="tau=%{x:.3f}<br>TWAP=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=col,
        )

    fig.update_layout(
        coloraxis=dict(
            colorscale="Cividis",
            cmin=0.0,
            cmax=zmax,
            colorbar=dict(title="Density"),
        ),
        legend=dict(orientation="h", x=0.0, y=1.12),
        height=700,
        width=1300,
    )
    fig.update_xaxes(title_text="Normalized execution time", range=[0.0, 1.0], row=1, col=1)
    fig.update_xaxes(title_text="Normalized execution time", range=[0.0, 1.0], row=1, col=2)
    fig.update_yaxes(title_text="Cumulative volume fraction", range=[0.0, 1.0], row=1, col=1)
    return fig


def _validate_schedule_columns(df: pd.DataFrame, *, label: str) -> None:
    required = {"child_time_norm", "child_volume_fraction"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            f"[{label}] Missing required schedule columns: {sorted(missing)}. "
            "Rerun scripts/metaorder_computation.py with COMPUTE_EXECUTION_SCHEDULES=true."
        )


def _load_group_table(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[{label}] Missing input file: {path}")
    df = pd.read_parquet(path)
    _validate_schedule_columns(df, label=label)
    return df


def _write_tables(
    proprietary_summary: GroupScheduleSummary,
    client_summary: GroupScheduleSummary,
) -> tuple[Path, Path]:
    curve_table = pd.concat(
        [
            _build_curve_table(proprietary_summary),
            _build_curve_table(client_summary),
        ],
        ignore_index=True,
    )
    heatmap_table = pd.concat(
        [
            _build_heatmap_table(proprietary_summary),
            _build_heatmap_table(client_summary),
        ],
        ignore_index=True,
    )

    curve_stem = with_member_nationality_tag(
        "cumulative_execution_schedule_prop_vs_client",
        MEMBER_NATIONALITY,
    )
    heatmap_stem = with_member_nationality_tag(
        "execution_schedule_heatmap_prop_vs_client",
        MEMBER_NATIONALITY,
    )
    curve_path = SUMMARY_OUTPUT_DIR / f"{curve_stem}.parquet"
    heatmap_path = SUMMARY_OUTPUT_DIR / f"{heatmap_stem}.parquet"
    curve_table.to_parquet(curve_path, index=False)
    heatmap_table.to_parquet(heatmap_path, index=False)
    curve_table.to_csv(SUMMARY_OUTPUT_DIR / f"{curve_stem}.csv", index=False)
    heatmap_table.to_csv(SUMMARY_OUTPUT_DIR / f"{heatmap_stem}.csv", index=False)
    return curve_path, heatmap_path


def _write_manifest(
    proprietary_summary: GroupScheduleSummary,
    client_summary: GroupScheduleSummary,
    *,
    curve_table_path: Path,
    heatmap_table_path: Path,
) -> Path:
    manifest = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "script": str(Path(__file__).resolve()),
        "config_path": str(_CONFIG_PATH.resolve()),
        "dataset_name": DATASET_NAME,
        "level": LEVEL,
        "member_nationality": MEMBER_NATIONALITY_TAG,
        "run_execution_schedule": bool(RUN_EXECUTION_SCHEDULE),
        "n_time_grid": int(N_TIME_GRID),
        "n_heatmap_bins_y": int(N_HEATMAP_BINS_Y),
        "input_paths": {
            "proprietary": str(PROPRIETARY_PATH.resolve()),
            "client": str(CLIENT_PATH.resolve()),
        },
        "output_paths": {
            "curve_table": str(curve_table_path.resolve()),
            "heatmap_table": str(heatmap_table_path.resolve()),
            "figure_dir": str(IMG_DIR.resolve()),
        },
        "groups": {
            "proprietary": {
                "n_input_rows": int(proprietary_summary.n_input_rows),
                "n_valid_metaorders": int(proprietary_summary.n_valid_metaorders),
                "skipped_reasons": proprietary_summary.skipped_reasons,
            },
            "client": {
                "n_input_rows": int(client_summary.n_input_rows),
                "n_valid_metaorders": int(client_summary.n_valid_metaorders),
                "skipped_reasons": client_summary.skipped_reasons,
            },
        },
    }
    manifest_path = SUMMARY_OUTPUT_DIR / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main() -> None:
    log_path = OUTPUT_ROOT / "logs" / with_member_nationality_tag(
        f"metaorder_execution_schedule_{LEVEL}.log",
        MEMBER_NATIONALITY,
    )
    logger = setup_file_logger(Path(__file__).stem, log_path, mode="a")
    with PrintTee(logger):
        print("[Intro] Metaorder execution-schedule run started...")
        print(
            "[Intro] Parameters — \n"
            f"  DATASET={DATASET_NAME}, LEVEL={LEVEL}, MEMBER_NATIONALITY={MEMBER_NATIONALITY_TAG},\n"
            f"  RUN_EXECUTION_SCHEDULE={RUN_EXECUTION_SCHEDULE}, N_TIME_GRID={N_TIME_GRID}, "
            f"N_HEATMAP_BINS_Y={N_HEATMAP_BINS_Y}"
        )
        print(
            "[Intro] Paths — \n"
            f"  OUTPUT_ROOT={OUTPUT_ROOT}\n"
            f"  SUMMARY_OUTPUT_DIR={SUMMARY_OUTPUT_DIR}\n"
            f"  IMG_DIR={IMG_DIR}\n"
            f"  PROPRIETARY_PATH={PROPRIETARY_PATH}\n"
            f"  CLIENT_PATH={CLIENT_PATH}"
        )

        if not RUN_EXECUTION_SCHEDULE:
            print("[Execution schedule] RUN_EXECUTION_SCHEDULE is false; nothing to do.")
            return

        ensure_plot_dirs(PLOT_OUTPUT_DIRS)
        SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        proprietary_df = _load_group_table(PROPRIETARY_PATH, label="proprietary")
        client_df = _load_group_table(CLIENT_PATH, label="client")
        tau_grid = np.linspace(0.0, 1.0, N_TIME_GRID)

        proprietary_summary = _aggregate_group(
            proprietary_df,
            group="proprietary",
            display_name="Proprietary",
            tau_grid=tau_grid,
            n_heatmap_bins_y=N_HEATMAP_BINS_Y,
        )
        client_summary = _aggregate_group(
            client_df,
            group="client",
            display_name="Client",
            tau_grid=tau_grid,
            n_heatmap_bins_y=N_HEATMAP_BINS_Y,
        )

        print(
            f"[Execution schedule] Proprietary valid metaorders: "
            f"{proprietary_summary.n_valid_metaorders}/{proprietary_summary.n_input_rows}"
        )
        print(
            f"[Execution schedule] Client valid metaorders: "
            f"{client_summary.n_valid_metaorders}/{client_summary.n_input_rows}"
        )
        if proprietary_summary.skipped_reasons:
            print(f"[Execution schedule] Proprietary skipped rows: {proprietary_summary.skipped_reasons}")
        if client_summary.skipped_reasons:
            print(f"[Execution schedule] Client skipped rows: {client_summary.skipped_reasons}")

        curve_table_path, heatmap_table_path = _write_tables(proprietary_summary, client_summary)
        manifest_path = _write_manifest(
            proprietary_summary,
            client_summary,
            curve_table_path=curve_table_path,
            heatmap_table_path=heatmap_table_path,
        )

        cumulative_fig = _plot_cumulative_schedule(proprietary_summary, client_summary)
        cumulative_stem = with_member_nationality_tag(
            "cumulative_execution_schedule_prop_vs_client",
            MEMBER_NATIONALITY,
        )
        save_plotly_figure(
            cumulative_fig,
            stem=cumulative_stem,
            dirs=PLOT_OUTPUT_DIRS,
            write_html=True,
            write_png=True,
            strict_png=False,
        )

        heatmap_fig = _plot_heatmap(proprietary_summary, client_summary)
        heatmap_stem = with_member_nationality_tag(
            "execution_schedule_heatmap_prop_vs_client",
            MEMBER_NATIONALITY,
        )
        save_plotly_figure(
            heatmap_fig,
            stem=heatmap_stem,
            dirs=PLOT_OUTPUT_DIRS,
            write_html=True,
            write_png=True,
            strict_png=False,
        )

        print(f"[Execution schedule] Saved cumulative schedule table to {curve_table_path}")
        print(f"[Execution schedule] Saved heatmap table to {heatmap_table_path}")
        print(f"[Execution schedule] Saved run manifest to {manifest_path}")
        print(f"[Execution schedule] Saved figures under {IMG_DIR}")


if __name__ == "__main__":
    main()
