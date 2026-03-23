#!/usr/bin/env python3
"""
Morning-versus-evening impact analysis on existing metaorder tables.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from datetime import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
from moimpact.impact_fits import (
    filter_metaorders_info_for_fits,
    fit_logarithmic_from_binned,
    fit_power_law_logbins_wls_new,
    plot_fit,
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
    COLOR_CLIENT,
    COLOR_PROPRIETARY,
    PlotOutputDirs,
    ensure_plot_dirs,
    make_plot_output_dirs,
    save_plotly_figure as _save_plotly_figure,
)


_CONFIG_ENV_VAR = "METAORDER_INTRADAY_CONFIG"
_config_override = os.environ.get(_CONFIG_ENV_VAR)
if _config_override:
    _CONFIG_PATH = Path(_config_override).expanduser()
    if not _CONFIG_PATH.is_absolute():
        _CONFIG_PATH = (_REPO_ROOT / _CONFIG_PATH).resolve()
else:
    _CONFIG_PATH = _REPO_ROOT / "config_ymls" / "metaorder_intraday_analysis.yml"
_CFG = load_yaml_mapping(_CONFIG_PATH)


def _cfg_require(key: str) -> Any:
    return cfg_require(_CFG, key, _CONFIG_PATH)


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_repo_path(_REPO_ROOT, value)


def _format_path_template(template: str, context: Mapping[str, str]) -> str:
    return format_path_template(template, context)


def _parse_time_string(raw_value: object, *, label: str) -> time:
    parsed = pd.to_datetime(str(raw_value))
    if pd.isna(parsed):
        raise ValueError(f"Invalid time string for {label}: {raw_value!r}")
    return parsed.time()


def _parse_session_windows(raw_value: object) -> "OrderedDict[str, tuple[time, time]]":
    if not isinstance(raw_value, Mapping) or len(raw_value) == 0:
        raise ValueError("SESSION_WINDOWS must be a non-empty mapping of session name to [start, end].")

    windows: "OrderedDict[str, tuple[time, time]]" = OrderedDict()
    for session_name, bounds in raw_value.items():
        if not isinstance(session_name, str) or not session_name.strip():
            raise ValueError("Session names in SESSION_WINDOWS must be non-empty strings.")
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError(
                f"SESSION_WINDOWS[{session_name!r}] must contain exactly two time strings: [start, end]."
            )
        start = _parse_time_string(bounds[0], label=f"{session_name}.start")
        end = _parse_time_string(bounds[1], label=f"{session_name}.end")
        if start >= end:
            raise ValueError(f"SESSION_WINDOWS[{session_name!r}] must satisfy start < end.")
        windows[str(session_name)] = (start, end)
    return windows


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


def _classify_intraday_session(
    timestamp: pd.Timestamp,
    session_windows: "OrderedDict[str, tuple[time, time]]",
) -> Optional[str]:
    if pd.isna(timestamp):
        return None

    timestamp_time = timestamp.time()
    items = list(session_windows.items())
    for idx, (session_name, (start_time, end_time)) in enumerate(items):
        is_last = idx == len(items) - 1
        if start_time <= timestamp_time < end_time:
            return session_name
        if is_last and start_time <= timestamp_time <= end_time:
            return session_name
    return None


def _attach_intraday_session_columns(
    df: pd.DataFrame,
    session_windows: "OrderedDict[str, tuple[time, time]]",
) -> pd.DataFrame:
    if "Period" not in df.columns:
        raise KeyError("Missing required column 'Period' in the metaorder info table.")

    out = df.copy()
    start_ns = out["Period"].apply(lambda value: _period_endpoint_ns(value, 0))
    end_ns = out["Period"].apply(lambda value: _period_endpoint_ns(value, 1))
    out["StartTimestamp"] = pd.to_datetime(start_ns, errors="coerce")
    out["EndTimestamp"] = pd.to_datetime(end_ns, errors="coerce")
    out["StartSession"] = out["StartTimestamp"].apply(
        lambda ts: _classify_intraday_session(ts, session_windows)
    )
    out["EndSession"] = out["EndTimestamp"].apply(
        lambda ts: _classify_intraday_session(ts, session_windows)
    )
    out["Session"] = np.where(
        out["StartSession"].eq(out["EndSession"]),
        out["StartSession"],
        None,
    )
    return out


def _session_color_map(session_names: Iterable[str]) -> dict[str, str]:
    palette = [
        THEME_COLORWAY[1],
        THEME_COLORWAY[3] if len(THEME_COLORWAY) > 3 else THEME_COLORWAY[0],
        THEME_COLORWAY[4] if len(THEME_COLORWAY) > 4 else THEME_COLORWAY[2],
    ]
    return {name: palette[idx % len(palette)] for idx, name in enumerate(session_names)}


def save_plotly_figure(fig, *args, **kwargs):
    """
    Summary
    -------
    Save a Plotly figure after removing the top-level title.

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
    This keeps exports consistent with the other metaorder analysis scripts.
    """
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)


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

LEVEL = str(_cfg_require("LEVEL"))
DATASET_NAME = str(_CFG.get("DATASET_NAME") or "ftsemib")
MEMBER_NATIONALITY = parse_member_nationality(_CFG.get("MEMBER_NATIONALITY"))
MEMBER_NATIONALITY_TAG = MEMBER_NATIONALITY or "all"
MIN_QV = float(_cfg_require("MIN_QV"))
MAX_PARTICIPATION_RATE = float(_cfg_require("MAX_PARTICIPATION_RATE"))
N_LOGBIN = int(_cfg_require("N_LOGBIN"))
MIN_COUNT = int(_cfg_require("MIN_COUNT"))
SESSION_WINDOWS = _parse_session_windows(_cfg_require("SESSION_WINDOWS"))

_path_context = {
    "DATASET_NAME": DATASET_NAME,
    "LEVEL": LEVEL,
    "MEMBER_NATIONALITY_TAG": MEMBER_NATIONALITY_TAG,
}
OUTPUT_ROOT = _resolve_repo_path(_format_path_template(str(_cfg_require("OUTPUT_FILE_PATH")), _path_context))
IMG_ROOT = _resolve_repo_path(_format_path_template(str(_cfg_require("IMG_OUTPUT_PATH")), _path_context))
ANALYSIS_OUTPUT_DIR = OUTPUT_ROOT / f"{LEVEL}_metaorder_intraday_analysis"
PLOT_OUTPUT_DIRS: PlotOutputDirs = make_plot_output_dirs(
    IMG_ROOT / f"{LEVEL}_metaorder_intraday_analysis",
    use_subdirs=True,
)


def _default_info_path(group_tag: str) -> Path:
    filename = with_member_nationality_tag(
        f"metaorders_info_sameday_{LEVEL}_{group_tag}.parquet",
        MEMBER_NATIONALITY,
    )
    return OUTPUT_ROOT / filename


PROPRIETARY_INFO_PATH = resolve_opt_repo_path(
    _REPO_ROOT,
    _CFG.get("PROPRIETARY_INFO_PATH"),
    _default_info_path("proprietary"),
)
CLIENT_INFO_PATH = resolve_opt_repo_path(
    _REPO_ROOT,
    _CFG.get("CLIENT_INFO_PATH"),
    _default_info_path("non_proprietary"),
)


def _save_session_tables(
    df_session: pd.DataFrame,
    df_filtered: pd.DataFrame,
    *,
    session_name: str,
    group_tag: str,
) -> tuple[Path, Path]:
    base_name = f"metaorders_info_sameday_{LEVEL}_{group_tag}_{session_name}.parquet"
    filtered_name = f"metaorders_info_sameday_filtered_{LEVEL}_{group_tag}_{session_name}.parquet"
    unfiltered_path = ANALYSIS_OUTPUT_DIR / with_member_nationality_tag(base_name, MEMBER_NATIONALITY)
    filtered_path = ANALYSIS_OUTPUT_DIR / with_member_nationality_tag(filtered_name, MEMBER_NATIONALITY)
    df_session.to_parquet(unfiltered_path, index=False)
    df_filtered.to_parquet(filtered_path, index=False)
    return unfiltered_path, filtered_path


def _empty_summary_row(group_tag: str, group_label: str, session_name: str) -> dict[str, object]:
    return {
        "group_tag": group_tag,
        "group_label": group_label,
        "session": session_name,
        "n_detected": 0,
        "n_after_qv_filter": 0,
        "n_fit_sample_after_pr_cap": 0,
        "n_bins_used": np.nan,
        "fit_status": "no_rows",
        "power_prefactor": np.nan,
        "power_prefactor_se": np.nan,
        "power_gamma": np.nan,
        "power_gamma_se": np.nan,
        "power_r2_log": np.nan,
        "power_r2_lin": np.nan,
        "log_a": np.nan,
        "log_a_se": np.nan,
        "log_b": np.nan,
        "log_b_se": np.nan,
        "log_r2_lin": np.nan,
    }


def _build_counts_figure(summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    group_specs = [
        ("Proprietary", COLOR_PROPRIETARY),
        ("Client", COLOR_CLIENT),
    ]
    sessions = summary_df["session"].dropna().astype(str).tolist()
    session_order = list(dict.fromkeys(sessions))
    for group_label, color in group_specs:
        sub = summary_df.loc[summary_df["group_label"] == group_label].copy()
        sub = sub.set_index("session").reindex(session_order).reset_index()
        fig.add_trace(
            go.Bar(
                x=sub["session"],
                y=sub["n_detected"],
                name=group_label,
                marker_color=color,
                customdata=np.stack(
                    [
                        sub["n_after_qv_filter"].fillna(0).to_numpy(dtype=float),
                        sub["n_fit_sample_after_pr_cap"].fillna(0).to_numpy(dtype=float),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "Session=%{x}<br>"
                    "Detected=%{y}<br>"
                    "After Q/V filter=%{customdata[0]:.0f}<br>"
                    "Fit sample after PR cap=%{customdata[1]:.0f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="group",
        title="Detected metaorders by intraday session",
        xaxis_title="Session",
        yaxis_title="Number of metaorders",
        legend_title_text="Flow group",
    )
    return fig


def _build_group_fit_figure(
    group_label: str,
    session_fits: Dict[str, dict[str, object]],
    session_colors: Mapping[str, str],
) -> Optional[go.Figure]:
    fig = go.Figure()
    any_trace = False
    for session_name, fit_payload in session_fits.items():
        binned = fit_payload.get("binned")
        params = fit_payload.get("params")
        if binned is None or params is None:
            continue
        plot_fit(
            fig,
            binned,
            params,
            label_prefix=session_name.capitalize(),
            label_size=LABEL_FONT_SIZE,
            legend_size=LEGEND_FONT_SIZE,
            log_params=fit_payload.get("log_params"),
            series_color=session_colors[session_name],
            log_line_color=session_colors[session_name],
        )
        any_trace = True
    if not any_trace:
        return None
    fig.update_layout(
        title=f"{group_label}: morning vs evening impact fits",
        xaxis_title="Q/V",
        yaxis_title=r"$I/\sigma$",
    )
    return fig


def _run_group_session_analysis(
    df_group: pd.DataFrame,
    *,
    group_tag: str,
    group_label: str,
    session_colors: Mapping[str, str],
) -> tuple[list[dict[str, object]], Dict[str, dict[str, object]]]:
    df_with_sessions = _attach_intraday_session_columns(df_group, SESSION_WINDOWS)
    kept_mask = df_with_sessions["Session"].isin(list(SESSION_WINDOWS.keys()))
    n_cross_session = int((~kept_mask).sum())
    print(
        f"[Intraday] {group_label}: loaded {len(df_group)} metaorders | "
        f"kept in-session={int(kept_mask.sum())} | cross-session/excluded={n_cross_session}"
    )

    summary_rows: list[dict[str, object]] = []
    session_fits: Dict[str, dict[str, object]] = {}
    for session_name in tqdm(
        list(SESSION_WINDOWS.keys()),
        desc=f"[Intraday] {group_label} sessions",
        dynamic_ncols=True,
    ):
        session_df = df_with_sessions.loc[df_with_sessions["Session"] == session_name].copy()
        session_df["FlowGroup"] = group_label
        session_df["FlowGroupTag"] = group_tag

        row = _empty_summary_row(group_tag, group_label, session_name)
        row["n_detected"] = int(len(session_df))

        filtered_df = filter_metaorders_info_for_fits(session_df, min_qv=MIN_QV)
        row["n_after_qv_filter"] = int(len(filtered_df))
        _save_session_tables(session_df, filtered_df, session_name=session_name, group_tag=group_tag)

        fit_df = filtered_df.copy()
        if "Participation Rate" in fit_df.columns:
            fit_df = fit_df.loc[fit_df["Participation Rate"] < MAX_PARTICIPATION_RATE].reset_index(drop=True)
        row["n_fit_sample_after_pr_cap"] = int(len(fit_df))

        if fit_df.empty:
            row["fit_status"] = "no_rows_after_filters"
            summary_rows.append(row)
            continue

        try:
            binned, params = fit_power_law_logbins_wls_new(
                fit_df,
                n_logbins=N_LOGBIN,
                min_count=MIN_COUNT,
                use_median=False,
                control_cols=None,
            )
        except Exception as exc:
            row["fit_status"] = f"power_fit_failed: {exc}"
            summary_rows.append(row)
            print(f"[Intraday] {group_label} / {session_name}: power-law fit skipped: {exc}")
            continue

        log_params = None
        row["fit_status"] = "ok"
        try:
            log_params = fit_logarithmic_from_binned(binned)
        except Exception as exc:
            row["fit_status"] = f"log_fit_failed: {exc}"
            print(f"[Intraday] {group_label} / {session_name}: logarithmic fit skipped: {exc}")

        y_hat, y_se, gamma_hat, gamma_se, r2_log, r2_lin, _, _ = params
        row.update(
            {
                "n_bins_used": int(len(binned)),
                "power_prefactor": float(y_hat),
                "power_prefactor_se": float(y_se),
                "power_gamma": float(gamma_hat),
                "power_gamma_se": float(gamma_se),
                "power_r2_log": float(r2_log),
                "power_r2_lin": float(r2_lin),
            }
        )
        if log_params is not None:
            a_hat, a_se, b_hat, b_se, r2_lin_log = log_params
            row.update(
                {
                    "log_a": float(a_hat),
                    "log_a_se": float(a_se),
                    "log_b": float(b_hat),
                    "log_b_se": float(b_se),
                    "log_r2_lin": float(r2_lin_log),
                }
            )

        session_fits[session_name] = {
            "binned": binned,
            "params": params,
            "log_params": log_params,
            "series_color": session_colors[session_name],
        }
        summary_rows.append(row)
    return summary_rows, session_fits


def main() -> None:
    """
    Summary
    -------
    Compare morning and evening impact fits for proprietary and client metaorders.

    Parameters
    ----------
    None
        Configuration is loaded from `config_ymls/metaorder_intraday_analysis.yml`
        or from the `METAORDER_INTRADAY_CONFIG` environment variable.

    Returns
    -------
    None
        The script writes session-filtered parquet tables, one summary table, and
        comparison figures under the configured output folders.

    Notes
    -----
    The analysis is intentionally post-detection: it loads the existing
    `metaorders_info_sameday_*` tables produced by `metaorder_computation.py`,
    derives each metaorder's session from its `Period`, drops cross-session
    executions, and then reuses the same Q/V and WLS fit logic as the full-day
    script.

    Examples
    --------
    Run from the repository root:

    >>> # doctest: +SKIP
    >>> # python scripts/metaorder_intraday_analysis.py
    """
    log_path = OUTPUT_ROOT / "logs" / with_member_nationality_tag(
        f"metaorder_intraday_analysis_{LEVEL}_prop_vs_client.log",
        MEMBER_NATIONALITY,
    )
    logger = setup_file_logger(Path(__file__).stem, log_path, mode="a")
    with PrintTee(logger):
        print("[Intro] Metaorder intraday analysis started...")
        print(
            "[Intro] Parameters — \n"
            f"  DATASET={DATASET_NAME}, LEVEL={LEVEL}, MEMBER_NATIONALITY={MEMBER_NATIONALITY_TAG}, "
            f"MIN_QV={MIN_QV}, MAX_PARTICIPATION_RATE={MAX_PARTICIPATION_RATE}, "
            f"N_LOGBIN={N_LOGBIN}, MIN_COUNT={MIN_COUNT}, SESSION_WINDOWS={dict(SESSION_WINDOWS)}"
        )
        print(
            "[Intro] Paths — \n"
            f"  OUTPUT_ROOT={OUTPUT_ROOT}\n"
            f"  ANALYSIS_OUTPUT_DIR={ANALYSIS_OUTPUT_DIR}\n"
            f"  IMG_DIR={PLOT_OUTPUT_DIRS.base_dir}\n"
            f"  PROPRIETARY_INFO_PATH={PROPRIETARY_INFO_PATH}\n"
            f"  CLIENT_INFO_PATH={CLIENT_INFO_PATH}"
        )

        ensure_plot_dirs(PLOT_OUTPUT_DIRS)
        ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

        group_specs = [
            ("proprietary", "Proprietary", PROPRIETARY_INFO_PATH, COLOR_PROPRIETARY),
            ("non_proprietary", "Client", CLIENT_INFO_PATH, COLOR_CLIENT),
        ]
        session_colors = _session_color_map(SESSION_WINDOWS.keys())
        all_summary_rows: list[dict[str, object]] = []

        for group_tag, group_label, table_path, _group_color in tqdm(
            group_specs,
            desc="[Intraday] Flow groups",
            dynamic_ncols=True,
        ):
            if not table_path.exists():
                raise FileNotFoundError(
                    f"Missing input parquet for {group_label}: {table_path}. "
                    "Run scripts/metaorder_computation.py for both proprietary and non-proprietary groups first."
                )
            df_group = pd.read_parquet(table_path)
            summary_rows, session_fits = _run_group_session_analysis(
                df_group,
                group_tag=group_tag,
                group_label=group_label,
                session_colors=session_colors,
            )
            all_summary_rows.extend(summary_rows)

            fit_fig = _build_group_fit_figure(group_label, session_fits, session_colors)
            if fit_fig is not None:
                stem = with_member_nationality_tag(
                    f"intraday_impact_fits_{LEVEL}_{group_tag}",
                    MEMBER_NATIONALITY,
                )
                save_plotly_figure(
                    fit_fig,
                    stem=stem,
                    dirs=PLOT_OUTPUT_DIRS,
                    write_html=True,
                    write_png=True,
                    strict_png=False,
                )
            else:
                print(f"[Intraday] {group_label}: no valid session fit produced a comparison figure.")

        summary_df = pd.DataFrame(all_summary_rows)
        if summary_df.empty:
            raise ValueError("No intraday summary rows were produced.")

        summary_csv_path = ANALYSIS_OUTPUT_DIR / with_member_nationality_tag(
            "intraday_session_summary_prop_vs_client.csv",
            MEMBER_NATIONALITY,
        )
        summary_parquet_path = ANALYSIS_OUTPUT_DIR / with_member_nationality_tag(
            "intraday_session_summary_prop_vs_client.parquet",
            MEMBER_NATIONALITY,
        )
        summary_df.to_csv(summary_csv_path, index=False)
        summary_df.to_parquet(summary_parquet_path, index=False)
        print(f"[Intraday] Saved summary table to {summary_parquet_path}")

        counts_fig = _build_counts_figure(summary_df)
        save_plotly_figure(
            counts_fig,
            stem=with_member_nationality_tag(
                f"intraday_metaorder_counts_{LEVEL}_prop_vs_client",
                MEMBER_NATIONALITY,
            ),
            dirs=PLOT_OUTPUT_DIRS,
            write_html=True,
            write_png=True,
            strict_png=False,
        )


if __name__ == "__main__":
    main()
