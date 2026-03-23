#!/usr/bin/env python3
"""
Metaorder summary statistics and comparison figures.

What this script does
---------------------
This script loads the canonical proprietary and client metaorder dictionaries,
recomputes the aggregated metaorder samples from the per-ISIN trade tapes, and
produces only the non-distribution summary outputs:

- aggressive-member nationality share
- nationality context versus overall traded volume
- member-level metaorder rank plot plus scatter of total child orders vs total trades
- mean daily metaorder-volume share by ISIN

By default the figures condition on the client/proprietary split. A CLI flag can
instead pool both flow groups into one "All metaorders" view.

How to run
----------
1) Edit `config_ymls/metaorder_summary_statistics.yml`, or point
   `METAORDER_SUMMARY_STATS_CONFIG` to an alternate YAML file.
2) Run:

    python scripts/metaorder_summary_statistics.py

   To pool client and proprietary metaorders into one group:

    python scripts/metaorder_summary_statistics.py --condition-on-client-proprietary false

Outputs
-------
- Figures:
  `images/{DATASET_NAME}/{LEVEL}_metaorder_summary_statistics/png/` and
  `.../html/`
- Tables:
  `out_files/{DATASET_NAME}/{LEVEL}_metaorder_summary_statistics/`
- Log file:
  `out_files/{DATASET_NAME}/logs/metaorder_summary_statistics_{LEVEL}_{prop_vs_client|all_metaorders}[...].log`
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure repository-root imports (e.g., `moimpact`) work when running
# `python scripts/metaorder_summary_statistics.py` from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.config import cfg_require, format_path_template, load_yaml_mapping, resolve_repo_path
from moimpact.logging_utils import PrintTee, setup_file_logger
from moimpact.metaorder_distribution_samples import (
    AGGRESSIVE_MEMBER_NATIONALITY_COL,
    MetaorderDistributionSamples,
    collect_metaorder_distribution_samples,
    list_metaorder_parquet_paths,
    load_trades_filtered_for_stats,
    parse_member_nationality,
    with_member_nationality_tag,
)
from moimpact.plot_style import (
    PLOTLY_TEMPLATE_NAME,
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


_CONFIG_ENV_VAR = "METAORDER_SUMMARY_STATS_CONFIG"
_config_override = os.environ.get(_CONFIG_ENV_VAR)
if _config_override:
    _CONFIG_PATH = Path(_config_override).expanduser()
    if not _CONFIG_PATH.is_absolute():
        _CONFIG_PATH = (_REPO_ROOT / _CONFIG_PATH).resolve()
else:
    _CONFIG_PATH = _REPO_ROOT / "config_ymls" / "metaorder_summary_statistics.yml"
_CFG = load_yaml_mapping(_CONFIG_PATH)


def _cfg_require(key: str):
    return cfg_require(_CFG, key, _CONFIG_PATH)


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_repo_path(_REPO_ROOT, value)


def _format_path_template(template: str, context: Mapping[str, str]) -> str:
    return format_path_template(template, context)


def _axis_ref_name(plotly_axis_name: str) -> str:
    return plotly_axis_name.replace("axis", "")


def _default_dict_path(
    output_dir: Path,
    level: str,
    proprietary: bool,
    member_nationality: Optional[str],
) -> Path:
    """Return the canonical metaorder-dictionary path for one flow group."""
    proprietary_tag = "proprietary" if proprietary else "non_proprietary"
    return output_dir / with_member_nationality_tag(
        f"metaorders_dict_all_{level}_{proprietary_tag}.pkl",
        member_nationality,
    )


def save_plotly_figure(fig, *args, **kwargs):
    """
    Summary
    -------
    Save a Plotly figure after removing its top-level title.

    Parameters
    ----------
    fig
        Plotly figure object.
    *args, **kwargs
        Forwarded to `moimpact.plotting.save_plotly_figure`.

    Returns
    -------
    tuple[Optional[Path], Optional[Path]]
        Output HTML/PNG paths returned by the shared plotting helper.

    Notes
    -----
    Summary figures are exported without top titles because captions live in the
    surrounding documentation or paper.
    """
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)


def _parse_cli_bool(value: object) -> bool:
    """
    Summary
    -------
    Parse a command-line boolean value.

    Parameters
    ----------
    value : object
        Raw CLI value.

    Returns
    -------
    bool
        Parsed boolean value.

    Notes
    -----
    Accepted truthy values are ``true``, ``1``, ``yes``, ``y``, and ``on``.
    Accepted falsy values are ``false``, ``0``, ``no``, ``n``, and ``off``.
    """
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value for --condition-on-client-proprietary "
        "(for example: true/false, yes/no, or 1/0)."
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Summary
    -------
    Parse the command-line arguments for the summary-statistics script.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Optional explicit argument list. When None, argparse uses ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build non-distribution summary figures for metaorders, either "
            "split by client/proprietary flow or pooled across both groups."
        )
    )
    parser.add_argument(
        "--condition-on-client-proprietary",
        type=_parse_cli_bool,
        default=True,
        metavar="{true,false}",
        help=(
            "If true (default), keep client and proprietary metaorders as "
            "separate groups. If false, pool both groups into a single "
            "'All metaorders' view."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
TICK_FONT_SIZE = int(_cfg_require("TICK_FONT_SIZE"))
LABEL_FONT_SIZE = int(_cfg_require("LABEL_FONT_SIZE"))
TITLE_FONT_SIZE = int(_cfg_require("TITLE_FONT_SIZE"))
LEGEND_FONT_SIZE = int(_cfg_require("LEGEND_FONT_SIZE"))
ANNOTATION_FONT_SIZE = int(_CFG.get("ANNOTATION_FONT_SIZE", 14))

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
COLOR_ALL_METAORDERS = "#4a4a4a"
KNOWN_MEMBER_NATIONALITIES: Tuple[str, str] = ("it", "foreign")
NATIONALITY_LABELS: Dict[str, str] = {"it": "IT", "foreign": "Foreign"}
NATIONALITY_COLORS: Dict[str, str] = {"it": THEME_COLORWAY[0], "foreign": THEME_COLORWAY[2]}

DATASET_NAME = str(_CFG.get("DATASET_NAME") or "ftsemib")
LEVEL = str(_cfg_require("LEVEL"))
TRADING_HOURS = tuple(_cfg_require("TRADING_HOURS"))
RUN_METAORDER_SUMMARY_STATISTICS = bool(_cfg_require("RUN_METAORDER_SUMMARY_STATISTICS"))
MEMBER_NATIONALITY = parse_member_nationality(_CFG.get("MEMBER_NATIONALITY"))
MEMBER_NATIONALITY_TAG = MEMBER_NATIONALITY or "all"

_path_context = {
    "DATASET_NAME": DATASET_NAME,
    "LEVEL": LEVEL,
    "MEMBER_NATIONALITY_TAG": MEMBER_NATIONALITY_TAG,
}

PARQUET_DIR = _resolve_repo_path(
    _format_path_template(str(_cfg_require("PARQUET_PATH")), _path_context)
)
OUTPUT_DIR = _resolve_repo_path(
    _format_path_template(str(_cfg_require("OUTPUT_FILE_PATH")), _path_context)
)
IMG_BASE_DIR = _resolve_repo_path(
    _format_path_template(str(_cfg_require("IMG_OUTPUT_PATH")), _path_context)
)
SUMMARY_OUTPUT_DIR = OUTPUT_DIR / f"{LEVEL}_metaorder_summary_statistics"
IMG_DIR = IMG_BASE_DIR / f"{LEVEL}_metaorder_summary_statistics"
MEMBERS_NATIONALITY_PATH = _resolve_repo_path("data/members_nationality.parquet")

_prop_override = _CFG.get("PROPRIETARY_DICT_PATH")
_client_override = _CFG.get("CLIENT_DICT_PATH")
PROPRIETARY_DICT_PATH = (
    _resolve_repo_path(_format_path_template(str(_prop_override), _path_context))
    if _prop_override is not None
    else _default_dict_path(OUTPUT_DIR, LEVEL, proprietary=True, member_nationality=MEMBER_NATIONALITY)
)
CLIENT_DICT_PATH = (
    _resolve_repo_path(_format_path_template(str(_client_override), _path_context))
    if _client_override is not None
    else _default_dict_path(OUTPUT_DIR, LEVEL, proprietary=False, member_nationality=MEMBER_NATIONALITY)
)


def _daily_total_market_volume(trades: pd.DataFrame) -> pd.Series:
    """Aggregate total traded volume by trading day for one ISIN."""
    if trades.empty:
        return pd.Series(dtype=float)
    volume = pd.to_numeric(trades["Total Quantity Buy"], errors="coerce").fillna(0.0)
    volume = volume + pd.to_numeric(trades["Total Quantity Sell"], errors="coerce").fillna(0.0)
    return volume.groupby(pd.to_datetime(trades["Trade Time"]).dt.date).sum().astype(float)


def _daily_metaorder_volume(
    metaorders_dict: Mapping[object, Sequence[Sequence[int]]],
    trades: pd.DataFrame,
) -> pd.Series:
    """Aggregate total metaorder volume by day for one ISIN and one flow group."""
    if trades.empty or not metaorders_dict:
        return pd.Series(dtype=float)

    times = pd.to_datetime(trades["Trade Time"]).to_numpy()
    buy_qty = pd.to_numeric(trades["Total Quantity Buy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sell_qty = pd.to_numeric(trades["Total Quantity Sell"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    volumes_by_day: Dict[dt.date, float] = {}
    n_trades = len(trades)
    for metas in metaorders_dict.values():
        for meta in metas:
            if not meta:
                continue
            meta_indices = np.asarray(meta, dtype=np.int64)
            if np.any((meta_indices < 0) | (meta_indices >= n_trades)):
                raise IndexError(
                    "Metaorder indices are out of bounds for the filtered trades slice. "
                    "Check that the dictionary and the selected filters match."
                )
            trade_date = pd.Timestamp(times[int(meta_indices[0])]).date()
            volume = float(buy_qty[meta_indices].sum() + sell_qty[meta_indices].sum())
            volumes_by_day[trade_date] = volumes_by_day.get(trade_date, 0.0) + volume

    return pd.Series(volumes_by_day, dtype=float)


def _infer_metaorder_member_nationality(labels: Sequence[object]) -> Tuple[Optional[str], bool]:
    """
    Infer one nationality tag for a metaorder from its child-trade labels.

    The rule mirrors the majority-vote logic used by the shared sample
    collectors so the counts reported here stay aligned with the existing
    nationality-share figure.
    """
    normalized = (
        pd.Series(labels, dtype="object")
        .dropna()
        .astype("string")
        .str.strip()
        .str.lower()
    )
    normalized = normalized.loc[normalized.isin(KNOWN_MEMBER_NATIONALITIES)]
    if normalized.empty:
        return None, False

    counts = normalized.value_counts()
    top_count = counts.max()
    tied = sorted(str(label) for label in counts[counts == top_count].index.tolist())
    inferred = tied[0]
    is_mixed = len(counts) > 1
    return inferred, is_mixed


def _compute_trade_volume_by_nationality(
    parquet_dir: Path,
    proprietary: bool,
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
    member_nationality: Optional[str] = None,
    members_nationality_path: Optional[Path] = None,
) -> Tuple[Dict[str, float], float]:
    """
    Aggregate flow-specific traded volume by executing-member nationality.

    Returns `(known_volume_by_nationality, unknown_volume)`, where the known map
    contains the canonical `it` / `foreign` labels only.
    """
    parquet_paths = list_metaorder_parquet_paths(parquet_dir)
    known_volume_by_nationality = {nationality: 0.0 for nationality in KNOWN_MEMBER_NATIONALITIES}
    unknown_volume = 0.0

    for path in parquet_paths:
        trades = load_trades_filtered_for_stats(
            path,
            proprietary=proprietary,
            trading_hours=trading_hours,
            member_nationality=member_nationality,
            members_nationality_path=members_nationality_path,
        )
        if trades.empty:
            continue

        volume = (
            pd.to_numeric(trades["Total Quantity Buy"], errors="coerce").fillna(0.0)
            + pd.to_numeric(trades["Total Quantity Sell"], errors="coerce").fillna(0.0)
        ).astype(float)
        nationality = (
            trades[AGGRESSIVE_MEMBER_NATIONALITY_COL]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        frame = pd.DataFrame({"nationality": nationality, "volume": volume})

        known = frame.loc[frame["nationality"].isin(KNOWN_MEMBER_NATIONALITIES)].copy()
        if not known.empty:
            grouped = known.groupby("nationality")["volume"].sum()
            for nat_label, total_volume in grouped.items():
                nat = str(nat_label)
                if nat in known_volume_by_nationality:
                    known_volume_by_nationality[nat] += float(total_volume)

        unknown_mask = ~frame["nationality"].isin(KNOWN_MEMBER_NATIONALITIES)
        unknown_volume += float(frame.loc[unknown_mask, "volume"].sum())

    return known_volume_by_nationality, float(unknown_volume)


def _compute_metaorder_activity_by_nationality(
    metaorders_dict_path: Path,
    parquet_dir: Path,
    proprietary: bool,
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
    member_nationality: Optional[str] = None,
    members_nationality_path: Optional[Path] = None,
) -> Tuple[Dict[str, int], Dict[str, float], int, float, int, float]:
    """
    Aggregate detected-metaorder counts and volumes by executing-member nationality.

    Returns
    -------
    tuple
        `(known_count_by_nationality, known_volume_by_nationality,
        unknown_count, unknown_volume, mixed_count, mixed_volume)`.
    """
    if not metaorders_dict_path.exists():
        raise FileNotFoundError(f"Metaorder dictionary not found: {metaorders_dict_path}")

    with metaorders_dict_path.open("rb") as handle:
        metaorders_dict_all = pickle.load(handle)

    parquet_paths = list_metaorder_parquet_paths(parquet_dir)
    known_count_by_nationality = {nationality: 0 for nationality in KNOWN_MEMBER_NATIONALITIES}
    known_volume_by_nationality = {nationality: 0.0 for nationality in KNOWN_MEMBER_NATIONALITIES}
    unknown_count = 0
    unknown_volume = 0.0
    mixed_count = 0
    mixed_volume = 0.0

    for path in parquet_paths:
        trades = load_trades_filtered_for_stats(
            path,
            proprietary=proprietary,
            trading_hours=trading_hours,
            member_nationality=member_nationality,
            members_nationality_path=members_nationality_path,
        )
        metaorders_dict = metaorders_dict_all.get(path.stem, {})
        if not metaorders_dict:
            continue

        n_trades = len(trades)
        nationality_arr = trades[AGGRESSIVE_MEMBER_NATIONALITY_COL].to_numpy()
        meta_volume_arr = (
            pd.to_numeric(trades["Total Quantity Buy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            + pd.to_numeric(trades["Total Quantity Sell"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )

        for metas in metaorders_dict.values():
            for meta in metas:
                if not meta:
                    continue
                meta_indices = np.asarray(meta, dtype=np.int64)
                if np.any((meta_indices < 0) | (meta_indices >= n_trades)):
                    raise IndexError(
                        f"Metaorder indices for ISIN {path.stem} fall outside the filtered trade tape. "
                        "Check that the dictionary and the selected filters match."
                    )

                meta_nat, is_mixed = _infer_metaorder_member_nationality(nationality_arr[meta_indices])
                meta_volume = float(meta_volume_arr[meta_indices].sum())

                if meta_nat in known_count_by_nationality:
                    known_count_by_nationality[meta_nat] += 1
                    known_volume_by_nationality[meta_nat] += meta_volume
                else:
                    unknown_count += 1
                    unknown_volume += meta_volume

                if is_mixed:
                    mixed_count += 1
                    mixed_volume += meta_volume

    return (
        known_count_by_nationality,
        known_volume_by_nationality,
        int(unknown_count),
        float(unknown_volume),
        int(mixed_count),
        float(mixed_volume),
    )


def _format_pct(value: float) -> str:
    """Return a compact percentage string, preserving missing values as `NA`."""
    return f"{value:.2f}%" if np.isfinite(value) else "NA"


def _build_flow_context_from_samples(samples: MetaorderDistributionSamples) -> Dict[str, object]:
    """Extract the nationality-context aggregates already stored in one sample bundle."""
    return {
        "trade_volume_by_nationality": {
            nationality: float(samples.trade_volume_by_nationality.get(nationality, 0.0))
            for nationality in KNOWN_MEMBER_NATIONALITIES
        },
        "metaorder_count_by_nationality": {
            nationality: int(samples.nationality_counts.get(nationality, 0))
            for nationality in KNOWN_MEMBER_NATIONALITIES
        },
        "metaorder_volume_by_nationality": {
            nationality: float(samples.metaorder_volume_by_nationality.get(nationality, 0.0))
            for nationality in KNOWN_MEMBER_NATIONALITIES
        },
        "unknown_trade_volume": float(samples.unknown_trade_volume),
        "unknown_metaorder_count": int(samples.unknown_nationality_metaorders),
        "unknown_metaorder_volume": float(samples.unknown_metaorder_volume),
        "mixed_metaorder_count": int(samples.mixed_nationality_metaorders),
        "mixed_metaorder_volume": float(samples.mixed_metaorder_volume),
    }


def _compute_flow_nationality_context(
    flow_label: str,
    metaorders_dict_path: Path,
    parquet_dir: Path,
    proprietary: bool,
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
    member_nationality: Optional[str] = None,
    members_nationality_path: Optional[Path] = None,
    show_progress: bool = False,
) -> Dict[str, object]:
    """
    Collect trade-volume and metaorder activity by nationality in one flow-specific pass.

    This combines the trade-tape denominator scan and the metaorder-volume scan
    so the nationality-context step does not reread every parquet file twice for
    the same flow group.
    """
    if not metaorders_dict_path.exists():
        raise FileNotFoundError(f"Metaorder dictionary not found: {metaorders_dict_path}")

    with metaorders_dict_path.open("rb") as handle:
        metaorders_dict_all = pickle.load(handle)

    parquet_paths = list_metaorder_parquet_paths(parquet_dir)
    n_paths = len(parquet_paths)
    context: Dict[str, object] = {
        "trade_volume_by_nationality": {nationality: 0.0 for nationality in KNOWN_MEMBER_NATIONALITIES},
        "metaorder_count_by_nationality": {nationality: 0 for nationality in KNOWN_MEMBER_NATIONALITIES},
        "metaorder_volume_by_nationality": {nationality: 0.0 for nationality in KNOWN_MEMBER_NATIONALITIES},
        "unknown_trade_volume": 0.0,
        "unknown_metaorder_count": 0,
        "unknown_metaorder_volume": 0.0,
        "mixed_metaorder_count": 0,
        "mixed_metaorder_volume": 0.0,
    }

    for idx, path in enumerate(parquet_paths, start=1):
        if show_progress and (idx == 1 or idx == n_paths or idx % 10 == 0):
            print(
                "[Metaorder summary][Nationality context] "
                f"flow={flow_label}, isin_progress={idx}/{n_paths}"
            )

        trades = load_trades_filtered_for_stats(
            path,
            proprietary=proprietary,
            trading_hours=trading_hours,
            member_nationality=member_nationality,
            members_nationality_path=members_nationality_path,
        )
        if trades.empty:
            continue

        volume = (
            pd.to_numeric(trades["Total Quantity Buy"], errors="coerce").fillna(0.0)
            + pd.to_numeric(trades["Total Quantity Sell"], errors="coerce").fillna(0.0)
        ).astype(float)
        nationality = (
            trades[AGGRESSIVE_MEMBER_NATIONALITY_COL]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        frame = pd.DataFrame({"nationality": nationality, "volume": volume})

        known = frame.loc[frame["nationality"].isin(KNOWN_MEMBER_NATIONALITIES)].copy()
        if not known.empty:
            grouped = known.groupby("nationality")["volume"].sum()
            for nat_label, total_volume in grouped.items():
                nat = str(nat_label)
                if nat in context["trade_volume_by_nationality"]:
                    context["trade_volume_by_nationality"][nat] += float(total_volume)

        unknown_mask = ~frame["nationality"].isin(KNOWN_MEMBER_NATIONALITIES)
        context["unknown_trade_volume"] += float(frame.loc[unknown_mask, "volume"].sum())

        metaorders_dict = metaorders_dict_all.get(path.stem, {})
        if not metaorders_dict:
            continue

        n_trades = len(trades)
        nationality_arr = trades[AGGRESSIVE_MEMBER_NATIONALITY_COL].to_numpy()
        meta_volume_arr = volume.to_numpy(dtype=float)
        for metas in metaorders_dict.values():
            for meta in metas:
                if not meta:
                    continue
                meta_indices = np.asarray(meta, dtype=np.int64)
                if np.any((meta_indices < 0) | (meta_indices >= n_trades)):
                    raise IndexError(
                        f"Metaorder indices for ISIN {path.stem} fall outside the filtered trade tape. "
                        "Check that the dictionary and the selected filters match."
                    )

                meta_nat, is_mixed = _infer_metaorder_member_nationality(nationality_arr[meta_indices])
                meta_volume = float(meta_volume_arr[meta_indices].sum())

                if meta_nat in context["metaorder_count_by_nationality"]:
                    context["metaorder_count_by_nationality"][meta_nat] += 1
                    context["metaorder_volume_by_nationality"][meta_nat] += meta_volume
                else:
                    context["unknown_metaorder_count"] += 1
                    context["unknown_metaorder_volume"] += meta_volume

                if is_mixed:
                    context["mixed_metaorder_count"] += 1
                    context["mixed_metaorder_volume"] += meta_volume

    return context


def compute_nationality_context_table(
    parquet_dir: Path,
    proprietary_dict_path: Path,
    client_dict_path: Path,
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
    member_nationality: Optional[str] = None,
    members_nationality_path: Optional[Path] = None,
    client_samples: Optional[MetaorderDistributionSamples] = None,
    proprietary_samples: Optional[MetaorderDistributionSamples] = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Summary
    -------
    Compare overall traded volume with detected metaorder activity by nationality.

    Parameters
    ----------
    parquet_dir : Path
        Directory containing per-ISIN parquet trade tapes.
    proprietary_dict_path : Path
        Path to the proprietary `metaorders_dict_all_*` pickle.
    client_dict_path : Path
        Path to the client `metaorders_dict_all_*` pickle.
    trading_hours : tuple[str, str], default=("09:30:00", "17:30:00")
        Inclusive trading-hours filter applied to the trade tapes.
    member_nationality : str | None, default=None
        Optional aggressive-member nationality filter. When provided, the
        reported shares are computed inside that selected slice only.
    members_nationality_path : Path | None, default=None
        Optional metadata parquet used to backfill nationality labels.
    client_samples : MetaorderDistributionSamples | None, default=None
        Optional precomputed client-flow samples. When provided together with
        `proprietary_samples`, the table is built directly from those already
        aggregated statistics instead of rescanning the parquet tapes.
    proprietary_samples : MetaorderDistributionSamples | None, default=None
        Optional precomputed proprietary-flow samples paired with
        `client_samples`.
    show_progress : bool, default=False
        If True, print coarse progress updates while the parquet tapes are
        scanned for the nationality-context aggregates.

    Returns
    -------
    pd.DataFrame
        One row per `(flow, nationality)` with trade-volume shares, metaorder
        shares, metaorder-volume shares, and the fraction of each nationality's
        traded volume that is captured by detected metaorders.

    Notes
    -----
    Shares across nationalities use only the subset of trades/metaorders with
    known executing-member nationality (`it` or `foreign`). This keeps the
    denominators aligned with the existing nationality-share table used in the
    paper.

    Examples
    --------
    >>> isinstance(
    ...     compute_nationality_context_table(
    ...         parquet_dir=Path("data/parquet"),
    ...         proprietary_dict_path=Path("out_files/prop.pkl"),
    ...         client_dict_path=Path("out_files/client.pkl"),
    ...     ),
    ...     pd.DataFrame,
    ... )
    True
    """
    flow_specs = (
        ("client", False, client_dict_path),
        ("proprietary", True, proprietary_dict_path),
    )
    flow_contexts: Dict[str, Dict[str, object]] = {}

    if client_samples is not None and proprietary_samples is not None:
        flow_contexts["client"] = _build_flow_context_from_samples(client_samples)
        flow_contexts["proprietary"] = _build_flow_context_from_samples(proprietary_samples)
    else:
        for flow_label, proprietary, dict_path in flow_specs:
            flow_contexts[flow_label] = _compute_flow_nationality_context(
                flow_label=flow_label,
                metaorders_dict_path=dict_path,
                parquet_dir=parquet_dir,
                proprietary=proprietary,
                trading_hours=trading_hours,
                member_nationality=member_nationality,
                members_nationality_path=members_nationality_path,
                show_progress=show_progress,
            )

    flow_contexts["all_metaorders"] = {
        "trade_volume_by_nationality": {
            nationality: float(flow_contexts["client"]["trade_volume_by_nationality"][nationality])
            + float(flow_contexts["proprietary"]["trade_volume_by_nationality"][nationality])
            for nationality in KNOWN_MEMBER_NATIONALITIES
        },
        "metaorder_count_by_nationality": {
            nationality: int(flow_contexts["client"]["metaorder_count_by_nationality"][nationality])
            + int(flow_contexts["proprietary"]["metaorder_count_by_nationality"][nationality])
            for nationality in KNOWN_MEMBER_NATIONALITIES
        },
        "metaorder_volume_by_nationality": {
            nationality: float(flow_contexts["client"]["metaorder_volume_by_nationality"][nationality])
            + float(flow_contexts["proprietary"]["metaorder_volume_by_nationality"][nationality])
            for nationality in KNOWN_MEMBER_NATIONALITIES
        },
        "unknown_trade_volume": float(flow_contexts["client"]["unknown_trade_volume"])
        + float(flow_contexts["proprietary"]["unknown_trade_volume"]),
        "unknown_metaorder_count": int(flow_contexts["client"]["unknown_metaorder_count"])
        + int(flow_contexts["proprietary"]["unknown_metaorder_count"]),
        "unknown_metaorder_volume": float(flow_contexts["client"]["unknown_metaorder_volume"])
        + float(flow_contexts["proprietary"]["unknown_metaorder_volume"]),
        "mixed_metaorder_count": int(flow_contexts["client"]["mixed_metaorder_count"])
        + int(flow_contexts["proprietary"]["mixed_metaorder_count"]),
        "mixed_metaorder_volume": float(flow_contexts["client"]["mixed_metaorder_volume"])
        + float(flow_contexts["proprietary"]["mixed_metaorder_volume"]),
    }

    rows: List[Dict[str, object]] = []
    for flow_label in ("client", "proprietary", "all_metaorders"):
        context = flow_contexts[flow_label]
        known_trade_volume_total = float(sum(context["trade_volume_by_nationality"].values()))
        known_metaorder_count_total = int(sum(context["metaorder_count_by_nationality"].values()))
        known_metaorder_volume_total = float(sum(context["metaorder_volume_by_nationality"].values()))

        for nationality in KNOWN_MEMBER_NATIONALITIES:
            trade_volume = float(context["trade_volume_by_nationality"][nationality])
            metaorder_count = int(context["metaorder_count_by_nationality"][nationality])
            metaorder_volume = float(context["metaorder_volume_by_nationality"][nationality])
            rows.append(
                {
                    "flow": flow_label,
                    "nationality": nationality,
                    "trade_volume": trade_volume,
                    "known_trade_volume_total": known_trade_volume_total,
                    "share_of_known_trade_volume_pct": (
                        100.0 * trade_volume / known_trade_volume_total if known_trade_volume_total > 0.0 else np.nan
                    ),
                    "metaorder_count": metaorder_count,
                    "known_metaorder_count_total": known_metaorder_count_total,
                    "share_of_known_metaorders_pct": (
                        100.0 * metaorder_count / known_metaorder_count_total
                        if known_metaorder_count_total > 0
                        else np.nan
                    ),
                    "metaorder_volume": metaorder_volume,
                    "known_metaorder_volume_total": known_metaorder_volume_total,
                    "share_of_known_metaorder_volume_pct": (
                        100.0 * metaorder_volume / known_metaorder_volume_total
                        if known_metaorder_volume_total > 0.0
                        else np.nan
                    ),
                    "share_of_nationality_volume_in_metaorders_pct": (
                        100.0 * metaorder_volume / trade_volume if trade_volume > 0.0 else np.nan
                    ),
                    "unknown_trade_volume": float(context["unknown_trade_volume"]),
                    "unknown_metaorder_count": int(context["unknown_metaorder_count"]),
                    "unknown_metaorder_volume": float(context["unknown_metaorder_volume"]),
                    "mixed_metaorder_count": int(context["mixed_metaorder_count"]),
                    "mixed_metaorder_volume": float(context["mixed_metaorder_volume"]),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    flow_order = pd.CategoricalDtype(["client", "proprietary", "all_metaorders"], ordered=True)
    nationality_order = pd.CategoricalDtype(list(KNOWN_MEMBER_NATIONALITIES), ordered=True)
    table["flow"] = table["flow"].astype(flow_order)
    table["nationality"] = table["nationality"].astype(nationality_order)
    table = table.sort_values(["flow", "nationality"]).reset_index(drop=True)
    return table


def build_nationality_context_figure(
    nationality_context_table: pd.DataFrame,
    condition_on_client_proprietary: bool = True,
) -> go.Figure:
    """
    Summary
    -------
    Build the nationality context figure linking overall flow to metaorder activity.

    Parameters
    ----------
    nationality_context_table : pd.DataFrame
        Output of `compute_nationality_context_table`.
    condition_on_client_proprietary : bool, default=True
        If True, show client and proprietary flow in separate panels. If False,
        show the pooled `all_metaorders` panel only.

    Returns
    -------
    go.Figure
        Plotly grouped-bar figure with the paper-ready nationality metrics.

    Notes
    -----
    The three displayed metrics are:
    1. share of known traded volume by nationality,
    2. share of known detected metaorders by nationality,
    3. share of each nationality's traded volume executed through metaorders.
    """
    required_columns = {
        "flow",
        "nationality",
        "trade_volume",
        "known_trade_volume_total",
        "share_of_known_trade_volume_pct",
        "metaorder_count",
        "known_metaorder_count_total",
        "share_of_known_metaorders_pct",
        "metaorder_volume",
        "share_of_nationality_volume_in_metaorders_pct",
    }
    missing_columns = required_columns.difference(nationality_context_table.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns for nationality context figure: {sorted(missing_columns)}")
    if nationality_context_table.empty:
        raise ValueError("The nationality context table is empty.")

    metric_specs = (
        ("share_of_known_trade_volume_pct", "Share of known<br>traded volume"),
        ("share_of_known_metaorders_pct", "Share of known<br>metaorders"),
        ("share_of_nationality_volume_in_metaorders_pct", "Own volume<br>in metaorders"),
    )
    flow_labels = {
        "client": "Client",
        "proprietary": "Proprietary",
        "all_metaorders": "All metaorders",
    }
    flows = ["client", "proprietary"] if condition_on_client_proprietary else ["all_metaorders"]
    fig = make_subplots(
        rows=1,
        cols=len(flows),
        shared_yaxes=condition_on_client_proprietary,
        horizontal_spacing=0.08,
        column_titles=[flow_labels[flow] for flow in flows],
    )

    panel_max = 0.0
    for col_idx, flow in enumerate(flows, start=1):
        flow_table = nationality_context_table.loc[nationality_context_table["flow"] == flow].copy()
        if flow_table.empty:
            subplot = fig.get_subplot(1, col_idx)
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"{_axis_ref_name(subplot.xaxis.plotly_name)} domain",
                yref=f"{_axis_ref_name(subplot.yaxis.plotly_name)} domain",
                text="No nationality data",
                showarrow=False,
                font=dict(size=ANNOTATION_FONT_SIZE),
            )
            continue

        for nationality in KNOWN_MEMBER_NATIONALITIES:
            row = flow_table.loc[flow_table["nationality"] == nationality]
            if row.empty:
                y_values: List[Optional[float]] = [None, None, None]
                customdata = np.full((len(metric_specs), 5), np.nan, dtype=float)
            else:
                record = row.iloc[0]
                y_values = []
                for column, _ in metric_specs:
                    value = pd.to_numeric(pd.Series([record[column]]), errors="coerce").iloc[0]
                    if pd.notna(value):
                        panel_max = max(panel_max, float(value))
                        y_values.append(float(value))
                    else:
                        y_values.append(None)
                customdata = np.tile(
                    np.asarray(
                        [
                            float(record["trade_volume"]),
                            float(record["known_trade_volume_total"]),
                            float(record["metaorder_count"]),
                            float(record["known_metaorder_count_total"]),
                            float(record["metaorder_volume"]),
                        ],
                        dtype=float,
                    ),
                    (len(metric_specs), 1),
                )

            fig.add_trace(
                go.Bar(
                    x=[label for _, label in metric_specs],
                    y=y_values,
                    name=NATIONALITY_LABELS[nationality],
                    marker_color=NATIONALITY_COLORS[nationality],
                    legendgroup=nationality,
                    offsetgroup=nationality,
                    customdata=customdata,
                    hovertemplate=(
                        "Nationality: %{fullData.name}"
                        "<br>Metric: %{x}"
                        "<br>Value: %{y:.2f}%"
                        "<br>Traded volume: %{customdata[0]:,.0f}"
                        "<br>Known total traded volume: %{customdata[1]:,.0f}"
                        "<br>Metaorders: %{customdata[2]:,.0f}"
                        "<br>Known total metaorders: %{customdata[3]:,.0f}"
                        "<br>Metaorder volume: %{customdata[4]:,.0f}<extra></extra>"
                    ),
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(row=1, col=col_idx)

    y_upper = max(100.0, panel_max + 12.0)
    for col_idx in range(1, len(flows) + 1):
        fig.update_yaxes(title_text="Percentage (%)" if col_idx == 1 else None, range=[0.0, y_upper], row=1, col=col_idx)

    fig.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        barmode="group",
        margin=dict(l=70, r=25, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def _normalize_member_identifier(member: object) -> Optional[str]:
    """Normalize a member identifier so trade-tape values and dictionary keys can be merged."""
    if pd.isna(member):
        return None

    if isinstance(member, (int, np.integer)):
        return str(int(member))

    if isinstance(member, (float, np.floating)):
        member_float = float(member)
        if not np.isfinite(member_float):
            return None
        if member_float.is_integer():
            return str(int(member_float))
        return format(member_float, "g")

    member_text = str(member).strip()
    if member_text == "" or member_text.lower() in {"nan", "none"}:
        return None

    try:
        numeric_value = float(member_text)
    except ValueError:
        return member_text

    if not np.isfinite(numeric_value):
        return member_text
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return member_text


def _canonicalize_member_count_map(counts: Mapping[object, int] | pd.Series) -> pd.Series:
    """Canonicalize per-member counts to a string-indexed integer series."""
    if isinstance(counts, pd.Series):
        if counts.empty:
            return pd.Series(dtype="int64")
        series = counts.copy()
    else:
        if len(counts) == 0:
            return pd.Series(dtype="int64")
        series = pd.Series(dict(counts))

    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return pd.Series(dtype="int64")

    series.index = pd.Index([_normalize_member_identifier(member) for member in series.index], dtype="object")
    valid_mask = [member is not None for member in series.index]
    series = series.loc[valid_mask]
    if series.empty:
        return pd.Series(dtype="int64")

    return series.groupby(level=0).sum().round().astype("int64").sort_index()


def _sum_member_count_maps(*counts_by_member: Mapping[object, int] | pd.Series) -> pd.Series:
    """
    Summary
    -------
    Sum one or more per-member count mappings on a canonical member index.

    Parameters
    ----------
    *counts_by_member : Mapping[object, int] | pd.Series
        Per-member count mappings or series to add together.

    Returns
    -------
    pd.Series
        Integer counts indexed by canonicalized member identifier.

    Notes
    -----
    Missing members in any individual input are treated as zero.
    """
    total = pd.Series(dtype="float64")
    for counts in counts_by_member:
        mapping = counts.to_dict() if isinstance(counts, pd.Series) else dict(counts)
        series = _canonicalize_member_count_map(mapping)
        if series.empty:
            continue
        total = series.astype(float) if total.empty else total.add(series.astype(float), fill_value=0.0)

    if total.empty:
        return pd.Series(dtype="int64")
    return total.round().astype("int64").sort_index()


def _compute_total_trades_by_member(
    parquet_dir: Path,
    proprietary: bool,
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
    member_nationality: Optional[str] = None,
    members_nationality_path: Optional[Path] = None,
) -> pd.Series:
    """
    Summary
    -------
    Count filtered aggressive trades for each member across all ISIN tapes.

    Parameters
    ----------
    parquet_dir : Path
        Directory containing per-ISIN parquet trade tapes.
    proprietary : bool
        If True, keep only proprietary aggressive trades; otherwise keep only
        client aggressive trades.
    trading_hours : tuple[str, str], default=("09:30:00", "17:30:00")
        Inclusive trading-hours filter applied to the trade tapes.
    member_nationality : str | None, default=None
        Optional aggressive-member nationality filter.
    members_nationality_path : Path | None, default=None
        Optional metadata parquet used to backfill nationality labels.

    Returns
    -------
    pd.Series
        Integer trade counts indexed by canonicalized member identifier.

    Notes
    -----
    The filtering intentionally mirrors the metaorder-sample construction so the
    trade counts are aligned with the loaded metaorder dictionaries.

    Examples
    --------
    >>> isinstance(_compute_total_trades_by_member(Path("data/parquet"), True), pd.Series)
    True
    """
    parquet_paths = list_metaorder_parquet_paths(parquet_dir)
    if not parquet_paths:
        return pd.Series(dtype="int64")

    total_counts: Dict[str, int] = {}
    for path in parquet_paths:
        trades = load_trades_filtered_for_stats(
            path,
            proprietary=proprietary,
            trading_hours=trading_hours,
            member_nationality=member_nationality,
            members_nationality_path=members_nationality_path,
        )
        if trades.empty or "ID Member" not in trades.columns:
            continue

        normalized_members = pd.Series(
            [_normalize_member_identifier(member) for member in trades["ID Member"].tolist()],
            dtype="object",
        ).dropna()
        if normalized_members.empty:
            continue

        per_isin_counts = normalized_members.value_counts(sort=False)
        for member, count in per_isin_counts.items():
            total_counts[str(member)] = total_counts.get(str(member), 0) + int(count)

    if not total_counts:
        return pd.Series(dtype="int64")
    return pd.Series(total_counts, dtype="int64").sort_index()


def _compute_total_child_orders_by_member(metaorders_dict_path: Path) -> pd.Series:
    """
    Summary
    -------
    Count total child orders across all detected metaorders for each member.

    Parameters
    ----------
    metaorders_dict_path : Path
        Pickle path to the `metaorders_dict_all_*` object for one flow slice.

    Returns
    -------
    pd.Series
        Integer child-order counts indexed by canonicalized member identifier.

    Notes
    -----
    Each child order corresponds to one trade index stored inside a detected
    metaorder. The returned count therefore sums metaorder lengths member by
    member across all ISINs.
    """
    if not metaorders_dict_path.exists():
        raise FileNotFoundError(f"Metaorder dictionary not found: {metaorders_dict_path}")

    with metaorders_dict_path.open("rb") as handle:
        metaorders_dict_all = pickle.load(handle)

    total_child_orders: Dict[object, int] = {}
    for metaorders_dict in metaorders_dict_all.values():
        for member, metas in metaorders_dict.items():
            total_child_orders[member] = total_child_orders.get(member, 0) + sum(len(meta) for meta in metas if meta)

    if not total_child_orders:
        return pd.Series(dtype="int64")
    return _canonicalize_member_count_map(total_child_orders)


def _build_member_profile_table(
    metaorder_counts: Mapping[object, int] | pd.Series,
    child_order_counts: Mapping[object, int] | pd.Series,
    trade_counts: Mapping[object, int] | pd.Series,
    flow_label: str,
) -> pd.DataFrame:
    """
    Summary
    -------
    Build the member-level table used by the rank-and-scatter profile figure.

    Parameters
    ----------
    metaorder_counts : Mapping[object, int] | pd.Series
        Number of detected metaorders for each member.
    child_order_counts : Mapping[object, int] | pd.Series
        Total number of child orders across all detected metaorders for each
        member.
    trade_counts : Mapping[object, int] | pd.Series
        Total filtered aggressive trades for each member.
    flow_label : str
        Flow-group label attached to every returned row.

    Returns
    -------
    pd.DataFrame
        Table with columns `Member`, `n_metaorders`, `total_child_orders`,
        `total_trades`, and `flow`, one row per member.

    Notes
    -----
    Members that traded in the filtered tape but have zero detected metaorders
    are retained with `n_metaorders = 0` and `total_child_orders = 0`, because
    they still define valid scatter points.

    Examples
    --------
    >>> out = _build_member_profile_table({1: 2}, {1: 7}, {1: 5, 2: 3}, "client")
    >>> out[["Member", "n_metaorders", "total_child_orders", "total_trades"]].to_dict("records")
    [{'Member': '1', 'n_metaorders': 2, 'total_child_orders': 7, 'total_trades': 5}, {'Member': '2', 'n_metaorders': 0, 'total_child_orders': 0, 'total_trades': 3}]
    """
    metaorder_series = _canonicalize_member_count_map(metaorder_counts)
    child_order_series = _canonicalize_member_count_map(child_order_counts)
    trade_series = _canonicalize_member_count_map(trade_counts)

    member_index = sorted(set(metaorder_series.index).union(set(child_order_series.index)).union(set(trade_series.index)))
    if not member_index:
        return pd.DataFrame(columns=["Member", "n_metaorders", "total_child_orders", "total_trades", "flow"])

    table = pd.DataFrame(index=pd.Index(member_index, name="Member"))
    table["n_metaorders"] = metaorder_series.reindex(table.index, fill_value=0).astype("int64")
    table["total_child_orders"] = child_order_series.reindex(table.index, fill_value=0).astype("int64")
    table["total_trades"] = trade_series.reindex(table.index, fill_value=0).astype("int64")
    table = table.loc[
        (table["n_metaorders"] > 0) | (table["total_child_orders"] > 0) | (table["total_trades"] > 0)
    ].copy()
    table["flow"] = str(flow_label)
    return table.reset_index()


def _build_member_metaorder_rank_table(member_table: pd.DataFrame) -> pd.DataFrame:
    """
    Summary
    -------
    Rank members by decreasing number of detected metaorders.

    Parameters
    ----------
    member_table : pd.DataFrame
        Member-level table produced by `_build_member_profile_table`.

    Returns
    -------
    pd.DataFrame
        Table with columns `Member`, `n_metaorders`, `flow`, and `rank`.

    Notes
    -----
    Members with `n_metaorders = 0` are excluded because the rank plot uses
    log-scaled axes and only the detected-metaorder profile is shown.

    Examples
    --------
    >>> ranked = _build_member_metaorder_rank_table(
    ...     pd.DataFrame({"Member": ["1", "2"], "n_metaorders": [5, 2], "total_trades": [10, 4], "flow": ["client", "client"]})
    ... )
    >>> ranked[["Member", "rank"]].to_dict("records")
    [{'Member': '1', 'rank': 1}, {'Member': '2', 'rank': 2}]
    """
    if member_table.empty:
        return pd.DataFrame(columns=["Member", "n_metaorders", "flow", "rank"])

    ranked = member_table.loc[:, ["Member", "n_metaorders", "flow"]].copy()
    ranked["Member"] = ranked["Member"].astype(str)
    ranked["n_metaorders"] = pd.to_numeric(ranked["n_metaorders"], errors="coerce").fillna(0.0)
    ranked = ranked.loc[ranked["n_metaorders"] > 0].copy()
    if ranked.empty:
        return pd.DataFrame(columns=["Member", "n_metaorders", "flow", "rank"])

    ranked.sort_values(["n_metaorders", "Member"], ascending=[False, True], kind="mergesort", inplace=True)
    ranked.reset_index(drop=True, inplace=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    return ranked


def build_daily_metaorder_share_table(
    total_market_volume_by_day: pd.Series,
    proprietary_metaorder_volume_by_day: pd.Series,
    client_metaorder_volume_by_day: pd.Series,
) -> pd.DataFrame:
    """
    Summary
    -------
    Build the day-level proprietary/client metaorder-share table.

    Parameters
    ----------
    total_market_volume_by_day : pd.Series
        Total daily market volume for one ISIN.
    proprietary_metaorder_volume_by_day : pd.Series
        Total proprietary metaorder volume for the same ISIN.
    client_metaorder_volume_by_day : pd.Series
        Total client metaorder volume for the same ISIN.

    Returns
    -------
    pd.DataFrame
        One row per day with denominator volume, group numerators, and ratios.

    Notes
    -----
    Missing proprietary/client days are filled with zero before the daily ratios
    are computed.
    """
    total = pd.Series(total_market_volume_by_day, dtype=float)
    total.index = pd.Index(pd.to_datetime(total.index).date, name="Date")
    total = total.groupby(level=0).sum().sort_index()
    if total.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "total_market_volume",
                "proprietary_metaorder_volume",
                "client_metaorder_volume",
                "proprietary_ratio",
                "client_ratio",
                "total_ratio",
            ]
        )

    prop = pd.Series(proprietary_metaorder_volume_by_day, dtype=float)
    prop.index = pd.Index(pd.to_datetime(prop.index).date, name="Date")
    prop = prop.groupby(level=0).sum()

    client = pd.Series(client_metaorder_volume_by_day, dtype=float)
    client.index = pd.Index(pd.to_datetime(client.index).date, name="Date")
    client = client.groupby(level=0).sum()

    daily = pd.DataFrame({"total_market_volume": total})
    daily["proprietary_metaorder_volume"] = prop.reindex(daily.index, fill_value=0.0).astype(float)
    daily["client_metaorder_volume"] = client.reindex(daily.index, fill_value=0.0).astype(float)
    daily = daily.loc[daily["total_market_volume"] > 0.0].copy()
    if daily.empty:
        return daily.reset_index()

    denominator = daily["total_market_volume"].astype(float)
    daily["proprietary_ratio"] = daily["proprietary_metaorder_volume"] / denominator
    daily["client_ratio"] = daily["client_metaorder_volume"] / denominator
    daily["total_ratio"] = daily["proprietary_ratio"] + daily["client_ratio"]
    return daily.reset_index()


def compute_daily_metaorder_share_table(
    parquet_dir: Path,
    proprietary_dict_path: Path,
    client_dict_path: Path,
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
    member_nationality: Optional[str] = None,
    members_nationality_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Summary
    -------
    Compute day-level proprietary/client metaorder shares for each ISIN.

    Parameters
    ----------
    parquet_dir : Path
        Directory containing per-ISIN parquet trade tapes.
    proprietary_dict_path : Path
        Path to the proprietary `metaorders_dict_all_*` pickle.
    client_dict_path : Path
        Path to the client `metaorders_dict_all_*` pickle.
    trading_hours : tuple[str, str], default=("09:30:00", "17:30:00")
        Inclusive trading-hours filter applied to all numerator/denominator
        volumes.
    member_nationality : str | None, default=None
        Optional aggressive-member nationality filter applied only to the
        numerator-side metaorder slices.
    members_nationality_path : Path | None, default=None
        Optional member metadata parquet used to backfill nationality labels.

    Returns
    -------
    pd.DataFrame
        Day-level table with one row per `(ISIN, Date)`.

    Notes
    -----
    The denominator always uses the full capacity-unfiltered tape for the ISIN.
    """
    with proprietary_dict_path.open("rb") as handle:
        metaorders_prop_all = pickle.load(handle)
    with client_dict_path.open("rb") as handle:
        metaorders_client_all = pickle.load(handle)

    parquet_paths = list_metaorder_parquet_paths(parquet_dir)
    if not parquet_paths:
        return pd.DataFrame()

    daily_tables: List[pd.DataFrame] = []
    for path in parquet_paths:
        trades_full = load_trades_filtered_for_stats(
            path,
            proprietary=None,
            trading_hours=trading_hours,
            member_nationality=None,
            members_nationality_path=members_nationality_path,
        )
        trades_prop = load_trades_filtered_for_stats(
            path,
            proprietary=True,
            trading_hours=trading_hours,
            member_nationality=member_nationality,
            members_nationality_path=members_nationality_path,
        )
        trades_client = load_trades_filtered_for_stats(
            path,
            proprietary=False,
            trading_hours=trading_hours,
            member_nationality=member_nationality,
            members_nationality_path=members_nationality_path,
        )

        daily_table = build_daily_metaorder_share_table(
            _daily_total_market_volume(trades_full),
            _daily_metaorder_volume(metaorders_prop_all.get(path.stem, {}), trades_prop),
            _daily_metaorder_volume(metaorders_client_all.get(path.stem, {}), trades_client),
        )
        if daily_table.empty:
            continue
        daily_table.insert(0, "ISIN", path.stem)
        daily_tables.append(daily_table)

    if not daily_tables:
        return pd.DataFrame(
            columns=[
                "ISIN",
                "Date",
                "total_market_volume",
                "proprietary_metaorder_volume",
                "client_metaorder_volume",
                "proprietary_ratio",
                "client_ratio",
                "total_ratio",
            ]
        )
    return pd.concat(daily_tables, ignore_index=True)


def build_mean_daily_metaorder_share_figure(
    daily_share_table: pd.DataFrame,
    condition_on_client_proprietary: bool = True,
) -> go.Figure:
    """
    Summary
    -------
    Build the mean daily metaorder-volume-share figure by ISIN.

    Parameters
    ----------
    daily_share_table : pd.DataFrame
        Output of `compute_daily_metaorder_share_table`.
    condition_on_client_proprietary : bool, default=True
        If True, show proprietary and client contributions separately. If
        False, plot their pooled total share only.

    Returns
    -------
    go.Figure
        Plotly figure of mean daily metaorder-volume share by ISIN.

    Notes
    -----
    Ratios are converted from decimals to percentages for plotting. The pooled
    mode uses the `total_ratio` column when available, or reconstructs it as
    `proprietary_ratio + client_ratio` for caller-provided helper tables.
    """
    required_cols = {"ISIN", "Date"}
    if condition_on_client_proprietary:
        required_cols.update({"proprietary_ratio", "client_ratio"})
    else:
        required_cols.add("total_ratio")
    missing = required_cols.difference(daily_share_table.columns)
    if missing:
        raise KeyError(f"Missing required columns for the daily-share figure: {sorted(missing)}")
    if daily_share_table.empty:
        raise ValueError("The daily-share table is empty.")

    grouped = daily_share_table.assign(ISIN=daily_share_table["ISIN"].astype(str))
    if condition_on_client_proprietary:
        grouped = grouped.assign(
            proprietary_ratio=pd.to_numeric(daily_share_table["proprietary_ratio"], errors="coerce"),
            client_ratio=pd.to_numeric(daily_share_table["client_ratio"], errors="coerce"),
        )
        # Tests and ad hoc callers sometimes pass only the split ratios; derive
        # the pooled total here so the plotting helper stays easy to reuse.
        if "total_ratio" in daily_share_table.columns:
            grouped = grouped.assign(
                total_ratio=pd.to_numeric(daily_share_table["total_ratio"], errors="coerce"),
            )
        else:
            grouped = grouped.assign(
                total_ratio=grouped["proprietary_ratio"] + grouped["client_ratio"],
            )
        grouped = (
            grouped.groupby("ISIN", as_index=False)
            .agg(
                proprietary_ratio=("proprietary_ratio", "mean"),
                client_ratio=("client_ratio", "mean"),
                total_ratio=("total_ratio", "mean"),
                n_days=("Date", "nunique"),
            )
            .sort_values("ISIN")
            .reset_index(drop=True)
        )
    else:
        grouped = grouped.assign(
            total_ratio=pd.to_numeric(daily_share_table["total_ratio"], errors="coerce"),
        )
        grouped = (
            grouped.groupby("ISIN", as_index=False)
            .agg(
                total_ratio=("total_ratio", "mean"),
                n_days=("Date", "nunique"),
            )
            .sort_values("ISIN")
            .reset_index(drop=True)
        )

    fig = go.Figure()
    if condition_on_client_proprietary:
        trace_specs = (
            ("Proprietary", "proprietary_ratio", COLOR_PROPRIETARY),
            ("Client", "client_ratio", COLOR_CLIENT),
        )
    else:
        trace_specs = (("All metaorders", "total_ratio", COLOR_ALL_METAORDERS),)

    for label, column, color in trace_specs:
        fig.add_trace(
            go.Bar(
                x=grouped["ISIN"].tolist(),
                y=(100.0 * grouped[column].to_numpy(dtype=float)).tolist(),
                name=label,
                marker_color=color,
                customdata=np.column_stack(
                    [
                        100.0 * grouped["total_ratio"].to_numpy(dtype=float),
                        grouped["n_days"].to_numpy(dtype=int),
                    ]
                ),
                hovertemplate=(
                    "ISIN %{x}"
                    f"<br>{label} mean daily share: %{{y:.2f}}%"
                    "<br>Total mean daily share: %{customdata[0]:.2f}%"
                    "<br>Trading days: %{customdata[1]:.0f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        xaxis_title="ISIN",
        yaxis_title="Percentage of daily market volume (%)",
        barmode="stack" if condition_on_client_proprietary else "group",
        showlegend=condition_on_client_proprietary,
        xaxis_tickangle=90,
        bargap=0.2,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def build_nationality_share_figure(
    client_samples: MetaorderDistributionSamples,
    proprietary_samples: MetaorderDistributionSamples,
    condition_on_client_proprietary: bool = True,
) -> go.Figure:
    """
    Summary
    -------
    Build the aggressive-member nationality share figure.

    Parameters
    ----------
    client_samples : MetaorderDistributionSamples
        Aggregated samples for the client flow slice.
    proprietary_samples : MetaorderDistributionSamples
        Aggregated samples for the proprietary flow slice.
    condition_on_client_proprietary : bool, default=True
        If True, show client and proprietary shares in separate panels. If
        False, pool both flow groups into one overall panel.

    Returns
    -------
    go.Figure
        Plotly bar figure for the selected grouping mode.

    Notes
    -----
    Percentages are computed on metaorders with known aggressive-member
    nationality only.
    """
    if not condition_on_client_proprietary:
        counts = {
            "it": int(client_samples.nationality_counts["it"] + proprietary_samples.nationality_counts["it"]),
            "foreign": int(
                client_samples.nationality_counts["foreign"] + proprietary_samples.nationality_counts["foreign"]
            ),
        }
        known = int(counts["it"] + counts["foreign"])
        fig = go.Figure()
        if known > 0:
            pct_it = 100.0 * counts["it"] / known
            pct_foreign = 100.0 * counts["foreign"] / known
            fig.add_trace(
                go.Bar(
                    x=["it", "foreign"],
                    y=[pct_it, pct_foreign],
                    marker_color=[THEME_COLORWAY[0], THEME_COLORWAY[2]],
                    text=[
                        f"{pct_it:.1f}%<br>(n={counts['it']})",
                        f"{pct_foreign:.1f}%<br>(n={counts['foreign']})",
                    ],
                    textposition="outside",
                    showlegend=False,
                    hovertemplate="Nationality %{x}<br>Share %{y:.2f}%<extra></extra>",
                )
            )
            y_upper = max(100.0, pct_it, pct_foreign) + 12.0
        else:
            y_upper = 100.0
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="x domain",
                yref="y domain",
                text="No known nationality",
                showarrow=False,
                font=dict(size=ANNOTATION_FONT_SIZE),
            )

        fig.update_layout(
            template=PLOTLY_TEMPLATE_NAME,
            xaxis_title="Member nationality",
            yaxis_title="Percentage of metaorders (%)",
            margin=dict(l=70, r=25, t=50, b=55),
        )
        fig.update_yaxes(range=[0.0, y_upper])
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        column_titles=["Client", "Proprietary"],
    )

    panel_max = 0.0
    panel_specs = (
        (client_samples, 1),
        (proprietary_samples, 2),
    )
    for samples, col_idx in panel_specs:
        counts = samples.nationality_counts
        known = int(counts["it"] + counts["foreign"])
        if known > 0:
            pct_it = 100.0 * counts["it"] / known
            pct_foreign = 100.0 * counts["foreign"] / known
            panel_max = max(panel_max, pct_it, pct_foreign)
            fig.add_trace(
                go.Bar(
                    x=["it", "foreign"],
                    y=[pct_it, pct_foreign],
                    marker_color=[THEME_COLORWAY[0], THEME_COLORWAY[2]],
                    text=[
                        f"{pct_it:.1f}%<br>(n={counts['it']})",
                        f"{pct_foreign:.1f}%<br>(n={counts['foreign']})",
                    ],
                    textposition="outside",
                    showlegend=False,
                    hovertemplate=(
                        "Nationality %{x}"
                        "<br>Share %{y:.2f}%<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
        else:
            subplot = fig.get_subplot(1, col_idx)
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref=f"{_axis_ref_name(subplot.xaxis.plotly_name)} domain",
                yref=f"{_axis_ref_name(subplot.yaxis.plotly_name)} domain",
                text="No known nationality",
                showarrow=False,
                font=dict(size=ANNOTATION_FONT_SIZE),
            )

        fig.update_xaxes(title_text="Member nationality", row=1, col=col_idx)

    fig.update_yaxes(title_text="Percentage of metaorders (%)", row=1, col=1)
    fig.update_yaxes(range=[0.0, max(100.0, panel_max + 12.0)], row=1, col=1)
    fig.update_yaxes(range=[0.0, max(100.0, panel_max + 12.0)], row=1, col=2)
    fig.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        margin=dict(l=70, r=25, t=50, b=55),
    )
    return fig


def _build_member_metaorder_profile_figure(
    member_table_specs: Sequence[Tuple[pd.DataFrame, str, str]],
    show_legend: bool = True,
) -> go.Figure:
    """
    Summary
    -------
    Build the member-profile figure with rank and trade-count comparisons.

    Parameters
    ----------
    member_table_specs : Sequence[tuple[pd.DataFrame, str, str]]
        Sequence of `(member_table, label, color)` triples defining the traces
        to overlay in the rank and scatter panels.
    show_legend : bool, default=True
        Whether to display the figure legend.

    Returns
    -------
    go.Figure
        Two-panel Plotly figure with a member-rank plot and a scatter plot.

    Notes
    -----
    The left panel shows the descending rank plot of detected metaorders per
    member. The right panel keeps members with trades but no detected
    metaorders at `x = 0`, so that subplot remains on the original count scale.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
    )

    max_child_orders = 0.0
    max_trades = 0.0
    rank_panel_has_data = False
    scatter_panel_has_data = False

    for table, label, color in member_table_specs:
        rank_table = _build_member_metaorder_rank_table(table)
        if not rank_table.empty:
            rank_panel_has_data = True
            fig.add_trace(
                go.Scatter(
                    x=rank_table["rank"].tolist(),
                    y=rank_table["n_metaorders"].tolist(),
                    mode="lines+markers",
                    line=dict(color=color, width=2.4),
                    marker=dict(color=color, size=6),
                    text=rank_table["Member"].tolist(),
                    name=label,
                    legendgroup=label,
                    hovertemplate=(
                        "Member %{text}"
                        "<br>Rank %{x:.0f}"
                        "<br>Number of metaorders %{y:.0f}<extra></extra>"
                    ),
                    showlegend=show_legend,
                ),
                row=1,
                col=1,
            )

        if table.empty:
            continue

        plot_table = table.copy()
        plot_table["Member"] = plot_table["Member"].astype(str)
        plot_table["n_metaorders"] = pd.to_numeric(plot_table["n_metaorders"], errors="coerce").fillna(0.0)
        plot_table["total_child_orders"] = pd.to_numeric(plot_table["total_child_orders"], errors="coerce").fillna(0.0)
        plot_table["total_trades"] = pd.to_numeric(plot_table["total_trades"], errors="coerce").fillna(0.0)
        if plot_table.empty:
            continue

        scatter_panel_has_data = True
        max_child_orders = max(max_child_orders, float(plot_table["total_child_orders"].max()))
        max_trades = max(max_trades, float(plot_table["total_trades"].max()))

        fig.add_trace(
            go.Scatter(
                x=plot_table["total_child_orders"].tolist(),
                y=plot_table["total_trades"].tolist(),
                mode="markers",
                marker=dict(
                    color=color,
                    size=9,
                    opacity=0.75,
                    line=dict(width=0.5, color=THEME_BG_COLOR),
                ),
                text=plot_table["Member"].tolist(),
                name=label,
                legendgroup=label,
                customdata=np.column_stack(
                    [
                        plot_table["total_child_orders"].to_numpy(dtype=float),
                        plot_table["total_trades"].to_numpy(dtype=float),
                        plot_table["n_metaorders"].to_numpy(dtype=float),
                    ]
                ),
                hovertemplate=(
                    "Member %{text}"
                    "<br>Total number of child orders: %{customdata[0]:.0f}"
                    "<br>Total number of trades: %{customdata[1]:.0f}"
                    "<br>Number of metaorders: %{customdata[2]:.0f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    if not rank_panel_has_data:
        subplot = fig.get_subplot(1, 1)
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref=f"{_axis_ref_name(subplot.xaxis.plotly_name)} domain",
            yref=f"{_axis_ref_name(subplot.yaxis.plotly_name)} domain",
            text="No member metaorders",
            showarrow=False,
            font=dict(size=ANNOTATION_FONT_SIZE),
        )

    if not scatter_panel_has_data:
        subplot = fig.get_subplot(1, 2)
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref=f"{_axis_ref_name(subplot.xaxis.plotly_name)} domain",
            yref=f"{_axis_ref_name(subplot.yaxis.plotly_name)} domain",
            text="No member data",
            showarrow=False,
            font=dict(size=ANNOTATION_FONT_SIZE),
        )
    else:
        # Give points lying on the axes a small visual margin so they do not
        # overlap the plot frame.
        x_lower = -0.03 * max_child_orders if max_child_orders > 0.0 else -0.05
        x_upper = max_child_orders * 1.05 if max_child_orders > 0.0 else 1.0
        y_lower = -0.03 * max_trades if max_trades > 0.0 else -0.05
        y_upper = max_trades * 1.05 if max_trades > 0.0 else 1.0
        fig.update_xaxes(range=[x_lower, x_upper], row=1, col=2)
        fig.update_yaxes(range=[y_lower, y_upper], row=1, col=2)

    fig.update_xaxes(
        title_text="Rank",
        type="log",
        exponentformat="power",
        showexponent="all",
        minexponent=0,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=r"$N_m$",
        type="log",
        exponentformat="power",
        showexponent="all",
        minexponent=0,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="# Child orders",
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="# Trades", row=1, col=2)
    fig.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        hovermode="closest",
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=70, r=25, t=70, b=60),
    )
    return fig


def _collect_group_samples(metaorders_dict_path: Path, proprietary: bool) -> MetaorderDistributionSamples:
    """Load one flow-group dictionary and recompute its aggregated samples."""
    return collect_metaorder_distribution_samples(
        metaorders_dict_path=metaorders_dict_path,
        parquet_dir=PARQUET_DIR,
        proprietary=proprietary,
        trading_hours=TRADING_HOURS,
        member_nationality=MEMBER_NATIONALITY,
        members_nationality_path=MEMBERS_NATIONALITY_PATH,
        include_counts_by_member=True,
        show_progress=True,
    )


def _mode_output_stem(base_stem: str, condition_on_client_proprietary: bool) -> str:
    """Return the output stem for split vs pooled runs."""
    if condition_on_client_proprietary:
        return base_stem
    stem_path = Path(base_stem)
    if stem_path.suffix:
        return f"{stem_path.stem}_all_metaorders{stem_path.suffix}"
    return f"{base_stem}_all_metaorders"


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Summary
    -------
    Run the YAML-configured metaorder summary-statistics pipeline.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Optional explicit CLI argument list.

    Returns
    -------
    None

    Notes
    -----
    The script always loads both client and proprietary dictionaries in one run.
    The CLI flag `--condition-on-client-proprietary {true,false}` decides
    whether the outputs preserve the client/proprietary split or pool both
    groups into one combined view.
    """
    args = _parse_args(argv)
    condition_on_client_proprietary = bool(args.condition_on_client_proprietary)
    mode_tag = "prop_vs_client" if condition_on_client_proprietary else "all_metaorders"

    log_path = OUTPUT_DIR / "logs" / with_member_nationality_tag(
        f"metaorder_summary_statistics_{LEVEL}_{mode_tag}.log",
        MEMBER_NATIONALITY,
    )
    logger = setup_file_logger(Path(__file__).stem, log_path, mode="a")
    with PrintTee(logger):
        print("[Intro] Metaorder summary-statistics run started...")
        print(
            "[Intro] Parameters — \n"
            f"  DATASET={DATASET_NAME}, LEVEL={LEVEL}, "
            f"MEMBER_NATIONALITY={MEMBER_NATIONALITY_TAG}, "
            f"TRADING_HOURS={TRADING_HOURS}, "
            f"RUN_METAORDER_SUMMARY_STATISTICS={RUN_METAORDER_SUMMARY_STATISTICS}, "
            f"CONDITION_ON_CLIENT_PROPRIETARY={condition_on_client_proprietary}"
        )
        print(
            "[Intro] Paths — \n"
            f"  PARQUET_DIR={PARQUET_DIR}\n"
            f"  OUTPUT_DIR={OUTPUT_DIR}\n"
            f"  IMG_DIR={IMG_DIR}\n"
            f"  PROPRIETARY_DICT_PATH={PROPRIETARY_DICT_PATH}\n"
            f"  CLIENT_DICT_PATH={CLIENT_DICT_PATH}\n"
            f"  MEMBERS_NATIONALITY_PATH={MEMBERS_NATIONALITY_PATH}"
        )

        if not RUN_METAORDER_SUMMARY_STATISTICS:
            print("[Metaorder summary] RUN_METAORDER_SUMMARY_STATISTICS is false; nothing to do.")
            return

        plot_dirs: PlotOutputDirs = make_plot_output_dirs(IMG_DIR, use_subdirs=True)
        ensure_plot_dirs(plot_dirs)
        SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        try:
            client_samples = _collect_group_samples(CLIENT_DICT_PATH, proprietary=False)
            proprietary_samples = _collect_group_samples(PROPRIETARY_DICT_PATH, proprietary=True)
        except Exception as exc:
            print(f"[Metaorder summary] Failed to collect comparison samples: {exc}")
            return

        combined_nationality_counts = {
            "it": int(client_samples.nationality_counts["it"] + proprietary_samples.nationality_counts["it"]),
            "foreign": int(
                client_samples.nationality_counts["foreign"] + proprietary_samples.nationality_counts["foreign"]
            ),
        }
        if condition_on_client_proprietary:
            print(
                "[Metaorder summary] Sample counts — "
                f"client_metaorders={client_samples.total_metaorders:,}, "
                f"proprietary_metaorders={proprietary_samples.total_metaorders:,}"
            )
            print(
                "[Metaorder summary][Nationality] "
                f"client_it={client_samples.nationality_counts['it']}, "
                f"client_foreign={client_samples.nationality_counts['foreign']}, "
                f"prop_it={proprietary_samples.nationality_counts['it']}, "
                f"prop_foreign={proprietary_samples.nationality_counts['foreign']}"
            )
        else:
            print(
                "[Metaorder summary] Sample counts — "
                f"all_metaorders={client_samples.total_metaorders + proprietary_samples.total_metaorders:,} "
                f"(client={client_samples.total_metaorders:,}, proprietary={proprietary_samples.total_metaorders:,})"
            )
            print(
                "[Metaorder summary][Nationality] "
                f"all_it={combined_nationality_counts['it']}, "
                f"all_foreign={combined_nationality_counts['foreign']}"
            )

        try:
            nationality_context_table = compute_nationality_context_table(
                parquet_dir=PARQUET_DIR,
                proprietary_dict_path=PROPRIETARY_DICT_PATH,
                client_dict_path=CLIENT_DICT_PATH,
                trading_hours=TRADING_HOURS,
                member_nationality=MEMBER_NATIONALITY,
                members_nationality_path=MEMBERS_NATIONALITY_PATH,
                client_samples=client_samples,
                proprietary_samples=proprietary_samples,
            )
        except Exception as exc:
            print(f"[Metaorder summary] Failed to build nationality context table: {exc}")
            return

        nationality_context_parquet_path = SUMMARY_OUTPUT_DIR / with_member_nationality_tag(
            "nationality_overall_context.parquet",
            MEMBER_NATIONALITY,
        )
        nationality_context_csv_path = SUMMARY_OUTPUT_DIR / with_member_nationality_tag(
            "nationality_overall_context.csv",
            MEMBER_NATIONALITY,
        )
        nationality_context_table.to_parquet(nationality_context_parquet_path, index=False)
        nationality_context_table.to_csv(nationality_context_csv_path, index=False)
        print(
            "[Metaorder summary] Nationality context table paths="
            f"{nationality_context_parquet_path} and {nationality_context_csv_path}"
        )

        for flow_label in ("client", "proprietary", "all_metaorders"):
            flow_table = nationality_context_table.loc[nationality_context_table["flow"] == flow_label].copy()
            if flow_table.empty:
                continue
            it_row = flow_table.loc[flow_table["nationality"] == "it"].iloc[0]
            foreign_row = flow_table.loc[flow_table["nationality"] == "foreign"].iloc[0]
            print(
                "[Metaorder summary][Nationality context] "
                f"flow={flow_label}, "
                f"it_trade_volume_share={_format_pct(float(it_row['share_of_known_trade_volume_pct']))}, "
                f"it_metaorder_share={_format_pct(float(it_row['share_of_known_metaorders_pct']))}, "
                f"it_metaorder_volume_share={_format_pct(float(it_row['share_of_known_metaorder_volume_pct']))}, "
                f"it_own_volume_in_metaorders={_format_pct(float(it_row['share_of_nationality_volume_in_metaorders_pct']))}, "
                f"foreign_trade_volume_share={_format_pct(float(foreign_row['share_of_known_trade_volume_pct']))}, "
                f"foreign_metaorder_share={_format_pct(float(foreign_row['share_of_known_metaorders_pct']))}, "
                f"foreign_metaorder_volume_share={_format_pct(float(foreign_row['share_of_known_metaorder_volume_pct']))}, "
                f"foreign_own_volume_in_metaorders={_format_pct(float(foreign_row['share_of_nationality_volume_in_metaorders_pct']))}, "
                f"unknown_trade_volume={float(flow_table['unknown_trade_volume'].iloc[0]):,.0f}, "
                f"unknown_metaorders={int(flow_table['unknown_metaorder_count'].iloc[0]):,}, "
                f"mixed_metaorders={int(flow_table['mixed_metaorder_count'].iloc[0]):,}"
            )

        fig = build_nationality_context_figure(
            nationality_context_table,
            condition_on_client_proprietary=condition_on_client_proprietary,
        )
        html_path, png_path = save_plotly_figure(
            fig,
            stem=with_member_nationality_tag(
                _mode_output_stem("nationality_overall_context", condition_on_client_proprietary),
                MEMBER_NATIONALITY,
            ),
            dirs=plot_dirs,
            width=1300 if condition_on_client_proprietary else 760,
            height=560,
            scale=2,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        print(
            "[Metaorder summary] Saved nationality-context figure "
            f"to HTML={html_path} PNG={png_path}"
        )

        fig = build_nationality_share_figure(
            client_samples,
            proprietary_samples,
            condition_on_client_proprietary=condition_on_client_proprietary,
        )
        html_path, png_path = save_plotly_figure(
            fig,
            stem=with_member_nationality_tag(
                "nationality_share_prop_vs_client"
                if condition_on_client_proprietary
                else "nationality_share_all_metaorders",
                MEMBER_NATIONALITY,
            ),
            dirs=plot_dirs,
            width=1200 if condition_on_client_proprietary else 700,
            height=560,
            scale=2,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        print(f"[Metaorder summary] Saved nationality-share figure to HTML={html_path} PNG={png_path}")

        if LEVEL != "member":
            print(
                "[Metaorder summary] Skipping member metaorder profile figure: "
                f"LEVEL={LEVEL!r} is not member."
            )
        else:
            try:
                client_child_order_counts = _compute_total_child_orders_by_member(CLIENT_DICT_PATH)
                proprietary_child_order_counts = _compute_total_child_orders_by_member(PROPRIETARY_DICT_PATH)
                client_trade_counts = _compute_total_trades_by_member(
                    parquet_dir=PARQUET_DIR,
                    proprietary=False,
                    trading_hours=TRADING_HOURS,
                    member_nationality=MEMBER_NATIONALITY,
                    members_nationality_path=MEMBERS_NATIONALITY_PATH,
                )
                proprietary_trade_counts = _compute_total_trades_by_member(
                    parquet_dir=PARQUET_DIR,
                    proprietary=True,
                    trading_hours=TRADING_HOURS,
                    member_nationality=MEMBER_NATIONALITY,
                    members_nationality_path=MEMBERS_NATIONALITY_PATH,
                )
            except Exception as exc:
                print(f"[Metaorder summary] Failed to build member-level counts: {exc}")
                return

            if condition_on_client_proprietary:
                client_member_table = _build_member_profile_table(
                    metaorder_counts=client_samples.counts_by_member,
                    child_order_counts=client_child_order_counts,
                    trade_counts=client_trade_counts,
                    flow_label="client",
                )
                proprietary_member_table = _build_member_profile_table(
                    metaorder_counts=proprietary_samples.counts_by_member,
                    child_order_counts=proprietary_child_order_counts,
                    trade_counts=proprietary_trade_counts,
                    flow_label="proprietary",
                )
                member_profile_table = pd.concat(
                    [client_member_table, proprietary_member_table],
                    ignore_index=True,
                )
                member_profile_specs = (
                    (client_member_table, "Client", COLOR_CLIENT),
                    (proprietary_member_table, "Proprietary", COLOR_PROPRIETARY),
                )
                print(
                    "[Metaorder summary] Member metaorder profile table: "
                    f"client_members={len(client_member_table):,}, "
                    f"prop_members={len(proprietary_member_table):,}"
                )
            else:
                pooled_member_table = _build_member_profile_table(
                    metaorder_counts=_sum_member_count_maps(
                        client_samples.counts_by_member,
                        proprietary_samples.counts_by_member,
                    ),
                    child_order_counts=_sum_member_count_maps(
                        client_child_order_counts,
                        proprietary_child_order_counts,
                    ),
                    trade_counts=_sum_member_count_maps(
                        client_trade_counts,
                        proprietary_trade_counts,
                    ),
                    flow_label="all_metaorders",
                )
                member_profile_table = pooled_member_table.copy()
                member_profile_specs = ((pooled_member_table, "All metaorders", COLOR_ALL_METAORDERS),)
                print(
                    "[Metaorder summary] Member metaorder profile table: "
                    f"members={len(pooled_member_table):,}"
                )

            profile_table_path = SUMMARY_OUTPUT_DIR / with_member_nationality_tag(
                _mode_output_stem("member_metaorder_profiles.parquet", condition_on_client_proprietary),
                MEMBER_NATIONALITY,
            )
            member_profile_table.to_parquet(profile_table_path, index=False)
            print(f"[Metaorder summary] Member metaorder profile table path={profile_table_path}")

            fig = _build_member_metaorder_profile_figure(
                member_profile_specs,
                show_legend=condition_on_client_proprietary,
            )
            html_path, png_path = save_plotly_figure(
                fig,
                stem=with_member_nationality_tag(
                    _mode_output_stem("member_metaorder_profiles", condition_on_client_proprietary),
                    MEMBER_NATIONALITY,
                ),
                dirs=plot_dirs,
                width=1400,
                height=620,
                scale=2,
                write_html=True,
                write_png=True,
                strict_png=False,
            )
            print(
                "[Metaorder summary] Saved member metaorder profile figure "
                f"to HTML={html_path} PNG={png_path}"
            )

        try:
            daily_share_table = compute_daily_metaorder_share_table(
                parquet_dir=PARQUET_DIR,
                proprietary_dict_path=PROPRIETARY_DICT_PATH,
                client_dict_path=CLIENT_DICT_PATH,
                trading_hours=TRADING_HOURS,
                member_nationality=MEMBER_NATIONALITY,
                members_nationality_path=MEMBERS_NATIONALITY_PATH,
            )
        except Exception as exc:
            print(f"[Metaorder summary] Failed to build the daily-share table: {exc}")
            return

        if daily_share_table.empty:
            print("[Metaorder summary] Skipping mean daily share figure: no valid denominator days.")
            return

        mean_prop_ratio = float(daily_share_table["proprietary_ratio"].mean())
        mean_client_ratio = float(daily_share_table["client_ratio"].mean())
        mean_total_ratio = float(daily_share_table["total_ratio"].mean())
        if condition_on_client_proprietary:
            print(
                "[Metaorder summary] Mean daily metaorder-volume share by ISIN: "
                f"proprietary={100.0 * mean_prop_ratio:.4f}%, "
                f"client={100.0 * mean_client_ratio:.4f}%, "
                f"total={100.0 * mean_total_ratio:.4f}% "
                f"(n_isins={daily_share_table['ISIN'].nunique()}, n_isin_days={len(daily_share_table)})"
            )
        else:
            print(
                "[Metaorder summary] Mean daily metaorder-volume share by ISIN: "
                f"all_metaorders={100.0 * mean_total_ratio:.4f}% "
                f"(n_isins={daily_share_table['ISIN'].nunique()}, n_isin_days={len(daily_share_table)})"
            )

        fig = build_mean_daily_metaorder_share_figure(
            daily_share_table,
            condition_on_client_proprietary=condition_on_client_proprietary,
        )
        html_path, png_path = save_plotly_figure(
            fig,
            stem=with_member_nationality_tag(
                _mode_output_stem("mean_daily_metaorder_volume_share", condition_on_client_proprietary),
                MEMBER_NATIONALITY,
            ),
            dirs=plot_dirs,
            width=1500,
            height=720,
            scale=2,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        print(f"[Metaorder summary] Saved mean daily share figure to HTML={html_path} PNG={png_path}")


if __name__ == "__main__":
    main()
