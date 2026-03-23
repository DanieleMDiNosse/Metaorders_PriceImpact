#!/usr/bin/env python3
"""
Metaorder distribution comparison figures with best-tail-model overlays.

What this script does
---------------------
This script loads the canonical proprietary and client metaorder dictionaries,
recomputes the aggregated metaorder samples from the per-ISIN trade tapes, and
produces a single organized comparison figure for the core distribution
diagnostics:

- metaorder duration
- inter-arrival time within metaorders
- metaorder volume `Q`
- relative size `Q/V`
- participation rate `eta`

Optional Clauset-style tail fits are computed with the shared
`moimpact.power_law_fits` helpers. For each panel, the script scores the common
tail against power-law, lognormal, exponential, and truncated-power-law models,
plots the best-scoring overlay, and writes the likelihood-ratio diagnostics to
the machine-readable fit table. It also exports a Markdown review file with the
best-vs-second fit table and the mean/median summary table.

How to run
----------
1) Edit `config_ymls/metaorder_distributions.yml`, or point
   `METAORDER_DISTRIBUTIONS_CONFIG` to an alternate YAML file.
2) Run:

    python scripts/metaorder_distributions.py

Outputs
-------
- Figure:
  `images/{DATASET_NAME}/{LEVEL}_metaorder_distributions/png/` and `.../html/`
- Figure-regeneration plot data:
  `out_files/{DATASET_NAME}/{LEVEL}_metaorder_distributions/`
- Fit summary tables:
  `out_files/{DATASET_NAME}/{LEVEL}_metaorder_distributions/`
- Markdown review file:
  `out_files/{DATASET_NAME}/{LEVEL}_metaorder_distributions/`
- Log file:
  `out_files/{DATASET_NAME}/logs/metaorder_distributions_{LEVEL}_prop_vs_client[...].log`
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence as SequenceCollection
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure repository-root imports (e.g., `moimpact`) work when running
# `python scripts/metaorder_distributions.py` from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.config import cfg_require, format_path_template, load_yaml_mapping, resolve_repo_path
from moimpact.logging_utils import PrintTee, setup_file_logger
from moimpact.metaorder_distribution_samples import (
    MetaorderDistributionSamples,
    collect_metaorder_distribution_samples,
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
from moimpact.power_law_fits import (
    FullPipelineBootstrapSummary,
    PowerLawComparisonResult,
    TailModelComparisonResult,
    bootstrap_full_clauset_pipeline_parameters,
    fit_power_law_with_alternatives,
    power_law_pdf,
)

BEST_MODEL_TRACE_COLOR = THEME_COLORWAY[2]
POWERLAW_TRACE_COLOR = "#4a4a4a"


@dataclass(frozen=True)
class MetricSpec:
    """Definition of one distribution row in the comparison figure."""

    panel_title: str
    xaxis_title: str
    field_name: str
    bins: int = 50
    plot_kind: str = "density"


@dataclass(frozen=True)
class PanelSpec:
    """Metadata and sample values for one client/proprietary distribution panel."""

    panel_idx: int
    total_panels: int
    row_idx: int
    col_idx: int
    metric: MetricSpec
    label: str
    group_tag: str
    color: str
    panel_values: np.ndarray
    discrete: bool = False


@dataclass(frozen=True)
class PanelFitResult:
    """Serializable output of one panel-level tail fit and bootstrap job."""

    panel_idx: int
    fit_row: Dict[str, object]
    annotation_text: str
    best_model: str
    comparison_summary: str
    fit_success: bool
    draw_best_fit: bool
    xmin: float
    alpha: float
    ks_stat: float
    n_tail: int
    best_fit_aic: float
    x_grid: np.ndarray
    y_grid: np.ndarray
    powerlaw_x_grid: np.ndarray
    powerlaw_y_grid: np.ndarray
    elapsed_seconds: float


def _distribution_metric_specs() -> tuple[MetricSpec, ...]:
    """Return the ordered panel definitions used by the combined figure."""
    return (
        MetricSpec("Metaorder duration", r"$T$", "durations_minutes"),
        MetricSpec("Inter-arrival times", r"$\Delta t$", "inter_arrivals_minutes"),
        MetricSpec("Metaorder volumes", r"$Q$", "meta_volumes"),
        MetricSpec("Relative size", r"$\phi$", "q_over_v"),
        MetricSpec("Participation rate", r"$\eta$", "participation_rates"),
    )


_CONFIG_ENV_VAR = "METAORDER_DISTRIBUTIONS_CONFIG"
_config_override = os.environ.get(_CONFIG_ENV_VAR)
if _config_override:
    _CONFIG_PATH = Path(_config_override).expanduser()
    if not _CONFIG_PATH.is_absolute():
        _CONFIG_PATH = (_REPO_ROOT / _CONFIG_PATH).resolve()
else:
    _CONFIG_PATH = _REPO_ROOT / "config_ymls" / "metaorder_distributions.yml"
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


def _figure_stem() -> str:
    """Return the canonical figure stem for the current nationality slice."""
    return with_member_nationality_tag("metaorder_distributions_prop_vs_client", MEMBER_NATIONALITY)


def _fit_summary_stem() -> str:
    """Return the canonical fit-summary stem for the current nationality slice."""
    return with_member_nationality_tag(
        "metaorder_distribution_fit_summary_prop_vs_client",
        MEMBER_NATIONALITY,
    )


def _review_stem() -> str:
    """Return the canonical fit-review stem for the current nationality slice."""
    return with_member_nationality_tag(
        "metaorder_distribution_fit_review_prop_vs_client",
        MEMBER_NATIONALITY,
    )


def _plot_data_stem() -> str:
    """Return the canonical saved plot-data stem for the current nationality slice."""
    return with_member_nationality_tag(
        "metaorder_distribution_plot_data_prop_vs_client",
        MEMBER_NATIONALITY,
    )


def _saved_output_paths() -> Dict[str, Path]:
    """Return the file paths used by normal and load-only figure regeneration."""
    fit_stem = _fit_summary_stem()
    review_stem = _review_stem()
    plot_data_stem = _plot_data_stem()
    return {
        "fit_summary_csv": DISTRIBUTIONS_OUTPUT_DIR / f"{fit_stem}.csv",
        "fit_summary_parquet": DISTRIBUTIONS_OUTPUT_DIR / f"{fit_stem}.parquet",
        "review_csv": DISTRIBUTIONS_OUTPUT_DIR / f"{review_stem}.csv",
        "review_markdown": DISTRIBUTIONS_OUTPUT_DIR / f"{review_stem}.md",
        "plot_data_csv": DISTRIBUTIONS_OUTPUT_DIR / f"{plot_data_stem}.csv",
        "plot_data_parquet": DISTRIBUTIONS_OUTPUT_DIR / f"{plot_data_stem}.parquet",
    }


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
    The surrounding paper/docs provide the caption, so the exported figure is
    saved without a separate top title.
    """
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)


TICK_FONT_SIZE = int(_cfg_require("TICK_FONT_SIZE"))
LABEL_FONT_SIZE = int(_cfg_require("LABEL_FONT_SIZE"))
TITLE_FONT_SIZE = int(_cfg_require("TITLE_FONT_SIZE"))
LEGEND_FONT_SIZE = int(_cfg_require("LEGEND_FONT_SIZE"))
ANNOTATION_FONT_SIZE = int(_CFG.get("ANNOTATION_FONT_SIZE", 18))

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
TRADING_HOURS = tuple(_cfg_require("TRADING_HOURS"))
RUN_METAORDER_DISTRIBUTIONS = bool(_cfg_require("RUN_METAORDER_DISTRIBUTIONS"))
MEMBER_NATIONALITY = parse_member_nationality(_CFG.get("MEMBER_NATIONALITY"))
MEMBER_NATIONALITY_TAG = MEMBER_NATIONALITY or "all"
POWERLAW_FIT_ENABLED = bool(_CFG.get("POWERLAW_FIT_ENABLED", True))
POWERLAW_FIT_METHOD = str(_CFG.get("POWERLAW_FIT_METHOD", "approx")).strip().lower()
if POWERLAW_FIT_METHOD not in {"approx", "powerlaw"}:
    raise ValueError(
        f"Invalid POWERLAW_FIT_METHOD={POWERLAW_FIT_METHOD!r} in {_CONFIG_PATH}. "
        "Expected one of {'approx', 'powerlaw'}."
    )
POWERLAW_MIN_TAIL = max(int(_CFG.get("POWERLAW_MIN_TAIL", 50)), 2)
POWERLAW_NUM_CANDIDATES = max(int(_CFG.get("POWERLAW_NUM_CANDIDATES", 200)), 5)
POWERLAW_REFINE_WINDOW = max(int(_CFG.get("POWERLAW_REFINE_WINDOW", 50)), 0)
POWERLAW_FULL_BOOTSTRAP_ENABLED = bool(_CFG.get("POWERLAW_FULL_BOOTSTRAP_ENABLED", False)) and POWERLAW_FIT_ENABLED
POWERLAW_FULL_BOOTSTRAP_RUNS = int(_CFG.get("POWERLAW_FULL_BOOTSTRAP_RUNS", 1000))
POWERLAW_FULL_BOOTSTRAP_ALPHA = float(_CFG.get("POWERLAW_FULL_BOOTSTRAP_ALPHA", 0.05))
_powerlaw_full_bootstrap_random_state_cfg = _CFG.get("POWERLAW_FULL_BOOTSTRAP_RANDOM_STATE", 0)
POWERLAW_FULL_BOOTSTRAP_RANDOM_STATE: Optional[int] = (
    None if _powerlaw_full_bootstrap_random_state_cfg is None else int(_powerlaw_full_bootstrap_random_state_cfg)
)
_powerlaw_full_bootstrap_num_candidates_cfg = _CFG.get("POWERLAW_FULL_BOOTSTRAP_NUM_CANDIDATES")
POWERLAW_FULL_BOOTSTRAP_NUM_CANDIDATES = max(
    int(_powerlaw_full_bootstrap_num_candidates_cfg),
    5,
) if _powerlaw_full_bootstrap_num_candidates_cfg is not None else min(POWERLAW_NUM_CANDIDATES, 50)
_powerlaw_full_bootstrap_refine_window_cfg = _CFG.get("POWERLAW_FULL_BOOTSTRAP_REFINE_WINDOW")
POWERLAW_FULL_BOOTSTRAP_REFINE_WINDOW = max(
    int(_powerlaw_full_bootstrap_refine_window_cfg),
    0,
) if _powerlaw_full_bootstrap_refine_window_cfg is not None else min(POWERLAW_REFINE_WINDOW, 10)
_powerlaw_fit_max_workers_cfg = _CFG.get("POWERLAW_FIT_MAX_WORKERS")
POWERLAW_FIT_MAX_WORKERS: Optional[int] = (
    None if _powerlaw_fit_max_workers_cfg is None else int(_powerlaw_fit_max_workers_cfg)
)
POWERLAW_FULL_BOOTSTRAP_RESAMPLING_SCHEME = "nonparametric_full_sample"
if POWERLAW_FULL_BOOTSTRAP_ENABLED and POWERLAW_FULL_BOOTSTRAP_RUNS < 1:
    raise ValueError("POWERLAW_FULL_BOOTSTRAP_RUNS must be >= 1 when POWERLAW_FULL_BOOTSTRAP_ENABLED is true.")
if (not np.isfinite(POWERLAW_FULL_BOOTSTRAP_ALPHA)) or POWERLAW_FULL_BOOTSTRAP_ALPHA <= 0.0 or POWERLAW_FULL_BOOTSTRAP_ALPHA >= 1.0:
    raise ValueError("POWERLAW_FULL_BOOTSTRAP_ALPHA must be a finite float in the open interval (0, 1).")


def _coerce_powerlaw_alternatives(value: object) -> Tuple[str, ...]:
    """Normalize configured alternative-model labels for likelihood-ratio checks."""
    if value is None:
        raw_values: Sequence[object] = ("lognormal", "exponential", "truncated_power_law")
    elif isinstance(value, str):
        raw_values = tuple(part.strip() for part in value.split(","))
    elif isinstance(value, SequenceCollection):
        raw_values = value
    else:
        raise TypeError("POWERLAW_COMPARE_ALTERNATIVES must be null, a string, or a sequence of strings.")

    normalized: list[str] = []
    for raw in raw_values:
        label = str(raw).strip()
        if label and label not in normalized:
            normalized.append(label)
    return tuple(normalized)


def _candidate_models_from_alternatives(alternatives: Sequence[str]) -> Tuple[str, ...]:
    """Return the ordered candidate-model list with the power law first."""
    return tuple(dict.fromkeys(("power_law", *alternatives)))


def _coerce_bootstrap_distributions(
    value: object,
    *,
    default: Sequence[str],
    allowed: Sequence[str],
) -> Tuple[str, ...]:
    """Normalize and validate the configured model list used for bootstrap refits."""
    if value is None:
        raw_values: Sequence[object] = default
    elif isinstance(value, str):
        raw_values = tuple(part.strip() for part in value.split(","))
    elif isinstance(value, SequenceCollection):
        raw_values = value
    else:
        raise TypeError("BOOTSTRAP_DISTRIBUTION must be null, a string, or a sequence of strings.")

    normalized: list[str] = []
    for raw in raw_values:
        label = str(raw).strip()
        if label and label not in normalized:
            normalized.append(label)

    if not normalized:
        raise ValueError("BOOTSTRAP_DISTRIBUTION must contain at least one fitted model label.")

    allowed_set = set(allowed)
    invalid = [label for label in normalized if label not in allowed_set]
    if invalid:
        raise ValueError(
            "BOOTSTRAP_DISTRIBUTION must be a subset of the fitted candidate models "
            f"{tuple(allowed)}; got invalid labels {tuple(invalid)}."
        )
    return tuple(normalized)


POWERLAW_COMPARE_ENABLED = bool(_CFG.get("POWERLAW_COMPARE_ENABLED", True)) and POWERLAW_FIT_ENABLED
POWERLAW_COMPARE_ALTERNATIVES = _coerce_powerlaw_alternatives(
    _CFG.get("POWERLAW_COMPARE_ALTERNATIVES", ("lognormal", "exponential", "truncated_power_law"))
)
if not POWERLAW_COMPARE_ALTERNATIVES:
    POWERLAW_COMPARE_ENABLED = False
POWERLAW_CANDIDATE_MODELS = _candidate_models_from_alternatives(POWERLAW_COMPARE_ALTERNATIVES)
BOOTSTRAP_DISTRIBUTION = _coerce_bootstrap_distributions(
    _CFG.get("BOOTSTRAP_DISTRIBUTION"),
    default=POWERLAW_CANDIDATE_MODELS,
    allowed=POWERLAW_CANDIDATE_MODELS,
)
BOOTSTRAP_ALTERNATIVES = tuple(model for model in BOOTSTRAP_DISTRIBUTION if model != "power_law")

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
DISTRIBUTIONS_OUTPUT_DIR = OUTPUT_DIR / f"{LEVEL}_metaorder_distributions"
IMG_DIR = IMG_BASE_DIR / f"{LEVEL}_metaorder_distributions"
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


def _filter_positive_finite(data: Sequence[float] | np.ndarray) -> np.ndarray:
    """Keep strictly positive finite values for log-log density plotting."""
    arr = np.asarray(data, dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def _density_curve(data: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a log-binned empirical density curve for one positive sample."""
    x_min = float(np.nanmin(data))
    x_max = float(np.nanmax(data))
    if x_max <= x_min:
        edges = np.array([x_min / 1.01, x_min * 1.01], dtype=float)
        edges[0] = max(edges[0], np.nextafter(0.0, 1.0))
    else:
        edges = np.geomspace(x_min, x_max, int(bins) + 1)
    density, edges = np.histogram(data, bins=edges, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    valid = np.isfinite(centers) & np.isfinite(density) & (density > 0)
    return centers[valid], density[valid]


def _distribution_group_specs() -> tuple[tuple[str, str, str, int], ...]:
    """Return the ordered client/proprietary panel metadata."""
    return (
        ("Client", "client", COLOR_CLIENT, 1),
        ("Proprietary", "proprietary", COLOR_PROPRIETARY, 2),
    )


def _empty_distribution_plot_data_frame() -> pd.DataFrame:
    """Return the canonical empty plot-data frame used for saved artifacts."""
    return pd.DataFrame(
        columns=[
            "metric",
            "panel_title",
            "group",
            "label",
            "row_idx",
            "col_idx",
            "trace_kind",
            "point_index",
            "x",
            "y",
        ]
    )


def _build_distribution_figure_shell() -> go.Figure:
    """Create the empty subplot grid shared by normal and load-only modes."""
    metrics: Sequence[MetricSpec] = _distribution_metric_specs()
    fig = make_subplots(
        rows=len(metrics),
        cols=2,
        shared_yaxes="rows",
        column_titles=["Client", "Proprietary"],
        vertical_spacing=0.045,
        horizontal_spacing=0.08,
    )

    for row_idx, metric in enumerate(metrics, start=1):
        for _, _, _, col_idx in _distribution_group_specs():
            fig.update_xaxes(
                type="log",
                title_text=metric.xaxis_title,
                exponentformat="power",
                showexponent="all",
                minexponent=0,
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                type="log",
                exponentformat="power",
                showexponent="all",
                minexponent=0,
                row=row_idx,
                col=col_idx,
            )
        fig.update_yaxes(title_text="Density", row=row_idx, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE_NAME,
        height=2080,
        width=1400,
        margin=dict(l=85, r=35, t=45, b=50),
        showlegend=False,
    )
    return fig


def _curve_rows_from_xy(
    *,
    panel: PanelSpec,
    trace_kind: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> list[dict[str, object]]:
    """Serialize one plotted curve into the saved long-format plot-data table."""
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0:
        return []
    common_size = min(x_arr.size, y_arr.size)
    if common_size == 0:
        return []
    x_arr = x_arr[:common_size]
    y_arr = y_arr[:common_size]
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0.0) & (y_arr > 0.0)
    if not bool(np.any(valid)):
        return []

    rows: list[dict[str, object]] = []
    valid_indices = np.flatnonzero(valid)
    for point_index, idx in enumerate(valid_indices):
        rows.append(
            {
                "metric": _metric_summary_name(panel.metric),
                "panel_title": panel.metric.panel_title,
                "group": panel.group_tag,
                "label": panel.label,
                "row_idx": int(panel.row_idx),
                "col_idx": int(panel.col_idx),
                "trace_kind": trace_kind,
                "point_index": int(point_index),
                "x": float(x_arr[idx]),
                "y": float(y_arr[idx]),
            }
        )
    return rows


def _bootstrap_ci_label(fit_row: Mapping[str, object]) -> str:
    """Return the label used for bootstrap intervals in panel annotations."""
    enabled = bool(fit_row.get("bootstrap_full_pipeline_enabled", False))
    alpha = fit_row.get("bootstrap_alpha", np.nan)
    try:
        alpha_value = float(alpha)
    except Exception:
        alpha_value = np.nan
    if enabled and np.isfinite(alpha_value) and 0.0 < alpha_value < 1.0:
        return f"{100.0 * (1.0 - alpha_value):.0f}% CI"
    return "CI"


def _format_annotation_scalar(value: object, *, format_spec: str) -> str:
    """Format one scalar value for subplot annotation text."""
    try:
        number = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return format(number, format_spec)


def _format_annotation_interval(
    *,
    label: str,
    estimate: object,
    ci_low: object,
    ci_high: object,
    format_spec: str,
    ci_label: str,
) -> str:
    """Format one estimate line with an inline confidence interval when available."""
    estimate_text = _format_annotation_scalar(estimate, format_spec=format_spec)
    line = f"{label} = {estimate_text}"
    try:
        low_value = float(ci_low)
        high_value = float(ci_high)
    except Exception:
        return line
    if not (np.isfinite(low_value) and np.isfinite(high_value)):
        return line
    ci_prefix = f"{ci_label} " if ci_label else ""
    return (
        f"{line} ({ci_prefix}"
        f"[{_format_annotation_scalar(low_value, format_spec=format_spec)}, "
        f"{_format_annotation_scalar(high_value, format_spec=format_spec)}])"
    )


def _annotation_text_from_fit_row(fit_row: Mapping[str, object]) -> str:
    """Build the per-panel annotation box text from one saved fit-summary row."""
    sample_size = fit_row.get("sample_size", 0)
    try:
        sample_size_value = int(sample_size)
    except Exception:
        sample_size_value = 0
    if sample_size_value <= 0:
        return "No positive finite data"

    if not bool(fit_row.get("fit_success", False)):
        return "No stable fit"

    def _with_text_color(text: str, color: str) -> str:
        """Wrap one annotation line in a font tag so Plotly renders it in `color`."""
        return f"<span style='color:{color}'>{text}</span>"

    best_model = str(fit_row.get("best_fit_model", "unavailable"))
    ci_label = _bootstrap_ci_label(fit_row)
    lines = [
        f"Best fit = {_pretty_model_name(best_model)}",
        _with_text_color(
            _format_annotation_interval(
                label="x_min",
                estimate=fit_row.get("xmin", np.nan),
                ci_low=fit_row.get("bootstrap_xmin_ci_low", np.nan),
                ci_high=fit_row.get("bootstrap_xmin_ci_high", np.nan),
                format_spec=".4g",
                ci_label=ci_label,
            ),
            POWERLAW_TRACE_COLOR,
        ),
        _with_text_color(
            _format_annotation_interval(
                label="alpha",
                estimate=fit_row.get("alpha", np.nan),
                ci_low=fit_row.get("power_law_alpha_ci_low", np.nan),
                ci_high=fit_row.get("power_law_alpha_ci_high", np.nan),
                format_spec=".4f",
                ci_label=ci_label,
            ),
            POWERLAW_TRACE_COLOR,
        ),
    ]
    return "<br>".join(lines)


def _resolve_powerlaw_fit_worker_count(panel_count: int) -> int:
    """
    Summary
    -------
    Resolve the number of worker processes used for panel-level tail fits.

    Parameters
    ----------
    panel_count : int
        Number of fit jobs to run.

    Returns
    -------
    int
        Number of worker processes. Returns ``1`` when fitting is disabled or
        when fewer than two jobs are available.

    Notes
    -----
    When ``POWERLAW_FIT_MAX_WORKERS`` is null or non-positive, the worker count
    defaults to the minimum of the available CPU count and ``panel_count``.
    """
    panel_count = max(int(panel_count), 0)
    if panel_count <= 1 or not POWERLAW_FIT_ENABLED:
        return 1

    cpu_count = max(os.cpu_count() or 1, 1)
    configured = POWERLAW_FIT_MAX_WORKERS
    if configured is None or configured <= 0:
        if POWERLAW_FULL_BOOTSTRAP_ENABLED:
            # The full-sample Clauset bootstrap is memory-hungry on the million-
            # observation inter-arrival panels. A conservative auto-cap avoids
            # swapping while still using multiple cores. Users can override this
            # by setting a positive POWERLAW_FIT_MAX_WORKERS in the YAML.
            cpu_count = min(cpu_count, 4)
        return max(1, min(panel_count, cpu_count))
    return max(1, min(panel_count, int(configured)))


def _alternative_slug(label: str) -> str:
    """Convert an alternative-model label into a stable table-column suffix."""
    return re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")


def _format_comparison_summary(comparisons: Sequence[PowerLawComparisonResult]) -> str:
    """Build a compact one-line summary of likelihood-ratio decisions."""
    if not comparisons:
        return "none"
    parts: list[str] = []
    for comparison in comparisons:
        alt_slug = _alternative_slug(comparison.alternative)
        parts.append(f"{alt_slug}={comparison.favored_model}")
    return ", ".join(parts)


def _pretty_model_name(model: str) -> str:
    """Render model labels for plot annotations and hover text."""
    return str(model).replace("_", " ")


def _comparison_row_fields(comparisons: Sequence[PowerLawComparisonResult]) -> Dict[str, object]:
    """Flatten alternative-model comparisons into fit-summary table columns."""
    comparison_map = {str(comp.alternative): comp for comp in comparisons}
    fields: Dict[str, object] = {
        "powerlaw_compare_enabled": POWERLAW_COMPARE_ENABLED,
        "powerlaw_compare_alternatives": ",".join(POWERLAW_COMPARE_ALTERNATIVES),
        "powerlaw_compare_summary": _format_comparison_summary(comparisons),
    }
    power_law_wins = 0
    non_power_law_winners: list[str] = []
    undecided: list[str] = []
    unavailable: list[str] = []

    for alternative in POWERLAW_COMPARE_ALTERNATIVES:
        slug = _alternative_slug(alternative)
        comparison = comparison_map.get(alternative)
        if comparison is None:
            fields[f"compare_{slug}_favored_model"] = "unavailable"
            fields[f"compare_{slug}_loglikelihood_ratio"] = np.nan
            fields[f"compare_{slug}_p_value"] = np.nan
            fields[f"compare_{slug}_nested"] = False
            unavailable.append(alternative)
            continue

        fields[f"compare_{slug}_favored_model"] = comparison.favored_model
        fields[f"compare_{slug}_loglikelihood_ratio"] = (
            float(comparison.loglikelihood_ratio)
            if comparison.loglikelihood_ratio is not None
            else np.nan
        )
        fields[f"compare_{slug}_p_value"] = (
            float(comparison.p_value) if comparison.p_value is not None else np.nan
        )
        fields[f"compare_{slug}_nested"] = bool(comparison.nested)

        if comparison.favored_model == "power_law":
            power_law_wins += 1
        elif comparison.favored_model == "undecided":
            undecided.append(alternative)
        elif comparison.favored_model == "unavailable":
            unavailable.append(alternative)
        else:
            non_power_law_winners.append(comparison.favored_model)

    fields["compare_power_law_wins_count"] = power_law_wins
    fields["compare_non_power_law_winners"] = ",".join(non_power_law_winners)
    fields["compare_undecided_alternatives"] = ",".join(undecided)
    fields["compare_unavailable_alternatives"] = ",".join(unavailable)
    return fields


def _pairwise_comparison_row_fields(fit_summary: Optional[Any]) -> Dict[str, object]:
    """Flatten direct model-vs-model comparisons into fit-summary table columns."""
    pairwise_map: Dict[tuple[str, str], TailModelComparisonResult] = {}
    if fit_summary is not None:
        pairwise_map = {
            (comparison.model_a, comparison.model_b): comparison
            for comparison in fit_summary.pairwise_comparisons
        }

    fields: Dict[str, object] = {}
    for index, model_a in enumerate(POWERLAW_CANDIDATE_MODELS):
        for model_b in POWERLAW_CANDIDATE_MODELS[index + 1 :]:
            model_a_slug = _alternative_slug(model_a)
            model_b_slug = _alternative_slug(model_b)
            key = f"pairwise_compare_{model_a_slug}_vs_{model_b_slug}"
            comparison = pairwise_map.get((model_a, model_b))
            if comparison is None:
                fields[f"{key}_favored_model"] = "unavailable"
                fields[f"{key}_loglikelihood_ratio"] = np.nan
                fields[f"{key}_p_value"] = np.nan
                fields[f"{key}_nested"] = False
                continue
            fields[f"{key}_favored_model"] = comparison.favored_model
            fields[f"{key}_loglikelihood_ratio"] = (
                float(comparison.loglikelihood_ratio)
                if comparison.loglikelihood_ratio is not None
                else np.nan
            )
            fields[f"{key}_p_value"] = (
                float(comparison.p_value) if comparison.p_value is not None else np.nan
            )
            fields[f"{key}_nested"] = bool(comparison.nested)
    return fields


def _model_score_row_fields(fit_summary: Optional[Any]) -> Dict[str, object]:
    """Flatten per-model score diagnostics into fit-summary table columns."""
    score_map = {}
    if fit_summary is not None:
        score_map = {score.model: score for score in fit_summary.model_scores}

    best_model = fit_summary.best_model if fit_summary is not None else "unavailable"
    best_score = score_map.get(best_model)
    fields: Dict[str, object] = {
        "best_fit_model": best_model,
        "best_fit_criterion": fit_summary.best_model_criterion if fit_summary is not None else "aic",
        "best_fit_loglikelihood": best_score.loglikelihood if best_score is not None else np.nan,
        "best_fit_aic": best_score.aic if best_score is not None else np.nan,
        "best_fit_bic": best_score.bic if best_score is not None else np.nan,
        "best_fit_valid": bool(best_score.valid) if best_score is not None else False,
    }

    for model in POWERLAW_CANDIDATE_MODELS:
        slug = _alternative_slug(model)
        score = score_map.get(model)
        fields[f"model_{slug}_loglikelihood"] = score.loglikelihood if score is not None else np.nan
        fields[f"model_{slug}_aic"] = score.aic if score is not None else np.nan
        fields[f"model_{slug}_bic"] = score.bic if score is not None else np.nan
        fields[f"model_{slug}_ks_stat"] = score.ks_stat if score is not None else np.nan
        fields[f"model_{slug}_parameter_count"] = score.parameter_count if score is not None else 0
        fields[f"model_{slug}_noise_flag"] = bool(score.noise_flag) if score is not None else False
        fields[f"model_{slug}_valid"] = bool(score.valid) if score is not None else False

    return fields


def _model_parameter_row_fields(fit_summary: Optional[Any]) -> Dict[str, object]:
    """Flatten fitted model parameters into one-row table fields."""
    if fit_summary is None or fit_summary.raw_fit is None:
        return {}

    fields: Dict[str, object] = {}
    for model in POWERLAW_CANDIDATE_MODELS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                distribution = getattr(fit_summary.raw_fit, model)
                parameter_names = tuple(getattr(distribution, "parameter_names", ()))
        except Exception:
            continue

        model_slug = _alternative_slug(model)
        for parameter_name in parameter_names:
            param_slug = _alternative_slug(parameter_name)
            raw_value = getattr(distribution, parameter_name, np.nan)
            try:
                value = float(raw_value)
            except Exception:
                value = np.nan
            fields[f"param_{model_slug}_{param_slug}"] = value
    return fields


def _bootstrap_metadata_row_fields(bootstrap_summary: Optional[FullPipelineBootstrapSummary]) -> Dict[str, object]:
    """Return row-level bootstrap provenance fields for one panel."""
    enabled = bool(POWERLAW_FULL_BOOTSTRAP_ENABLED and POWERLAW_FIT_ENABLED)
    return {
        "bootstrap_full_pipeline_enabled": enabled,
        "bootstrap_distribution": ",".join(BOOTSTRAP_DISTRIBUTION) if enabled else "",
        "bootstrap_resampling_scheme": (
            bootstrap_summary.resampling_scheme
            if bootstrap_summary is not None
            else (POWERLAW_FULL_BOOTSTRAP_RESAMPLING_SCHEME if enabled else "disabled")
        ),
        "bootstrap_runs_requested": int(POWERLAW_FULL_BOOTSTRAP_RUNS) if enabled else 0,
        "bootstrap_alpha": float(POWERLAW_FULL_BOOTSTRAP_ALPHA) if enabled else np.nan,
        "bootstrap_random_state": POWERLAW_FULL_BOOTSTRAP_RANDOM_STATE if enabled else np.nan,
        "bootstrap_pipeline_valid_runs": (
            int(bootstrap_summary.successful_pipeline_runs) if bootstrap_summary is not None else 0
        ),
    }


def _bootstrap_xmin_row_fields(bootstrap_summary: Optional[FullPipelineBootstrapSummary]) -> Dict[str, object]:
    """Return bootstrap CI fields for the refitted Clauset cutoff `xmin`."""
    if bootstrap_summary is None or bootstrap_summary.xmin_summary is None:
        return {
            "bootstrap_xmin_valid_runs": 0,
            "bootstrap_xmin_ci_low": np.nan,
            "bootstrap_xmin_ci_high": np.nan,
            "bootstrap_xmin_std": np.nan,
        }

    xmin_summary = bootstrap_summary.xmin_summary
    return {
        "bootstrap_xmin_valid_runs": int(xmin_summary.valid_runs),
        "bootstrap_xmin_ci_low": (
            float(xmin_summary.ci_low)
            if xmin_summary.ci_low is not None and np.isfinite(xmin_summary.ci_low)
            else np.nan
        ),
        "bootstrap_xmin_ci_high": (
            float(xmin_summary.ci_high)
            if xmin_summary.ci_high is not None and np.isfinite(xmin_summary.ci_high)
            else np.nan
        ),
        "bootstrap_xmin_std": (
            float(xmin_summary.std)
            if xmin_summary.std is not None and np.isfinite(xmin_summary.std)
            else np.nan
        ),
    }


def _model_parameter_bootstrap_row_fields(
    bootstrap_summary: Optional[FullPipelineBootstrapSummary],
) -> Dict[str, object]:
    """Flatten per-model bootstrap CI summaries into one-row export fields."""
    if bootstrap_summary is None:
        return {
            f"model_{_alternative_slug(model)}_bootstrap_valid_runs": 0
            for model in BOOTSTRAP_DISTRIBUTION
        }

    summary_by_model = {summary.model: summary for summary in bootstrap_summary.model_summaries}
    fields: Dict[str, object] = {}
    for model in BOOTSTRAP_DISTRIBUTION:
        model_slug = _alternative_slug(model)
        model_summary = summary_by_model.get(model)
        fields[f"model_{model_slug}_bootstrap_valid_runs"] = (
            int(model_summary.valid_runs) if model_summary is not None else 0
        )
        if model_summary is None:
            continue
        for interval in model_summary.parameter_intervals:
            param_slug = _alternative_slug(interval.parameter)
            if interval.ci_low is not None and np.isfinite(interval.ci_low):
                fields[f"param_{model_slug}_{param_slug}_ci_low"] = float(interval.ci_low)
            if interval.ci_high is not None and np.isfinite(interval.ci_high):
                fields[f"param_{model_slug}_{param_slug}_ci_high"] = float(interval.ci_high)
            if interval.std is not None and np.isfinite(interval.std):
                fields[f"param_{model_slug}_{param_slug}_std"] = float(interval.std)
    return fields


def _build_fit_row(
    *,
    metric: MetricSpec,
    group_tag: str,
    sample_size: int,
    fit_enabled: bool,
    fit_summary: Optional[Any],
    bootstrap_summary: Optional[FullPipelineBootstrapSummary],
) -> Dict[str, object]:
    """Assemble one flat fit-summary row for CSV/Parquet export."""
    fit_result = None if fit_summary is None else fit_summary.fit_result
    comparisons = tuple() if fit_summary is None else fit_summary.comparisons
    return {
        "metric": _metric_summary_name(metric),
        "panel_title": metric.panel_title,
        "group": group_tag,
        "sample_size": int(sample_size),
        "fit_enabled": bool(fit_enabled),
        "fit_method": POWERLAW_FIT_METHOD,
        "fit_success": bool(fit_result is not None),
        "alpha": float(fit_result.alpha) if fit_result is not None else np.nan,
        "xmin": float(fit_result.xmin) if fit_result is not None else np.nan,
        "ks_stat": float(fit_result.ks_stat) if fit_result is not None else np.nan,
        "n_tail": int(fit_result.n_tail) if fit_result is not None else 0,
        "powerlaw_min_tail": POWERLAW_MIN_TAIL,
        "powerlaw_num_candidates": POWERLAW_NUM_CANDIDATES,
        "powerlaw_refine_window": POWERLAW_REFINE_WINDOW,
        "dataset_name": DATASET_NAME,
        "level": LEVEL,
        "member_nationality": MEMBER_NATIONALITY_TAG,
        **_bootstrap_metadata_row_fields(bootstrap_summary),
        **_bootstrap_xmin_row_fields(bootstrap_summary),
        **_comparison_row_fields(comparisons),
        **_pairwise_comparison_row_fields(fit_summary),
        **_model_score_row_fields(fit_summary),
        **_model_parameter_row_fields(fit_summary),
        **_model_parameter_bootstrap_row_fields(bootstrap_summary),
    }


def _compact_fit_summary_table(fit_summary: pd.DataFrame) -> pd.DataFrame:
    """Keep only the compact fit-summary columns requested for export."""
    column_map: Dict[str, str] = {
        "metric": "metric",
        "panel_title": "panel_title",
        "group": "group",
        "sample_size": "sample_size",
        "fit_enabled": "fit_enabled",
        "fit_success": "fit_success",
        "fit_method": "fit_method",
        "alpha": "alpha",
        "xmin": "xmin",
        "ks_stat": "ks_stat",
        "n_tail": "n_tail",
        "best_fit_model": "best_fit_model",
        "best_fit_criterion": "best_fit_criterion",
        "best_fit_aic": "best_fit_aic",
        "powerlaw_compare_summary": "powerlaw_compare_summary",
        "bootstrap_full_pipeline_enabled": "bootstrap_full_pipeline_enabled",
        "bootstrap_distribution": "bootstrap_distribution",
        "bootstrap_resampling_scheme": "bootstrap_resampling_scheme",
        "bootstrap_runs_requested": "bootstrap_runs_requested",
        "bootstrap_alpha": "bootstrap_alpha",
        "bootstrap_random_state": "bootstrap_random_state",
        "bootstrap_pipeline_valid_runs": "bootstrap_pipeline_valid_runs",
        "bootstrap_xmin_valid_runs": "bootstrap_xmin_valid_runs",
        "bootstrap_xmin_ci_low": "bootstrap_xmin_ci_low",
        "bootstrap_xmin_ci_high": "bootstrap_xmin_ci_high",
        "bootstrap_xmin_std": "bootstrap_xmin_std",
    }

    for model in POWERLAW_CANDIDATE_MODELS:
        model_slug = _alternative_slug(model)
        src = f"model_{model_slug}_loglikelihood"
        if src in fit_summary.columns:
            column_map[src] = f"{model_slug}_loglikelihood"
        valid_runs_src = f"model_{model_slug}_bootstrap_valid_runs"
        if valid_runs_src in fit_summary.columns:
            column_map[valid_runs_src] = f"{model_slug}_bootstrap_valid_runs"

    for index, model_a in enumerate(POWERLAW_CANDIDATE_MODELS):
        for model_b in POWERLAW_CANDIDATE_MODELS[index + 1 :]:
            model_a_slug = _alternative_slug(model_a)
            model_b_slug = _alternative_slug(model_b)
            ratio_src = f"pairwise_compare_{model_a_slug}_vs_{model_b_slug}_loglikelihood_ratio"
            p_src = f"pairwise_compare_{model_a_slug}_vs_{model_b_slug}_p_value"
            if ratio_src in fit_summary.columns:
                column_map[ratio_src] = f"loglikelihood_ratio_{model_a_slug}_vs_{model_b_slug}"
            if p_src in fit_summary.columns:
                column_map[p_src] = f"p_value_{model_a_slug}_vs_{model_b_slug}"
            # Backward-compatible fallback for older in-memory rows that only
            # stored the power-law-vs-alternative comparison fields.
            if model_a == "power_law":
                legacy_ratio_src = f"compare_{model_b_slug}_loglikelihood_ratio"
                legacy_p_src = f"compare_{model_b_slug}_p_value"
                if legacy_ratio_src in fit_summary.columns and ratio_src not in fit_summary.columns:
                    column_map[legacy_ratio_src] = f"loglikelihood_ratio_{model_a_slug}_vs_{model_b_slug}"
                if legacy_p_src in fit_summary.columns and p_src not in fit_summary.columns:
                    column_map[legacy_p_src] = f"p_value_{model_a_slug}_vs_{model_b_slug}"

    for model in POWERLAW_CANDIDATE_MODELS:
        model_slug = _alternative_slug(model)
        param_cols = sorted(col for col in fit_summary.columns if col.startswith(f"param_{model_slug}_"))
        for src in param_cols:
            column_map[src] = src.removeprefix("param_")

    compact = fit_summary.loc[:, [col for col in column_map if col in fit_summary.columns]].copy()
    return compact.rename(columns=column_map)


def _parameter_count_from_summary_columns(model: str, columns: Sequence[str]) -> int:
    """Infer one model's parameter count from the compact summary columns."""
    model_slug = _alternative_slug(model)
    count = sum(
        1
        for column in columns
        if column.startswith(f"{model_slug}_") and (not column.endswith("_loglikelihood"))
        and (not column.endswith("_ci_low"))
        and (not column.endswith("_ci_high"))
        and (not column.endswith("_std"))
        and (not column.endswith("_bootstrap_valid_runs"))
    )
    return max(int(count), 1)


def _signed_pairwise_ratio_for_models(
    row: pd.Series,
    *,
    best_model: str,
    second_model: str,
    candidate_models: Sequence[str],
) -> tuple[float, float]:
    """Return the pairwise LR sign-adjusted so positive values favor `best_model`."""
    if best_model == second_model:
        return np.nan, np.nan

    best_index = candidate_models.index(best_model)
    second_index = candidate_models.index(second_model)
    if best_index < second_index:
        left_model, right_model = best_model, second_model
        sign = 1.0
    else:
        left_model, right_model = second_model, best_model
        sign = -1.0

    ratio_value = row.get(
        f"loglikelihood_ratio_{_alternative_slug(left_model)}_vs_{_alternative_slug(right_model)}",
        np.nan,
    )
    p_value = row.get(
        f"p_value_{_alternative_slug(left_model)}_vs_{_alternative_slug(right_model)}",
        np.nan,
    )
    if pd.isna(ratio_value) or pd.isna(p_value):
        return np.nan, np.nan
    return float(sign * float(ratio_value)), float(p_value)


def _build_best_vs_second_review_table(fit_summary: pd.DataFrame) -> pd.DataFrame:
    """Build the paper-style review table from the compact fit summary."""
    parameter_counts = {
        model: _parameter_count_from_summary_columns(model, fit_summary.columns)
        for model in POWERLAW_CANDIDATE_MODELS
    }
    rows: list[dict[str, object]] = []

    for _, row in fit_summary.iterrows():
        loglikelihoods: dict[str, float] = {}
        for model in POWERLAW_CANDIDATE_MODELS:
            value = row.get(f"{_alternative_slug(model)}_loglikelihood", np.nan)
            if pd.notna(value):
                loglikelihoods[model] = float(value)
        if len(loglikelihoods) < 2:
            rows.append(
                {
                    "Group": row.get("group", ""),
                    "Metric": row.get("metric", ""),
                    "Best by AIC": "unavailable",
                    "Best fit parameters": "unavailable",
                    "2nd": "unavailable",
                    "2nd fit parameters": "unavailable",
                    "LR for best vs 2nd": np.nan,
                    "p-value": np.nan,
                    "Review": "unavailable",
                }
            )
            continue

        ranked = sorted(
            (
                (
                    model,
                    2.0 * float(parameter_counts.get(model, 1)) - 2.0 * loglikelihood,
                )
                for model, loglikelihood in loglikelihoods.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
        best_model = ranked[0][0]
        second_model = ranked[1][0]
        signed_ratio, p_value = _signed_pairwise_ratio_for_models(
            row,
            best_model=best_model,
            second_model=second_model,
            candidate_models=POWERLAW_CANDIDATE_MODELS,
        )
        if (not np.isfinite(signed_ratio)) or (not np.isfinite(p_value)):
            review = "unavailable"
        elif p_value >= 0.1:
            review = "tie / undecided"
        elif signed_ratio > 0.0:
            review = "decisive"
        else:
            review = "pairwise favors 2nd"

        rows.append(
            {
                "Group": row.get("group", ""),
                "Metric": row.get("metric", ""),
                "Best by AIC": best_model,
                "Best fit parameters": _format_model_parameters_for_review(row, best_model),
                "2nd": second_model,
                "2nd fit parameters": _format_model_parameters_for_review(row, second_model),
                "LR for best vs 2nd": signed_ratio,
                "p-value": p_value,
                "Review": review,
            }
        )

    return pd.DataFrame(rows)


def _format_review_value(value: object, *, precision: int) -> str:
    """Format review-table scalars for Markdown output."""
    if value is None or pd.isna(value):
        return "NA"
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    if number == 0.0:
        return "~0"
    if abs(number) < 1.0e-4:
        return f"{number:.1e}"
    return f"{number:.{precision}f}"


def _format_model_parameters_for_review(row: pd.Series, model: str) -> str:
    """Format one fitted model's parameters and optional bootstrap CIs for review tables."""
    model_slug = _alternative_slug(model)
    parameter_columns = sorted(
        [
        column
        for column in row.index
        if column.startswith(f"{model_slug}_")
        and (not column.endswith("_loglikelihood"))
        and (not column.endswith("_ci_low"))
        and (not column.endswith("_ci_high"))
        and (not column.endswith("_std"))
        and (not column.endswith("_bootstrap_valid_runs"))
        ]
    )
    if not parameter_columns:
        return "NA"

    formatted_parameters: list[str] = []
    for parameter_column in parameter_columns:
        parameter_name = parameter_column.removeprefix(f"{model_slug}_")
        parameter_value = row.get(parameter_column, np.nan)
        formatted = f"{parameter_name}={_format_review_value(parameter_value, precision=4)}"
        ci_low = row.get(f"{parameter_column}_ci_low", np.nan)
        ci_high = row.get(f"{parameter_column}_ci_high", np.nan)
        if pd.notna(ci_low) and pd.notna(ci_high):
            formatted += (
                f" [{_format_review_value(ci_low, precision=4)}, "
                f"{_format_review_value(ci_high, precision=4)}]"
            )
        formatted_parameters.append(formatted)
    return "; ".join(formatted_parameters)


def _best_vs_second_review_markdown(review_table: pd.DataFrame) -> str:
    """Render the review table as Markdown with fixed numeric formatting."""
    header = (
        "| Group | Metric | Best by AIC | Best fit parameters | 2nd | 2nd fit parameters | LR for best vs 2nd | p-value | Review |\n"
        "|---|---|---|---|---|---|---:|---:|---|\n"
    )
    lines = [header]
    for _, row in review_table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["Group"]),
                    str(row["Metric"]),
                    str(row["Best by AIC"]),
                    str(row["Best fit parameters"]),
                    str(row["2nd"]),
                    str(row["2nd fit parameters"]),
                    _format_review_value(row["LR for best vs 2nd"], precision=2),
                    _format_review_value(row["p-value"], precision=4),
                    str(row["Review"]),
                )
            )
            + " |\n"
        )
    return "".join(lines)


def _distribution_summary_values(
    samples: MetaorderDistributionSamples,
    metric: MetricSpec,
) -> np.ndarray:
    """Return the sample array used for one panel's summary statistics."""
    return _filter_positive_finite(getattr(samples, metric.field_name))


def _metric_summary_name(metric: MetricSpec) -> str:
    """Return the exported metric label used in summary tables."""
    return metric.field_name


def _build_distribution_location_summary_table(
    client_samples: MetaorderDistributionSamples,
    proprietary_samples: MetaorderDistributionSamples,
) -> pd.DataFrame:
    """Build the mean/median table for all panels in the combined figure."""
    rows: list[dict[str, object]] = []
    datasets = (
        ("client", client_samples),
        ("proprietary", proprietary_samples),
    )
    for metric in _distribution_metric_specs():
        for group, samples in datasets:
            values = _distribution_summary_values(samples, metric)
            rows.append(
                {
                    "Group": group,
                    "Metric": _metric_summary_name(metric),
                    "Sample size": int(values.size),
                    "Mean": float(np.mean(values)) if values.size else np.nan,
                    "Median": float(np.median(values)) if values.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _distribution_location_summary_markdown(location_summary: pd.DataFrame) -> str:
    """Render the mean/median summary table as Markdown."""
    header = (
        "| Group | Metric | Sample size | Mean | Median |\n"
        "|---|---|---:|---:|---:|\n"
    )
    lines = [header]
    for _, row in location_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["Group"]),
                    str(row["Metric"]),
                    str(int(row["Sample size"])),
                    _format_review_value(row["Mean"], precision=4),
                    _format_review_value(row["Median"], precision=4),
                )
            )
            + " |\n"
        )
    return "".join(lines)


def _bootstrap_review_note(fit_summary: pd.DataFrame) -> str:
    """Return a Markdown note describing bootstrap intervals when enabled."""
    if fit_summary.empty:
        return ""
    enabled_series = fit_summary.get("bootstrap_full_pipeline_enabled")
    if enabled_series is None or not bool(pd.Series(enabled_series).fillna(False).any()):
        return ""

    first_row = fit_summary.iloc[0]
    alpha = first_row.get("bootstrap_alpha", np.nan)
    runs = first_row.get("bootstrap_runs_requested", np.nan)
    scheme = str(first_row.get("bootstrap_resampling_scheme", POWERLAW_FULL_BOOTSTRAP_RESAMPLING_SCHEME))
    if scheme == "nonparametric_full_sample":
        scheme_label = "nonparametric full-sample"
    else:
        scheme_label = scheme.replace("_", " ")
    bootstrap_models_label = ", ".join(BOOTSTRAP_DISTRIBUTION)
    return (
        "_Parameter intervals below are percentile bootstrap confidence intervals "
        f"from the {scheme_label} bootstrap with alpha="
        f"{_format_review_value(alpha, precision=4)} and "
        f"{_format_review_value(runs, precision=0)} requested replicates. "
        f"Each replicate reruns xmin selection and the models in [{bootstrap_models_label}]. "
        "The machine-readable fit summary also stores bootstrap standard deviations for each reported parameter._\n\n"
    )


def _combined_review_markdown(
    review_table: pd.DataFrame,
    location_summary: pd.DataFrame,
    *,
    bootstrap_note: str = "",
) -> str:
    """Render the exported Markdown report with both review and location tables."""
    return (
        "## Best Vs Second Fit Review\n\n"
        + bootstrap_note
        + _best_vs_second_review_markdown(review_table)
        + "\n## Distribution Means And Medians\n\n"
        + _distribution_location_summary_markdown(location_summary)
    )


def _evaluate_model_overlay_curve(
    fit_summary: Any,
    data: np.ndarray,
    *,
    model: str,
    discrete: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate one fitted model on a plotting grid covering the fitted tail."""
    fit_result = fit_summary.fit_result
    x_max = float(np.nanmax(data))
    if (not np.isfinite(x_max)) or x_max <= fit_result.xmin:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    if discrete:
        x_grid = np.arange(
            int(max(1, np.ceil(fit_result.xmin))),
            int(np.floor(x_max)) + 1,
            dtype=float,
        )
    else:
        x_grid = np.logspace(np.log10(fit_result.xmin), np.log10(x_max), 250)
    if x_grid.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    tail_mass = fit_result.n_tail / float(data.size)
    if (model == "power_law") and (not discrete):
        y_grid = tail_mass * power_law_pdf(x_grid, fit_result.alpha, fit_result.xmin)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            distribution = getattr(fit_summary.raw_fit, model)
            y_grid = tail_mass * np.asarray(distribution.pdf(x_grid), dtype=float)
    y_grid = np.ravel(y_grid)
    if y_grid.size != x_grid.size:
        # Some third-party fitted distributions can drop one endpoint when
        # evaluating the PDF on a log grid. Trim to the common support so the
        # diagnostic overlay remains usable instead of failing the whole panel.
        common_size = min(x_grid.size, y_grid.size)
        if common_size == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        x_grid = x_grid[:common_size]
        y_grid = y_grid[:common_size]
    valid = np.isfinite(y_grid) & (y_grid > 0)
    return x_grid[valid], y_grid[valid]


def _fit_power_law_overlay(
    data: np.ndarray,
    *,
    discrete: bool,
    show_progress: bool = False,
) -> Tuple[
    Optional[Any],
    Optional[FullPipelineBootstrapSummary],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit the common tail and return both best-model and power-law overlay curves."""
    if (not POWERLAW_FIT_ENABLED) or data.size == 0:
        empty = np.empty(0, dtype=float)
        return None, None, empty, empty, empty, empty

    effective_min_tail = POWERLAW_MIN_TAIL
    if discrete:
        # Member-count samples contain one observation per member and are much
        # smaller than trade-level panels, so clamp the tail requirement to the
        # available sample size to allow a discrete fit.
        effective_min_tail = min(POWERLAW_MIN_TAIL, max(2, int(data.size)))
    fit_summary = fit_power_law_with_alternatives(
        data,
        discrete=discrete,
        fit_method=POWERLAW_FIT_METHOD,
        min_tail=effective_min_tail,
        num_candidates=POWERLAW_NUM_CANDIDATES,
        refine_window=POWERLAW_REFINE_WINDOW,
        alternatives=POWERLAW_COMPARE_ALTERNATIVES if POWERLAW_COMPARE_ENABLED else (),
    )
    if fit_summary is None:
        empty = np.empty(0, dtype=float)
        return None, None, empty, empty, empty, empty

    bootstrap_summary: Optional[FullPipelineBootstrapSummary] = None
    if POWERLAW_FULL_BOOTSTRAP_ENABLED:
        bootstrap_summary = bootstrap_full_clauset_pipeline_parameters(
            data,
            discrete=discrete,
            fit_method=POWERLAW_FIT_METHOD,
            min_tail=effective_min_tail,
            num_candidates=POWERLAW_FULL_BOOTSTRAP_NUM_CANDIDATES,
            refine_window=POWERLAW_FULL_BOOTSTRAP_REFINE_WINDOW,
            bootstrap_runs=POWERLAW_FULL_BOOTSTRAP_RUNS,
            alpha=POWERLAW_FULL_BOOTSTRAP_ALPHA,
            random_state=POWERLAW_FULL_BOOTSTRAP_RANDOM_STATE,
            alternatives=BOOTSTRAP_ALTERNATIVES,
            empirical_fit=fit_summary,
            show_progress=show_progress,
        )

    best_x, best_y = _evaluate_model_overlay_curve(
        fit_summary,
        data,
        model=fit_summary.best_model,
        discrete=discrete,
    )
    power_x, power_y = _evaluate_model_overlay_curve(
        fit_summary,
        data,
        model="power_law",
        discrete=discrete,
    )
    if fit_summary.best_model == "power_law":
        power_x = np.empty(0, dtype=float)
        power_y = np.empty(0, dtype=float)
    return fit_summary, bootstrap_summary, best_x, best_y, power_x, power_y


def _run_panel_fit(panel: PanelSpec) -> PanelFitResult:
    """Fit and bootstrap one panel sample and return only serializable outputs."""
    started_at = time.perf_counter()
    fit_enabled_for_panel = POWERLAW_FIT_ENABLED
    empty = np.empty(0, dtype=float)

    if fit_enabled_for_panel:
        fit_summary, bootstrap_summary, x_grid, y_grid, powerlaw_x_grid, powerlaw_y_grid = _fit_power_law_overlay(
            panel.panel_values,
            discrete=panel.discrete,
            show_progress=False,
        )
    else:
        fit_summary, bootstrap_summary, x_grid, y_grid, powerlaw_x_grid, powerlaw_y_grid = (
            None,
            None,
            empty,
            empty,
            empty,
            empty,
        )

    fit_result = None if fit_summary is None else fit_summary.fit_result
    comparisons = tuple() if fit_summary is None else fit_summary.comparisons
    best_model = "unavailable" if fit_summary is None else fit_summary.best_model
    best_score_map = {} if fit_summary is None else {score.model: score for score in fit_summary.model_scores}
    best_score = best_score_map.get(best_model)

    fit_row = _build_fit_row(
        metric=panel.metric,
        group_tag=panel.group_tag,
        sample_size=panel.panel_values.size,
        fit_enabled=fit_enabled_for_panel,
        fit_summary=fit_summary,
        bootstrap_summary=bootstrap_summary,
    )
    draw_best_fit = bool(fit_result is not None and x_grid.size > 0)

    return PanelFitResult(
        panel_idx=panel.panel_idx,
        fit_row=fit_row,
        annotation_text=_annotation_text_from_fit_row(fit_row),
        best_model=best_model,
        comparison_summary=_format_comparison_summary(comparisons),
        fit_success=bool(fit_result is not None),
        draw_best_fit=draw_best_fit,
        xmin=float(fit_result.xmin) if fit_result is not None else np.nan,
        alpha=float(fit_result.alpha) if fit_result is not None else np.nan,
        ks_stat=float(fit_result.ks_stat) if fit_result is not None else np.nan,
        n_tail=int(fit_result.n_tail) if fit_result is not None else 0,
        best_fit_aic=(
            float(best_score.aic)
            if best_score is not None and best_score.aic is not None
            else np.nan
        ),
        x_grid=x_grid,
        y_grid=y_grid,
        powerlaw_x_grid=powerlaw_x_grid,
        powerlaw_y_grid=powerlaw_y_grid,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def _add_panel_annotation(
    fig: go.Figure,
    *,
    row_idx: int,
    col_idx: int,
    text: str,
    color: str = "#444444",
) -> None:
    """Add a small summary box inside one subplot."""
    subplot = fig.get_subplot(row_idx, col_idx)
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref=f"{_axis_ref_name(subplot.xaxis.plotly_name)} domain",
        yref=f"{_axis_ref_name(subplot.yaxis.plotly_name)} domain",
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        align="left",
        font=dict(color=color, size=ANNOTATION_FONT_SIZE),
        bordercolor=color,
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(255,255,255,0.85)",
    )


def _panel_progress_tail(panel: PanelSpec) -> str:
    """Render one compact progress label for a panel-level fit task."""
    return (
        f"group={panel.group_tag}, metric={_metric_summary_name(panel.metric)}, "
        f"n={panel.panel_values.size:,}"
    )


def _log_panel_completion(panel: PanelSpec, result: PanelFitResult) -> None:
    """Print the canonical completion line for one panel fit task."""
    fit_status = "fit" if result.fit_success else "no-fit"
    print(
        "[Metaorder distributions] Panel "
        f"{panel.panel_idx}/{panel.total_panels} completed in {result.elapsed_seconds:.1f}s "
        f"(best={result.best_model}; {fit_status}; comparisons={result.comparison_summary})"
    )


def _compute_panel_fit_results(
    panels: Sequence[PanelSpec],
    *,
    show_progress: bool = True,
) -> Dict[int, PanelFitResult]:
    """Execute panel-level tail fits sequentially or in a process pool."""
    if not panels:
        return {}

    worker_count = _resolve_powerlaw_fit_worker_count(len(panels))
    if worker_count <= 1:
        results: Dict[int, PanelFitResult] = {}
        for panel in panels:
            result = _run_panel_fit(panel)
            results[panel.panel_idx] = result
            if show_progress:
                _log_panel_completion(panel, result)
        return results

    if show_progress:
        print(
            "[Metaorder distributions] Parallel tail fits enabled — "
            f"workers={worker_count}, queued_panels={len(panels)}"
        )

    results: Dict[int, PanelFitResult] = {}
    # Execute the largest panels first and recycle the pool between waves so
    # finished workers release their resident memory before the next wave starts.
    execution_order = sorted(
        panels,
        key=lambda panel: panel.panel_values.size,
        reverse=True,
    )
    batch_count = int(np.ceil(len(execution_order) / worker_count))
    for batch_index, batch_start in enumerate(range(0, len(execution_order), worker_count), start=1):
        batch = execution_order[batch_start : batch_start + worker_count]
        if show_progress:
            print(
                "[Metaorder distributions] Parallel tail fit batch "
                f"{batch_index}/{batch_count} — workers={len(batch)}, queued_panels={len(batch)}"
            )
        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            future_to_panel = {
                executor.submit(_run_panel_fit, panel): panel
                for panel in batch
            }
            for future in as_completed(future_to_panel):
                panel = future_to_panel[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive logging path
                    raise RuntimeError(
                        "[Metaorder distributions] Panel "
                        f"{panel.panel_idx}/{panel.total_panels} failed "
                        f"({_panel_progress_tail(panel)})"
                    ) from exc
                results[panel.panel_idx] = result
                if show_progress:
                    _log_panel_completion(panel, result)
    return results


def _apply_panel_fit_result(fig: go.Figure, *, panel: PanelSpec, result: PanelFitResult) -> None:
    """Add one fitted tail overlay, cutoff line, and annotation to the figure."""
    if result.draw_best_fit:
        fig.add_trace(
            go.Scatter(
                x=result.x_grid,
                y=result.y_grid,
                mode="lines",
                line=dict(
                    color=POWERLAW_TRACE_COLOR if result.best_model == "power_law" else BEST_MODEL_TRACE_COLOR,
                    width=2.2,
                    dash="dash",
                ),
                showlegend=False,
                hovertemplate=(
                    f"{panel.metric.panel_title}<br>"
                    f"{panel.label} best-fit tail<br>"
                    f"model={_pretty_model_name(result.best_model)}<br>"
                    f"xmin={result.xmin:.6g}<br>"
                    f"KS={result.ks_stat:.4f}<br>"
                    f"AIC={result.best_fit_aic:.4f}<br>"
                    f"n_tail={result.n_tail:,}<br>"
                    f"comparisons={result.comparison_summary}<extra></extra>"
                ),
            ),
            row=panel.row_idx,
            col=panel.col_idx,
        )
        if result.powerlaw_x_grid.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=result.powerlaw_x_grid,
                    y=result.powerlaw_y_grid,
                    mode="lines",
                    line=dict(color=POWERLAW_TRACE_COLOR, width=2.0),
                    showlegend=False,
                    hovertemplate=(
                        f"{panel.metric.panel_title}<br>"
                        f"{panel.label} power law tail<br>"
                        f"alpha={result.alpha:.4f}<br>"
                        f"xmin={result.xmin:.6g}<br>"
                        f"KS={result.ks_stat:.4f}<extra></extra>"
                    ),
                ),
                row=panel.row_idx,
                col=panel.col_idx,
            )
        fig.add_vline(
            x=float(result.xmin),
            line=dict(color=BEST_MODEL_TRACE_COLOR, width=1.5, dash="dot"),
            opacity=0.8,
            row=panel.row_idx,
            col=panel.col_idx,
        )

    _add_panel_annotation(
        fig,
        row_idx=panel.row_idx,
        col_idx=panel.col_idx,
        text=result.annotation_text,
        color=panel.color,
    )


def _collect_group_samples(metaorders_dict_path: Path, proprietary: bool) -> MetaorderDistributionSamples:
    """Load one flow-group dictionary and recompute its aggregated samples."""
    return collect_metaorder_distribution_samples(
        metaorders_dict_path=metaorders_dict_path,
        parquet_dir=PARQUET_DIR,
        proprietary=proprietary,
        trading_hours=TRADING_HOURS,
        member_nationality=MEMBER_NATIONALITY,
        members_nationality_path=MEMBERS_NATIONALITY_PATH,
        include_counts_by_member=False,
        show_progress=True,
    )


def build_distribution_figure(
    client_samples: MetaorderDistributionSamples,
    proprietary_samples: MetaorderDistributionSamples,
    *,
    show_progress: bool = True,
) -> Tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
    """
    Summary
    -------
    Build the combined comparison figure and the fit summary table.

    Parameters
    ----------
    client_samples : MetaorderDistributionSamples
        Aggregated samples for the client flow slice.
    proprietary_samples : MetaorderDistributionSamples
        Aggregated samples for the proprietary flow slice.
    show_progress : bool, default=True
        If True, print one progress line per panel so long tail-model fitting
        stages do not appear stalled.

    Returns
    -------
    tuple[go.Figure, pd.DataFrame, pd.DataFrame]
        Plotly figure, one row per fitted metric/group with fit diagnostics, and
        a long-format curve table used to regenerate the figure without
        recomputing the pipeline inputs.

    Notes
    -----
    All rows are plotted on log-scaled axes using continuous log-binned density
    curves with optional tail fits. The expensive tail fit plus optional
    full-pipeline bootstrap for each panel is an independent CPU-bound task, so
    those jobs are dispatched through a `ProcessPoolExecutor` when multiple
    workers are enabled.
    """
    metrics: Sequence[MetricSpec] = _distribution_metric_specs()
    datasets = (
        ("Client", "client", client_samples, COLOR_CLIENT, 1),
        ("Proprietary", "proprietary", proprietary_samples, COLOR_PROPRIETARY, 2),
    )

    fig = _build_distribution_figure_shell()

    fit_rows_by_panel: dict[int, dict[str, object]] = {}
    panels_to_fit: list[PanelSpec] = []
    ordered_panels: list[PanelSpec] = []
    plot_rows: list[dict[str, object]] = []
    total_panels = len(metrics) * len(datasets)
    for row_idx, metric in enumerate(metrics, start=1):
        for label, group_tag, samples, color, col_idx in datasets:
            panel_values = _distribution_summary_values(samples, metric)
            panel_idx = (row_idx - 1) * len(datasets) + col_idx
            panel = PanelSpec(
                panel_idx=panel_idx,
                total_panels=total_panels,
                row_idx=row_idx,
                col_idx=col_idx,
                metric=metric,
                label=label,
                group_tag=group_tag,
                color=color,
                panel_values=panel_values,
            )
            ordered_panels.append(panel)
            panel_started_at = time.perf_counter()
            if show_progress:
                print(
                    "[Metaorder distributions] Panel "
                    f"{panel_idx}/{total_panels} — {_panel_progress_tail(panel)}"
                )

            if panel_values.size == 0:
                _add_panel_annotation(
                    fig,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    text="No positive finite data",
                )
                fit_rows_by_panel[panel_idx] = _build_fit_row(
                    metric=metric,
                    group_tag=group_tag,
                    sample_size=0,
                    fit_enabled=POWERLAW_FIT_ENABLED,
                    fit_summary=None,
                    bootstrap_summary=None,
                )
                if show_progress:
                    elapsed = time.perf_counter() - panel_started_at
                    print(
                        "[Metaorder distributions] Panel "
                        f"{panel_idx}/{total_panels} completed in {elapsed:.1f}s "
                        "(no positive finite data)"
                    )
                continue

            centers, density = _density_curve(panel_values, bins=metric.bins)
            y_label = "density"
            hover_template = (
                f"{metric.panel_title}<br>"
                f"{label}<br>"
                "x=%{x:.4g}<br>"
                f"{y_label}=%{{y:.4g}}<extra></extra>"
            )
            fig.add_trace(
                go.Scatter(
                    x=centers,
                    y=density,
                    mode="lines+markers",
                    line=dict(color=color, width=2.6),
                    marker=dict(color=color, size=6),
                    showlegend=False,
                    hovertemplate=hover_template,
                ),
                row=row_idx,
                col=col_idx,
            )
            plot_rows.extend(
                _curve_rows_from_xy(
                    panel=panel,
                    trace_kind="density",
                    x_values=centers,
                    y_values=density,
                )
            )
            panels_to_fit.append(panel)

    fit_results = _compute_panel_fit_results(panels_to_fit, show_progress=show_progress)
    fit_rows: list[dict[str, object]] = []
    for panel in ordered_panels:
        result = fit_results.get(panel.panel_idx)
        if result is not None:
            _apply_panel_fit_result(fig, panel=panel, result=result)
            fit_rows_by_panel[panel.panel_idx] = result.fit_row
            plot_rows.extend(
                _curve_rows_from_xy(
                    panel=panel,
                    trace_kind="best_fit",
                    x_values=result.x_grid,
                    y_values=result.y_grid,
                )
            )
            plot_rows.extend(
                _curve_rows_from_xy(
                    panel=panel,
                    trace_kind="power_law_overlay",
                    x_values=result.powerlaw_x_grid,
                    y_values=result.powerlaw_y_grid,
                )
            )
        fit_rows.append(fit_rows_by_panel[panel.panel_idx])
    plot_data = (
        pd.DataFrame(plot_rows)
        if plot_rows
        else _empty_distribution_plot_data_frame()
    )
    return fig, pd.DataFrame(fit_rows), plot_data


def _sorted_saved_curve_xy(
    plot_data: pd.DataFrame,
    *,
    metric: str,
    group: str,
    trace_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one saved curve, ordered by its serialized point index."""
    if plot_data.empty:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    subset = plot_data.loc[
        (plot_data["metric"] == metric)
        & (plot_data["group"] == group)
        & (plot_data["trace_kind"] == trace_kind)
    ].copy()
    if subset.empty:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    subset.sort_values("point_index", inplace=True)
    return (
        subset["x"].to_numpy(dtype=float),
        subset["y"].to_numpy(dtype=float),
    )


def build_distribution_figure_from_saved_outputs(
    fit_summary: pd.DataFrame,
    plot_data: pd.DataFrame,
    *,
    show_progress: bool = True,
) -> go.Figure:
    """
    Summary
    -------
    Rebuild the distribution figure from saved fit and curve outputs only.

    Parameters
    ----------
    fit_summary : pd.DataFrame
        Compact fit-summary table previously written by this script.
    plot_data : pd.DataFrame
        Long-format curve table previously written by this script.
    show_progress : bool, default=True
        If True, print one progress line per reconstructed panel.

    Returns
    -------
    go.Figure
        Plotly figure equivalent to the normal pipeline output.

    Notes
    -----
    This load-only path avoids reading the parquet trade tapes and avoids
    rerunning the tail-fit/bootstrap jobs. It depends on the saved
    `metaorder_distribution_plot_data_*` and fit-summary artifacts.
    """
    if fit_summary.empty:
        raise ValueError("Saved fit summary is empty; cannot regenerate the distribution figure.")

    required_fit_columns = {"metric", "group", "sample_size", "fit_success", "panel_title"}
    missing_fit_columns = sorted(required_fit_columns.difference(fit_summary.columns))
    if missing_fit_columns:
        raise ValueError(
            "Saved fit summary is missing required columns for load mode: "
            f"{missing_fit_columns}"
        )

    required_plot_columns = {"metric", "group", "trace_kind", "point_index", "x", "y"}
    missing_plot_columns = sorted(required_plot_columns.difference(plot_data.columns))
    if missing_plot_columns:
        raise ValueError(
            "Saved plot data is missing required columns for load mode: "
            f"{missing_plot_columns}"
        )

    fig = _build_distribution_figure_shell()
    fit_rows = {
        (str(row["metric"]), str(row["group"])): row
        for _, row in fit_summary.iterrows()
    }

    for row_idx, metric in enumerate(_distribution_metric_specs(), start=1):
        for label, group_tag, color, col_idx in _distribution_group_specs():
            panel_key = (metric.field_name, group_tag)
            fit_row = fit_rows.get(panel_key)
            if fit_row is None:
                raise ValueError(
                    "Saved fit summary does not contain the required panel "
                    f"metric={metric.field_name!r}, group={group_tag!r}."
                )

            sample_size = int(fit_row.get("sample_size", 0) or 0)
            density_x, density_y = _sorted_saved_curve_xy(
                plot_data,
                metric=metric.field_name,
                group=group_tag,
                trace_kind="density",
            )
            if sample_size > 0 and density_x.size == 0:
                raise ValueError(
                    "Saved plot data does not contain the empirical density curve for "
                    f"metric={metric.field_name!r}, group={group_tag!r}."
                )
            if density_x.size > 0:
                fig.add_trace(
                    go.Scatter(
                        x=density_x,
                        y=density_y,
                        mode="lines+markers",
                        line=dict(color=color, width=2.6),
                        marker=dict(color=color, size=6),
                        showlegend=False,
                        hovertemplate=(
                            f"{metric.panel_title}<br>"
                            f"{label}<br>"
                            "x=%{x:.4g}<br>"
                            "density=%{y:.4g}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            best_x, best_y = _sorted_saved_curve_xy(
                plot_data,
                metric=metric.field_name,
                group=group_tag,
                trace_kind="best_fit",
            )
            powerlaw_x, powerlaw_y = _sorted_saved_curve_xy(
                plot_data,
                metric=metric.field_name,
                group=group_tag,
                trace_kind="power_law_overlay",
            )
            best_model = str(fit_row.get("best_fit_model", "unavailable"))
            if best_x.size > 0:
                xmin_value = pd.to_numeric(pd.Series([fit_row.get("xmin", np.nan)]), errors="coerce").iloc[0]
                ks_value = pd.to_numeric(pd.Series([fit_row.get("ks_stat", np.nan)]), errors="coerce").iloc[0]
                aic_value = pd.to_numeric(pd.Series([fit_row.get("best_fit_aic", np.nan)]), errors="coerce").iloc[0]
                n_tail_value = int(pd.to_numeric(pd.Series([fit_row.get("n_tail", 0)]), errors="coerce").fillna(0).iloc[0])
                fig.add_trace(
                    go.Scatter(
                        x=best_x,
                        y=best_y,
                        mode="lines",
                        line=dict(
                            color=POWERLAW_TRACE_COLOR if best_model == "power_law" else BEST_MODEL_TRACE_COLOR,
                            width=2.2,
                            dash="dash",
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"{metric.panel_title}<br>"
                            f"{label} best-fit tail<br>"
                            f"model={_pretty_model_name(best_model)}<br>"
                            f"xmin={_format_annotation_scalar(xmin_value, format_spec='.6g')}<br>"
                            f"KS={_format_annotation_scalar(ks_value, format_spec='.4f')}<br>"
                            f"AIC={_format_annotation_scalar(aic_value, format_spec='.4f')}<br>"
                            f"n_tail={n_tail_value:,}<br>"
                            f"comparisons={fit_row.get('powerlaw_compare_summary', 'none')}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                if powerlaw_x.size > 0:
                    alpha_value = pd.to_numeric(pd.Series([fit_row.get("alpha", np.nan)]), errors="coerce").iloc[0]
                    fig.add_trace(
                        go.Scatter(
                            x=powerlaw_x,
                            y=powerlaw_y,
                            mode="lines",
                            line=dict(color=POWERLAW_TRACE_COLOR, width=2.0),
                            showlegend=False,
                            hovertemplate=(
                                f"{metric.panel_title}<br>"
                                f"{label} power law tail<br>"
                                f"alpha={_format_annotation_scalar(alpha_value, format_spec='.4f')}<br>"
                                f"xmin={_format_annotation_scalar(xmin_value, format_spec='.6g')}<br>"
                                f"KS={_format_annotation_scalar(ks_value, format_spec='.4f')}<extra></extra>"
                            ),
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
                if np.isfinite(xmin_value):
                    fig.add_vline(
                        x=float(xmin_value),
                        line=dict(color=BEST_MODEL_TRACE_COLOR, width=1.5, dash="dot"),
                        opacity=0.8,
                        row=row_idx,
                        col=col_idx,
                    )

            _add_panel_annotation(
                fig,
                row_idx=row_idx,
                col_idx=col_idx,
                text=_annotation_text_from_fit_row(fit_row),
                color=color,
            )
            if show_progress:
                print(
                    "[Metaorder distributions] Rebuilt panel from saved outputs — "
                    f"group={group_tag}, metric={metric.field_name}, sample_size={sample_size:,}"
                )

    return fig


def _load_saved_dataframe(parquet_path: Path, csv_path: Path, *, artifact_name: str) -> pd.DataFrame:
    """Load a saved DataFrame from parquet, falling back to CSV when needed."""
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"Missing saved {artifact_name}. Expected either {parquet_path} or {csv_path}."
    )


def _load_saved_distribution_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the compact fit summary and saved plot-data artifact for `--load` mode."""
    output_paths = _saved_output_paths()
    fit_summary = _load_saved_dataframe(
        output_paths["fit_summary_parquet"],
        output_paths["fit_summary_csv"],
        artifact_name="fit summary",
    )
    plot_data = _load_saved_dataframe(
        output_paths["plot_data_parquet"],
        output_paths["plot_data_csv"],
        artifact_name="plot data",
    )
    return fit_summary, plot_data


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Summary
    -------
    Build the command-line parser for the distribution figure script.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Parser exposing the optional load-only regeneration mode.

    Notes
    -----
    The default run remains configuration-driven via the YAML file. `--load`
    skips the expensive sample reconstruction and tail-fit pipeline, rebuilding
    the figure from saved script outputs only.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the metaorder distribution comparison figure, or regenerate it "
            "from saved script outputs only."
        )
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help=(
            "Load the saved fit summary and saved plot-data artifact written by a "
            "previous run, then regenerate the HTML/PNG figure without reading the "
            "trade tapes or recomputing the fits."
        ),
    )
    return parser


def main() -> None:
    """
    Summary
    -------
    Run the YAML-configured metaorder-distributions pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The script always loads both client and proprietary dictionaries in one run
    and writes one combined comparison figure for the selected nationality slice.
    """
    args = build_arg_parser().parse_args()
    log_path = OUTPUT_DIR / "logs" / with_member_nationality_tag(
        f"metaorder_distributions_{LEVEL}_prop_vs_client.log",
        MEMBER_NATIONALITY,
    )
    logger = setup_file_logger(Path(__file__).stem, log_path, mode="a")
    with PrintTee(logger):
        print("[Intro] Metaorder distributions run started...")
        print(
            "[Intro] Parameters — \n"
            f"  DATASET={DATASET_NAME}, LEVEL={LEVEL}, "
            f"MEMBER_NATIONALITY={MEMBER_NATIONALITY_TAG}, "
            f"TRADING_HOURS={TRADING_HOURS}, "
                f"RUN_METAORDER_DISTRIBUTIONS={RUN_METAORDER_DISTRIBUTIONS}, "
                f"POWERLAW_FIT_ENABLED={POWERLAW_FIT_ENABLED}, "
                f"POWERLAW_FIT_METHOD={POWERLAW_FIT_METHOD}, "
                f"POWERLAW_MIN_TAIL={POWERLAW_MIN_TAIL}, "
                f"POWERLAW_NUM_CANDIDATES={POWERLAW_NUM_CANDIDATES}, "
                f"POWERLAW_REFINE_WINDOW={POWERLAW_REFINE_WINDOW}, "
                f"POWERLAW_COMPARE_ENABLED={POWERLAW_COMPARE_ENABLED}, "
                f"POWERLAW_COMPARE_ALTERNATIVES={POWERLAW_COMPARE_ALTERNATIVES}, "
                f"POWERLAW_FULL_BOOTSTRAP_ENABLED={POWERLAW_FULL_BOOTSTRAP_ENABLED}, "
                f"BOOTSTRAP_DISTRIBUTION={BOOTSTRAP_DISTRIBUTION}, "
                f"POWERLAW_FULL_BOOTSTRAP_RUNS={POWERLAW_FULL_BOOTSTRAP_RUNS}, "
                f"POWERLAW_FULL_BOOTSTRAP_ALPHA={POWERLAW_FULL_BOOTSTRAP_ALPHA}, "
                f"POWERLAW_FULL_BOOTSTRAP_RANDOM_STATE={POWERLAW_FULL_BOOTSTRAP_RANDOM_STATE}, "
                f"POWERLAW_FULL_BOOTSTRAP_NUM_CANDIDATES={POWERLAW_FULL_BOOTSTRAP_NUM_CANDIDATES}, "
                f"POWERLAW_FULL_BOOTSTRAP_REFINE_WINDOW={POWERLAW_FULL_BOOTSTRAP_REFINE_WINDOW}, "
                f"POWERLAW_FIT_MAX_WORKERS={POWERLAW_FIT_MAX_WORKERS}, "
                f"LOAD_ONLY={bool(args.load)}"
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

        plot_dirs: PlotOutputDirs = make_plot_output_dirs(IMG_DIR, use_subdirs=True)
        ensure_plot_dirs(plot_dirs)
        DISTRIBUTIONS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_paths = _saved_output_paths()

        if args.load:
            print("[Metaorder distributions] Load mode enabled; rebuilding the figure from saved outputs only...")
            fit_summary, plot_data = _load_saved_distribution_outputs()
            fig = build_distribution_figure_from_saved_outputs(
                fit_summary,
                plot_data,
                show_progress=True,
            )
            print("[Metaorder distributions] Exporting regenerated Plotly figure to HTML/PNG...")
            html_path, png_path = save_plotly_figure(
                fig,
                stem=_figure_stem(),
                dirs=plot_dirs,
                width=1400,
                height=2080,
                scale=2,
                write_html=True,
                write_png=True,
                strict_png=False,
            )
            print(
                "[Metaorder distributions] Saved regenerated comparison figure to "
                f"HTML={html_path} PNG={png_path}"
            )
            return

        if not RUN_METAORDER_DISTRIBUTIONS:
            print("[Metaorder distributions] RUN_METAORDER_DISTRIBUTIONS is false; nothing to do.")
            return

        try:
            client_samples = _collect_group_samples(CLIENT_DICT_PATH, proprietary=False)
            proprietary_samples = _collect_group_samples(PROPRIETARY_DICT_PATH, proprietary=True)
        except Exception as exc:
            print(f"[Metaorder distributions] Failed to collect comparison samples: {exc}")
            return

        print(
            "[Metaorder distributions] Sample counts — "
            f"client_metaorders={client_samples.total_metaorders:,}, "
            f"proprietary_metaorders={proprietary_samples.total_metaorders:,}"
        )

        print("[Metaorder distributions] Building comparison figure and best-fit tail overlays...")
        figure_started_at = time.perf_counter()
        fig, fit_summary, plot_data = build_distribution_figure(
            client_samples,
            proprietary_samples,
            show_progress=True,
        )
        print(
            "[Metaorder distributions] Figure assembly completed in "
            f"{time.perf_counter() - figure_started_at:.1f}s"
        )
        fit_summary = _compact_fit_summary_table(fit_summary)
        fit_review_table = _build_best_vs_second_review_table(fit_summary)
        location_summary_table = _build_distribution_location_summary_table(
            client_samples,
            proprietary_samples,
        )
        print("[Metaorder distributions] Exporting Plotly figure to HTML/PNG...")
        html_path, png_path = save_plotly_figure(
            fig,
            stem=_figure_stem(),
            dirs=plot_dirs,
            width=1400,
            height=2080,
            scale=2,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        print(f"[Metaorder distributions] Saved comparison figure to HTML={html_path} PNG={png_path}")

        fit_summary.to_csv(output_paths["fit_summary_csv"], index=False)
        fit_summary.to_parquet(output_paths["fit_summary_parquet"], index=False)
        print(
            "[Metaorder distributions] Saved fit summary to "
            f"CSV={output_paths['fit_summary_csv']} PARQUET={output_paths['fit_summary_parquet']}"
        )

        plot_data.to_csv(output_paths["plot_data_csv"], index=False)
        plot_data.to_parquet(output_paths["plot_data_parquet"], index=False)
        print(
            "[Metaorder distributions] Saved figure-regeneration plot data to "
            f"CSV={output_paths['plot_data_csv']} PARQUET={output_paths['plot_data_parquet']}"
        )

        fit_review_table.to_csv(output_paths["review_csv"], index=False)
        output_paths["review_markdown"].write_text(
            _combined_review_markdown(
                fit_review_table,
                location_summary_table,
                bootstrap_note=_bootstrap_review_note(fit_summary),
            ),
            encoding="utf-8",
        )
        print(
            "[Metaorder distributions] Saved fit-review outputs to "
            f"CSV={output_paths['review_csv']} MARKDOWN={output_paths['review_markdown']}"
        )


if __name__ == "__main__":
    main()
