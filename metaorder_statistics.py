#!/usr/bin/env python3
"""
Metaorder dictionary statistics (durations, volumes, inter-arrivals, participation).

What this script does
---------------------
This script produces distributional summaries and plots from the metaorder-index
dictionary produced by `metaorder_computation.py` (the `metaorders_dict_all_*`
pickle). It loads the corresponding per-ISIN trade tapes (parquet files) and
computes metaorder-level quantities such as:

- durations
- inter-arrival times
- metaorder volumes
- Q/V and participation-rate proxies
- metaorders-per-member counts

It then exports figures under the configured output folder.

How to run
----------
1) Edit `config_ymls/metaorder_statistics.yml` (dataset name, dict path, tape folder, proprietary flag).
2) Run:

    python metaorder_statistics.py

Outputs
-------
- Figures: `images/{DATASET_NAME}/{METAORDER_STATS_LEVEL}_{proprietary_tag}/png/` and `html/`
- Log file: `metaorder_statistics.log`
"""

from __future__ import annotations

import datetime as dt
import gc
import math
import builtins
import logging
import pickle
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (good for scripts / servers)
import matplotlib.pyplot as plt
from matplotlib import cycler
import seaborn as sns

import plotly.express as px
import plotly.io as pio

# ---------------------------------------------------------------------
# Configuration loader (YAML)
# ---------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_CONFIG_PATH = _SCRIPT_DIR / "config_ymls" / "metaorder_statistics.yml"
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config file: {_CONFIG_PATH}")

_CFG = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
if not isinstance(_CFG, dict):
    raise TypeError(f"Config must be a mapping (YAML dict): {_CONFIG_PATH}")


def _cfg_require(key: str):
    if key not in _CFG:
        raise KeyError(f"Missing required key '{key}' in {_CONFIG_PATH}")
    return _CFG[key]


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (_SCRIPT_DIR / path).resolve()
    return path


def _format_path_template(template: str, context: Mapping[str, str]) -> str:
    """
    Format a path template with a restricted placeholder set.

    Allowed placeholders are the keys in `context`.
    """
    if "{" not in template:
        return template
    try:
        return template.format(**context)
    except KeyError as e:
        allowed = ", ".join(sorted(context.keys()))
        raise KeyError(
            f"Unknown placeholder {e} in path template '{template}'. "
            f"Allowed placeholders: {allowed}."
        ) from e


# Sizes tuned for print-friendly plots (loaded from YAML)
TICK_FONT_SIZE = int(_cfg_require("TICK_FONT_SIZE"))
LABEL_FONT_SIZE = int(_cfg_require("LABEL_FONT_SIZE"))
TITLE_FONT_SIZE = int(_cfg_require("TITLE_FONT_SIZE"))
LEGEND_FONT_SIZE = int(_cfg_require("LEGEND_FONT_SIZE"))
DEFAULT_FIGSIZE = tuple(_cfg_require("DEFAULT_FIGSIZE"))
THEME_COLORWAY = ["#5B8FF9", "#91CC75", "#EE6666", "#5470C6", "#FAC858", "#73C0DE"]
THEME_GRID_COLOR = "#E5ECF6"
THEME_BG_COLOR = "#FFFFFF"
THEME_FONT_FAMILY = "DejaVu Sans"

sns.set_theme(style="whitegrid", palette=THEME_COLORWAY)
pio.templates.default = "plotly_white"

plt.rcParams.update({
    "font.size": TICK_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "font.family": THEME_FONT_FAMILY,
    "axes.labelsize": LABEL_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.grid": True,
    "axes.facecolor": THEME_BG_COLOR,
    "axes.prop_cycle": cycler(color=THEME_COLORWAY),
    "figure.facecolor": THEME_BG_COLOR,
    "savefig.facecolor": THEME_BG_COLOR,
    "grid.color": THEME_GRID_COLOR,
    "grid.linestyle": ":",
    "grid.alpha": 0.7,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "figure.figsize": DEFAULT_FIGSIZE,
})

# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
LOG_PATH = Path(__file__).with_suffix(".log")
logger = logging.getLogger(Path(__file__).stem)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _formatter = logging.Formatter("%(asctime)s - %(message)s")
    _file_handler = logging.FileHandler(LOG_PATH, mode="a")
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)

_original_print = builtins.print


def log_print(*args, **kwargs):
    """
    Summary
    -------
    Print to stdout and append the same message to this script's log file.

    Parameters
    ----------
    *args
        Positional arguments forwarded to `print`.
    **kwargs
        Keyword arguments forwarded to `print` (e.g. `sep`, `end`, `flush`).

    Returns
    -------
    None

    Notes
    -----
    - The module monkey-patches `builtins.print` to this function so that
      printed diagnostics are also persisted to `metaorder_statistics.log`.

    Examples
    --------
    >>> log_print(\"hello\", 1, 2, sep=\"|\")
    hello|1|2
    """
    sep = kwargs.get("sep", " ")
    message = sep.join(str(a) for a in args)
    logger.info(message)
    _original_print(*args, **kwargs)


builtins.print = log_print

# ---------------------------------------------------------------------
# Configuration (loaded from YAML)
# ---------------------------------------------------------------------
DATASET_NAME = str(_CFG.get("DATASET_NAME") or "ftsemib")

# Trading-hours filter applied when loading the per-ISIN trade tapes.
TRADING_HOURS = tuple(_cfg_require("TRADING_HOURS"))

# Metaorder dictionary stats (raw metaorder indices, not the per-metaorder info parquet)
RUN_METAORDER_DICT_STATS = bool(_cfg_require("RUN_METAORDER_DICT_STATS"))
METAORDER_STATS_LEVEL = str(_cfg_require("METAORDER_STATS_LEVEL"))
METAORDER_STATS_PROPRIETARY = bool(_cfg_require("METAORDER_STATS_PROPRIETARY"))
METAORDER_STATS_PROPRIETARY_TAG = "proprietary" if METAORDER_STATS_PROPRIETARY else "non_proprietary"

_path_context = {
    "DATASET_NAME": DATASET_NAME,
    "METAORDER_STATS_LEVEL": METAORDER_STATS_LEVEL,
    "METAORDER_STATS_PROPRIETARY_TAG": METAORDER_STATS_PROPRIETARY_TAG,
}

PARQUET_PATH = str(_cfg_require("PARQUET_PATH"))
PARQUET_DIR = _resolve_repo_path(_format_path_template(PARQUET_PATH, _path_context))
log_print(f"Using parquet directory: {PARQUET_DIR}")

OUTPUT_FILE_PATH = str(_cfg_require("OUTPUT_FILE_PATH"))
OUTPUT_DIR = _resolve_repo_path(_format_path_template(OUTPUT_FILE_PATH, _path_context))

IMG_OUTPUT_PATH = str(_cfg_require("IMG_OUTPUT_PATH"))
IMG_BASE_DIR = _resolve_repo_path(_format_path_template(IMG_OUTPUT_PATH, _path_context))

_stats_dict_override = _CFG.get("METAORDER_STATS_DICT_PATH")
if _stats_dict_override is None:
    METAORDER_STATS_DICT_PATH = (
        OUTPUT_DIR / f"metaorders_dict_all_{METAORDER_STATS_LEVEL}_{METAORDER_STATS_PROPRIETARY_TAG}.pkl"
    )
else:
    METAORDER_STATS_DICT_PATH = _resolve_repo_path(
        _format_path_template(str(_stats_dict_override), _path_context)
    )

PLOT_DIR = IMG_BASE_DIR / f"{METAORDER_STATS_LEVEL}_{METAORDER_STATS_PROPRIETARY_TAG}"
PNG_DIR = Path(PLOT_DIR) / "png"
HTML_DIR = Path(PLOT_DIR) / "html"
for _figure_dir in (Path(PLOT_DIR), PNG_DIR, HTML_DIR):
    _figure_dir.mkdir(parents=True, exist_ok=True)

METAORDER_POWERLAW_FIT_ENABLED = bool(_CFG.get("METAORDER_POWERLAW_FIT_ENABLED", True))
METAORDER_POWERLAW_MIN_TAIL = max(int(_CFG.get("METAORDER_POWERLAW_MIN_TAIL", 50)), 2)
METAORDER_POWERLAW_NUM_CANDIDATES = max(int(_CFG.get("METAORDER_POWERLAW_NUM_CANDIDATES", 200)), 5)
METAORDER_POWERLAW_REFINE_WINDOW = max(int(_CFG.get("METAORDER_POWERLAW_REFINE_WINDOW", 50)), 0)


def _png_output_path(path_like: str | Path) -> Path:
    """Return a PNG output path under the canonical `png/` directory."""
    return PNG_DIR / Path(path_like).name


def _html_output_path(path_like: str | Path) -> Path:
    """Return an HTML output path under the canonical `html/` directory."""
    return HTML_DIR / Path(path_like).name


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def plot_pdf_line(
    ax: plt.Axes,
    data: Iterable[float],
    bins: int | str | Sequence[float] = 50,
    label: str | None = None,
    color: str | None = None,
    marker: str = "o",
    markersize: float = 2.5,
    logx: bool = False,
    logy: bool = False,
) -> bool:
    """
    Summary
    -------
    Plot a 1D density line using histogram-based PDF estimates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    data : Iterable[float]
        Sample values.
    bins : int | str | Sequence[float], default=50
        Histogram bin specification forwarded to `numpy.histogram`.
    label : str | None, default=None
        Legend label.
    color : str | None, default=None
        Line color.
    marker : str, default=\"o\"
        Matplotlib marker.
    markersize : float, default=2.5
        Marker size.
    logx : bool, default=False
        If True, use a log scale on the x axis (non-positive values are dropped).
    logy : bool, default=False
        If True, use a log scale on the y axis.

    Returns
    -------
    plotted : bool
        True if a line was drawn, False if no valid data were available.

    Notes
    -----
    - This helper avoids KDE dependencies and behaves robustly for small samples.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> _ = plot_pdf_line(ax, [1, 2, 3], bins=3)
    """
    arr = pd.to_numeric(pd.Series(list(data)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if logx:
        arr = arr[arr > 0]
    if arr.size == 0:
        return False

    density, edges = np.histogram(arr, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax.plot(
        centers,
        density,
        label=label,
        color=color,
        marker=marker,
        linestyle="-",
        markersize=markersize,
    )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    return True


def _plot_kde_line(
    ax: plt.Axes,
    data: Iterable[float],
    label: str | None = None,
    color: str | None = None,
    bw_adjust: float = 1.0,
    logx: bool = False,
    logy: bool = False,
) -> bool:
    """Plot a 1D KDE line and return False when density estimation is not feasible."""
    arr = pd.to_numeric(pd.Series(list(data)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if logx:
        arr = arr[arr > 0]
    if arr.size < 2:
        return False

    # KDE is undefined for near-constant samples (singular covariance).
    if np.allclose(arr, arr[0]):
        return False

    try:
        sns.kdeplot(
            x=arr,
            ax=ax,
            label=label,
            color=color,
            bw_adjust=bw_adjust,
            fill=False,
            cut=0,
            linewidth=1.5,
        )
    except Exception:
        return False

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    return True


@dataclass(frozen=True)
class PowerLawFitResult:
    """Container for the fitted parameters of a Clauset-style continuous power law."""

    alpha: float
    xmin: float
    ks_stat: float
    n_tail: int


def fit_power_law_clauset_continuous(
    data: Iterable[float],
    min_tail: int = 50,
    num_candidates: int = 200,
    refine_window: int = 50,
) -> Optional[PowerLawFitResult]:
    """
    Summary
    -------
    Fit a continuous power law using the Clauset et al. MLE + KS xmin selection.

    Parameters
    ----------
    data : Iterable[float]
        One-dimensional sample of positive observations.
    min_tail : int, default=50
        Minimum number of observations required in the fitted tail x >= xmin.
    num_candidates : int, default=200
        Number of coarse candidate cutoffs built from log-spaced tail sizes.
    refine_window : int, default=50
        Number of start-index positions explored on each side of the best
        coarse candidate during local refinement.

    Returns
    -------
    Optional[PowerLawFitResult]
        Best-fit parameters and KS distance, or None when no valid tail exists.

    Notes
    -----
    - Only finite values strictly greater than zero are used.
    - Coarse candidate cutoffs are generated from log-spaced tail sizes.
    - A local refinement pass evaluates neighboring indices around the best
      coarse candidate.
    - For each candidate xmin, alpha is estimated with:
      alpha = 1 + n / sum(log(x_i / xmin)).
    - The selected xmin minimizes the KS distance between empirical and model CDF.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = 1.0 * np.power(1.0 - rng.random(5000), -1.0 / (2.5 - 1.0))
    >>> fit = fit_power_law_clauset_continuous(x, min_tail=100)
    >>> fit is not None
    True
    """
    if isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=float)
    else:
        arr = np.asarray(list(data), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    min_tail = max(int(min_tail), 2)
    num_candidates = max(int(num_candidates), 5)
    refine_window = max(int(refine_window), 0)

    n = arr.size
    if n < min_tail:
        return None

    x_sorted = np.sort(arr)
    if x_sorted.size < 2:
        return None

    log_sorted = np.log(x_sorted)
    suffix_log_sum = np.cumsum(log_sorted[::-1])[::-1]
    ranks = np.arange(1, n + 1, dtype=float)

    def _evaluate_start_index(start_idx: int) -> Optional[PowerLawFitResult]:
        # The maximum value is not used as xmin: tail above it is degenerate.
        if start_idx < 0 or start_idx >= (n - 1):
            return None

        n_tail = n - start_idx
        if n_tail < min_tail:
            return None

        xmin = float(x_sorted[start_idx])
        if xmin <= 0 or not np.isfinite(xmin):
            return None

        # sum(log(x_i / xmin)) = sum(log(x_i)) - n_tail * log(xmin)
        denom = float(suffix_log_sum[start_idx] - n_tail * math.log(xmin))
        if (not np.isfinite(denom)) or denom <= 0:
            return None

        alpha = 1.0 + n_tail / denom
        if (not np.isfinite(alpha)) or alpha <= 1.0:
            return None

        tail = x_sorted[start_idx:]
        model_cdf = 1.0 - np.power(tail / xmin, 1.0 - alpha)
        model_cdf = np.clip(model_cdf, 0.0, 1.0)

        empirical_cdf_low = (ranks[:n_tail] - 1.0) / n_tail
        empirical_cdf_high = ranks[:n_tail] / n_tail
        ks_stat = float(
            max(
                np.max(np.abs(model_cdf - empirical_cdf_low)),
                np.max(np.abs(model_cdf - empirical_cdf_high)),
            )
        )
        return PowerLawFitResult(alpha=float(alpha), xmin=xmin, ks_stat=ks_stat, n_tail=int(n_tail))

    def _candidate_start_indices(tail_sizes: np.ndarray) -> np.ndarray:
        start_indices = n - tail_sizes
        start_indices = start_indices[(start_indices >= 0) & (start_indices < (n - 1))]
        if start_indices.size == 0:
            return np.empty(0, dtype=int)

        start_indices = np.unique(start_indices.astype(int))
        # Keep one index per distinct xmin to avoid repeated evaluations.
        distinct: List[int] = []
        last_xmin: float | None = None
        for idx in np.sort(start_indices):
            xmin = float(x_sorted[idx])
            if last_xmin is None or xmin != last_xmin:
                distinct.append(int(idx))
                last_xmin = xmin
        return np.asarray(distinct, dtype=int)

    # Coarse pass on log-spaced tail sizes (rank space, not raw x space).
    n_coarse = min(num_candidates, max(n - min_tail + 1, 1))
    coarse_tail_sizes = np.rint(np.geomspace(min_tail, n, num=n_coarse)).astype(int)
    coarse_tail_sizes = np.clip(coarse_tail_sizes, min_tail, n)
    coarse_indices = _candidate_start_indices(np.unique(coarse_tail_sizes))
    if coarse_indices.size == 0:
        return None

    best_result: Optional[PowerLawFitResult] = None
    best_start_idx: Optional[int] = None
    for start_idx in coarse_indices:
        fit = _evaluate_start_index(int(start_idx))
        if fit is None:
            continue
        if best_result is None:
            best_result = fit
            best_start_idx = int(start_idx)
            continue
        if fit.ks_stat < best_result.ks_stat:
            best_result = fit
            best_start_idx = int(start_idx)
        elif np.isclose(fit.ks_stat, best_result.ks_stat, rtol=0.0, atol=1e-12) and fit.n_tail > best_result.n_tail:
            best_result = fit
            best_start_idx = int(start_idx)

    if best_result is None or best_start_idx is None:
        return None

    # Local refinement around the best coarse candidate.
    if refine_window > 0:
        refine_lo = max(0, best_start_idx - refine_window)
        refine_hi = min(n - 2, best_start_idx + refine_window)
        refine_tail_sizes = n - np.arange(refine_lo, refine_hi + 1, dtype=int)
        refine_indices = _candidate_start_indices(refine_tail_sizes)
        for start_idx in refine_indices:
            fit = _evaluate_start_index(int(start_idx))
            if fit is None:
                continue
            if fit.ks_stat < best_result.ks_stat:
                best_result = fit
            elif np.isclose(fit.ks_stat, best_result.ks_stat, rtol=0.0, atol=1e-12) and fit.n_tail > best_result.n_tail:
                best_result = fit

    return best_result


def power_law_pdf(x: Iterable[float], alpha: float, xmin: float) -> np.ndarray:
    """
    Summary
    -------
    Evaluate the continuous power-law PDF on x for a fitted (alpha, xmin) pair.

    Parameters
    ----------
    x : Iterable[float]
        Evaluation points.
    alpha : float
        Power-law exponent, expected > 1.
    xmin : float
        Lower cutoff of the power-law tail, expected > 0.

    Returns
    -------
    np.ndarray
        Array with density values; points below xmin are set to 0.

    Notes
    -----
    Uses p(x) = (alpha - 1) / xmin * (x / xmin)^(-alpha) for x >= xmin.

    Examples
    --------
    >>> vals = power_law_pdf([1.0, 2.0, 4.0], alpha=2.5, xmin=1.0)
    >>> vals.shape
    (3,)
    """
    if isinstance(x, np.ndarray):
        x_arr = np.asarray(x, dtype=float)
    else:
        x_arr = np.asarray(list(x), dtype=float)
    density = np.zeros_like(x_arr, dtype=float)
    if alpha <= 1.0 or xmin <= 0.0:
        return density
    mask = np.isfinite(x_arr) & (x_arr >= xmin)
    if np.any(mask):
        density[mask] = ((alpha - 1.0) / xmin) * np.power(x_arr[mask] / xmin, -alpha)
    return density


# ---------------------------------------------------------------------
# Metaorder dictionary statistics (durations, volumes, inter-arrivals)
# ---------------------------------------------------------------------
def list_metaorder_parquet_paths(data_dir: Path) -> List[Path]:
    """
    Summary
    -------
    List per-ISIN parquet trade-tape files in a directory.

    Parameters
    ----------
    data_dir : Path
        Directory expected to contain `*.parquet` files, one per ISIN.

    Returns
    -------
    list[Path]
        Sorted list of parquet paths in `data_dir`.

    Notes
    -----
    - This helper does not recurse into subdirectories.

    Examples
    --------
    >>> paths = list_metaorder_parquet_paths(Path(\"data/parquet\"))
    """
    if not data_dir.exists():
        print(f"[Metaorder stats] Parquet directory not found: {data_dir}")
        return []
    paths: List[Path] = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() != ".parquet":
            continue
        paths.append(path)
    return paths


def load_trades_filtered_for_stats(
    path: Path,
    proprietary: Optional[bool],
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
) -> pd.DataFrame:
    """
    Summary
    -------
    Load a per-ISIN trade tape and apply basic filters used in metaorder statistics.

    Parameters
    ----------
    path : Path
        Parquet path for one ISIN's trade tape.
    proprietary : bool | None
        If True, keep only proprietary aggressive trades; if False, keep only
        non-proprietary aggressive trades; if None, keep all trades.
    trading_hours : tuple[str, str], default=(\"09:30:00\", \"17:30:00\")
        Inclusive trading-hours filter applied on `Trade Time`.

    Returns
    -------
    pd.DataFrame
        Filtered trade tape with a stable ordering and an `ISIN` column.

    Notes
    -----
    - The function adds a `__row_id__` column to ensure stable sorting when multiple
      trades share the same timestamp.

    Examples
    --------
    >>> trades = load_trades_filtered_for_stats(Path(\"data/parquet/ENEL.parquet\"), proprietary=True)
    """
    trades = pd.read_parquet(path)
    if proprietary is True:
        trades = trades[trades["Trade Type Aggressive"] == "Dealing_on_own_account"].copy()
    elif proprietary is False:
        trades = trades[trades["Trade Type Aggressive"] != "Dealing_on_own_account"].copy()
    # proprietary=None -> no filter (full tape)
    start, end = trading_hours
    if trading_hours != None:
        trades = trades[
            (trades["Trade Time"].dt.time >= pd.to_datetime(start).time())
            & (trades["Trade Time"].dt.time <= pd.to_datetime(end).time())
        ].copy()
    trades = trades.reset_index(drop=True)
    trades["__row_id__"] = np.arange(len(trades), dtype=np.int64)
    trades.sort_values(["Trade Time", "__row_id__"], kind="mergesort", inplace=True)
    trades.reset_index(drop=True, inplace=True)
    trades["ISIN"] = path.stem
    return trades


def run_metaorder_dict_statistics(
    metaorders_dict_path: Path,
    parquet_dir: Path,
    img_dir: Path,
    proprietary: bool,
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
) -> None:
    """
    Compute distributional summaries from a metaorder-index dictionary and save figures.

    Parameters
    ----------
    metaorders_dict_path:
        Pickle path to the `metaorders_dict_all_*` object produced by `metaorder_computation.py`.
        Expected structure: `{isin: {member_or_client_id: [list_of_metaorders_as_trade_index_lists]}}`.
    parquet_dir:
        Directory containing per-ISIN `*.parquet` trade tapes (filename stem is treated as ISIN).
    img_dir:
        Output directory for figures.
    proprietary:
        Whether the metaorders dictionary corresponds to proprietary (True) or non-proprietary (False) metaorders.
        This is used only to choose the trade filtering for the per-ISIN tapes.
    trading_hours:
        Inclusive (start, end) trading-hours filter as `("HH:MM:SS", "HH:MM:SS")`.

    Returns
    -------
    None

    Notes
    -----
    - Figures are primarily matplotlib-based for robustness (PNG output without extra deps).
    - For the "metaorders per member" plot, we also write a Plotly HTML companion and
      try to export a Plotly PNG (requires `kaleido`); when that export fails, we
      fall back to a matplotlib line plot while keeping the historical PNG filename.
    """
    if not metaorders_dict_path.exists():
        print(f"[Metaorder stats] Metaorder dictionary not found: {metaorders_dict_path}")
        return

    img_dir.mkdir(parents=True, exist_ok=True)
    html_dir = img_dir / "html"
    png_dir = img_dir / "png"
    html_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    try:
        metaorders_dict_all = pickle.load(metaorders_dict_path.open("rb"))
    except Exception as exc:
        print(f"[Metaorder stats] Failed to load {metaorders_dict_path}: {exc}")
        return

    parquet_paths = list_metaorder_parquet_paths(parquet_dir)
    if not parquet_paths:
        print(f"[Metaorder stats] No parquet files found in {parquet_dir}.")
        return

    print("[Metaorder stats] Building aggregated statistics and density plots...")
    total_metaorders = 0
    counts_by_member: Dict[int, int] = {}
    durations: List[float] = []
    inter_arrivals: List[float] = []
    meta_volumes: List[float] = []
    q_over_v: List[float] = []
    participation_rates: List[float] = []

    def _volume_over_window(ts_ns: np.ndarray, csum_vol: np.ndarray, start_ns: np.int64, end_ns: np.int64) -> float:
        start_idx = np.searchsorted(ts_ns, start_ns, side="left")
        end_idx = np.searchsorted(ts_ns, end_ns, side="right") - 1
        if end_idx < start_idx or end_idx < 0 or start_idx >= ts_ns.size:
            return 0.0
        prev = csum_vol[start_idx - 1] if start_idx > 0 else 0.0
        return float(csum_vol[end_idx] - prev)

    for path in tqdm(parquet_paths, desc="ISINs"):
        trades_full = load_trades_filtered_for_stats(path, proprietary=None, trading_hours=trading_hours)
        trades = load_trades_filtered_for_stats(path, proprietary=proprietary, trading_hours=trading_hours)

        times = trades["Trade Time"].to_numpy()
        q_buy = trades["Total Quantity Buy"].to_numpy()
        q_sell = trades["Total Quantity Sell"].to_numpy()

        times_full_ns = trades_full["Trade Time"].to_numpy("int64")
        vol_full = (
            trades_full["Total Quantity Buy"].to_numpy(dtype=float)
            + trades_full["Total Quantity Sell"].to_numpy(dtype=float)
        )
        csum_vol_full = np.cumsum(vol_full)

        daily_vols = (
            trades_full.groupby(trades_full["Trade Time"].dt.date)[["Total Quantity Buy", "Total Quantity Sell"]]
            .sum()
            .sum(axis=1)
            .to_dict()
        )

        metaorders_dict = metaorders_dict_all.get(path.stem, {})
        total_metaorders += sum(len(v) for v in metaorders_dict.values())

        for member, metas in metaorders_dict.items():
            counts_by_member[member] = counts_by_member.get(member, 0) + len(metas)

        for metas in metaorders_dict.values():
            for meta in metas:
                if not meta:
                    continue
                start_idx, end_idx = meta[0], meta[-1]

                start_time_np = times[start_idx]
                end_time_np = times[end_idx]

                dur_seconds = (end_time_np - start_time_np) / np.timedelta64(1, "s")
                durations.append(float(dur_seconds))

                meta_indices = np.asarray(meta, dtype=np.int64)
                vols = float(q_buy[meta_indices].sum() + q_sell[meta_indices].sum())
                meta_volumes.append(vols)

                start_date = pd.Timestamp(start_time_np).date()
                day_volume = daily_vols.get(start_date, 0.0)
                if day_volume != 0:
                    q_over_v.append(float(vols / day_volume))

                start_ns = np.int64(pd.Timestamp(start_time_np).value)
                end_ns = np.int64(pd.Timestamp(end_time_np).value)
                slice_volume = _volume_over_window(times_full_ns, csum_vol_full, start_ns, end_ns)
                if slice_volume != 0:
                    participation_rates.append(float(vols / slice_volume))

                if len(meta) > 1:
                    meta_times = times[meta_indices]
                    diffs = (meta_times[1:] - meta_times[:-1]) / np.timedelta64(1, "s")
                    inter_arrivals.extend(diffs.tolist())

        del trades
        gc.collect()

    print(f"[Metaorder stats] Total metaorders (all ISINs): {total_metaorders}")
    if counts_by_member:
        # Distribution profile: sort members by metaorder count and plot the profile as a line.
        # This is more readable than a histogram when counts are heavy-tailed and member IDs are many.
        sorted_items = sorted(counts_by_member.items(), key=lambda x: x[1], reverse=True)
        members, counts = zip(*sorted_items)
        counts_arr = np.asarray(counts, dtype=float)
        ranks = np.arange(1, len(counts_arr) + 1, dtype=int)

        counts_df = pd.DataFrame(
            {
                "rank": ranks,
                "n_metaorders": counts_arr,
                "member": [str(m) for m in members],
            }
        )

        fig_profile = px.line(
            counts_df,
            x="rank",
            y="n_metaorders",
            markers=True,
            labels={
                "rank": "Member",
                "n_metaorders": "Number of metaorders",
            },
        )
        fig_profile.update_traces(
            line=dict(color="#5B8FF9", width=2.0),
            marker=dict(color="#5B8FF9", size=6),
            customdata=counts_df[["member"]].to_numpy(),
            hovertemplate="Rank %{x}<br>Member %{customdata[0]}<br>Metaorders %{y}<extra></extra>",
        )
        fig_profile.update_yaxes(type="log")
        fig_profile.update_layout(
            template="plotly_white",
            font={"family": THEME_FONT_FAMILY, "size": TICK_FONT_SIZE, "color": "#1F2937"},
            colorway=THEME_COLORWAY,
            paper_bgcolor=THEME_BG_COLOR,
            plot_bgcolor=THEME_BG_COLOR,
            margin=dict(l=60, r=20, t=60, b=60),
        )

        html_path = html_dir / "metaorders_per_member_all.html"
        fig_profile.write_html(html_path)
        print(f"[Metaorder stats] Saved HTML to {html_path}")

        plotly_png_path = png_dir / "metaorders_per_member_all.png"
        try:
            fig_profile.write_image(plotly_png_path)
            print(f"[Metaorder stats] Saved PNG to {plotly_png_path}")
        except Exception as exc:
            print(f"[Metaorder stats][warn] Could not export Plotly PNG (kaleido missing?): {exc}")
            # Matplotlib fallback for environments without Plotly static export support.
            plt.figure(figsize=(12, 6.5))
            plt.plot(ranks, counts_arr, color="#5B8FF9", linewidth=1.8)
            plt.scatter(ranks, counts_arr, color="#5B8FF9", s=10, alpha=0.7)
            plt.xlabel("Member rank (sorted by # metaorders)")
            plt.ylabel("Number of metaorders")
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(plotly_png_path)
            plt.close()
            print(f"[Metaorder stats] Saved PNG (matplotlib fallback) to {plotly_png_path}")

    def _pdf_plot(
        data: List[float],
        title: str,
        xlabel: str,
        filename: str,
        logy: bool = True,
        logx: bool = False,
        bins: int = 50,
    ):
        if not data:
            tqdm.write(f"Skipping plot {title}: no data")
            return
        fig, ax = plt.subplots(figsize=(10, 5.5))
        plotted = plot_pdf_line(
            ax,
            data,
            bins=bins,
            color="tab:blue",
            label="Empirical density",
            logx=logx,
            logy=logy,
        )
        if not plotted:
            tqdm.write(f"Skipping plot {title}: no valid numeric data")
            plt.close(fig)
            return

        fit_data = pd.to_numeric(pd.Series(data), errors="coerce").to_numpy(dtype=float)
        fit_data = fit_data[np.isfinite(fit_data)]
        if logx:
            fit_data = fit_data[fit_data > 0]

        if not METAORDER_POWERLAW_FIT_ENABLED:
            print(
                f"[Metaorder stats] Power-law fit disabled by config for {title} "
                "(METAORDER_POWERLAW_FIT_ENABLED=False)."
            )
        else:
            print(f"[Metaorder stats] Fitting power law for {title}...")
            fit_result = fit_power_law_clauset_continuous(
                fit_data,
                min_tail=METAORDER_POWERLAW_MIN_TAIL,
                num_candidates=METAORDER_POWERLAW_NUM_CANDIDATES,
                refine_window=METAORDER_POWERLAW_REFINE_WINDOW,
            )
            if fit_result is None:
                print(
                    f"[Metaorder stats] Power-law fit skipped for {title}: "
                    f"not enough valid tail data (min_tail={METAORDER_POWERLAW_MIN_TAIL})."
                )
            else:
                x_max = float(np.nanmax(fit_data)) if fit_data.size else float("nan")
                if np.isfinite(x_max) and x_max > fit_result.xmin:
                    if logx:
                        x_grid = np.logspace(np.log10(fit_result.xmin), np.log10(x_max), 250)
                    else:
                        x_grid = np.linspace(fit_result.xmin, x_max, 250)
                    # Histogram density is unconditional on the whole sample; scale the
                    # fitted tail density by tail mass for an apples-to-apples overlay.
                    tail_mass = fit_result.n_tail / fit_data.size
                    prefactor = tail_mass * (fit_result.alpha - 1.0) / fit_result.xmin
                    y_grid = tail_mass * power_law_pdf(x_grid, fit_result.alpha, fit_result.xmin)
                    valid = np.isfinite(y_grid) & (y_grid > 0)
                    if np.any(valid):
                        ax.plot(
                            x_grid[valid],
                            y_grid[valid],
                            color="tab:orange",
                            linewidth=1.5,
                            alpha=0.7,
                            label=(
                                "Fit: "
                                fr"$f(x)={prefactor:.2g}\frac{{x}}{{{fit_result.xmin:.2g}}}^{{-{fit_result.alpha:.2f}}}$, "
                            ),
                        )
                        ax.axvline(
                            fit_result.xmin,
                            color="tab:orange",
                            linestyle="--",
                            linewidth=1,
                            alpha=0.5,
                        )
                        print(
                            f"[Metaorder stats] Power-law fit for {title}: "
                            f"alpha={fit_result.alpha:.4f}, xmin={fit_result.xmin:.6g}, "
                            f"KS={fit_result.ks_stat:.4f}, n_tail={fit_result.n_tail}"
                        )
                    else:
                        print(
                            f"[Metaorder stats] Power-law fit skipped for {title}: "
                            "fitted density is non-positive on the plotting grid."
                        )
                else:
                    print(
                        f"[Metaorder stats] Power-law fit skipped for {title}: "
                        "invalid support range after filtering."
                    )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        plt.tight_layout()
        fig.savefig(png_dir / filename)
        plt.close(fig)

    _pdf_plot(
        [d / 60 for d in durations],
        "Metaorder duration",
        "Minutes",
        "metaorder_duration_all.png",
        logx=True,
    )
    _pdf_plot([t / 60 for t in inter_arrivals], "Inter-arrival times", "Minutes", "interarrival_all.png", logx=True)
    _pdf_plot(meta_volumes, "Metaorder volumes", "Q", "volumes_all.png", logx=True)
    _pdf_plot([q for q in q_over_v], "Q/V ", "Q/V", "q_over_v_all.png", logx=True)
    _pdf_plot(
        [r for r in participation_rates],
        r"$\eta$",
        r"$\eta$",
        "participation_rate_all.png",
        logx=True,
    )


# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Summary
    -------
    Run metaorder-dictionary distribution statistics as configured in YAML.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - Inputs/outputs are controlled by `config_ymls/metaorder_statistics.yml`.
    - This script is intentionally focused on metaorder-dictionary summaries;
      crowding/imbalance analysis lives in `crowding_analysis.py`.

    Examples
    --------
    >>> # From the repository root:
    >>> # python metaorder_statistics.py
    """
    if not RUN_METAORDER_DICT_STATS:
        print("[Metaorder stats] RUN_METAORDER_DICT_STATS is false; nothing to do.")
        return

    run_metaorder_dict_statistics(
        metaorders_dict_path=METAORDER_STATS_DICT_PATH,
        parquet_dir=PARQUET_DIR,
        img_dir=PLOT_DIR,
        proprietary=METAORDER_STATS_PROPRIETARY,
        trading_hours=TRADING_HOURS,
    )


if __name__ == "__main__":
    main()
