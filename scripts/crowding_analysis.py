#!/usr/bin/env python3
"""
Crowding / imbalance analysis for proprietary vs client metaorders.

What this script does
---------------------
This script compares proprietary and client flow after metaorders have been
constructed by `metaorder_computation.py`. It attaches several imbalance proxies
and computes crowding statistics as correlations between metaorder direction and
those imbalances:

- Within-group (leave-one-out) imbalance on `(ISIN, Date)`.
- Cross-group environment imbalance built from the other group on `(ISIN, Date)`.
- All-vs-all leave-one-out imbalance on the concatenated sample.
- Member-level crowding between proprietary direction and client flow aggregated
  at `(Member, Date)` (optional).
- Crowding conditioned on participation rate (eta), delegated to the
  `crowding_vs_part_rate.py` workflow from the same config.

For each case it reports both global correlations and mean daily correlations,
with Date-cluster bootstrap confidence intervals (resampling trading dates with
replacement). It can also export plots (daily time series, imbalance
distributions, participation diagnostics, ACF of signs).

How to run
----------
1) Edit `config_ymls/crowding_analysis.yml` (paths and toggles).
2) Run:

    python scripts/crowding_analysis.py

Outputs
-------
- Figures: `images/{DATASET_NAME}/prop_vs_nonprop/` (and `html/` when used)
- Participation-conditioned figures/tables: `images/{DATASET_NAME}/crowding_vs_part_rate/`
  and `out_files/{DATASET_NAME}/crowding_vs_part_rate/` by default
- Log file: `crowding_analysis.log`
- The input metaorder parquets may be rewritten to persist newly computed
  imbalance/return columns.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure repository-root imports (e.g., `moimpact`) work when running
# `python scripts/crowding_analysis.py` from the repo root.
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
from moimpact.stats.correlation import (
    corr_with_bootstrap_p as _corr_with_bootstrap_p,
    corr_with_ci as _corr_with_ci,
    corr_with_cluster_bootstrap_ci_and_permutation_p as _corr_with_cluster_bootstrap_ci_and_permutation_p,
)

# ---------------------------------------------------------------------
# Configuration loader (YAML)
# ---------------------------------------------------------------------
_CONFIG_ENV_VAR = "CROWDING_CONFIG"
_config_override = os.environ.get(_CONFIG_ENV_VAR)
if _config_override:
    _CONFIG_PATH = Path(_config_override).expanduser()
    if not _CONFIG_PATH.is_absolute():
        _CONFIG_PATH = (_REPO_ROOT / _CONFIG_PATH).resolve()
else:
    _CONFIG_PATH = _REPO_ROOT / "config_ymls" / "crowding_analysis.yml"
_CFG = load_yaml_mapping(_CONFIG_PATH)


def _cfg_require(key: str):
    return cfg_require(_CFG, key, _CONFIG_PATH)


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_repo_path(_REPO_ROOT, value)


def _format_path_template(template: str, context: Mapping[str, str]) -> str:
    return format_path_template(template, context)


def _resolve_opt_repo_path(value: Optional[str | Path], default: Path) -> Path:
    """Resolve a path, falling back to `default` when the config value is None."""
    return resolve_opt_repo_path(_REPO_ROOT, value, default)


# Sizes tuned for print-friendly plots (loaded from YAML)
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
        Output HTML/PNG paths returned by the shared plotting helper.

    Notes
    -----
    This analysis exports title-less figures to keep panel labeling external.
    """
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)

# ---------------------------------------------------------------------
# Configuration (loaded from YAML)
# ---------------------------------------------------------------------
DATASET_NAME = str(_CFG.get("DATASET_NAME") or "ftsemib")
_path_context = {"DATASET_NAME": DATASET_NAME}

PARQUET_PATH = str(_cfg_require("PARQUET_PATH"))
PARQUET_DIR = _resolve_repo_path(_format_path_template(PARQUET_PATH, _path_context))

OUTPUT_FILE_PATH = str(_cfg_require("OUTPUT_FILE_PATH"))
OUTPUT_DIR = _resolve_repo_path(_format_path_template(OUTPUT_FILE_PATH, _path_context))

IMG_OUTPUT_PATH = str(_cfg_require("IMG_OUTPUT_PATH"))
IMG_BASE_DIR = _resolve_repo_path(_format_path_template(IMG_OUTPUT_PATH, _path_context))

_prop_path_cfg = _CFG.get("PROP_PATH")
_client_path_cfg = _CFG.get("CLIENT_PATH")
PROP_PATH = _resolve_opt_repo_path(
    _format_path_template(str(_prop_path_cfg), _path_context) if _prop_path_cfg is not None else None,
    OUTPUT_DIR / "metaorders_info_sameday_filtered_member_proprietary.parquet",
)
CLIENT_PATH = _resolve_opt_repo_path(
    _format_path_template(str(_client_path_cfg), _path_context) if _client_path_cfg is not None else None,
    OUTPUT_DIR / "metaorders_info_sameday_filtered_member_non_proprietary.parquet",
)

ALPHA = float(_cfg_require("ALPHA"))  # significance level for confidence intervals
BOOTSTRAP_RUNS = int(_cfg_require("BOOTSTRAP_RUNS"))  # number of permutation/bootstraps for p-values
_random_state_cfg = _CFG.get("RANDOM_STATE")
RANDOM_STATE: Optional[int] = None if _random_state_cfg is None else int(_random_state_cfg)
BOOTSTRAP_HEATMAP = bool(_cfg_require("BOOTSTRAP_HEATMAP"))  # toggle permutation test + significance filtering in heatmaps
P_VALUE_THRESHOLD = float(_cfg_require("P_VALUE_THRESHOLD"))  # significance cutoff for filtering plotted correlations
MIN_N = int(_cfg_require("MIN_N"))  # minimum number of metaorders per day to include in the analysis
SMOOTHING_DAYS = int(_cfg_require("SMOOTHING_DAYS"))  # number of days to smooth the correlation
MIN_METAORDERS_PER_MEMBER = int(_cfg_require("MIN_METAORDERS_PER_MEMBER"))
N_MIN_PER_MEMBER_CLIENT = int(_cfg_require("N_MIN_PER_MEMBER_CLIENT"))
MEMBER_WINDOW_DAYS = int(_cfg_require("MEMBER_WINDOW_DAYS"))

# Daily returns / imbalance-return scatter
ATTACH_DAILY_RETURNS = bool(_cfg_require("ATTACH_DAILY_RETURNS"))
PLOT_IMBALANCE_VS_RETURNS = bool(_cfg_require("PLOT_IMBALANCE_VS_RETURNS"))
_returns_dir_cfg = _CFG.get("RETURNS_DATA_DIR")
RETURNS_DATA_DIR = _resolve_opt_repo_path(
    _format_path_template(str(_returns_dir_cfg), _path_context) if _returns_dir_cfg is not None else None,
    PARQUET_DIR,
)
RETURNS_TRADING_HOURS = tuple(_cfg_require("RETURNS_TRADING_HOURS"))
DAILY_RETURN_COL = str(_cfg_require("DAILY_RETURN_COL"))

# Toggles for imbalance-specific analyses
ACF_IMBALANCE = bool(_cfg_require("ACF_IMBALANCE"))
DISTRIBUTIONS_IMBALANCE = bool(_cfg_require("DISTRIBUTIONS_IMBALANCE"))

# Plotting parameters
ACF_MAX_LAG = int(_cfg_require("ACF_MAX_LAG"))
ACF_BOOTSTRAP_SAMPLES = int(_cfg_require("ACF_BOOTSTRAP_SAMPLES"))
IMBALANCE_HIST_BINS = int(_cfg_require("IMBALANCE_HIST_BINS"))
IMBALANCE_USE_KDE = bool(_CFG.get("IMBALANCE_USE_KDE", False))
IMBALANCE_KDE_BW_ADJUST = float(_CFG.get("IMBALANCE_KDE_BW_ADJUST", 1.0))
if IMBALANCE_KDE_BW_ADJUST <= 0:
    raise ValueError("IMBALANCE_KDE_BW_ADJUST must be strictly positive.")
ACF_OUTPUT_DIRNAME = str(_cfg_require("ACF_OUTPUT_DIRNAME"))

# Integrated crowding-vs-participation (eta) analysis.
RUN_CROWDING_VS_PART_RATE = bool(_CFG.get("RUN_CROWDING_VS_PART_RATE", True))
CROWDING_VS_PART_RATE_ANALYSIS_TAG = str(_CFG.get("CROWDING_VS_PART_RATE_ANALYSIS_TAG", "crowding_vs_part_rate"))
CROWDING_VS_PART_RATE_IMBALANCE_KIND = str(_CFG.get("CROWDING_VS_PART_RATE_IMBALANCE_KIND", "local,cross"))
CROWDING_VS_PART_RATE_ETA_BINS = int(_CFG.get("CROWDING_VS_PART_RATE_ETA_BINS", 10))
CROWDING_VS_PART_RATE_ETA_BINNING = str(_CFG.get("CROWDING_VS_PART_RATE_ETA_BINNING", "pooled_quantiles"))
CROWDING_VS_PART_RATE_ETA_MAX = float(_CFG.get("CROWDING_VS_PART_RATE_ETA_MAX", 1.0))
CROWDING_VS_PART_RATE_MIN_N_BIN = int(_CFG.get("CROWDING_VS_PART_RATE_MIN_N_BIN", 100))
CROWDING_VS_PART_RATE_MIN_N_DAY_BIN = int(_CFG.get("CROWDING_VS_PART_RATE_MIN_N_DAY_BIN", 50))
CROWDING_VS_PART_RATE_BOOTSTRAP_RUNS = int(_CFG.get("CROWDING_VS_PART_RATE_BOOTSTRAP_RUNS", BOOTSTRAP_RUNS))
CROWDING_VS_PART_RATE_CLUSTER_CI = str(_CFG.get("CROWDING_VS_PART_RATE_CLUSTER_CI", "date"))
CROWDING_VS_PART_RATE_ALPHA = float(_CFG.get("CROWDING_VS_PART_RATE_ALPHA", ALPHA))
_crowding_vs_part_rate_seed_default = 0 if RANDOM_STATE is None else RANDOM_STATE
CROWDING_VS_PART_RATE_SEED = int(_CFG.get("CROWDING_VS_PART_RATE_SEED", _crowding_vs_part_rate_seed_default))
CROWDING_VS_PART_RATE_PLOTS = str(_CFG.get("CROWDING_VS_PART_RATE_PLOTS", "plotly"))
CROWDING_VS_PART_RATE_RUN_REGRESSIONS = bool(_CFG.get("CROWDING_VS_PART_RATE_RUN_REGRESSIONS", True))
CROWDING_VS_PART_RATE_REG_SAMPLE_FRAC = float(_CFG.get("CROWDING_VS_PART_RATE_REG_SAMPLE_FRAC", 1.0))
CROWDING_VS_PART_RATE_PERMUTATION_RUNS = int(_CFG.get("CROWDING_VS_PART_RATE_PERMUTATION_RUNS", 0))
CROWDING_VS_PART_RATE_RUN_2D = bool(_CFG.get("CROWDING_VS_PART_RATE_RUN_2D", False))
CROWDING_VS_PART_RATE_QV_BINS = int(_CFG.get("CROWDING_VS_PART_RATE_QV_BINS", 10))
CROWDING_VS_PART_RATE_WRITE_PARQUET = bool(_CFG.get("CROWDING_VS_PART_RATE_WRITE_PARQUET", False))

# Output directories
# By default, keep prop-vs-client outputs in a stable dataset-level folder.
PLOT_DIR = IMG_BASE_DIR / "prop_vs_nonprop"
PLOT_OUTPUT_DIRS: PlotOutputDirs = make_plot_output_dirs(PLOT_DIR, use_subdirs=True)
PNG_DIR = PLOT_OUTPUT_DIRS.png_dir


def _png_output_path(path_like: str | Path) -> Path:
    """Return a PNG output path under the canonical `png/` directory."""
    return PNG_DIR / Path(path_like).name


# ---------------------------------------------------------------------
# Correlation with confidence interval
# ---------------------------------------------------------------------
def corr_with_ci(
    x: Iterable[float],
    y: Iterable[float],
    alpha: float = 0.05,
) -> Tuple[float, float, float, int]:
    """
    Summary
    -------
    Compute Pearson correlation and a two-sided confidence interval.

    Parameters
    ----------
    x : Iterable[float]
        1D array-like of observations.
    y : Iterable[float]
        1D array-like of observations (same length as `x`).
    alpha : float, default=0.05
        Significance level for a two-sided `(1 - alpha)` CI.

    Returns
    -------
    r : float
        Pearson correlation coefficient.
    lo : float
        Lower bound of the percentile-bootstrap CI.
    hi : float
        Upper bound of the percentile-bootstrap CI.
    n : int
        Number of valid `(x, y)` pairs used (after dropping NaNs).

    Notes
    -----
    - The CI uses a non-parametric percentile bootstrap over `(x, y)` pairs with
      `BOOTSTRAP_RUNS` replications (loaded from YAML config).
    - If fewer than 4 valid observations are available, the function returns NaNs
      for `(r, lo, hi)` and the corresponding `n`.

    Examples
    --------
    >>> r, lo, hi, n = corr_with_ci([1, 2, 3, 4], [1.0, 2.0, 1.0, 2.0], alpha=0.05)
    """
    return _corr_with_ci(
        x,
        y,
        alpha=alpha,
        n_bootstrap=BOOTSTRAP_RUNS,
        random_state=RANDOM_STATE,
    )


def corr_with_bootstrap_p(
    x: Iterable[float],
    y: Iterable[float],
    n_bootstrap: int = BOOTSTRAP_RUNS,
) -> Tuple[float, float, int]:
    """
    Summary
    -------
    Compute Pearson correlation and a two-sided permutation p-value.

    Parameters
    ----------
    x : Iterable[float]
        1D array-like of observations.
    y : Iterable[float]
        1D array-like of observations (same length as `x`).
    n_bootstrap : int, default=BOOTSTRAP_RUNS
        Number of permutations used to approximate the null distribution.

    Returns
    -------
    r : float
        Pearson correlation coefficient.
    p_value : float
        Two-sided permutation p-value (NaN when `n_bootstrap <= 0` or when the
        correlation is undefined).
    n : int
        Number of valid `(x, y)` pairs used (after dropping NaNs).

    Notes
    -----
    - Despite the name, this is a permutation test (shuffle `y`), not a bootstrap.
    - The p-value is computed as `2 * min(P(r_perm >= r_obs), P(r_perm <= r_obs))`.

    Examples
    --------
    >>> r, p, n = corr_with_bootstrap_p([1, -1, 1], [0.1, -0.2, 0.0], n_bootstrap=100)
    """
    return _corr_with_bootstrap_p(
        x,
        y,
        n_bootstrap=n_bootstrap,
        random_state=RANDOM_STATE,
    )


def corr_with_cluster_bootstrap_ci_and_permutation_p(
    x: Iterable[float],
    y: Iterable[float],
    cluster: Iterable[object],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = BOOTSTRAP_RUNS,
    n_permutations: Optional[int] = None,
    y_const_tol: float = 0.0,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, float, int, int]:
    """
    Compute a cluster-robust correlation CI and p-value via resampling.

    Parameters
    ----------
    x
        1D iterable of numeric values (e.g., metaorder directions in {-1, +1}).
    y
        1D iterable of numeric values (e.g., an imbalance proxy).
    cluster
        1D iterable of cluster labels with the same length as `x` and `y`
        (e.g., trading `Date`). The bootstrap resamples clusters with replacement.
    alpha
        Significance level for the (1 - alpha) percentile confidence interval.
    n_bootstrap
        Number of cluster bootstrap replications used to form the CI.
    n_permutations
        Number of cluster permutations used to approximate the two-sided p-value.
        If None, defaults to `n_bootstrap`.
    y_const_tol
        Tolerance used to check that `y` is (approximately) constant within each
        cluster when computing the permutation p-value. If this condition fails,
        the function returns `p = NaN` because the cluster-label permutation would
        not preserve within-cluster structure.
    random_state
        Optional integer seed for NumPy's RNG to make the resampling reproducible.

    Returns
    -------
    r
        Pearson correlation computed on the original sample.
    lo
        Lower bound of the cluster bootstrap percentile CI (alpha / 2 quantile).
    hi
        Upper bound of the cluster bootstrap percentile CI (1 - alpha / 2 quantile).
    p
        Two-sided permutation p-value computed by permuting cluster-level `y`
        labels across clusters (NaN if `y` varies within clusters beyond tolerance).
    n_obs
        Number of (x, y) pairs used after dropping NaNs/infs.
    n_clusters
        Number of unique clusters used after dropping NaNs/infs.

    Notes
    -----
    - Cluster bootstrap: resample clusters (e.g., trading days) with replacement,
      concatenating all observations within selected clusters; this accounts for
      arbitrary dependence within a cluster.
    - Cluster permutation p-value: under the null of no relationship between `x`
      and `y`, we permute the cluster-level `y` labels across clusters and
      recompute the correlation. This requires `y` to be (approximately) constant
      within each cluster, which holds by construction for member-day environment
      imbalances.

    Examples
    --------
    >>> x = [1, -1, 1, 1, -1]
    >>> y = [0.2, 0.2, -0.1, -0.1, -0.1]
    >>> d = ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"]
    >>> r, lo, hi, p, n_obs, n_clusters = corr_with_cluster_bootstrap_ci_and_permutation_p(x, y, d, n_bootstrap=200, random_state=0)
    """
    seed = RANDOM_STATE if random_state is None else random_state
    return _corr_with_cluster_bootstrap_ci_and_permutation_p(
        x,
        y,
        cluster,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        y_const_tol=y_const_tol,
        random_state=seed,
    )


def _date_cluster_corr_ci(
    x: Iterable[float],
    y: Iterable[float],
    cluster: Iterable[object],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = BOOTSTRAP_RUNS,
) -> Tuple[float, float, float, int, int]:
    """
    Summary
    -------
    Compute Pearson correlation with a Date-cluster bootstrap confidence interval.

    Parameters
    ----------
    x : Iterable[float]
        First variable (typically metaorder direction).
    y : Iterable[float]
        Second variable (typically an imbalance measure).
    cluster : Iterable[object]
        Cluster labels used for resampling (typically trading dates).
    alpha : float, default=0.05
        Significance level for the two-sided percentile interval.
    n_bootstrap : int, default=BOOTSTRAP_RUNS
        Number of cluster-bootstrap replications.

    Returns
    -------
    r : float
        Sample Pearson correlation on valid observations.
    lo : float
        Lower confidence bound.
    hi : float
        Upper confidence bound.
    n_obs : int
        Number of valid row-level observations used.
    n_clusters : int
        Number of valid clusters used.

    Notes
    -----
    - This wrapper disables permutation testing and keeps only the cluster
      bootstrap CI, which is the inference scheme used for crowding-vs-eta.
    - Rows with NaN/inf in `x` or `y`, or missing cluster labels, are excluded.

    Examples
    --------
    >>> r, lo, hi, n_obs, n_days = _date_cluster_corr_ci([1, -1], [0.2, -0.3], ["2024-01-02", "2024-01-03"], n_bootstrap=10)
    >>> isinstance(n_obs, int) and isinstance(n_days, int)
    True
    """
    r, lo, hi, _, n_obs, n_clusters = corr_with_cluster_bootstrap_ci_and_permutation_p(
        x,
        y,
        cluster,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        n_permutations=0,
    )
    return float(r), float(lo), float(hi), int(n_obs), int(n_clusters)


def _cluster_bootstrap_mean_ci(
    values: Iterable[float],
    cluster: Iterable[object],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = BOOTSTRAP_RUNS,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, int, int]:
    """
    Summary
    -------
    Compute a cluster-bootstrap percentile CI for a sample mean.

    Parameters
    ----------
    values : Iterable[float]
        Numeric sample values (e.g., daily correlations).
    cluster : Iterable[object]
        Cluster labels associated with `values` (typically trading dates).
    alpha : float, default=0.05
        Significance level for the two-sided percentile interval.
    n_bootstrap : int, default=BOOTSTRAP_RUNS
        Number of cluster-bootstrap replications.
    random_state : Optional[int], default=None
        Optional seed for reproducibility.

    Returns
    -------
    mean_obs : float
        Sample mean computed on valid observations.
    lo : float
        Lower confidence bound.
    hi : float
        Upper confidence bound.
    n_obs : int
        Number of valid observations used.
    n_clusters : int
        Number of distinct valid clusters used.

    Notes
    -----
    - Resampling is performed at the cluster level with replacement.
    - The bootstrap mean is computed by aggregating sufficient statistics
      (`sum(values)` and `count`) from sampled clusters.
    - If fewer than two clusters are available, CIs are returned as NaN.

    Examples
    --------
    >>> m, lo, hi, n_obs, n_days = _cluster_bootstrap_mean_ci([0.1, 0.2], ["2024-01-02", "2024-01-03"], n_bootstrap=10, random_state=0)
    >>> n_obs == 2 and n_days == 2
    True
    """
    values_arr = np.asarray(values, dtype=float)
    cluster_arr = np.asarray(list(cluster), dtype=object)
    if values_arr.shape != cluster_arr.shape:
        raise ValueError("values and cluster must have the same length.")

    mask = np.isfinite(values_arr) & (~pd.isna(cluster_arr))
    values_arr = values_arr[mask]
    cluster_arr = cluster_arr[mask]

    n_obs = int(values_arr.size)
    if n_obs < 1:
        return float("nan"), float("nan"), float("nan"), n_obs, 0

    cluster_codes, _ = pd.factorize(cluster_arr, sort=False)
    n_clusters = int(cluster_codes.max() + 1) if cluster_codes.size else 0
    if n_clusters < 1:
        return float("nan"), float("nan"), float("nan"), n_obs, 0

    tmp = pd.DataFrame({"cluster": cluster_codes, "value": values_arr})
    agg = (
        tmp.groupby("cluster", sort=False, dropna=False)
        .agg(
            n=("value", "size"),
            sum_value=("value", "sum"),
        )
        .reset_index(drop=True)
    )
    n_k = agg["n"].to_numpy(dtype=float)
    sum_value_k = agg["sum_value"].to_numpy(dtype=float)

    n_total = float(n_k.sum())
    sum_total = float(sum_value_k.sum())
    mean_obs = float(sum_total / n_total) if n_total > 0 else float("nan")

    lo = hi = float("nan")
    if n_bootstrap > 0 and np.isfinite(mean_obs) and n_clusters >= 2:
        rng = np.random.default_rng(random_state)
        boot_means = np.empty(int(n_bootstrap), dtype=float)
        for b in range(int(n_bootstrap)):
            sampled = rng.integers(0, n_clusters, size=n_clusters)
            freq = np.bincount(sampled, minlength=n_clusters).astype(float)
            n_b = float(np.sum(freq * n_k))
            sum_b = float(np.sum(freq * sum_value_k))
            boot_means[b] = sum_b / n_b if n_b > 0.0 else np.nan

        valid = boot_means[np.isfinite(boot_means)]
        if valid.size > 0:
            lo = float(np.quantile(valid, alpha / 2.0))
            hi = float(np.quantile(valid, 1.0 - alpha / 2.0))

    return float(mean_obs), float(lo), float(hi), n_obs, n_clusters


def extract_date(period_list):
    """
    Summary
    -------
    Extract and normalize the first timestamp from a list-like container.

    Parameters
    ----------
    period_list : Sequence[object]
        A sequence whose first element can be parsed as a datetime-like value.

    Returns
    -------
    pd.Timestamp | None
        The normalized date (midnight) as a pandas Timestamp, or None when the
        input is empty or unparseable.

    Notes
    -----
    - This helper is mainly used when converting legacy inputs where a `Period`
      column stores lists of timestamps.
    - Returning a pandas Timestamp avoids object-dtype `datetime.date` columns,
      which some parquet engines cannot serialize reliably.

    Examples
    --------
    >>> extract_date(["2024-01-02 10:00:00"])
    Timestamp('2024-01-02 00:00:00')
    """
    if len(period_list) > 0:
        # Keep as pandas Timestamp to avoid object-dtype `datetime.date` columns,
        # which some parquet engines (e.g. fastparquet) can't serialize reliably.
        return pd.to_datetime(period_list[0]).normalize()
    return None


# ---------------------------------------------------------------------
# Daily returns helpers
# ---------------------------------------------------------------------
def list_isin_parquet_paths(data_dir: Path) -> List[Path]:
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
        Sorted list of parquet paths. The synthetic file `ALTRI_FTSEMIB.parquet`
        (if present) is excluded.

    Notes
    -----
    - This helper does not recurse into subdirectories.

    Examples
    --------
    >>> paths = list_isin_parquet_paths(Path(\"data/parquet\"))
    """
    if not data_dir.exists():
        print(f"[Daily returns] Data directory not found: {data_dir}")
        return []
    paths: List[Path] = []
    for path in data_dir.iterdir():
        name = path.name
        if not name.endswith(".parquet"):
            continue
        if path.stem.upper() == "ALTRI_FTSEMIB":
            continue
        paths.append(path)
    return sorted(paths)


def compute_daily_returns_for_path(parquet_path: Path, trading_hours: Tuple[str, str]) -> pd.Series:
    """
    Summary
    -------
    Compute close-to-close daily log returns from a per-ISIN trade tape.

    Parameters
    ----------
    parquet_path : Path
        Path to a per-ISIN trade-tape parquet containing at least
        `Trade Time` and `Price Last Contract`.
    trading_hours : tuple[str, str]
        Inclusive trading-hours filter as `(\"HH:MM:SS\", \"HH:MM:SS\")`.

    Returns
    -------
    pd.Series
        Series indexed by date with daily log returns. Returns may contain NaN
        if prices are non-positive or insufficient data are available.

    Notes
    -----
    - The "close" is defined as the last trade price within `trading_hours`.
    - If fewer than 2 daily closes exist, returns an empty Series.

    Examples
    --------
    >>> r = compute_daily_returns_for_path(Path(\"data/parquet/ENEL.parquet\"), (\"09:30:00\", \"17:30:00\"))
    """
    trades = pd.read_parquet(parquet_path, columns=["Trade Time", "Price Last Contract"])
    if trades.empty:
        return pd.Series(dtype=float)

    trades = trades.dropna(subset=["Trade Time", "Price Last Contract"]).copy()
    trades["Trade Time"] = pd.to_datetime(trades["Trade Time"], errors="coerce")
    trades = trades.dropna(subset=["Trade Time"])
    start, end = trading_hours
    mask_hours = trades["Trade Time"].dt.time.between(pd.to_datetime(start).time(), pd.to_datetime(end).time())
    trades = trades.loc[mask_hours]
    trades = trades.sort_values("Trade Time", kind="mergesort")
    trades = trades[trades["Price Last Contract"] > 0]
    if trades.empty:
        return pd.Series(dtype=float)

    daily_close = trades.groupby(trades["Trade Time"].dt.date)["Price Last Contract"].last()
    daily_close = daily_close.sort_index()
    if daily_close.size < 2:
        return pd.Series(dtype=float)

    returns = np.log(daily_close).diff()
    return returns.replace([np.inf, -np.inf], np.nan)


def build_daily_returns_lookup(
    data_dir: Path,
    isins: Sequence[str],
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
) -> Dict[Tuple[str, dt.date], float]:
    """
    Summary
    -------
    Build a `(ISIN, Date) -> daily log return` lookup from per-ISIN trade tapes.

    Parameters
    ----------
    data_dir : Path
        Directory containing per-ISIN parquet files.
    isins : Sequence[str]
        List of ISINs to process. Files whose stem is not in this list are skipped.
    trading_hours : tuple[str, str], default=(\"09:30:00\", \"17:30:00\")
        Inclusive trading-hours filter used when computing daily closes.

    Returns
    -------
    dict[tuple[str, datetime.date], float]
        Mapping from `(ISIN, Date)` to the daily log return for that ISIN/date.

    Notes
    -----
    - This function can be slow on large universes; it iterates over each ISIN
      file and computes daily returns.
    - Missing days are simply absent from the dictionary.

    Examples
    --------
    >>> lookup = build_daily_returns_lookup(Path(\"data/parquet\"), [\"ENEL\", \"ENI\"])
    >>> lookup.get((\"ENEL\", dt.date(2024, 1, 2)))
    """
    isin_set = {str(isin) for isin in isins if pd.notna(isin)}
    if not isin_set:
        return {}

    paths = list_isin_parquet_paths(data_dir)
    paths = [p for p in paths if p.stem in isin_set]
    lookup: Dict[Tuple[str, dt.date], float] = {}
    for path in tqdm(paths, desc="Computing daily returns per ISIN"):
        returns = compute_daily_returns_for_path(path, trading_hours)
        if returns.empty:
            continue
        isin = path.stem
        for date_key, ret in returns.items():
            try:
                date_obj = pd.to_datetime(date_key).date()
            except Exception:
                continue
            lookup[(isin, date_obj)] = float(ret) if pd.notnull(ret) else np.nan
    return lookup


def attach_daily_returns_column(
    metaorders: pd.DataFrame,
    daily_returns: Dict[Tuple[str, dt.date], float],
    new_col: str,
) -> Tuple[pd.DataFrame, bool]:
    """
    Summary
    -------
    Attach a daily-return column to a metaorder table via `(ISIN, Date)` keys.

    Parameters
    ----------
    metaorders : pd.DataFrame
        Metaorder table containing `ISIN` and `Date` columns.
    daily_returns : dict[tuple[str, datetime.date], float]
        Lookup produced by `build_daily_returns_lookup`.
    new_col : str
        Name of the column to create/overwrite in the output DataFrame.

    Returns
    -------
    out : pd.DataFrame
        Copy of `metaorders` with the new column attached.
    changed : bool
        True if the resulting column differs from an existing `new_col` (or if
        it did not exist previously), else False.

    Notes
    -----
    - If `ISIN` or `Date` is missing from `metaorders`, the function creates
      `new_col` filled with NaNs and reports `changed` accordingly.
    - `Date` values are normalized to `datetime.date` when forming dictionary keys.

    Examples
    --------
    >>> out, changed = attach_daily_returns_column(df, lookup, new_col=\"Daily Return\")
    """
    out = metaorders.copy()

    if ("ISIN" not in out.columns) or ("Date" not in out.columns):
        placeholder = pd.Series(np.nan, index=out.index, dtype=float)
        existing = out[new_col] if new_col in out.columns else None
        changed = existing is None or not placeholder.equals(existing)
        out[new_col] = placeholder
        return out, changed

    def _to_date(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        # datetime is a subclass of date -> check datetime first.
        if isinstance(val, dt.datetime):
            return val.date()
        if isinstance(val, dt.date):
            return val
        try:
            return pd.to_datetime(val).date()
        except Exception:
            return None

    values = []
    for isin, date_val in zip(out.get("ISIN", []), out.get("Date", [])):
        key = (str(isin), _to_date(date_val))
        values.append(daily_returns.get(key, np.nan))

    new_series = pd.Series(values, index=out.index, dtype=float)
    existing = out[new_col] if new_col in out.columns else None
    changed = existing is None or not new_series.equals(existing)
    out[new_col] = new_series
    return out, changed


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
# Plotting is centralized on Plotly + shared save helpers (`moimpact.plotting`).


# Local imbalance per (ISIN, Date)
# ---------------------------------------------------------------------
def add_daily_imbalance(
    df: pd.DataFrame,
    group_cols=("ISIN", "Date"),
    side_col: str = "Direction",
    vol_col: str = "Q",
    new_col: str = "imbalance_local",
) -> pd.DataFrame:
    """
    Summary
    -------
    Compute leave-one-out signed-volume imbalance within each `(group_cols)` cell.

    Parameters
    ----------
    df : pd.DataFrame
        Metaorder table.
    group_cols : tuple[str, ...], default=(\"ISIN\", \"Date\")
        Grouping columns (typically instrument and trading day).
    side_col : str, default=\"Direction\"
        Column with metaorder sign/direction (typically in {-1, +1}).
    vol_col : str, default=\"Q\"
        Column with metaorder volume/size (non-negative).
    new_col : str, default=\"imbalance_local\"
        Name of the column to create in the output.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with `new_col` added.

    Notes
    -----
    For each row `i` in a group `g`, the imbalance is:

    `imb_i = (Σ_{j∈g, j≠i} Q_j D_j) / (Σ_{j∈g, j≠i} Q_j)`

    If a group contains only one metaorder, the denominator is zero and the
    imbalance is set to NaN for that row.

    Examples
    --------
    >>> demo = pd.DataFrame(
    ...     {
    ...         "ISIN": ["A", "A"],
    ...         "Date": ["2024-01-01", "2024-01-01"],
    ...         "Q": [1.0, 2.0],
    ...         "Direction": [1, -1],
    ...     }
    ... )
    >>> out = add_daily_imbalance(demo)
    >>> "imbalance_local" in out.columns
    True
    """

    df = df.copy()
    group_cols = list(group_cols)
    grouped = df.groupby(group_cols, group_keys=False, dropna=False, sort=False)

    # Signed volume of each metaorder; temporary helper column
    df["__QD__"] = df[vol_col].to_numpy(dtype=float) * df[side_col].to_numpy(dtype=float)
    total_Q = grouped[vol_col].transform("sum")
    total_QD = grouped["__QD__"].transform("sum")

    denom = total_Q - df[vol_col]
    numer = total_QD - df["__QD__"]

    df[new_col] = np.where(denom > 0, numer / denom, np.nan)

    return df.drop(columns="__QD__")


# ---------------------------------------------------------------------
# Print information, correlations, and intuition
# ---------------------------------------------------------------------
def analyze_flow(
    df: pd.DataFrame,
    label: str,
    imb_col: str = "imbalance_local",
    side_col: str = "Direction",
    vol_col: str = "Q",
    date_col: str = "Date",
    alpha: float = 0.05,
) -> None:
    """
    Summary
    -------
    Print descriptive statistics and within-group crowding diagnostics for one flow sample.

    Parameters
    ----------
    df : pd.DataFrame
        Metaorder table containing at least direction, imbalance, and date columns.
    label : str
        Human-readable dataset label used in printed summaries.
    imb_col : str, default="imbalance_local"
        Column with the imbalance proxy used in correlation diagnostics.
    side_col : str, default="Direction"
        Column with trade direction values (typically in {-1, +1}).
    vol_col : str, default="Q"
        Column with metaorder volume used for the global-imbalance sanity check.
    date_col : str, default="Date"
        Date column used as the cluster unit for bootstrap confidence intervals.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    None
        This function prints diagnostics to stdout/log and does not return values.

    Notes
    -----
    - The main correlation CI is computed with a Date-cluster bootstrap when `date_col`
      is available; otherwise it falls back to the row-level bootstrap helper.
    - The final "global imbalance" block is a mechanical sanity check based on
      leave-one-out totals over the whole sample.

    Examples
    --------
    >>> demo = pd.DataFrame({"ISIN":["A","A"], "Date":["2024-01-01","2024-01-01"], "Direction":[1,-1], "Q":[1.0,2.0], "imbalance_local":[0.2,-0.2]})
    >>> analyze_flow(demo, "demo sample")
    """
    print("\n" + "=" * 80)
    print(f"Analysis for: {label}")
    print("=" * 80)
    print(f"Total metaorders        : {len(df):,}")
    print(f"Unique ISINs            : {df['ISIN'].nunique():,}")
    print(f"Unique trading dates    : {df['Date'].nunique():,}")

    if "Q/V" in df.columns:
        qv_min = df["Q/V"].min()
        qv_max = df["Q/V"].max()
        print(f"Q/V range               : {qv_min:.2e} – {qv_max:.2e}")

    if "Participation Rate" in df.columns:
        pr_mean = df["Participation Rate"].mean()
        pr_std = df["Participation Rate"].std()
        print(f"Participation rate mean±std : {pr_mean:.3f} ± {pr_std:.3f}")

    # Quality of local imbalance
    na_share = df[imb_col].isna().mean()
    print(f"\nShare of NaN {imb_col}: {na_share:.2%}")
    if na_share > 0:
        print("  -> NaNs come from (ISIN,Date) with a single metaorder (no 'others').")

    # Correlation between sign and *local* imbalance
    x = df[side_col].astype(float).to_numpy()
    y = df[imb_col].astype(float).to_numpy()
    if date_col in df.columns:
        r_loc, lo_loc, hi_loc, n_loc, n_days_loc = _date_cluster_corr_ci(
            x,
            y,
            df[date_col],
            alpha=alpha,
            n_bootstrap=BOOTSTRAP_RUNS,
        )
        ci_label = "Date-cluster bootstrap"
    else:
        r_loc, lo_loc, hi_loc, n_loc = corr_with_ci(x, y, alpha=alpha)
        n_days_loc = 0
        ci_label = "row bootstrap"

    if math.isnan(r_loc):
        print("\nCorr(Direction, local imbalance): not enough data (n < 3).")
    else:
        if n_days_loc > 0:
            print(
                f"\nCorr({side_col}, {imb_col}) = {r_loc:.3f} "
                f"(95% {ci_label} CI [{lo_loc:.3f}, {hi_loc:.3f}], n={n_loc}, days={n_days_loc})"
            )
        else:
            print(
                f"\nCorr({side_col}, {imb_col}) = {r_loc:.3f} "
                f"(95% {ci_label} CI [{lo_loc:.3f}, {hi_loc:.3f}], n={n_loc})"
            )

    # Conditional means of local imbalance
    for side, name in [(1, "buys"), (-1, "sells")]:
        mask = df[side_col] == side
        if mask.any():
            m = df.loc[mask, imb_col].mean()
            print(f"Mean {imb_col} | {name:5s}: {m:+.3f}")
        else:
            print(f"No {name} in this sample.")

    # Intuition from local correlation
    if not math.isnan(r_loc):
        if r_loc > 0.05:
            msg = (
                "trades tend to go WITH the *daily* metaorder imbalance "
                "(crowded / herding)."
            )
        elif r_loc < -0.05:
            msg = (
                "trades tend to go AGAINST the *daily* metaorder imbalance "
                "(contrarian / liquidity-providing)."
            )
        else:
            msg = (
                "trade direction is roughly UNCORRELATED with the *daily* imbalance "
                "(idiosyncratic flow)."
            )
        print(f"\nIntuition (local): For {label}, {msg}")

    # Optional: reproduce *global* imbalance and its (mechanically negative) correlation
    Q = df[vol_col].to_numpy(dtype=float)
    D = df[side_col].to_numpy(dtype=float)
    total_Q = Q.sum()
    total_QD = (Q * D).sum()
    denom = total_Q - Q
    global_imb = np.where(denom > 0, (total_QD - Q * D) / denom, np.nan)

    mask = denom > 0
    if mask.any():
        r_glob, lo_glob, hi_glob, n_glob = corr_with_ci(D[mask], global_imb[mask], alpha=alpha)
        print(
            f"\nSanity check: Corr(Direction, *global* imbalance) = {r_glob:.3f} "
            f"(95% CI [{lo_glob:.3f}, {hi_glob:.3f}], n={n_glob})"
        )
        print(
            "  -> This correlation is expected to be mechanically NEGATIVE even for\n"
            "     random signs, because each metaorder is subtracted from the total\n"
            "     when computing its own 'others' imbalance."
        )


# ---------------------------------------------------------------------
# Daily correlation time series
# ---------------------------------------------------------------------
def daily_crowding_ts(
    df: pd.DataFrame,
    side_col: str = "Direction",
    imb_col: str = "imbalance_local",
    date_col: str = "Date",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Summary
    -------
    Compute a daily time series of Corr(Direction, imbalance) with pointwise CIs.

    Parameters
    ----------
    df : pd.DataFrame
        Metaorder table containing at least `date_col`, `side_col`, and `imb_col`.
    side_col : str, default=\"Direction\"
        Direction column.
    imb_col : str, default=\"imbalance_local\"
        Imbalance proxy column.
    date_col : str, default=\"Date\"
        Trading-date column.
    alpha : float, default=0.05
        Significance level for pointwise (per-day) bootstrap CIs.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: `Date`, `r`, `lo`, `hi`, `p`, `n`.

    Notes
    -----
    - Per-day CIs are computed by `corr_with_ci` (row bootstrap within that day).
    - The `p` column is a simple within-day permutation p-value.

    Examples
    --------
    >>> ts = daily_crowding_ts(demo_df)
    >>> set(["Date", "r", "lo", "hi", "n"]).issubset(ts.columns)
    True
    """
    rows = []
    for d, g in df.groupby(date_col, sort=True):
        x = g[side_col].to_numpy(dtype=float)
        y = g[imb_col].to_numpy(dtype=float)
        r, lo, hi, n = corr_with_ci(x, y, alpha=alpha)
        r_b, p_val, _ = corr_with_bootstrap_p(x, y, n_bootstrap=BOOTSTRAP_RUNS)
        rows.append({"Date": d, "r": r, "lo": lo, "hi": hi, "p": p_val, "n": n})

    out = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return out


def compute_environment_imbalance(
    source_df: pd.DataFrame,
    group_cols=("ISIN", "Date"),
    side_col: str = "Direction",
    vol_col: str = "Q",
    new_col: str = "imbalance_env",
) -> pd.DataFrame:
    """
    Summary
    -------
    Compute signed-volume imbalance from an "environment" sample.

    Parameters
    ----------
    source_df : pd.DataFrame
        Metaorders used to define the environment (e.g., client flow when
        studying proprietary direction).
    group_cols : tuple[str, ...], default=(\"ISIN\", \"Date\")
        Grouping columns over which the environment imbalance is computed.
    side_col : str, default=\"Direction\"
        Direction column in `source_df`.
    vol_col : str, default=\"Q\"
        Volume column in `source_df`.
    new_col : str, default=\"imbalance_env\"
        Name of the imbalance column in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        A table with `group_cols` and `new_col`, where:
        `new_col = Σ(QD) / Σ(Q)` within each group.

    Notes
    -----
    - Unlike `add_daily_imbalance`, this is *not* leave-one-out; it computes a
      single environment imbalance value per `(group_cols)` cell.

    Examples
    --------
    >>> env = compute_environment_imbalance(client_df, group_cols=("ISIN", "Date"), new_col="imb_client")
    >>> "imb_client" in env.columns
    True
    """
    tmp = source_df.copy()
    tmp["__QD__"] = tmp[vol_col].to_numpy(dtype=float) * tmp[side_col].to_numpy(dtype=float)
    group_cols = list(group_cols)
    agg = (
        tmp.groupby(group_cols, dropna=False, sort=False)
        .agg(total_Q=(vol_col, "sum"), total_QD=("__QD__", "sum"))
        .reset_index()
    )
    agg[new_col] = np.where(agg["total_Q"] > 0, agg["total_QD"] / agg["total_Q"], np.nan)
    return agg[group_cols + [new_col]]


def attach_environment_imbalance(
    target_df: pd.DataFrame,
    environment_df: pd.DataFrame,
    new_col: str,
    group_cols=("ISIN", "Date"),
    side_col: str = "Direction",
    vol_col: str = "Q",
) -> pd.DataFrame:
    """
    Summary
    -------
    Attach an environment imbalance column to each row of a target metaorder table.

    Parameters
    ----------
    target_df : pd.DataFrame
        Metaorders for which the environment imbalance is attached.
    environment_df : pd.DataFrame
        Metaorders used to compute the environment imbalance.
    new_col : str
        Name of the attached imbalance column.
    group_cols : tuple[str, ...], default=(\"ISIN\", \"Date\")
        Join keys.
    side_col : str, default=\"Direction\"
        Direction column in `environment_df`.
    vol_col : str, default=\"Q\"
        Volume column in `environment_df`.

    Returns
    -------
    pd.DataFrame
        Copy of `target_df` with `new_col` merged on `group_cols`.

    Notes
    -----
    - Missing environment groups lead to NaN values in `new_col`.

    Examples
    --------
    >>> out = attach_environment_imbalance(prop_df, client_df, new_col=\"imb_client_env\")
    """
    env_imb = compute_environment_imbalance(
        environment_df,
        group_cols=group_cols,
        side_col=side_col,
        vol_col=vol_col,
        new_col=new_col,
    )
    target_clean = target_df.drop(columns=[new_col], errors="ignore")
    return target_clean.merge(env_imb, on=list(group_cols), how="left", sort=False)


def attach_member_client_imbalance(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    new_col: str = "imbalance_client_member_env",
    member_col: str = "Member",
    date_col: str = "Date",
    side_col: str = "Direction",
    vol_col: str = "Q",
) -> pd.DataFrame:
    """
    Attach client-flow imbalance aggregated at the (Member, Date) level
    to each proprietary metaorder.

    For each (Member, Date), we compute the signed-volume imbalance using only
    client metaorders (across all ISINs for that member/day), and merge it onto
    the proprietary dataframe keyed by (Member, Date).

    Parameters
    ----------
    metaorders_proprietary : pd.DataFrame
        Proprietary metaorders containing `member_col` and `date_col`.
    metaorders_non_proprietary : pd.DataFrame
        Client metaorders used to build the member-day environment imbalance.
    new_col : str, default=\"imbalance_client_member_env\"
        Name of the attached environment imbalance column.
    member_col : str, default=\"Member\"
        Member identifier column used as part of the environment key.
    date_col : str, default=\"Date\"
        Trading-date column used as part of the environment key.
    side_col : str, default=\"Direction\"
        Direction column in the client metaorders table.
    vol_col : str, default=\"Q\"
        Volume column in the client metaorders table.

    Returns
    -------
    pd.DataFrame
        Copy of `metaorders_proprietary` with `new_col` attached.

    Notes
    -----
    - This is a member-level environment: it aggregates across all ISINs for a
      member on a given day.

    Examples
    --------
    >>> out = attach_member_client_imbalance(prop_df, client_df)
    """
    env_imb = compute_environment_imbalance(
        metaorders_non_proprietary,
        group_cols=(member_col, date_col),
        side_col=side_col,
        vol_col=vol_col,
        new_col=new_col,
    )
    target_clean = metaorders_proprietary.drop(columns=[new_col], errors="ignore")
    return target_clean.merge(env_imb, on=[member_col, date_col], how="left", sort=False)


def plot_daily_crowding(
    daily_prop: pd.DataFrame,
    daily_client: pd.DataFrame,
    out_prefix: str = "daily_crowding",
    smoothing_days: int = 20,
    label_prop: str = "Prop: r(Direction, imbalance)",
    label_client: str = "Client: r(Direction, imbalance)",
    title: str = "Daily crowding: Corr(Direction, daily imbalance_local)",
    smoothed_title: str | None = None,
) -> None:
    """
    Summary
    -------
    Save daily and smoothed crowding time-series plots for two groups.

    Parameters
    ----------
    daily_prop : pd.DataFrame
        Per-day correlation table for proprietary flow (output of `daily_crowding_ts`).
    daily_client : pd.DataFrame
        Per-day correlation table for client flow.
    out_prefix : str, default=\"daily_crowding\"
        Filename prefix used for outputs.
    smoothing_days : int, default=20
        Window length for the rolling-mean plot.
    label_prop : str, default=\"Prop: r(Direction, imbalance)\"
        Legend label for proprietary series.
    label_client : str, default=\"Client: r(Direction, imbalance)\"
        Legend label for client series.
    title : str, default=\"Daily crowding: Corr(Direction, daily imbalance_local)\"
        Title for the raw daily correlation plot.
    smoothed_title : str | None, default=None
        Title for the rolling-mean plot. If None, a default is constructed.

    Returns
    -------
    None
        Writes PNG figures to disk.

    Notes
    -----
    - Produces two PNGs:
      1) `{out_prefix}_daily_corr.png` with pointwise CIs.
      2) `{out_prefix}_rolling_{N}d.png` with rolling means.

    Examples
    --------
    >>> plot_daily_crowding(ts_prop, ts_client, out_prefix=\"images/daily_crowding\", smoothing_days=10)
    """
    if smoothing_days < 1:
        raise ValueError("smoothing_days must be >= 1")

    out_prefix_name = Path(out_prefix).name
    # 1) Raw daily correlations with CI bands
    fig_daily = go.Figure()
    x_prop = pd.to_datetime(daily_prop["Date"])
    x_client = pd.to_datetime(daily_client["Date"])

    fig_daily.add_trace(go.Scatter(x=x_prop, y=daily_prop["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_daily.add_trace(
        go.Scatter(
            x=x_prop,
            y=daily_prop["lo"],
            mode="lines",
            fill="tonexty",
            fillcolor=COLOR_BAND_PROPRIETARY,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_daily.add_trace(
        go.Scatter(
            x=x_prop,
            y=daily_prop["r"],
            mode="lines",
            name=label_prop,
            line=dict(color=COLOR_PROPRIETARY, width=2),
        )
    )

    fig_daily.add_trace(go.Scatter(x=x_client, y=daily_client["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_daily.add_trace(
        go.Scatter(
            x=x_client,
            y=daily_client["lo"],
            mode="lines",
            fill="tonexty",
            fillcolor=COLOR_BAND_CLIENT,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_daily.add_trace(
        go.Scatter(
            x=x_client,
            y=daily_client["r"],
            mode="lines",
            name=label_client,
            line=dict(color=COLOR_CLIENT, width=2, dash="dash"),
        )
    )
    fig_daily.add_hline(y=0.0, line=dict(color=COLOR_NEUTRAL, width=1, dash="dot"))
    fig_daily.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Daily correlation",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _, out_path1 = save_plotly_figure(
        fig_daily,
        stem=f"{out_prefix_name}_daily_corr",
        dirs=PLOT_OUTPUT_DIRS,
        write_html=True,
        write_png=True,
        strict_png=False,
    )
    print(f"\n[Daily plots] Saved daily correlation plot to: {out_path1}")

    # 2) Rolling mean smoothing of the daily correlations
    daily_prop = daily_prop.copy()
    daily_client = daily_client.copy()
    # Allow rolling averages to appear once we have at least some data points.
    min_periods = min(5, smoothing_days)
    daily_prop["r_roll"] = daily_prop["r"].rolling(
        window=smoothing_days,
        min_periods=min_periods,
    ).mean()
    daily_client["r_roll"] = daily_client["r"].rolling(
        window=smoothing_days,
        min_periods=min_periods,
    ).mean()

    fig_roll = go.Figure()
    fig_roll.add_trace(
        go.Scatter(
            x=x_prop,
            y=daily_prop["r_roll"],
            mode="lines",
            name=f"Prop {smoothing_days}-day mean",
            line=dict(color=COLOR_PROPRIETARY, width=2),
        )
    )
    fig_roll.add_trace(
        go.Scatter(
            x=x_client,
            y=daily_client["r_roll"],
            mode="lines",
            name=f"Client {smoothing_days}-day mean",
            line=dict(color=COLOR_CLIENT, width=2),
        )
    )
    fig_roll.add_hline(y=0.0, line=dict(color=COLOR_NEUTRAL, width=1, dash="dot"))
    fig_roll.update_layout(
        title=smoothed_title or f"{smoothing_days}-day rolling crowding correlation",
        xaxis_title="Date",
        yaxis_title=f"{smoothing_days}-day rolling correlation",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _, out_path2 = save_plotly_figure(
        fig_roll,
        stem=f"{out_prefix_name}_rolling_{smoothing_days}d",
        dirs=PLOT_OUTPUT_DIRS,
        write_html=True,
        write_png=True,
        strict_png=False,
    )
    print(f"[Daily plots] Saved rolling correlation plot to: {out_path2}")


def run_daily_crowding_analysis(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    alpha: float = 0.05,
    min_n: int = 100,
    out_prefix: str = "daily_crowding",
    make_plots: bool = True,
    smoothing_days: int = 20,
) -> None:
    """
    Summary
    -------
    Compute within-group daily crowding series and report filtered mean-daily statistics.

    Parameters
    ----------
    metaorders_proprietary : pd.DataFrame
        Proprietary metaorders containing `Direction`, `Date`, and `imbalance_local`.
    metaorders_non_proprietary : pd.DataFrame
        Client metaorders containing `Direction`, `Date`, and `imbalance_local`.
    alpha : float, default=0.05
        Significance level used for confidence intervals.
    min_n : int, default=100
        Minimum per-day sample size required for inclusion in filtered summaries/plots.
    out_prefix : str, default="daily_crowding"
        Filename prefix for exported plot artifacts.
    make_plots : bool, default=True
        If True, save daily and smoothed correlation plots.
    smoothing_days : int, default=20
        Window length used for the rolling-mean plot.

    Returns
    -------
    None
        Prints summary statistics and optionally saves figures.

    Notes
    -----
    - Mean daily correlations are computed on the filtered day set (`n >= min_n`)
      and use a Date-cluster bootstrap percentile CI.
    - Daily pointwise CIs in the plotted time series are computed by `daily_crowding_ts`.

    Examples
    --------
    >>> run_daily_crowding_analysis(prop_df, client_df, min_n=50, make_plots=False)
    """
    print("\n" + "=" * 80)
    print("Daily crowding analysis (per trading day)")
    print("=" * 80)

    daily_prop = daily_crowding_ts(metaorders_proprietary, alpha=alpha)
    daily_client = daily_crowding_ts(metaorders_non_proprietary, alpha=alpha)

    # Filter by minimum sample size per day
    daily_prop_f = daily_prop[daily_prop["n"] >= min_n].reset_index(drop=True)
    daily_client_f = daily_client[daily_client["n"] >= min_n].reset_index(drop=True)

    print(f"\nProprietary days with n >= {min_n}: {len(daily_prop_f)} "
          f"(out of {len(daily_prop)})")
    print(f"Client days with n >= {min_n}: {len(daily_client_f)} "
          f"(out of {len(daily_client)})")

    if not daily_prop_f.empty:
        mean_prop_f, lo_prop_f, hi_prop_f, n_days_prop, _ = _cluster_bootstrap_mean_ci(
            daily_prop_f["r"],
            daily_prop_f["Date"],
            alpha=alpha,
            n_bootstrap=BOOTSTRAP_RUNS,
        )
        print(
            f"\nProp mean daily correlation (unfiltered): {daily_prop['r'].mean():.3f}"
        )
        print(
            f"Prop mean daily correlation (n >= {min_n}): {mean_prop_f:.3f} "
            f"(95% Date-cluster bootstrap CI [{lo_prop_f:.3f}, {hi_prop_f:.3f}], days={n_days_prop})"
        )
    if not daily_client_f.empty:
        mean_client_f, lo_client_f, hi_client_f, n_days_client, _ = _cluster_bootstrap_mean_ci(
            daily_client_f["r"],
            daily_client_f["Date"],
            alpha=alpha,
            n_bootstrap=BOOTSTRAP_RUNS,
        )
        print(
            f"\nClient mean daily correlation (unfiltered): {daily_client['r'].mean():.3f}"
        )
        print(
            f"Client mean daily correlation (n >= {min_n}): {mean_client_f:.3f} "
            f"(95% Date-cluster bootstrap CI [{lo_client_f:.3f}, {hi_client_f:.3f}], days={n_days_client})"
        )

    if make_plots and (not daily_prop_f.empty) and (not daily_client_f.empty):
        plot_daily_crowding(
            daily_prop_f,
            daily_client_f,
            out_prefix=out_prefix,
            smoothing_days=smoothing_days,
        )
    else:
        print("\n[Daily plots] Skipped plot generation (no data after filtering or disabled).")


def run_cross_group_crowding_analysis(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    alpha: float = 0.05,
    min_n: int = 100,
    out_prefix: str = "cross_crowding",
    make_plots: bool = True,
    smoothing_days: int = 20,
) -> None:
    """
    Summary
    -------
    Compute cross-group crowding summaries for prop|client and client|prop directions.

    Parameters
    ----------
    metaorders_proprietary : pd.DataFrame
        Proprietary metaorders containing `Direction`, `Date`, `ISIN`, and `Q`.
    metaorders_non_proprietary : pd.DataFrame
        Client metaorders containing `Direction`, `Date`, `ISIN`, and `Q`.
    alpha : float, default=0.05
        Significance level used for confidence intervals.
    min_n : int, default=100
        Minimum per-day sample size used to filter daily correlations.
    out_prefix : str, default="cross_crowding"
        Filename prefix for exported cross-group plots.
    make_plots : bool, default=True
        If True, save daily and smoothed cross-group correlation plots.
    smoothing_days : int, default=20
        Window length used for the rolling-mean plot.

    Returns
    -------
    None
        Prints global and daily summary statistics and optionally saves plots.

    Notes
    -----
    - The environment imbalance for each group is built from the other group on
      the same `(ISIN, Date)` pair.
    - Both global and filtered mean-daily correlations are reported with
      Date-cluster bootstrap percentile CIs.

    Examples
    --------
    >>> run_cross_group_crowding_analysis(prop_df, client_df, min_n=50, make_plots=False)
    """
    print("\n" + "=" * 80)
    print("Cross-group crowding (prop vs client)")
    print("=" * 80)

    prop_env_col = "imbalance_client_env"
    client_env_col = "imbalance_prop_env"
    prop_with_client = attach_environment_imbalance(
        metaorders_proprietary,
        metaorders_non_proprietary,
        new_col=prop_env_col,
    )
    client_with_prop = attach_environment_imbalance(
        metaorders_non_proprietary,
        metaorders_proprietary,
        new_col=client_env_col,
    )

    if ("Date" in prop_with_client.columns) and ("Date" in client_with_prop.columns):
        r_prop_global, lo_prop_global, hi_prop_global, n_prop_global, n_days_prop_global = _date_cluster_corr_ci(
            prop_with_client["Direction"],
            prop_with_client[prop_env_col],
            prop_with_client["Date"],
            alpha=alpha,
            n_bootstrap=BOOTSTRAP_RUNS,
        )
        r_client_global, lo_client_global, hi_client_global, n_client_global, n_days_client_global = _date_cluster_corr_ci(
            client_with_prop["Direction"],
            client_with_prop[client_env_col],
            client_with_prop["Date"],
            alpha=alpha,
            n_bootstrap=BOOTSTRAP_RUNS,
        )

        if math.isnan(r_prop_global):
            print("\nGlobal Corr(Direction_prop, client imbalance): not enough data (n < 3).")
        else:
            print(
                f"\nGlobal Corr(Direction_prop, client imbalance) = {r_prop_global:.3f} "
                f"(95% Date-cluster bootstrap CI [{lo_prop_global:.3f}, {hi_prop_global:.3f}], "
                f"n={n_prop_global}, days={n_days_prop_global})"
            )
        if math.isnan(r_client_global):
            print("Global Corr(Direction_client, prop imbalance): not enough data (n < 3).")
        else:
            print(
                f"Global Corr(Direction_client, prop imbalance) = {r_client_global:.3f} "
                f"(95% Date-cluster bootstrap CI [{lo_client_global:.3f}, {hi_client_global:.3f}], "
                f"n={n_client_global}, days={n_days_client_global})"
            )
    else:
        print("\n[Cross-group] Missing Date column; skipped Date-cluster global correlation summaries.")

    daily_prop_env = daily_crowding_ts(prop_with_client, imb_col=prop_env_col, alpha=alpha)
    daily_client_env = daily_crowding_ts(client_with_prop, imb_col=client_env_col, alpha=alpha)

    daily_prop_f = daily_prop_env[daily_prop_env["n"] >= min_n].reset_index(drop=True)
    daily_client_f = daily_client_env[daily_client_env["n"] >= min_n].reset_index(drop=True)

    print(f"\nProp vs client days with n >= {min_n}: {len(daily_prop_f)} "
          f"(out of {len(daily_prop_env)})")
    print(f"Client vs prop days with n >= {min_n}: {len(daily_client_f)} "
          f"(out of {len(daily_client_env)})")

    if not daily_prop_f.empty:
        mean_prop_f, lo_prop_f, hi_prop_f, n_days_prop, _ = _cluster_bootstrap_mean_ci(
            daily_prop_f["r"],
            daily_prop_f["Date"],
            alpha=alpha,
            n_bootstrap=BOOTSTRAP_RUNS,
        )
        print(
            f"\nProp vs client mean daily correlation (unfiltered): {daily_prop_env['r'].mean():.3f}"
        )
        print(
            f"Prop vs client mean daily correlation (n >= {min_n}): {mean_prop_f:.3f} "
            f"(95% Date-cluster bootstrap CI [{lo_prop_f:.3f}, {hi_prop_f:.3f}], days={n_days_prop})"
        )
    if not daily_client_f.empty:
        mean_client_f, lo_client_f, hi_client_f, n_days_client, _ = _cluster_bootstrap_mean_ci(
            daily_client_f["r"],
            daily_client_f["Date"],
            alpha=alpha,
            n_bootstrap=BOOTSTRAP_RUNS,
        )
        print(
            f"\nClient vs prop mean daily correlation (unfiltered): {daily_client_env['r'].mean():.3f}"
        )
        print(
            f"Client vs prop mean daily correlation (n >= {min_n}): {mean_client_f:.3f} "
            f"(95% Date-cluster bootstrap CI [{lo_client_f:.3f}, {hi_client_f:.3f}], days={n_days_client})"
        )

    if make_plots and (not daily_prop_f.empty) and (not daily_client_f.empty):
        plot_daily_crowding(
            daily_prop_f,
            daily_client_f,
            out_prefix=out_prefix,
            smoothing_days=smoothing_days,
            label_prop="Prop vs client: r(Direction_prop, client imbalance)",
            label_client="Client vs prop: r(Direction_client, prop imbalance)",
            title="Cross-group crowding: Corr(Direction, other-group imbalance)",
            smoothed_title=f"Smoothed cross-group crowding ({smoothing_days}-day rolling mean)",
        )
    else:
        print("\n[Cross-group plots] Skipped plot generation (no data after filtering or disabled).")


def run_all_vs_all_crowding_analysis(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    alpha: float = 0.05,
    min_n: int = 100,
    out_prefix: str = "all_vs_all_crowding",
    make_plots: bool = True,
    smoothing_days: int = 20,
) -> None:
    """
    Summary
    -------
    Analyze crowding versus "all other" metaorders (prop + client combined).

    Parameters
    ----------
    metaorders_proprietary : pd.DataFrame
        Proprietary metaorders containing `ISIN`, `Date`, `Direction`, and `Q`.
    metaorders_non_proprietary : pd.DataFrame
        Client metaorders containing the same core columns.
    alpha : float, default=0.05
        Significance level used for per-day correlation CIs.
    min_n : int, default=100
        Minimum number of metaorders per day used to filter daily correlations.
    out_prefix : str, default=\"all_vs_all_crowding\"
        Output prefix used for saved figures.
    make_plots : bool, default=True
        If True, save daily and rolling plots.
    smoothing_days : int, default=20
        Rolling window length for smoothed correlations.

    Returns
    -------
    None
        Prints summaries and optionally saves figures.

    Notes
    -----
    - For each metaorder, the imbalance is computed as a leave-one-out imbalance
      over the *concatenated* (prop + client) sample on the same `(ISIN, Date)`.

    Examples
    --------
    >>> run_all_vs_all_crowding_analysis(prop_df, client_df, make_plots=False)
    """
    print("\n" + "=" * 80)
    print("Crowding versus all others (prop + client)")
    print("=" * 80)

    combined = pd.concat([metaorders_proprietary, metaorders_non_proprietary], ignore_index=True)
    combined_with_all = add_daily_imbalance(
        combined,
        group_cols=("ISIN", "Date"),
        side_col="Direction",
        vol_col="Q",
        new_col="imbalance_all_others",
    )

    prop_all = combined_with_all[combined_with_all["Group"] == "prop"].reset_index(drop=True)
    client_all = combined_with_all[combined_with_all["Group"] == "client"].reset_index(drop=True)

    daily_prop_all = daily_crowding_ts(prop_all, imb_col="imbalance_all_others", alpha=alpha)
    daily_client_all = daily_crowding_ts(client_all, imb_col="imbalance_all_others", alpha=alpha)

    daily_prop_f = daily_prop_all[daily_prop_all["n"] >= min_n].reset_index(drop=True)
    daily_client_f = daily_client_all[daily_client_all["n"] >= min_n].reset_index(drop=True)

    print(f"\nProp vs all days with n >= {min_n}: {len(daily_prop_f)} "
          f"(out of {len(daily_prop_all)})")
    print(f"Client vs all days with n >= {min_n}: {len(daily_client_f)} "
          f"(out of {len(daily_client_all)})")

    if not daily_prop_f.empty:
        print(
            f"\nProp vs all mean daily correlation (unfiltered): {daily_prop_all['r'].mean():.3f}"
        )
        print(
            f"Prop vs all mean daily correlation (n >= {min_n}): {daily_prop_f['r'].mean():.3f}"
        )
    if not daily_client_f.empty:
        print(
            f"\nClient vs all mean daily correlation (unfiltered): {daily_client_all['r'].mean():.3f}"
        )
        print(
            f"Client vs all mean daily correlation (n >= {min_n}): {daily_client_f['r'].mean():.3f}"
        )

    if make_plots and (not daily_prop_f.empty) and (not daily_client_f.empty):
        plot_daily_crowding(
            daily_prop_f,
            daily_client_f,
            out_prefix=out_prefix,
            smoothing_days=smoothing_days,
            label_prop="Prop vs all others",
            label_client="Client vs all others",
            title="Crowding versus all other metaorders",
            smoothed_title=f"Smoothed crowding vs all ({smoothing_days}-day rolling mean)",
        )
    else:
        print("\n[All-others plots] Skipped plot generation (no data after filtering or disabled).")


def run_member_level_prop_client_crowding_analysis(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    env_col: str = "imbalance_client_member_env",
    alpha: float = 0.05,
    min_metaorders_per_member: int = MIN_METAORDERS_PER_MEMBER,
    out_prefix: str = "member_prop_client_crowding",
    make_plots: bool = True,
    member_window_days: int = MEMBER_WINDOW_DAYS,
    n_min_per_member_client: int = N_MIN_PER_MEMBER_CLIENT,
) -> None:
    """
    Summary
    -------
    Member-level crowding: Corr(Direction_prop, client imbalance at (Member, Date)).

    Parameters
    ----------
    metaorders_proprietary : pd.DataFrame
        Proprietary metaorders including `Member`, `Date`, `Direction`, and `env_col`.
    metaorders_non_proprietary : pd.DataFrame
        Client metaorders (used only if `env_col` must be constructed upstream).
    env_col : str, default=\"imbalance_client_member_env\"
        Column with client member-day environment imbalance attached to proprietary rows.
    alpha : float, default=0.05
        Significance level used for confidence intervals.
    min_metaorders_per_member : int, default=MIN_METAORDERS_PER_MEMBER
        Minimum proprietary metaorders required to report a member-level correlation.
    out_prefix : str, default=\"member_prop_client_crowding\"
        Output prefix for saved figures.
    make_plots : bool, default=True
        If True, export a member-vs-time-window heatmap of correlations.
    member_window_days : int, default=MEMBER_WINDOW_DAYS
        Window length (in trading days) for heatmap aggregation (non-overlapping).
    n_min_per_member_client : int, default=N_MIN_PER_MEMBER_CLIENT
        Minimum number of client metaorders per `(Member, Date)` required when
        constructing the environment imbalance.

    Returns
    -------
    None
        Prints global and per-member summaries and optionally saves plots.

    Notes
    -----
    - Confidence intervals are computed via Date-cluster bootstrap to account for
      dependence within trading days.

    Examples
    --------
    >>> run_member_level_prop_client_crowding_analysis(prop_df, client_df, make_plots=False)
    """
    print("\n" + "=" * 80)
    print("Member-level crowding (prop vs own clients)")
    print("=" * 80)

    required_cols = {"Member", "Direction", env_col, "Date"}
    missing = required_cols - set(metaorders_proprietary.columns)
    if missing:
        print(
            f"\n[Member crowding] Missing required columns in proprietary dataframe; "
            f"skipping member-level analysis. Missing: {missing}"
        )
        return
    if env_col not in metaorders_proprietary.columns:
        print(
            f"\n[Member crowding] Column '{env_col}' not found in proprietary dataframe; "
            "skipping member-level analysis."
        )
        return

    member_window_days = int(member_window_days)
    if member_window_days <= 0:
        raise ValueError("member_window_days must be >= 1")

    df = metaorders_proprietary[list(required_cols)].copy()
    df["Direction"] = pd.to_numeric(df["Direction"], errors="coerce")
    df[env_col] = pd.to_numeric(df[env_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Global correlation across all proprietary metaorders
    r_global, lo_global, hi_global, n_global = corr_with_ci(
        df["Direction"],
        df[env_col],
        alpha=alpha,
    )
    r_global_b, p_global, _ = corr_with_bootstrap_p(
        df["Direction"],
        df[env_col],
        n_bootstrap=BOOTSTRAP_RUNS,
    )
    if math.isnan(r_global):
        print("\n[Member crowding] Global Corr(Direction_prop, client member-imbalance): not enough data (n < 3).")
    else:
        print(
            f"\nGlobal Corr(Direction_prop, {env_col}) = {r_global:.3f} "
            f"(p (bootstrap) = {p_global:.3f}, 95% CI [{lo_global:.3f}, {hi_global:.3f}], n={n_global})"
        )

    # Per-member correlations
    rows = []
    for member, g in df.groupby("Member", sort=True):
        # Use Date as the resampling unit to account for dependence within the
        # same trading day (shared market conditions + member-day imbalance).
        r_m, lo_m, hi_m, p_m, n_m, n_days_m = corr_with_cluster_bootstrap_ci_and_permutation_p(
            g["Direction"],
            g[env_col],
            g["Date"],
            alpha=alpha,
            n_bootstrap=BOOTSTRAP_RUNS,
            n_permutations=BOOTSTRAP_RUNS,
            y_const_tol=1e-12,
        )
        rows.append(
            {
                "Member": member,
                "r": r_m,
                "lo": lo_m,
                "hi": hi_m,
                "p": p_m,
                "n": n_m,
                "n_days": n_days_m,
            }
        )

    stats = pd.DataFrame(rows)
    if stats.empty:
        print("\n[Member crowding] No proprietary metaorders available for member-level analysis.")
        return

    stats = stats.sort_values("Member").reset_index(drop=True)

    # Add total client-metaorder counts per member and apply symmetric thresholds
    client_counts_total = (
        metaorders_non_proprietary.groupby("Member").size().rename("n_client_total")
        if not metaorders_non_proprietary.empty
        else pd.Series(dtype=int)
    )
    stats = stats.merge(
        client_counts_total.reset_index(),
        on="Member",
        how="left",
    )
    stats["n_client_total"] = stats["n_client_total"].fillna(0).astype(int)

    stats_filtered = stats[
        (stats["n"] >= min_metaorders_per_member)
        & (stats["n_client_total"] >= min_metaorders_per_member)
    ].reset_index(drop=True)

    print(
        f"\n[Member crowding] Members with n_prop >= {min_metaorders_per_member} "
        f"and n_client >= {min_metaorders_per_member}: "
        f"{len(stats_filtered)} (out of {len(stats)})"
    )
    if not stats_filtered.empty:
        mean_r = stats_filtered["r"].mean(skipna=True)
        print(f"[Member crowding] Mean per-member correlation (n >= {min_metaorders_per_member}): {mean_r:.3f}")
    else:
        print("[Member crowding] No members pass the minimum metaorder threshold; skipping exports.")
        return

    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = out_prefix_path.parent / f"{out_prefix_path.name}_per_member.parquet"
    stats_filtered.to_parquet(parquet_path, index=False)
    print(f"[Member crowding] Saved per-member statistics to: {parquet_path}")

    if make_plots:
        r_vals = stats_filtered["r"].dropna()
        if r_vals.empty:
            print("[Member crowding] No finite per-member correlations to plot per-member bar chart.")
        else:
            # Sort members by correlation for a more informative plot
            plot_df = stats_filtered.dropna(subset=["r"]).copy()
            plot_df = plot_df.sort_values("r").reset_index(drop=True)
            members = plot_df["Member"].astype(str).tolist()
            r_sorted = plot_df["r"].to_numpy()
            p_sorted = plot_df["p"].to_numpy(dtype=float, copy=True)
            p_labels = [
                f"p={p:.3g}" if np.isfinite(p) else "p=NA"
                for p in p_sorted
            ]
            hover_text = [
                f"Member {m}<br>corr={r:.3f}<br>p={p:.3g}" if np.isfinite(p) else f"Member {m}<br>corr={r:.3f}<br>p=NA"
                for m, r, p in zip(members, r_sorted, p_sorted)
            ]
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=members,
                        y=r_sorted,
                        marker_color=COLOR_PROPRIETARY,
                        customdata=np.column_stack([p_sorted]),
                        text=p_labels,
                        textposition="outside",
                        hovertext=hover_text,
                        hoverinfo="text",
                    )
                ]
            )
            fig.add_hline(y=0.0, line=dict(color=COLOR_NEUTRAL, width=1, dash="dot"))
            fig.update_layout(
                title="Per-member prop/client crowding correlation",
                xaxis_title="Member",
                yaxis_title=r"$\mathrm{Corr}(\epsilon_i, \mathrm{imb}_{m, d_i})$",
                xaxis=dict(tickangle=90),
                uniformtext_minsize=8,
                uniformtext_mode="show",
            )
            _, bar_path = save_plotly_figure(
                fig,
                stem=f"{out_prefix_path.name}_hist",
                dirs=PLOT_OUTPUT_DIRS,
                write_html=True,
                write_png=True,
                strict_png=False,
            )
            print(f"[Member crowding] Saved per-member correlation bar chart to: {bar_path}")

        # Member–window (member_window_days-day, non-overlapping) heatmap with minimum counts
        # Build non-overlapping windows from the union of prop+client dates
        all_dates = sorted(
            set(pd.to_datetime(metaorders_proprietary["Date"]).dt.date.dropna()).union(
                set(pd.to_datetime(metaorders_non_proprietary.get("Date", pd.Series([], dtype=object))).dt.date.dropna())
            )
        )
        if not all_dates:
            print("[Member crowding] No dates available to build member–window heatmap.")
            return

        window_labels: dict = {}
        window_order: list[str] = []
        for idx in range(0, len(all_dates), member_window_days):
            chunk = all_dates[idx : idx + member_window_days]
            start = chunk[0]
            end = chunk[-1]
            label = f"{start}_to_{end}"
            window_order.append(label)
            for d in chunk:
                window_labels[d] = label

        # Assign windows to proprietary metaorders
        df_win = df.copy()
        df_win["__Date_dt__"] = pd.to_datetime(df_win["Date"]).dt.date
        df_win["Window"] = df_win["__Date_dt__"].map(window_labels)
        df_win = df_win.dropna(subset=["Window"])

        # Prop counts per (Member, Window)
        prop_counts = (
            df_win.groupby(["Member", "Window"], sort=False)
            .size()
            .rename("n_prop")
            .reset_index()
        )

        # Client counts per (Member, Window)
        client_dates = metaorders_non_proprietary.copy()
        client_dates["__Date_dt__"] = pd.to_datetime(client_dates["Date"]).dt.date
        client_dates["Window"] = client_dates["__Date_dt__"].map(window_labels)
        client_counts = (
            client_dates.dropna(subset=["Window"])
            .groupby(["Member", "Window"], sort=False)
            .size()
            .rename("n_client")
            .reset_index()
        )

        # Merge counts into prop data for filtering
        df_win = df_win.merge(prop_counts, on=["Member", "Window"], how="left")
        df_win = df_win.merge(client_counts, on=["Member", "Window"], how="left")
        df_win["n_client"] = df_win["n_client"].fillna(0).astype(int)

        # Compute correlations per (Member, Window) with minimum counts.
        # If BOOTSTRAP_HEATMAP is True, we also compute permutation p-values and
        # filter non-significant cells (set them to NaN) before plotting.
        window_rows = []
        grouped_win = df_win.groupby(["Member", "Window"], sort=True)
        for (member, window_label), g in grouped_win:
            n_prop = int(g["n_prop"].iloc[0]) if "n_prop" in g else len(g)
            n_client = int(g["n_client"].iloc[0]) if "n_client" in g else 0
            r_win = float("nan")
            p_win = float("nan")
            n_win = len(g)
            if n_prop >= n_min_per_member_client and n_client >= n_min_per_member_client:
                if BOOTSTRAP_HEATMAP:
                    # Cluster by Date: the member-day imbalance is constant within a day, and
                    # we want to preserve within-day dependence when testing significance.
                    r_win_b, _, _, p_win, n_win, _ = corr_with_cluster_bootstrap_ci_and_permutation_p(
                        g["Direction"],
                        g[env_col],
                        g["Date"],
                        alpha=alpha,
                        n_bootstrap=0,  # CI not needed for the heatmap; keep runtime low
                        n_permutations=BOOTSTRAP_RUNS,
                        y_const_tol=1e-12,
                    )
                    if (not np.isfinite(p_win)) or (p_win <= P_VALUE_THRESHOLD):
                        r_win = r_win_b
                else:
                    r_win_b, p_win, n_win = corr_with_bootstrap_p(
                        g["Direction"], g[env_col], n_bootstrap=0
                    )
                    r_win = r_win_b
            window_rows.append(
                {
                    "Member": member,
                    "Window": window_label,
                    "r": r_win,
                    "p": p_win,
                    "n_prop": n_prop,
                    "n_client": n_client,
                    "n_used": n_win,
                }
            )

        window_stats = pd.DataFrame(window_rows)
        if window_stats.empty:
            print("[Member crowding] No data available to build member–window heatmap.")
        else:
            # Order windows by their start date using window_order
            window_cat = pd.CategoricalDtype(categories=window_order, ordered=True)
            window_stats["Window"] = window_stats["Window"].astype(window_cat)
            pivot = window_stats.pivot(index="Window", columns="Member", values="r")
            pivot = pivot.sort_index()

            if pivot.empty:
                print("[Member crowding] Member–window heatmap pivot is empty; skipping plot.")
            else:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=pivot.to_numpy(dtype=float),
                        x=[str(x) for x in pivot.columns],
                        y=[str(y) for y in pivot.index],
                        colorscale="RdBu",
                        zmin=-1.0,
                        zmax=1.0,
                        zmid=0.0,
                        colorbar=dict(title=r"$\mathrm{Corr}(\epsilon_i, \mathrm{imb}_{m, d_i})$"),
                        hovertemplate="Window=%{y}<br>Member=%{x}<br>corr=%{z:.3f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title="Member-window prop/client crowding heatmap",
                    xaxis_title="Member",
                    yaxis_title=f"{member_window_days}-day window (non-overlapping)",
                )
                _, heatmap_path = save_plotly_figure(
                    fig,
                    stem=f"{out_prefix_path.name}_heatmap_{member_window_days}d",
                    dirs=PLOT_OUTPUT_DIRS,
                    write_html=True,
                    write_png=True,
                    strict_png=False,
                )
                print(f"[Member crowding] Saved member–window heatmap to: {heatmap_path}")


def compute_daily_metaorder_counts(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summary
    -------
    Compute daily metaorder counts for prop/client and their normalized imbalance.

    Parameters
    ----------
    metaorders_proprietary : pd.DataFrame
        Proprietary metaorders containing a `Date` column.
    metaorders_non_proprietary : pd.DataFrame
        Client metaorders containing a `Date` column.

    Returns
    -------
    pd.DataFrame
        Indexed by Date with integer columns `Proprietary`, `Client` and a float
        column `imbalance_counts = (N_prop - N_client) / (N_prop + N_client)`.

    Notes
    -----
    - Dates are normalized to pandas datetime index and sorted.

    Examples
    --------
    >>> daily = compute_daily_metaorder_counts(prop_df, client_df)
    >>> "imbalance_counts" in daily.columns
    True
    """
    prop_counts = metaorders_proprietary.groupby("Date").size().rename("Proprietary")
    client_counts = metaorders_non_proprietary.groupby("Date").size().rename("Client")

    daily_metaorders = (
        pd.concat([prop_counts, client_counts], axis=1)
        .fillna(0)
        .astype(int)
    )
    daily_metaorders.index = pd.to_datetime(daily_metaorders.index)
    daily_metaorders = daily_metaorders.sort_index()

    total_counts = daily_metaorders["Proprietary"] + daily_metaorders["Client"]
    with np.errstate(divide="ignore", invalid="ignore"):
        daily_metaorders["imbalance_counts"] = np.where(
            total_counts > 0,
            (daily_metaorders["Proprietary"] - daily_metaorders["Client"]) / total_counts,
            np.nan,
        )
    return daily_metaorders


def plot_daily_count_imbalance(daily_metaorders: pd.DataFrame, out_prefix: str) -> None:
    """
    Summary
    -------
    Save figures describing the daily count imbalance between prop and client flow.

    Parameters
    ----------
    daily_metaorders : pd.DataFrame
        Output of `compute_daily_metaorder_counts`.
    out_prefix : str
        Output prefix used for filenames.

    Returns
    -------
    None
        Writes PNG figures to disk.

    Notes
    -----
    - Produces two PNGs:
      1) `{out_prefix}_imbalance_timeseries.png` (time series).
      2) `{out_prefix}_imbalance_histogram.png` (density estimate).

    Examples
    --------
    >>> plot_daily_count_imbalance(daily, out_prefix=\"images/daily_counts\")
    """
    out_prefix_name = Path(out_prefix).name

    # Time-series plot
    fig_ts = go.Figure(
        data=[
            go.Scatter(
                x=pd.to_datetime(daily_metaorders.index),
                y=daily_metaorders["imbalance_counts"],
                mode="lines",
                line=dict(color=COLOR_PROPRIETARY, width=2),
                name="Daily count imbalance",
            )
        ]
    )
    fig_ts.add_hline(y=0.0, line=dict(color=COLOR_NEUTRAL, width=1, dash="dot"))
    fig_ts.update_layout(
        title="Daily count imbalance",
        xaxis_title="Date",
        yaxis_title="(N_prop - N_client) / (N_prop + N_client)",
    )
    _, ts_path = save_plotly_figure(
        fig_ts,
        stem=f"{out_prefix_name}_imbalance_timeseries",
        dirs=PLOT_OUTPUT_DIRS,
        write_html=True,
        write_png=True,
        strict_png=False,
    )
    print(f"\n[Daily count plots] Saved imbalance time-series to: {ts_path}")

    # PDF of imbalance distribution
    vals = pd.to_numeric(daily_metaorders["imbalance_counts"], errors="coerce")
    vals = vals[np.isfinite(vals)]
    fig_hist = go.Figure()
    if vals.size > 0:
        fig_hist.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=max(20, int(np.sqrt(vals.size))),
                histnorm="probability density",
                marker=dict(color=COLOR_PROPRIETARY),
                opacity=0.8,
                name="Imbalance density",
            )
        )
        hist_title = "Imbalance density"
    else:
        hist_title = "Imbalance density (no valid data)"
        fig_hist.add_annotation(
            text="No valid data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    fig_hist.update_layout(
        title=hist_title,
        xaxis_title="Imbalance",
        yaxis_title="Density",
        bargap=0.05,
    )
    _, hist_path = save_plotly_figure(
        fig_hist,
        stem=f"{out_prefix_name}_imbalance_histogram",
        dirs=PLOT_OUTPUT_DIRS,
        write_html=True,
        write_png=True,
        strict_png=False,
    )
    if vals.size > 0:
        print(f"[Daily count plots] Saved imbalance density plot to: {hist_path}")
    else:
        print(f"[Daily count plots] Imbalance density plot skipped (no data).")


def run_daily_count_imbalance_analysis(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    out_prefix: str,
    make_plots: bool = True,
) -> None:
    """
    Summary
    -------
    Compute and (optionally) plot daily count imbalance between prop and client flow.

    Parameters
    ----------
    metaorders_proprietary : pd.DataFrame
        Proprietary metaorders containing `Date`.
    metaorders_non_proprietary : pd.DataFrame
        Client metaorders containing `Date`.
    out_prefix : str
        Prefix for saved output files.
    make_plots : bool, default=True
        If True, save plots.

    Returns
    -------
    None

    Notes
    -----
    - This is a diagnostic intended to complement volume-weighted imbalance measures.

    Examples
    --------
    >>> run_daily_count_imbalance_analysis(prop_df, client_df, out_prefix=\"images/daily_counts\", make_plots=False)
    """
    print("\n" + "=" * 80)
    print("Daily count imbalance")
    print("=" * 80)

    daily_metaorders = compute_daily_metaorder_counts(
        metaorders_proprietary,
        metaorders_non_proprietary,
    )

    print("\nFirst few rows of daily counts and imbalance:")
    print(daily_metaorders.head())

    mean_imb = daily_metaorders["imbalance_counts"].mean()
    std_imb = daily_metaorders["imbalance_counts"].std()
    print(f"\nMean imbalance: {mean_imb:+.3f}")
    print(f"Std imbalance : {std_imb:.3f}")

    if make_plots and not daily_metaorders.empty:
        plot_daily_count_imbalance(daily_metaorders, out_prefix=out_prefix)
    else:
        print("\n[Daily count plots] Skipped plot generation (no data or disabled).")


def run_crowding_vs_part_rate_analysis(
    prop_path: Path,
    client_path: Path,
    output_base_dir: Path,
    image_base_dir: Path,
    config_path: Path = _CONFIG_PATH,
) -> None:
    """
    Summary
    -------
    Run the participation-rate crowding workflow from `crowding_vs_part_rate.py`.

    Parameters
    ----------
    prop_path : Path
        Path to proprietary metaorders parquet.
    client_path : Path
        Path to client (non-proprietary) metaorders parquet.
    output_base_dir : Path
        Base output directory (typically `out_files/{DATASET_NAME}`).
    image_base_dir : Path
        Base image directory (typically `images/{DATASET_NAME}`).
    config_path : Path, default=_CONFIG_PATH
        YAML config path passed to the delegated script.

    Returns
    -------
    None
        Writes the same CSV/Parquet tables and Plotly/Matplotlib figures as
        `crowding_vs_part_rate.py`.

    Notes
    -----
    - This helper keeps a single orchestration entrypoint (`crowding_analysis.py`)
      while reusing the validated eta-conditioned analysis implementation.
    - The delegated script recomputes missing imbalance columns if needed; here we
      still pass the already-enriched parquet paths to keep behavior deterministic.

    Examples
    --------
    >>> run_crowding_vs_part_rate_analysis(
    ...     Path("out_files/ftsemib/metaorders_info_sameday_filtered_member_proprietary.parquet"),
    ...     Path("out_files/ftsemib/metaorders_info_sameday_filtered_member_non_proprietary.parquet"),
    ...     Path("out_files/ftsemib"),
    ...     Path("images/ftsemib"),
    ... )
    """
    if not RUN_CROWDING_VS_PART_RATE:
        print("\n[Crowding vs eta] Skipped (RUN_CROWDING_VS_PART_RATE=false).")
        return

    if CROWDING_VS_PART_RATE_PLOTS not in {"plotly", "matplotlib", "both"}:
        raise ValueError(
            "CROWDING_VS_PART_RATE_PLOTS must be one of {'plotly', 'matplotlib', 'both'}."
        )
    if CROWDING_VS_PART_RATE_ETA_BINNING not in {"pooled_quantiles", "group_quantiles"}:
        raise ValueError(
            "CROWDING_VS_PART_RATE_ETA_BINNING must be one of {'pooled_quantiles', 'group_quantiles'}."
        )
    if CROWDING_VS_PART_RATE_CLUSTER_CI not in {"date", "none"}:
        raise ValueError("CROWDING_VS_PART_RATE_CLUSTER_CI must be either 'date' or 'none'.")
    if CROWDING_VS_PART_RATE_ETA_BINS < 1:
        raise ValueError("CROWDING_VS_PART_RATE_ETA_BINS must be >= 1.")
    if CROWDING_VS_PART_RATE_MIN_N_BIN < 1:
        raise ValueError("CROWDING_VS_PART_RATE_MIN_N_BIN must be >= 1.")
    if CROWDING_VS_PART_RATE_MIN_N_DAY_BIN < 1:
        raise ValueError("CROWDING_VS_PART_RATE_MIN_N_DAY_BIN must be >= 1.")
    if CROWDING_VS_PART_RATE_BOOTSTRAP_RUNS < 0:
        raise ValueError("CROWDING_VS_PART_RATE_BOOTSTRAP_RUNS must be >= 0.")
    if not (0 < CROWDING_VS_PART_RATE_REG_SAMPLE_FRAC <= 1):
        raise ValueError("CROWDING_VS_PART_RATE_REG_SAMPLE_FRAC must be in (0, 1].")

    try:
        from crowding_vs_part_rate import main as crowding_vs_part_rate_main
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Failed to import crowding_vs_part_rate.py.") from exc

    argv: List[str] = [
        "--config-path",
        str(config_path),
        "--dataset-name",
        DATASET_NAME,
        "--prop-path",
        str(prop_path),
        "--client-path",
        str(client_path),
        "--output-file-path",
        str(output_base_dir),
        "--img-output-path",
        str(image_base_dir),
        "--analysis-tag",
        CROWDING_VS_PART_RATE_ANALYSIS_TAG,
        "--imbalance-kind",
        CROWDING_VS_PART_RATE_IMBALANCE_KIND,
        "--eta-bins",
        str(CROWDING_VS_PART_RATE_ETA_BINS),
        "--eta-binning",
        CROWDING_VS_PART_RATE_ETA_BINNING,
        "--eta-max",
        str(CROWDING_VS_PART_RATE_ETA_MAX),
        "--min-n-bin",
        str(CROWDING_VS_PART_RATE_MIN_N_BIN),
        "--min-n-day-bin",
        str(CROWDING_VS_PART_RATE_MIN_N_DAY_BIN),
        "--bootstrap-runs",
        str(CROWDING_VS_PART_RATE_BOOTSTRAP_RUNS),
        "--cluster-ci",
        CROWDING_VS_PART_RATE_CLUSTER_CI,
        "--alpha",
        str(CROWDING_VS_PART_RATE_ALPHA),
        "--seed",
        str(CROWDING_VS_PART_RATE_SEED),
        "--reg-sample-frac",
        str(CROWDING_VS_PART_RATE_REG_SAMPLE_FRAC),
        "--permutation-runs",
        str(CROWDING_VS_PART_RATE_PERMUTATION_RUNS),
        "--qv-bins",
        str(CROWDING_VS_PART_RATE_QV_BINS),
        "--plots",
        CROWDING_VS_PART_RATE_PLOTS,
    ]

    argv.append("--run-regressions" if CROWDING_VS_PART_RATE_RUN_REGRESSIONS else "--no-run-regressions")
    if CROWDING_VS_PART_RATE_RUN_2D:
        argv.append("--run-2d")
    if CROWDING_VS_PART_RATE_WRITE_PARQUET:
        argv.append("--write-parquet")

    print("\n" + "=" * 80)
    print("Crowding conditioned on participation rate (eta)")
    print("=" * 80)
    exit_code = int(crowding_vs_part_rate_main(argv))
    if exit_code != 0:
        raise RuntimeError(f"Crowding-vs-participation workflow failed with exit code {exit_code}.")


def plot_imbalance_vs_daily_return(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    out_path: str | Path,
    imbalance_col: str = "imbalance_local",
    return_col: str = DAILY_RETURN_COL,
) -> None:
    """
    Summary
    -------
    Scatter plot of imbalance versus daily return for proprietary and client flow.

    Parameters
    ----------
    metaorders_proprietary : pd.DataFrame
        Proprietary metaorders containing `imbalance_col` and `return_col`.
    metaorders_non_proprietary : pd.DataFrame
        Client metaorders containing `imbalance_col` and `return_col`.
    out_path : str | Path
        Output filename (only basename is used under the configured plot folder).
    imbalance_col : str, default=\"imbalance_local\"
        Imbalance column name.
    return_col : str, default=DAILY_RETURN_COL
        Daily return column name.

    Returns
    -------
    None
        Writes a PNG figure to disk.

    Notes
    -----
    - This is a descriptive diagnostic; it does not control for other covariates.

    Examples
    --------
    >>> plot_imbalance_vs_daily_return(prop_df, client_df, out_path=\"imb_vs_ret.png\")
    """
    required_cols = {imbalance_col, return_col}
    missing_prop = required_cols - set(metaorders_proprietary.columns)
    missing_client = required_cols - set(metaorders_non_proprietary.columns)
    if missing_prop or missing_client:
        print(
            "\n[Imbalance vs returns] Missing columns; skip plot. "
            f"Prop missing: {missing_prop}, client missing: {missing_client}"
        )
        return

    frames = []
    for df, label in [
        (metaorders_proprietary, "Proprietary"),
        (metaorders_non_proprietary, "Client"),
    ]:
        if df.empty:
            continue
        subset = df[[imbalance_col, return_col]].copy()
        subset["Group"] = label
        frames.append(subset)

    if not frames:
        print("\n[Imbalance vs returns] No data available to build scatter plot.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined[imbalance_col] = pd.to_numeric(combined[imbalance_col], errors="coerce")
    combined[return_col] = pd.to_numeric(combined[return_col], errors="coerce")
    valid = combined.dropna(subset=[imbalance_col, return_col])
    if valid.empty:
        print("\n[Imbalance vs returns] No overlapping non-NaN imbalance/return pairs to plot.")
        return

    out_path = _png_output_path(Path(out_path).name)

    fig = go.Figure()
    for label, g in valid.groupby("Group"):
        color = COLOR_PROPRIETARY if label == "Proprietary" else COLOR_CLIENT
        fig.add_trace(
            go.Scatter(
                x=g[imbalance_col],
                y=g[return_col],
                mode="markers",
                marker=dict(size=6, opacity=0.55, color=color),
                name=label,
            )
        )
    fig.add_hline(y=0.0, line=dict(color=COLOR_NEUTRAL, width=1, dash="dot"))
    fig.add_vline(x=0.0, line=dict(color=COLOR_NEUTRAL, width=1, dash="dot"))
    fig.update_layout(
        title="Imbalance vs daily return",
        xaxis_title="Imbalance (others on same ISIN/day)",
        yaxis_title="Daily log return (close-to-close)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _, out_path_saved = save_plotly_figure(
        fig,
        stem=Path(out_path).stem,
        dirs=PLOT_OUTPUT_DIRS,
        write_html=True,
        write_png=True,
        strict_png=False,
    )
    print(f"\n[Imbalance vs returns] Saved scatter plot to: {out_path_saved}")


def plot_imbalance_distributions(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    plot_dir: str | Path,
    bins: int = 50,
    use_kde: bool = False,
    kde_bw_adjust: float = 1.0,
) -> None:
    """
    Summary
    -------
    Plot within-group and cross-group imbalance distributions split by buy/sell.

    Parameters
    ----------
    metaorders_proprietary : pd.DataFrame
        Proprietary-flow metaorders dataframe.
    metaorders_non_proprietary : pd.DataFrame
        Non-proprietary (client) metaorders dataframe.
    plot_dir : str | Path
        Kept for API compatibility; output location is managed by `_png_output_path`.
    bins : int, default=50
        Number of histogram bins when `use_kde=False` or when KDE falls back.
    use_kde : bool, default=False
        If True, estimate densities using KDE instead of histogram-based lines.
    kde_bw_adjust : float, default=1.0
        Seaborn bandwidth multiplier used only when `use_kde=True`.

    Returns
    -------
    None
        Saves the distribution figure to disk.

    Notes
    -----
    - If KDE cannot be estimated for a series (too few points or near-constant
      values), the function falls back to histogram density for that series.

    Examples
    --------
    >>> plot_imbalance_distributions(prop_df, client_df, "images", use_kde=True, kde_bw_adjust=0.8)
    """
    if kde_bw_adjust <= 0:
        raise ValueError("kde_bw_adjust must be strictly positive.")

    def _hist_density(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
        density, edges = np.histogram(values, bins=n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        valid = np.isfinite(centers) & np.isfinite(density) & (density > 0)
        return centers[valid], density[valid]

    def _kde_density(values: np.ndarray, bw_adjust: float) -> tuple[np.ndarray, np.ndarray]:
        from scipy.stats import gaussian_kde

        if values.size < 2:
            return np.array([]), np.array([])
        if np.allclose(values, values[0]):
            return np.array([]), np.array([])
        kde = gaussian_kde(values, bw_method="scott")
        kde.set_bandwidth(kde.factor * bw_adjust)
        x_grid = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), 250)
        y_grid = kde(x_grid)
        valid = np.isfinite(x_grid) & np.isfinite(y_grid) & (y_grid > 0)
        return x_grid[valid], y_grid[valid]

    def _add_density_trace(fig_obj: go.Figure, row: int, values: np.ndarray, label: str, color: str) -> bool:
        arr = values[np.isfinite(values)]
        if arr.size == 0:
            return False
        x, y = _kde_density(arr, kde_bw_adjust) if use_kde else _hist_density(arr, bins)
        if x.size == 0:
            x, y = _hist_density(arr, bins)
        if x.size == 0:
            return False
        fig_obj.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                legendgroup=label,
            ),
            row=row,
            col=1,
        )
        return True

    within_series = [
        (
            pd.to_numeric(metaorders_proprietary.loc[metaorders_proprietary["Direction"] == 1, "imbalance_local"], errors="coerce").to_numpy(dtype=float),
            "Prop buy",
            COLOR_PROPRIETARY,
        ),
        (
            pd.to_numeric(metaorders_proprietary.loc[metaorders_proprietary["Direction"] == -1, "imbalance_local"], errors="coerce").to_numpy(dtype=float),
            "Prop sell",
            THEME_COLORWAY[1],
        ),
        (
            pd.to_numeric(metaorders_non_proprietary.loc[metaorders_non_proprietary["Direction"] == 1, "imbalance_local"], errors="coerce").to_numpy(dtype=float),
            "Client buy",
            COLOR_CLIENT,
        ),
        (
            pd.to_numeric(metaorders_non_proprietary.loc[metaorders_non_proprietary["Direction"] == -1, "imbalance_local"], errors="coerce").to_numpy(dtype=float),
            "Client sell",
            THEME_COLORWAY[3],
        ),
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.14,
    )

    added_within = False
    for values, label, color in within_series:
        added_within = _add_density_trace(fig, row=1, values=values, label=label, color=color) or added_within
    if not added_within:
        fig.add_annotation(text="No within-group data", x=0.5, y=0.85, xref="paper", yref="paper", showarrow=False)

    added_cross = False
    has_cross_cols = ("imbalance_client_env" in metaorders_proprietary.columns) and (
        "imbalance_prop_env" in metaorders_non_proprietary.columns
    )
    if not has_cross_cols:
        fig.add_annotation(text="Cross-group imbalance columns not found", x=0.5, y=0.36, xref="paper", yref="paper", showarrow=False)
        print("\n[Imbalance distribution] Cross-group imbalance columns not found; skipping cross density plot.")
    else:
        cross_series = [
            (
                pd.to_numeric(metaorders_proprietary.loc[metaorders_proprietary["Direction"] == 1, "imbalance_client_env"], errors="coerce").to_numpy(dtype=float),
                "Prop buy (client env)",
                COLOR_PROPRIETARY,
            ),
            (
                pd.to_numeric(metaorders_proprietary.loc[metaorders_proprietary["Direction"] == -1, "imbalance_client_env"], errors="coerce").to_numpy(dtype=float),
                "Prop sell (client env)",
                THEME_COLORWAY[1],
            ),
            (
                pd.to_numeric(metaorders_non_proprietary.loc[metaorders_non_proprietary["Direction"] == 1, "imbalance_prop_env"], errors="coerce").to_numpy(dtype=float),
                "Client buy (prop env)",
                COLOR_CLIENT,
            ),
            (
                pd.to_numeric(metaorders_non_proprietary.loc[metaorders_non_proprietary["Direction"] == -1, "imbalance_prop_env"], errors="coerce").to_numpy(dtype=float),
                "Client sell (prop env)",
                THEME_COLORWAY[3],
            ),
        ]

        for values, label, color in cross_series:
            added_cross = _add_density_trace(fig, row=1, values=values, label=label, color=color) or added_cross
        if not added_cross:
            fig.add_annotation(text="No cross-group data", x=0.5, y=0.36, xref="paper", yref="paper", showarrow=False)

    fig.update_xaxes(title_text="Within-group imbalance", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Cross-group imbalance", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    fig.update_layout(
        title="Imbalance distributions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=900,
    )
    if added_within or added_cross:
        _, out_path = save_plotly_figure(
            fig,
            stem="imbalance_distribution",
            dirs=PLOT_OUTPUT_DIRS,
            write_html=True,
            write_png=True,
            strict_png=False,
        )
        print(f"\n[Imbalance distribution] Saved combined density plot to: {out_path}")
    else:
        print("\n[Imbalance distribution] Skipped density plot (no data).")


def compute_autocorr_fft(series: Iterable[float], max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Summary
    -------
    Compute an autocorrelation function (ACF) efficiently via FFT.

    Parameters
    ----------
    series : Iterable[float]
        1D sample (NaNs are dropped).
    max_lag : int
        Maximum lag returned.

    Returns
    -------
    lags : np.ndarray
        Integer lags from 0 to `max_lag` (inclusive).
    acf : np.ndarray
        Autocorrelation values for each lag (same length as `lags`).

    Notes
    -----
    - The series is demeaned before computing the ACF.
    - When variance is zero, returns an ACF of ones.

    Examples
    --------
    >>> lags, acf = compute_autocorr_fft([1, -1, 1, -1], max_lag=3)
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    arr = arr - arr.mean()
    max_lag = min(max_lag, n - 1)
    if max_lag < 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    var0 = float(np.dot(arr, arr))
    if var0 == 0.0:
        lags = np.arange(max_lag + 1, dtype=int)
        return lags, np.ones_like(lags, dtype=float)

    n_fft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(arr, n=n_fft)
    acf_full = np.fft.irfft(fx * np.conjugate(fx), n=n_fft)[:n]
    acf_full = acf_full / var0
    lags = np.arange(max_lag + 1, dtype=int)
    return lags, acf_full[: max_lag + 1]


def bootstrap_noise_band(series: Iterable[float], max_lag: int, n_bootstrap: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Summary
    -------
    Build a permutation-based noise band for the ACF under i.i.d. signs.

    Parameters
    ----------
    series : Iterable[float]
        1D sample (NaNs are dropped).
    max_lag : int
        Maximum lag (band excludes lag 0).
    n_bootstrap : int
        Number of permutations used.

    Returns
    -------
    lower : np.ndarray
        5th percentile band for lags 1..max_lag.
    upper : np.ndarray
        95th percentile band for lags 1..max_lag.

    Notes
    -----
    - This is a simple placebo band: it permutes the series to destroy temporal
      dependence while preserving the marginal distribution.

    Examples
    --------
    >>> lo, hi = bootstrap_noise_band([1, -1, 1, -1], max_lag=3, n_bootstrap=50)
    """
    base = np.asarray(series, dtype=float)
    base = base[~np.isnan(base)]
    n = base.size
    if n == 0 or max_lag < 1:
        return np.array([], dtype=float), np.array([], dtype=float)
    max_lag = min(max_lag, n - 1)

    boot = np.empty((n_bootstrap, max_lag))
    for b in range(n_bootstrap):
        perm = np.random.permutation(base)
        _, acf_tmp = compute_autocorr_fft(perm, max_lag)
        boot[b] = acf_tmp[1 : max_lag + 1]

    lower = np.quantile(boot, 0.05, axis=0)
    upper = np.quantile(boot, 0.95, axis=0)
    return lower, upper


def plot_direction_autocorrelation_per_isin(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    out_dir: str | Path,
    max_lag: int = 300,
    n_bootstrap: int = 100,
) -> None:
    """
    Summary
    -------
    Compute and save per-ISIN sign autocorrelation plots for prop and client flow.

    Parameters
    ----------
    metaorders_proprietary : pd.DataFrame
        Proprietary metaorders containing `ISIN` and `Direction`.
    metaorders_non_proprietary : pd.DataFrame
        Client metaorders containing `ISIN` and `Direction`.
    out_dir : str | Path
        Directory where per-ISIN PNGs are written.
    max_lag : int, default=300
        Maximum lag shown.
    n_bootstrap : int, default=100
        Number of permutations used for noise bands.

    Returns
    -------
    None
        Writes one `{ISIN}_acf.png` per ISIN.

    Notes
    -----
    - ISINs are collected as the union of both dataframes' `ISIN` columns.

    Examples
    --------
    >>> plot_direction_autocorrelation_per_isin(prop_df, client_df, out_dir=\"images/acf\", max_lag=50)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_isins = sorted(
        set(metaorders_proprietary.get("ISIN", pd.Series(dtype=object)).dropna().astype(str)).union(
            set(metaorders_non_proprietary.get("ISIN", pd.Series(dtype=object)).dropna().astype(str))
        )
    )
    if not all_isins:
        print("\n[ACF] No ISINs found; skipping per-ISIN autocorrelation plots.")
        return

    for isin in tqdm(all_isins, desc="Per-ISIN ACF"):
        prop_sub = metaorders_proprietary[metaorders_proprietary["ISIN"].astype(str) == isin]
        client_sub = metaorders_non_proprietary[metaorders_non_proprietary["ISIN"].astype(str) == isin]

        if prop_sub.empty and client_sub.empty:
            continue

        lags_prop, acf_prop = compute_autocorr_fft(prop_sub["Direction"], max_lag) if not prop_sub.empty else (np.array([]), np.array([]))
        lags_client, acf_client = compute_autocorr_fft(client_sub["Direction"], max_lag) if not client_sub.empty else (np.array([]), np.array([]))

        lower_prop, upper_prop = (
            bootstrap_noise_band(prop_sub["Direction"], max_lag, n_bootstrap) if not prop_sub.empty else (np.array([]), np.array([]))
        )
        lower_client, upper_client = (
            bootstrap_noise_band(client_sub["Direction"], max_lag, n_bootstrap) if not client_sub.empty else (np.array([]), np.array([]))
        )

        if (lags_prop.size <= 1) and (lags_client.size <= 1):
            continue

        fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
        configs = [
            ("Proprietary", lags_prop, acf_prop, lower_prop, upper_prop, COLOR_PROPRIETARY, 1),
            ("Client", lags_client, acf_client, lower_client, upper_client, COLOR_CLIENT, 2),
        ]

        for group_name, lags, acf_vals, lower, upper, color, col in configs:
            if lags.size <= 1:
                continue
            if lower.size and upper.size and lower.shape[0] == lags.size - 1:
                fig.add_trace(
                    go.Scatter(x=lags[1:], y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
                    row=1,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=lags[1:],
                        y=lower,
                        mode="lines",
                        fill="tonexty",
                        fillcolor="rgba(107,114,128,0.25)",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
            fig.add_trace(
                go.Scatter(
                    x=lags[1:],
                    y=acf_vals[1:],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{group_name} ACF",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            fig.update_xaxes(title_text="Lag", range=[0, max_lag], row=1, col=col)

        fig.update_yaxes(title_text="Autocorrelation", row=1, col=1)
        fig.update_layout(title=f"Metaorder sign autocorrelation ({isin})", height=700)
        save_plotly_figure(
            fig,
            stem=f"{isin}_acf",
            dirs=make_plot_output_dirs(out_dir, use_subdirs=False),
            write_html=False,
            write_png=True,
            strict_png=False,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def load_metaorders(path: str | Path) -> pd.DataFrame:
    """
    Summary
    -------
    Load a metaorder table from CSV, parquet, or pickle and normalize key columns.

    Parameters
    ----------
    path : str | Path
        Input path. Supported extensions: `.csv`, `.parquet`, `.pkl`, `.pickle`.

    Returns
    -------
    pd.DataFrame
        Loaded metaorders with a normalized `Date` column and required columns
        present (`ISIN`, `Q`, `Direction`).

    Notes
    -----
    - If `Date` is missing, it is derived from the first timestamp in `Period`.
    - `Date` is normalized to midnight timestamps to avoid object-dtype dates.

    Examples
    --------
    >>> df = load_metaorders(\"out_files/ftsemib/metaorders_info_sameday_filtered_member_proprietary.parquet\")
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path_obj)
    elif suffix in {".parquet"}:
        df = pd.read_parquet(path_obj)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path_obj)
    else:
        raise ValueError(
            "Unsupported file extension for metaorders: "
            f"{path_obj.name} (expected .csv, .parquet, .pkl, or .pickle)"
        )

    # Basic sanity checks / conversions
    if "Date" not in df.columns:
        print(f"Adding 'Date' column from 'Period' for file: {path_obj}")
        df["Date"] = df["Period"].apply(extract_date)

    df = df.copy()
    # Parquet writers generally expect a real datetime dtype, not object-dtype
    # Python `datetime.date` values.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

    required = ["ISIN", "Q", "Direction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df


def main() -> None:
    """
    Summary
    -------
    Run the full crowding/imbalance analysis pipeline for prop vs client metaorders.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - All inputs/outputs are controlled via `config_ymls/crowding_analysis.yml`.
    - The script may overwrite the input parquet files to persist computed
      imbalance/return columns.

    Examples
    --------
    >>> # From the repository root:
    >>> # python scripts/crowding_analysis.py
    """
    log_path = OUTPUT_DIR / "logs" / "crowding_analysis.log"
    logger = setup_file_logger(Path(__file__).stem, log_path, mode="a")
    with PrintTee(logger):
        print(f"Using parquet directory: {PARQUET_DIR}")
        ensure_plot_dirs(PLOT_OUTPUT_DIRS)

        # Load data
        metaorders_proprietary = load_metaorders(PROP_PATH)
        metaorders_non_proprietary = load_metaorders(CLIENT_PATH)
        metaorders_proprietary["Group"] = "prop"
        metaorders_non_proprietary["Group"] = "client"

        # Add within-group and cross-group imbalances, then persist if we added anything new
        prop_updated = False
        client_updated = False

        if "imbalance_local" in metaorders_proprietary.columns:
            print(
                "\n[Prop vs Non-Prop] Daily imbalance already present in proprietary metaorders dataframe. "
                "Skipping within-group calculation."
            )
        else:
            metaorders_proprietary = add_daily_imbalance(metaorders_proprietary)
            prop_updated = True

        if "imbalance_local" in metaorders_non_proprietary.columns:
            print(
                "\n[Prop vs Non-Prop] Daily imbalance already present in non-proprietary metaorders dataframe. "
                "Skipping within-group calculation."
            )
        else:
            metaorders_non_proprietary = add_daily_imbalance(metaorders_non_proprietary)
            client_updated = True

        prop_env_col = "imbalance_client_env"
        client_env_col = "imbalance_prop_env"

        if prop_env_col in metaorders_proprietary.columns:
            print(
                "\n[Prop vs Non-Prop] Cross-group imbalance already present in proprietary metaorders dataframe. "
                "Skipping client-environment calculation."
            )
        else:
            metaorders_proprietary = attach_environment_imbalance(
                metaorders_proprietary,
                metaorders_non_proprietary,
                new_col=prop_env_col,
            )
            prop_updated = True

        if client_env_col in metaorders_non_proprietary.columns:
            print(
                "\n[Prop vs Non-Prop] Cross-group imbalance already present in non-proprietary metaorders dataframe. "
                "Skipping proprietary-environment calculation."
            )
        else:
            metaorders_non_proprietary = attach_environment_imbalance(
                metaorders_non_proprietary,
                metaorders_proprietary,
                new_col=client_env_col,
            )
            client_updated = True

        member_env_col = "imbalance_client_member_env"
        if member_env_col in metaorders_proprietary.columns:
            print(
                "\n[Prop vs Non-Prop] Member-level client imbalance already present in proprietary metaorders dataframe. "
                "Skipping member-environment calculation."
            )
        else:
            metaorders_proprietary = attach_member_client_imbalance(
                metaorders_proprietary,
                metaorders_non_proprietary,
                new_col=member_env_col,
            )
            prop_updated = True

        if ATTACH_DAILY_RETURNS:
            all_isins: List[str] = sorted(
                set(metaorders_proprietary["ISIN"].astype(str)).union(
                    metaorders_non_proprietary["ISIN"].astype(str)
                )
            )
            daily_returns = build_daily_returns_lookup(
                RETURNS_DATA_DIR,
                isins=all_isins,
                trading_hours=RETURNS_TRADING_HOURS,
            )
            if not daily_returns and (DAILY_RETURN_COL in metaorders_proprietary.columns) and (
                DAILY_RETURN_COL in metaorders_non_proprietary.columns
            ):
                print("\n[Daily returns] Lookup is empty; keeping existing daily return columns unchanged.")
            else:
                metaorders_proprietary, prop_ret_updated = attach_daily_returns_column(
                    metaorders_proprietary,
                    daily_returns,
                    new_col=DAILY_RETURN_COL,
                )
                metaorders_non_proprietary, client_ret_updated = attach_daily_returns_column(
                    metaorders_non_proprietary,
                    daily_returns,
                    new_col=DAILY_RETURN_COL,
                )
                prop_updated = prop_updated or prop_ret_updated
                client_updated = client_updated or client_ret_updated

        if prop_updated:
            metaorders_proprietary.to_parquet(PROP_PATH)
            print(f"\n[Prop vs Non-Prop] Saved proprietary metaorders with imbalance columns to: {PROP_PATH}")
        if client_updated:
            metaorders_non_proprietary.to_parquet(CLIENT_PATH)
            print(f"\n[Prop vs Non-Prop] Saved non-proprietary metaorders with imbalance columns to: {CLIENT_PATH}")
        if not prop_updated and not client_updated:
            print("\n[Prop vs Non-Prop] No new columns to persist on disk.")

        if ATTACH_DAILY_RETURNS and PLOT_IMBALANCE_VS_RETURNS:
            plot_imbalance_vs_daily_return(
                metaorders_proprietary,
                metaorders_non_proprietary,
                out_path=PLOT_DIR / "imbalance_vs_daily_returns.png",
                imbalance_col="imbalance_local",
                return_col=DAILY_RETURN_COL,
            )

        run_crowding_vs_part_rate_analysis(
            prop_path=PROP_PATH,
            client_path=CLIENT_PATH,
            output_base_dir=OUTPUT_DIR,
            image_base_dir=IMG_BASE_DIR,
            config_path=_CONFIG_PATH,
        )

        if DISTRIBUTIONS_IMBALANCE:
            plot_imbalance_distributions(
                metaorders_proprietary,
                metaorders_non_proprietary,
                plot_dir=PLOT_DIR,
                bins=IMBALANCE_HIST_BINS,
                use_kde=IMBALANCE_USE_KDE,
                kde_bw_adjust=IMBALANCE_KDE_BW_ADJUST,
            )

        if ACF_IMBALANCE:
            plot_direction_autocorrelation_per_isin(
                metaorders_proprietary,
                metaorders_non_proprietary,
                out_dir=PNG_DIR / ACF_OUTPUT_DIRNAME,
                max_lag=ACF_MAX_LAG,
                n_bootstrap=ACF_BOOTSTRAP_SAMPLES,
            )

        # Global analysis on full sample
        analyze_flow(
            metaorders_proprietary,
            "Proprietary metaorders",
            alpha=ALPHA,
        )
        analyze_flow(
            metaorders_non_proprietary,
            "Non-proprietary (client) metaorders",
            alpha=ALPHA,
        )

        # Daily time-series analysis + plots
        run_daily_crowding_analysis(
            metaorders_proprietary,
            metaorders_non_proprietary,
            alpha=ALPHA,
            min_n=MIN_N,
            out_prefix=str(PLOT_DIR / "daily_crowding"),
            make_plots=True,
            smoothing_days=SMOOTHING_DAYS,
        )

        # Cross-group crowding: prop vs client environments
        run_cross_group_crowding_analysis(
            metaorders_proprietary,
            metaorders_non_proprietary,
            alpha=ALPHA,
            min_n=MIN_N,
            out_prefix=str(PLOT_DIR / "cross_crowding"),
            make_plots=True,
            smoothing_days=SMOOTHING_DAYS,
        )

        # Versus all others (prop + client) with self-exclusion
        run_all_vs_all_crowding_analysis(
            metaorders_proprietary,
            metaorders_non_proprietary,
            alpha=ALPHA,
            min_n=MIN_N,
            out_prefix=str(PLOT_DIR / "all_vs_all_crowding"),
            make_plots=True,
            smoothing_days=SMOOTHING_DAYS,
        )

        run_member_level_prop_client_crowding_analysis(
            metaorders_proprietary,
            metaorders_non_proprietary,
            env_col="imbalance_client_member_env",
            alpha=ALPHA,
            min_metaorders_per_member=MIN_METAORDERS_PER_MEMBER,
            out_prefix=str(PLOT_DIR / "member_prop_client_crowding"),
            make_plots=True,
        )

        run_daily_count_imbalance_analysis(
            metaorders_proprietary,
            metaorders_non_proprietary,
            out_prefix=str(PLOT_DIR / "daily_counts"),
            make_plots=True,
        )


if __name__ == "__main__":
    main()
