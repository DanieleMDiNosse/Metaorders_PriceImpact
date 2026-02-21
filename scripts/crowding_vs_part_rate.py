#!/usr/bin/env python3
"""
Crowding vs participation rate (eta) analysis for metaorders.

This script tests whether metaorders with higher (lower) participation rate
experience higher (lower) crowding, using multiple imbalance ("environment")
definitions and two complementary crowding metrics:

1) Crowding intensity: |imbalance|
2) Crowding alignment: Direction * imbalance  (positive = trade with the crowd)

It produces:
- bin-level curves vs participation-rate bins, with cluster bootstrap CIs
  (clustered by Date),
- effect sizes (top-minus-bottom bin) and monotonic trend tests,
- optional interaction regressions of Direction on imbalance, with an
  imbalance x log(eta) term and clustered standard errors (statsmodels),
- optional 2D heatmaps conditioning jointly on (Q/V, eta),
- optional within-Date permutation placebos.

The script is standalone by design: it does not import metaorder_statistics.py
to avoid side effects (e.g., monkey-patching print).

Usage
-----
Activate the repository's main conda environment, then run:

    python scripts/crowding_vs_part_rate.py --dataset-name ftsemib

Or override paths:

    python scripts/crowding_vs_part_rate.py --prop-path out_files/ftsemib/...parquet \
        --client-path out_files/ftsemib/...parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repository-root imports (e.g., `moimpact`) work when running
# `python scripts/crowding_vs_part_rate.py` from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise ImportError("Missing dependency: pyyaml is required to read config defaults.") from exc

from moimpact.config import format_path_template, resolve_repo_path
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
    COLOR_PROPRIETARY,
    PlotOutputDirs,
    ensure_plot_dirs,
    make_plot_output_dirs,
    save_plotly_figure as _save_plotly_figure,
)

_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config_ymls" / "metaorder_statistics.yml"

# Canonical column names in this repository's metaorder parquet outputs.
COL_ISIN = "ISIN"
COL_DATE = "Date"
COL_PERIOD = "Period"
COL_DIR = "Direction"
COL_Q = "Q"
COL_ETA = "Participation Rate"
COL_QV = "Q/V"
COL_VTV = "Vt/V"
COL_DAILY_VOL = "Daily Vol"
TICK_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 15
LEGEND_FONT_SIZE = 12
COLOR_NOISE_BAND = "rgba(107,114,128,0.22)"

try:
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
except ImportError:
    # Plotly is optional for this script; PNG/HTML exports are guarded elsewhere.
    pass


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
    This workflow exports title-less figures to keep labels in paper captions.
    """
    fig.update_layout(title=None)
    return _save_plotly_figure(fig, *args, **kwargs)


@dataclass(frozen=True)
class ResolvedPaths:
    """Resolved input/output paths and key identifiers."""

    dataset_name: str
    prop_path: Path
    client_path: Path
    out_dir: Path
    img_dir: Path
    config_path: Path


@dataclass(frozen=True)
class BinResults:
    """Container for bin-level summaries, Date-cluster panels, and bootstrap draws."""

    summary: pd.DataFrame
    daily_panel: pd.DataFrame
    sums_date: pd.DataFrame
    stat_cols: List[str]
    boot_date: Optional[Dict[str, np.ndarray]]
    edges: np.ndarray


def load_yaml_defaults(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration defaults if available.

    Parameters
    ----------
    config_path : Path
        Path to a YAML config file (expected to be a mapping).

    Returns
    -------
    cfg : dict
        Parsed YAML content (empty dict if file missing).

    Notes
    -----
    This script uses `config_ymls/metaorder_statistics.yml` only for convenient
    defaults (e.g., DATASET_NAME and base output/image paths). CLI arguments always
    override YAML values.

    Examples
    --------
    >>> from pathlib import Path
    >>> cfg = load_yaml_defaults(Path("config_ymls/metaorder_statistics.yml"))
    >>> isinstance(cfg, dict)
    True
    """
    if not config_path.exists():
        return {}
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a YAML mapping (dict): {config_path}")
    return cfg


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_repo_path(_REPO_ROOT, value)


def _format_path_template(template: str, context: Mapping[str, str]) -> str:
    return format_path_template(template, context)


def resolve_paths(cfg: Mapping[str, Any], args: argparse.Namespace) -> ResolvedPaths:
    """
    Resolve dataset name, input parquet paths, and output directories.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        YAML configuration mapping (may be empty).
    args : argparse.Namespace
        Parsed CLI args.

    Returns
    -------
    paths : ResolvedPaths
        Concrete paths for inputs and outputs.

    Notes
    -----
    Defaults follow the same conventions as metaorder_statistics.py:
    `out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_member_...parquet`.

    Examples
    --------
    >>> import argparse
    >>> ns = argparse.Namespace(dataset_name="ftsemib", prop_path=None, client_path=None, output_file_path="out_files", img_output_path="images", analysis_tag="crowding_vs_part_rate")
    >>> paths = resolve_paths({}, ns)
    >>> paths.dataset_name == "ftsemib"
    True
    """
    dataset_name = str(args.dataset_name or cfg.get("DATASET_NAME") or "ftsemib")
    path_context = {"DATASET_NAME": dataset_name}
    out_base_cfg = str(args.output_file_path or cfg.get("OUTPUT_FILE_PATH") or "out_files/{DATASET_NAME}")
    img_base_cfg = str(args.img_output_path or cfg.get("IMG_OUTPUT_PATH") or "images/{DATASET_NAME}")
    out_base = _resolve_repo_path(_format_path_template(out_base_cfg, path_context))
    img_base = _resolve_repo_path(_format_path_template(img_base_cfg, path_context))

    prop_default = out_base / "metaorders_info_sameday_filtered_member_proprietary.parquet"
    client_default = out_base / "metaorders_info_sameday_filtered_member_non_proprietary.parquet"

    prop_path = _resolve_repo_path(args.prop_path) if args.prop_path else prop_default
    client_path = _resolve_repo_path(args.client_path) if args.client_path else client_default

    out_dir = out_base / str(args.analysis_tag)
    img_dir = img_base / str(args.analysis_tag)

    config_path = _resolve_repo_path(args.config_path) if args.config_path else _DEFAULT_CONFIG_PATH

    return ResolvedPaths(
        dataset_name=dataset_name,
        prop_path=prop_path,
        client_path=client_path,
        out_dir=out_dir,
        img_dir=img_dir,
        config_path=config_path,
    )


def _read_parquet_with_fallback(path: Path, columns: Optional[Sequence[str]]) -> pd.DataFrame:
    """
    Read a parquet file, optionally projecting columns, with a robust fallback.

    Parameters
    ----------
    path : Path
        Parquet file path.
    columns : Optional[Sequence[str]]
        Columns to read. If None, reads all columns.

    Returns
    -------
    df : pd.DataFrame
        Loaded data.

    Notes
    -----
    When a column projection fails (e.g., missing column), we fall back to
    reading the full parquet and then selecting the available subset. This
    keeps the script robust across minor schema differences.

    Examples
    --------
    >>> import pandas as pd
    >>> # Example is illustrative; requires a real parquet on disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    if columns is None:
        return pd.read_parquet(path)
    cols = list(columns)
    try:
        return pd.read_parquet(path, columns=cols)
    except Exception:
        # Prefer a schema-based retry to avoid loading large blob columns when a
        # minor schema difference exists (e.g., Date missing but Period present).
        try:
            import pyarrow.parquet as pq  # type: ignore

            available = set(pq.ParquetFile(path).schema_arrow.names)
            cols2 = [c for c in cols if c in available]
            if cols2:
                return pd.read_parquet(path, columns=cols2)
        except Exception:
            pass
        df = pd.read_parquet(path)
        keep = [c for c in cols if c in df.columns]
        return df[keep].copy()


def validate_required_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    """
    Validate that a DataFrame contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    required : Sequence[str]
        Columns that must be present.
    label : str
        Human-readable label for error messages (e.g., "prop" or "client").

    Returns
    -------
    None

    Notes
    -----
    This is a hard validation step; missing columns raise a KeyError with the
    full missing set.

    Examples
    --------
    >>> import pandas as pd
    >>> validate_required_columns(pd.DataFrame({"a":[1]}), ["a"], "demo")
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"[{label}] Missing required columns: {missing}")


def _period_start_ns(period_value: Any) -> Optional[int]:
    """Extract the start timestamp (epoch-nanoseconds) from a Period field.

    Parameters
    ----------
    period_value : Any
        Value from the `Period` column. In this repository it is typically a
        2-element array-like `[start_ns, end_ns]` (see metaorder_computation.py).

    Returns
    -------
    start_ns : Optional[int]
        Start timestamp in epoch-nanoseconds, or None if it cannot be parsed.

    Notes
    -----
    - `metaorder_computation.py` stores Period endpoints as epoch-nanoseconds
      (Python ints) so parquet writers/readers can serialize them reliably.
    - When reading back, pandas may materialize each Period entry as a
      `numpy.ndarray` of int64.
    - We treat scalar integers as already being epoch-nanoseconds.

    Examples
    --------
    >>> _period_start_ns([1, 2])
    1
    >>> import numpy as np
    >>> _period_start_ns(np.array([3, 4]))
    3
    """
    if period_value is None:
        return None
    if isinstance(period_value, float) and np.isnan(period_value):
        return None
    if isinstance(period_value, (list, tuple, np.ndarray)):
        if len(period_value) == 0:
            return None
        try:
            return int(period_value[0])
        except Exception:
            return None
    if isinstance(period_value, (np.integer, int)):
        return int(period_value)
    if isinstance(period_value, pd.Timestamp):
        return int(period_value.value)
    if isinstance(period_value, str):
        s = period_value.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return None
            first = inner.split(",")[0].strip()
            try:
                return int(first)
            except Exception:
                return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def ensure_date_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Ensure a normalized Date column exists, deriving it from Period if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Metaorder dataframe (typically loaded from the `metaorders_info_*` parquets).
    label : str
        Human-readable label for error messages (e.g., "prop" or "client").

    Returns
    -------
    out : pd.DataFrame
        DataFrame guaranteed to contain `Date` as `datetime64[ns]` normalized to
        midnight.

    Notes
    -----
    - Some repository parquet outputs (from `metaorder_computation.py`) do not
      include a `Date` column. In that case we derive `Date` from the first
      endpoint of `Period` (start timestamp) and normalize it.
    - This assumes metaorders are single-day ("sameday") by construction.

    Examples
    --------
    >>> import pandas as pd
    >>> demo = pd.DataFrame({"ISIN": ["X"], "Period": [[1, 2]]})
    >>> out = ensure_date_column(demo, label="demo")
    >>> "Date" in out.columns
    True
    """
    if COL_DATE in df.columns:
        out = df.copy()
        out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce").dt.normalize()
        return out
    if COL_PERIOD not in df.columns:
        raise KeyError(f"[{label}] Missing {COL_DATE} and {COL_PERIOD}; cannot infer trading date.")

    out = df.copy()
    start_ns = out[COL_PERIOD].apply(_period_start_ns)
    out[COL_DATE] = pd.to_datetime(start_ns, errors="coerce").dt.normalize()
    if out[COL_DATE].isna().any():
        bad = out.loc[out[COL_DATE].isna(), COL_PERIOD].head(3).tolist()
        raise ValueError(f"[{label}] Failed to infer Date from Period for some rows. Examples: {bad}")
    return out


def compute_imbalance_local(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (COL_ISIN, COL_DATE),
    side_col: str = COL_DIR,
    vol_col: str = COL_Q,
    out_col: str = "imbalance_local",
) -> pd.DataFrame:
    """
    Compute leave-one-out signed-volume imbalance within groups.

    Parameters
    ----------
    df : pd.DataFrame
        Metaorder-level data containing `group_cols`, `side_col`, and `vol_col`.
        Each row corresponds to a metaorder.
    group_cols : Sequence[str]
        Columns defining the environment (default: (ISIN, Date)).
    side_col : str
        Direction column name (expected +/-1).
    vol_col : str
        Volume column name (metaorder volume Q).
    out_col : str
        Output imbalance column name.

    Returns
    -------
    out : pd.DataFrame
        Copy of df with an added `out_col`.

    Notes
    -----
    For each row i within a group G:
        imbalance_i = sum_{j in G, j!=i} Q_j * D_j / sum_{j in G, j!=i} Q_j
    If a group has a single row (no "others"), the denominator is zero and the
    imbalance is set to NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> demo = pd.DataFrame({\"ISIN\":[\"X\",\"X\"],\"Date\":[\"2024-01-01\",\"2024-01-01\"],\"Q\":[10,5],\"Direction\":[1,-1]})
    >>> out = compute_imbalance_local(demo)
    >>> \"imbalance_local\" in out.columns
    True
    """
    validate_required_columns(df, list(group_cols) + [side_col, vol_col], label="imbalance_local")
    out = df.copy()
    out["__Q__"] = pd.to_numeric(out[vol_col], errors="coerce")
    out["__D__"] = pd.to_numeric(out[side_col], errors="coerce")
    out["__QD__"] = out["__Q__"].to_numpy(dtype=float) * out["__D__"].to_numpy(dtype=float)

    grouped = out.groupby(list(group_cols), dropna=False, sort=False)
    total_q = grouped["__Q__"].transform("sum")
    total_qd = grouped["__QD__"].transform("sum")

    denom = total_q - out["__Q__"]
    numer = total_qd - out["__QD__"]
    out[out_col] = np.where(denom > 0, numer / denom, np.nan)

    return out.drop(columns=["__Q__", "__D__", "__QD__"])


def compute_environment_imbalance(
    source_df: pd.DataFrame,
    group_cols: Sequence[str],
    side_col: str = COL_DIR,
    vol_col: str = COL_Q,
    out_col: str = "imbalance_env",
) -> pd.DataFrame:
    """
    Compute signed-volume imbalance on an environment defined by `group_cols`.

    Parameters
    ----------
    source_df : pd.DataFrame
        Metaorders used to define the environment (e.g., the other group).
    group_cols : Sequence[str]
        Columns defining the environment (e.g., (ISIN, Date)).
    side_col : str
        Direction column name (expected +/-1).
    vol_col : str
        Volume column name (metaorder volume Q).
    out_col : str
        Name of the returned imbalance column.

    Returns
    -------
    env : pd.DataFrame
        DataFrame with columns `group_cols` + [out_col].

    Notes
    -----
    For each group g:
        imbalance(g) = sum_{i in g} Q_i * D_i / sum_{i in g} Q_i
    If sum(Q) == 0, imbalance is NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> demo = pd.DataFrame({\"ISIN\":[\"X\",\"X\"],\"Date\":[\"2024-01-01\",\"2024-01-01\"],\"Q\":[10,5],\"Direction\":[1,-1]})
    >>> env = compute_environment_imbalance(demo, group_cols=[\"ISIN\",\"Date\"], out_col=\"imb\")
    >>> set(env.columns) == {\"ISIN\",\"Date\",\"imb\"}
    True
    """
    validate_required_columns(source_df, list(group_cols) + [side_col, vol_col], label="environment_imbalance")
    tmp = source_df[list(group_cols) + [side_col, vol_col]].copy()
    tmp["__Q__"] = pd.to_numeric(tmp[vol_col], errors="coerce")
    tmp["__D__"] = pd.to_numeric(tmp[side_col], errors="coerce")
    tmp["__QD__"] = tmp["__Q__"].to_numpy(dtype=float) * tmp["__D__"].to_numpy(dtype=float)

    agg = (
        tmp.groupby(list(group_cols), dropna=False, sort=False)
        .agg(total_q=("__Q__", "sum"), total_qd=("__QD__", "sum"))
        .reset_index()
    )
    agg[out_col] = np.where(agg["total_q"] > 0, agg["total_qd"] / agg["total_q"], np.nan)
    return agg[list(group_cols) + [out_col]]


def attach_environment_imbalance(
    target_df: pd.DataFrame,
    environment_df: pd.DataFrame,
    group_cols: Sequence[str],
    out_col: str,
    side_col: str = COL_DIR,
    vol_col: str = COL_Q,
) -> pd.DataFrame:
    """
    Attach an environment imbalance computed on `environment_df` onto `target_df`.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame receiving the imbalance column.
    environment_df : pd.DataFrame
        DataFrame used to compute the environment imbalance.
    group_cols : Sequence[str]
        Environment keys (e.g., (ISIN, Date)).
    out_col : str
        Name of the attached imbalance column.
    side_col : str
        Direction column name.
    vol_col : str
        Volume column name.

    Returns
    -------
    out : pd.DataFrame
        target_df with `out_col` merged in.

    Notes
    -----
    This is used for cross-group crowding where the "environment" is defined by
    the other group's signed volume imbalance.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame({\"ISIN\":[\"X\"],\"Date\":[\"2024-01-01\"],\"Q\":[1],\"Direction\":[1]})
    >>> b = pd.DataFrame({\"ISIN\":[\"X\"],\"Date\":[\"2024-01-01\"],\"Q\":[2],\"Direction\":[-1]})
    >>> out = attach_environment_imbalance(a, b, group_cols=[\"ISIN\",\"Date\"], out_col=\"imb_env\")
    >>> \"imb_env\" in out.columns
    True
    """
    env = compute_environment_imbalance(
        environment_df,
        group_cols=group_cols,
        side_col=side_col,
        vol_col=vol_col,
        out_col=out_col,
    )
    out = target_df.drop(columns=[out_col], errors="ignore").merge(env, on=list(group_cols), how="left", sort=False)
    return out


def _safe_pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _quantile_bin_edges(values: pd.Series, n_bins: int) -> np.ndarray:
    """
    Compute quantile-based bin edges with duplicate-handling.

    Parameters
    ----------
    values : pd.Series
        Numeric values to bin.
    n_bins : int
        Target number of bins.

    Returns
    -------
    edges : np.ndarray
        Sorted unique edges of length >= 2.

    Notes
    -----
    If the data has many ties, some quantiles coincide and the number of bins
    is reduced automatically by taking unique edges.

    Examples
    --------
    >>> import pandas as pd
    >>> edges = _quantile_bin_edges(pd.Series([1,2,3,4,5]), n_bins=2)
    >>> len(edges) >= 2
    True
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        raise ValueError("Cannot bin: no finite values.")
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = v.quantile(qs, interpolation="linear").to_numpy(dtype=float)
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError("Cannot bin: all values identical after filtering.")
    return edges


def make_eta_bins(
    eta_prop: pd.Series,
    eta_client: pd.Series,
    n_bins: int,
    mode: str,
) -> Dict[str, Any]:
    """
    Build participation-rate bin edges for prop/client samples.

    Parameters
    ----------
    eta_prop : pd.Series
        Participation rates for proprietary metaorders (filtered to finite).
    eta_client : pd.Series
        Participation rates for client metaorders (filtered to finite).
    n_bins : int
        Desired number of bins (quantiles).
    mode : str
        One of {"pooled_quantiles", "group_quantiles"}.

    Returns
    -------
    bins : dict
        Dictionary with keys:
        - "mode"
        - "edges_prop" (np.ndarray)
        - "edges_client" (np.ndarray)

    Notes
    -----
    - `pooled_quantiles` uses pooled edges (same edges for both groups).
    - `group_quantiles` uses separate edges per group.

    Examples
    --------
    >>> import pandas as pd
    >>> bins = make_eta_bins(pd.Series([0.1,0.2]), pd.Series([0.3,0.4]), n_bins=2, mode="pooled_quantiles")
    >>> "edges_prop" in bins and "edges_client" in bins
    True
    """
    if mode not in {"pooled_quantiles", "group_quantiles"}:
        raise ValueError("mode must be one of: pooled_quantiles, group_quantiles")
    if mode == "pooled_quantiles":
        pooled = pd.concat([eta_prop, eta_client], ignore_index=True)
        edges = _quantile_bin_edges(pooled, n_bins=n_bins)
        return {"mode": mode, "edges_prop": edges, "edges_client": edges}
    edges_prop = _quantile_bin_edges(eta_prop, n_bins=n_bins)
    edges_client = _quantile_bin_edges(eta_client, n_bins=n_bins)
    return {"mode": mode, "edges_prop": edges_prop, "edges_client": edges_client}


def assign_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """
    Assign quantile-bin indices given explicit bin edges.

    Parameters
    ----------
    values : pd.Series
        Values to bin.
    edges : np.ndarray
        Sorted unique edges.

    Returns
    -------
    bins : pd.Series
        Integer bin indices (0..B-1) with pandas nullable dtype Int64.

    Notes
    -----
    Values outside the [min,max] edge range become NaN (rare if edges come from
    the same values). Missing values remain NaN.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> b = assign_bins(pd.Series([0.1, 0.2, 0.3]), edges=np.array([0.1, 0.2, 0.3]))
    >>> b.notna().sum() >= 1
    True
    """
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    codes = pd.cut(v, bins=edges, include_lowest=True, labels=False, right=True)
    return codes.astype("Int64")


def prepare_analysis_frame(
    df: pd.DataFrame,
    imb_col: str,
    eta_col: str = COL_ETA,
    direction_col: str = COL_DIR,
    eta_max: float = 1.0,
) -> pd.DataFrame:
    """
    Filter and augment a metaorder DataFrame for crowding-vs-eta analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input metaorders DataFrame.
    imb_col : str
        Column containing the imbalance measure to use.
    eta_col : str
        Participation-rate column name.
    direction_col : str
        Direction column name (+1 buy, -1 sell).
    eta_max : float
        Maximum participation rate to keep (default 1.0).

    Returns
    -------
    out : pd.DataFrame
        Filtered copy with added columns:
        - "__eta__", "__log_eta__", "__imb__", "__abs_imb__", "__align__"

    Notes
    -----
    - Keeps rows with 0 < eta <= eta_max, finite imbalance, and Direction in {-1,+1}.
    - `__align__ = Direction * imbalance` is the primary crowding metric.

    Examples
    --------
    >>> import pandas as pd
    >>> demo = pd.DataFrame({\"Participation Rate\":[0.1],\"Direction\":[1],\"imb\":[0.2]})
    >>> out = prepare_analysis_frame(demo.rename(columns={\"imb\":\"imbalance_local\"}), imb_col=\"imbalance_local\")
    >>> float(out[\"__align__\"].iloc[0]) == 0.2
    True
    """
    validate_required_columns(df, [eta_col, direction_col, imb_col], label="prepare")
    out = df.copy()

    out["__eta__"] = pd.to_numeric(out[eta_col], errors="coerce")
    out["__imb__"] = pd.to_numeric(out[imb_col], errors="coerce")
    out["__dir__"] = pd.to_numeric(out[direction_col], errors="coerce")

    mask = (
        np.isfinite(out["__eta__"].to_numpy(dtype=float))
        & np.isfinite(out["__imb__"].to_numpy(dtype=float))
        & np.isfinite(out["__dir__"].to_numpy(dtype=float))
        & (out["__eta__"] > 0)
        & (out["__eta__"] <= float(eta_max))
        & (out["__dir__"].isin([-1, 1]))
    )
    out = out.loc[mask].copy()

    out["__log_eta__"] = np.log(out["__eta__"].to_numpy(dtype=float))
    out["__abs_imb__"] = np.abs(out["__imb__"].to_numpy(dtype=float))
    out["__align__"] = out["__dir__"].to_numpy(dtype=float) * out["__imb__"].to_numpy(dtype=float)

    # Basic invariants (tolerant to floating noise).
    if not out.empty:
        if (out["__align__"].abs() > 1.0 + 1e-9).any():
            raise ValueError("Invariant violated: |Direction*imbalance| should be <= 1 (check imbalance definition).")

    return out


def compute_bin_summary(
    df: pd.DataFrame,
    bin_col: str,
    min_n: int,
) -> pd.DataFrame:
    """
    Compute per-bin summary statistics for a prepared analysis frame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `prepare_analysis_frame`, containing "__eta__", "__imb__", "__dir__",
        "__abs_imb__", and "__align__".
    bin_col : str
        Column with integer bin codes.
    min_n : int
        Minimum sample size to compute correlation in a bin.

    Returns
    -------
    summary : pd.DataFrame
        Per-bin statistics including n, bin-center eta, mean metrics, and corr(Direction, imbalance).

    Notes
    -----
    Correlations are set to NaN if the bin is too small or degenerate.

    Examples
    --------
    >>> import pandas as pd
    >>> demo = pd.DataFrame({\"__eta__\":[0.1,0.2],\"__imb__\":[0.2,0.1],\"__dir__\":[1,-1],\"__abs_imb__\":[0.2,0.1],\"__align__\":[0.2,-0.1],\"bin\":[0,0]})
    >>> s = compute_bin_summary(demo, bin_col=\"bin\", min_n=1)
    >>> int(s[\"n\"].iloc[0]) == 2
    True
    """
    validate_required_columns(df, ["__eta__", "__imb__", "__dir__", "__abs_imb__", "__align__", bin_col], label="bin_summary")
    g = df.groupby(bin_col, dropna=False, sort=True)
    rows = []
    for b, sub in g:
        if pd.isna(b):
            continue
        n = int(len(sub))
        eta_center = float(np.nanmedian(sub["__eta__"].to_numpy(dtype=float))) if n > 0 else float("nan")
        mean_align = float(np.nanmean(sub["__align__"].to_numpy(dtype=float))) if n > 0 else float("nan")
        mean_abs_imb = float(np.nanmean(sub["__abs_imb__"].to_numpy(dtype=float))) if n > 0 else float("nan")
        mean_eta = float(np.nanmean(sub["__eta__"].to_numpy(dtype=float))) if n > 0 else float("nan")
        corr = float("nan")
        if n >= max(3, int(min_n)):
            corr = _safe_pearson_corr(sub["__dir__"].to_numpy(dtype=float), sub["__imb__"].to_numpy(dtype=float))
        rows.append(
            {
                "bin": int(b),
                "n": n,
                "eta_center": eta_center,
                "eta_mean": mean_eta,
                "mean_align": mean_align,
                "mean_abs_imb": mean_abs_imb,
                "corr_dir_imb": corr,
            }
        )
    out = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
    return out


def compute_cluster_bin_sums(
    df: pd.DataFrame,
    cluster_cols: Sequence[str],
    bin_col: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute per-(cluster,bin) sufficient statistics for fast cluster bootstraps.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared analysis frame with "__eta__", "__imb__", "__dir__", "__abs_imb__", "__align__".
    cluster_cols : Sequence[str]
        Cluster keys (e.g., ("Date",) or ("ISIN","Date")).
    bin_col : str
        Integer bin column.

    Returns
    -------
    sums : pd.DataFrame
        One row per (cluster, bin) with sufficient statistics.
    stat_cols : List[str]
        Names of statistic columns in `sums` that can be aggregated by summation.

    Notes
    -----
    The returned stats allow computing per-bin means and Pearson correlation
    from aggregated sums in O(1) per bin.

    Examples
    --------
    >>> import pandas as pd
    >>> demo = pd.DataFrame({\"Date\":[\"2024-01-01\"],\"bin\":[0],\"__eta__\":[0.1],\"__imb__\":[0.2],\"__dir__\":[1],\"__abs_imb__\":[0.2],\"__align__\":[0.2]})
    >>> sums, stat_cols = compute_cluster_bin_sums(demo, cluster_cols=[\"Date\"], bin_col=\"bin\")
    >>> set(stat_cols).issubset(set(sums.columns))
    True
    """
    validate_required_columns(
        df,
        list(cluster_cols) + [bin_col, "__eta__", "__imb__", "__dir__", "__abs_imb__", "__align__"],
        label="cluster_bin_sums",
    )
    tmp = df[list(cluster_cols) + [bin_col, "__eta__", "__imb__", "__dir__", "__abs_imb__", "__align__"]].copy()
    tmp["__imb2__"] = tmp["__imb__"].to_numpy(dtype=float) ** 2
    tmp["__dir2__"] = tmp["__dir__"].to_numpy(dtype=float) ** 2
    tmp["__dir_imb__"] = tmp["__dir__"].to_numpy(dtype=float) * tmp["__imb__"].to_numpy(dtype=float)

    stat_cols = [
        "n",
        "sum_eta",
        "sum_abs_imb",
        "sum_align",
        "sum_dir",
        "sum_imb",
        "sum_dir_imb",
        "sum_dir2",
        "sum_imb2",
    ]

    agg = (
        tmp.groupby(list(cluster_cols) + [bin_col], dropna=False, sort=False)
        .agg(
            n=("__eta__", "size"),
            sum_eta=("__eta__", "sum"),
            sum_abs_imb=("__abs_imb__", "sum"),
            sum_align=("__align__", "sum"),
            sum_dir=("__dir__", "sum"),
            sum_imb=("__imb__", "sum"),
            sum_dir_imb=("__dir_imb__", "sum"),
            sum_dir2=("__dir2__", "sum"),
            sum_imb2=("__imb2__", "sum"),
        )
        .reset_index()
    )
    return agg, stat_cols


def _compute_metrics_from_sums(totals: np.ndarray, stat_cols: Sequence[str]) -> Dict[str, np.ndarray]:
    """
    Compute per-bin metrics from summed sufficient statistics.

    Parameters
    ----------
    totals : np.ndarray
        Array of shape (n_bins, n_stats) with summed statistics.
    stat_cols : Sequence[str]
        Stat column names aligned with the second axis of `totals`.

    Returns
    -------
    metrics : Dict[str, np.ndarray]
        Dict mapping metric name -> array of shape (n_bins,).

    Notes
    -----
    Metrics computed:
    - mean_align
    - mean_abs_imb
    - eta_mean
    - corr_dir_imb

    Examples
    --------
    >>> import numpy as np
    >>> cols = [\"n\",\"sum_eta\",\"sum_abs_imb\",\"sum_align\",\"sum_dir\",\"sum_imb\",\"sum_dir_imb\",\"sum_dir2\",\"sum_imb2\"]
    >>> totals = np.array([[10, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 10.0, 1.0]])
    >>> m = _compute_metrics_from_sums(totals, cols)
    >>> \"mean_align\" in m and m[\"mean_align\"].shape == (1,)
    True
    """
    idx = {name: i for i, name in enumerate(stat_cols)}
    n = totals[:, idx["n"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        eta_mean = totals[:, idx["sum_eta"]] / n
        mean_abs_imb = totals[:, idx["sum_abs_imb"]] / n
        mean_align = totals[:, idx["sum_align"]] / n

        mean_dir = totals[:, idx["sum_dir"]] / n
        mean_imb = totals[:, idx["sum_imb"]] / n
        exy = totals[:, idx["sum_dir_imb"]] / n
        ex2 = totals[:, idx["sum_dir2"]] / n
        ey2 = totals[:, idx["sum_imb2"]] / n

        cov = exy - mean_dir * mean_imb
        var_x = ex2 - mean_dir**2
        var_y = ey2 - mean_imb**2
        denom = np.sqrt(var_x * var_y)
        corr = np.where((n >= 3) & (denom > 0), cov / denom, np.nan)

    return {
        "n": n,
        "eta_mean": eta_mean,
        "mean_abs_imb": mean_abs_imb,
        "mean_align": mean_align,
        "corr_dir_imb": corr,
    }


def bootstrap_cluster_curves(
    sums: pd.DataFrame,
    cluster_cols: Sequence[str],
    bin_col: str,
    stat_cols: Sequence[str],
    n_bins: int,
    n_runs: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Cluster bootstrap per-bin curves using sufficient statistics.

    Parameters
    ----------
    sums : pd.DataFrame
        Output of `compute_cluster_bin_sums`.
    cluster_cols : Sequence[str]
        Cluster columns (defines bootstrap resampling units).
    bin_col : str
        Integer bin column.
    stat_cols : Sequence[str]
        Columns in `sums` to be aggregated by summation.
    n_bins : int
        Number of bins.
    n_runs : int
        Bootstrap replicates.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    curves : Dict[str, np.ndarray]
        Dict containing:
        - "rep_mean_align": (n_runs, n_bins)
        - "rep_mean_abs_imb": (n_runs, n_bins)
        - "rep_corr_dir_imb": (n_runs, n_bins)

    Notes
    -----
    Resampling is done with replacement over clusters. Each replicate computes
    metrics from the aggregated (resampled) sums, corresponding to a standard
    cluster bootstrap.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> demo = pd.DataFrame({\"Date\":[\"2024-01-01\"],\"bin\":[0],\"n\":[10],\"sum_eta\":[1.0],\"sum_abs_imb\":[2.0],\"sum_align\":[0.0],\"sum_dir\":[0.0],\"sum_imb\":[0.0],\"sum_dir_imb\":[0.0],\"sum_dir2\":[10.0],\"sum_imb2\":[1.0]})
    >>> out = bootstrap_cluster_curves(demo, cluster_cols=[\"Date\"], bin_col=\"bin\", stat_cols=[\"n\",\"sum_eta\",\"sum_abs_imb\",\"sum_align\",\"sum_dir\",\"sum_imb\",\"sum_dir_imb\",\"sum_dir2\",\"sum_imb2\"], n_bins=1, n_runs=10, seed=0)
    >>> out[\"rep_mean_align\"].shape == (10, 1)
    True
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    validate_required_columns(sums, list(cluster_cols) + [bin_col] + list(stat_cols), label="bootstrap")

    # Map clusters to contiguous ids.
    if len(cluster_cols) == 1:
        cluster_vals = sums[cluster_cols[0]].astype("object").to_numpy()
        cluster_codes, cluster_uniques = pd.factorize(cluster_vals, sort=False)
    else:
        cluster_index = pd.MultiIndex.from_frame(sums[list(cluster_cols)].astype("object"), names=list(cluster_cols))
        cluster_codes, cluster_uniques = pd.factorize(cluster_index, sort=False)
    n_clusters = int(len(cluster_uniques))

    bin_codes = pd.to_numeric(sums[bin_col], errors="coerce").astype("Int64")
    if bin_codes.isna().any():
        raise ValueError("Found NaN bin codes in sums; check bin assignment.")
    bin_codes = bin_codes.astype(int).to_numpy()
    if bin_codes.min() < 0 or bin_codes.max() >= n_bins:
        raise ValueError(f"Bin codes out of range [0,{n_bins-1}].")

    stats_arr = np.zeros((n_clusters, n_bins, len(stat_cols)), dtype=float)
    stats_arr[cluster_codes, bin_codes, :] = sums[list(stat_cols)].to_numpy(dtype=float)
    flat = stats_arr.reshape(n_clusters, -1)

    rng = np.random.default_rng(int(seed))
    p = np.full(n_clusters, 1.0 / n_clusters)

    rep_mean_align = np.full((n_runs, n_bins), np.nan, dtype=float)
    rep_mean_abs_imb = np.full((n_runs, n_bins), np.nan, dtype=float)
    rep_corr_dir_imb = np.full((n_runs, n_bins), np.nan, dtype=float)

    for r in range(n_runs):
        weights = rng.multinomial(n_clusters, pvals=p)
        totals_flat = weights @ flat
        totals = totals_flat.reshape(n_bins, len(stat_cols))
        metrics = _compute_metrics_from_sums(totals, stat_cols=stat_cols)
        rep_mean_align[r, :] = metrics["mean_align"]
        rep_mean_abs_imb[r, :] = metrics["mean_abs_imb"]
        rep_corr_dir_imb[r, :] = metrics["corr_dir_imb"]

    return {
        "rep_mean_align": rep_mean_align,
        "rep_mean_abs_imb": rep_mean_abs_imb,
        "rep_corr_dir_imb": rep_corr_dir_imb,
        "n_clusters": np.array([n_clusters], dtype=float),
    }


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a percentile bootstrap confidence interval.

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap samples with replicate dimension first.
    alpha : float
        Significance level (default 0.05 for a 95% CI).

    Returns
    -------
    lo : np.ndarray
        Lower percentile bound.
    hi : np.ndarray
        Upper percentile bound.

    Notes
    -----
    NaNs are ignored per element; if all replicates are NaN for an element,
    the CI bounds are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> lo, hi = percentile_ci(np.array([[0.0, 1.0],[2.0, 3.0]]), alpha=0.05)
    >>> lo.shape == hi.shape == (2,)
    True
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    x = np.asarray(samples, dtype=float)
    lo = np.nanpercentile(x, 100.0 * (alpha / 2.0), axis=0)
    hi = np.nanpercentile(x, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lo, hi


def bootstrap_centered_noise_band(rep: Optional[np.ndarray], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a per-bin bootstrap "noise region" centered at zero.

    Parameters
    ----------
    rep : Optional[np.ndarray]
        Bootstrap replicate matrix with shape (n_runs, n_bins), or None.
    alpha : float
        Significance level used for percentile bounds (default 0.05 for 95%).

    Returns
    -------
    lo : np.ndarray
        Lower noise-band bound per bin (around zero).
    hi : np.ndarray
        Upper noise-band bound per bin (around zero).

    Notes
    -----
    The replicate curves are centered bin-wise before computing percentiles:
        noise_{r,b} = rep_{r,b} - mean_r(rep_{r,b})
    so the resulting band represents sampling variability ("noise") rather than
    the estimated signal level itself.

    Examples
    --------
    >>> import numpy as np
    >>> lo, hi = bootstrap_centered_noise_band(np.array([[0.0, 1.0], [2.0, 3.0]]), alpha=0.05)
    >>> lo.shape == hi.shape == (2,)
    True
    """
    if rep is None:
        return np.array([], dtype=float), np.array([], dtype=float)
    x = np.asarray(rep, dtype=float)
    if x.ndim != 2 or x.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    with np.errstate(invalid="ignore"):
        center = np.nanmean(x, axis=0, keepdims=True)
    centered = x - center
    lo, hi = percentile_ci(centered, alpha=alpha)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    # Avoid scipy dependency by rank-transforming with pandas.
    xs = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ys = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _safe_pearson_corr(xs, ys)


def compute_effect_sizes_from_bootstrap(
    rep: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    """
    Compute top-minus-bottom effect size and CI from bootstrap replicates.

    Parameters
    ----------
    rep : np.ndarray
        Replicates of shape (n_runs, n_bins) for a metric.
    alpha : float
        Significance level for CI (default 0.05).

    Returns
    -------
    out : Dict[str, float]
        Dict with keys: delta, lo, hi.

    Notes
    -----
    Uses replicate-wise differences: rep[:, -1] - rep[:, 0].

    Examples
    --------
    >>> import numpy as np
    >>> out = compute_effect_sizes_from_bootstrap(np.array([[0.0, 1.0],[1.0, 2.0]]), alpha=0.05)
    >>> \"delta\" in out and \"lo\" in out and \"hi\" in out
    True
    """
    rep = np.asarray(rep, dtype=float)
    diffs = rep[:, -1] - rep[:, 0]
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan")}
    lo, hi = percentile_ci(diffs[:, None], alpha=alpha)
    return {"delta": float(np.nanmean(diffs)), "lo": float(lo[0]), "hi": float(hi[0])}


def bootstrap_spearman_ci(
    rep: np.ndarray,
    x: np.ndarray,
    alpha: float,
    min_bins: int = 3,
) -> Dict[str, float]:
    """
    Compute a Spearman trend estimate and CI from bootstrap replicate curves.

    Parameters
    ----------
    rep : np.ndarray
        Metric replicates of shape (n_runs, n_bins).
    x : np.ndarray
        Bin x-coordinates of shape (n_bins,) (e.g., eta bin centers).
    alpha : float
        Significance level (default 0.05).
    min_bins : int
        Minimum number of finite bins required to compute Spearman in a replicate.

    Returns
    -------
    out : Dict[str, float]
        Dict with keys: spearman, lo, hi.

    Notes
    -----
    This treats each bootstrap replicate curve as one draw of the bin-level
    relationship and computes Spearman(x, y_rep) per replicate.

    Examples
    --------
    >>> import numpy as np
    >>> rep = np.array([[0.0, 1.0, 2.0],[0.0, 1.0, 2.0]])
    >>> out = bootstrap_spearman_ci(rep, x=np.array([1.0, 2.0, 3.0]), alpha=0.05)
    >>> \"spearman\" in out
    True
    """
    rep = np.asarray(rep, dtype=float)
    x = np.asarray(x, dtype=float)
    if rep.ndim != 2:
        raise ValueError("rep must be 2D (n_runs, n_bins).")
    if x.ndim != 1 or x.size != rep.shape[1]:
        raise ValueError("x must be 1D with length equal to n_bins.")

    vals: List[float] = []
    for r in range(rep.shape[0]):
        y = rep[r, :]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < int(min_bins):
            continue
        vals.append(_spearman_corr(x[mask], y[mask]))
    if not vals:
        return {"spearman": float("nan"), "lo": float("nan"), "hi": float("nan")}
    arr = np.asarray(vals, dtype=float)
    lo, hi = percentile_ci(arr[:, None], alpha=alpha)
    return {"spearman": float(np.nanmean(arr)), "lo": float(lo[0]), "hi": float(hi[0])}


def compute_daily_panel_from_sums(
    sums: pd.DataFrame,
    cluster_cols: Sequence[str],
    bin_col: str,
    min_n_corr: int,
) -> pd.DataFrame:
    """
    Convert cluster-bin sufficient stats to a per-cluster (e.g., per-day) panel of metrics.

    Parameters
    ----------
    sums : pd.DataFrame
        Output of `compute_cluster_bin_sums`.
    cluster_cols : Sequence[str]
        Cluster columns present in `sums` (e.g., ("Date",)).
    bin_col : str
        Bin column name.
    min_n_corr : int
        Minimum n required to compute corr(Direction, imbalance) for a cell.

    Returns
    -------
    panel : pd.DataFrame
        DataFrame with columns: cluster_cols, bin, n, eta_mean, mean_abs_imb, mean_align, corr_dir_imb.

    Notes
    -----
    This is used as a "daily bin panel" when cluster_cols = ("Date",).

    Examples
    --------
    >>> import pandas as pd
    >>> demo = pd.DataFrame({\"Date\":[\"2024-01-01\"],\"bin\":[0],\"n\":[10],\"sum_eta\":[1.0],\"sum_abs_imb\":[2.0],\"sum_align\":[0.0],\"sum_dir\":[0.0],\"sum_imb\":[0.0],\"sum_dir_imb\":[0.0],\"sum_dir2\":[10.0],\"sum_imb2\":[1.0]})
    >>> panel = compute_daily_panel_from_sums(demo, cluster_cols=[\"Date\"], bin_col=\"bin\", min_n_corr=3)
    >>> set([\"mean_align\",\"mean_abs_imb\",\"corr_dir_imb\"]).issubset(panel.columns)
    True
    """
    required = list(cluster_cols) + [bin_col] + [
        "n",
        "sum_eta",
        "sum_abs_imb",
        "sum_align",
        "sum_dir",
        "sum_imb",
        "sum_dir_imb",
        "sum_dir2",
        "sum_imb2",
    ]
    validate_required_columns(sums, required, label="daily_panel")

    n = pd.to_numeric(sums["n"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta_mean = pd.to_numeric(sums["sum_eta"], errors="coerce").to_numpy(dtype=float) / n
        mean_abs_imb = pd.to_numeric(sums["sum_abs_imb"], errors="coerce").to_numpy(dtype=float) / n
        mean_align = pd.to_numeric(sums["sum_align"], errors="coerce").to_numpy(dtype=float) / n

        mean_dir = pd.to_numeric(sums["sum_dir"], errors="coerce").to_numpy(dtype=float) / n
        mean_imb = pd.to_numeric(sums["sum_imb"], errors="coerce").to_numpy(dtype=float) / n
        exy = pd.to_numeric(sums["sum_dir_imb"], errors="coerce").to_numpy(dtype=float) / n
        ex2 = pd.to_numeric(sums["sum_dir2"], errors="coerce").to_numpy(dtype=float) / n
        ey2 = pd.to_numeric(sums["sum_imb2"], errors="coerce").to_numpy(dtype=float) / n

        cov = exy - mean_dir * mean_imb
        var_x = ex2 - mean_dir**2
        var_y = ey2 - mean_imb**2
        denom = np.sqrt(var_x * var_y)
        corr = np.where((n >= max(3, int(min_n_corr))) & (denom > 0), cov / denom, np.nan)

    out = sums[list(cluster_cols) + [bin_col]].copy()
    out["n"] = sums["n"].astype(int)
    out["eta_mean"] = eta_mean
    out["mean_abs_imb"] = mean_abs_imb
    out["mean_align"] = mean_align
    out["corr_dir_imb"] = corr
    return out


def _try_import_plotly() -> Optional[Any]:
    try:
        import plotly.graph_objects as go  # noqa: F401
        from plotly.subplots import make_subplots  # noqa: F401
    except Exception:
        return None
    return True


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8")


def _try_git_hash() -> Optional[str]:
    """
    Try to read the current git commit hash for the repository.

    Returns
    -------
    sha : Optional[str]
        Commit hash if available, else None.

    Notes
    -----
    This is used only for the run manifest to support reproducibility.

    Examples
    --------
    >>> isinstance(_try_git_hash(), (str, type(None)))
    True
    """
    import subprocess

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True).strip()
    except Exception:
        return None
    return out or None


def _ensure_dirs(paths: ResolvedPaths, dry_run: bool) -> None:
    if dry_run:
        return
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_plot_dirs(make_plot_output_dirs(paths.img_dir, use_subdirs=True))


def _get_imbalance_col(imbalance_kind: str, group: str) -> str:
    if imbalance_kind == "local":
        return "imbalance_local"
    if imbalance_kind == "cross":
        return "imbalance_client_env" if group == "prop" else "imbalance_prop_env"
    if imbalance_kind == "all":
        return "imbalance_all_others"
    raise ValueError("imbalance_kind must be one of: local, cross, all")


def _select_columns_for_load(run_regressions: bool, run_2d: bool) -> List[str]:
    cols = [COL_ISIN, COL_DATE, COL_PERIOD, COL_DIR, COL_Q, COL_ETA]
    if run_regressions or run_2d:
        cols.append(COL_QV)
    if run_regressions:
        cols.extend([COL_VTV, COL_DAILY_VOL])
    return cols


def _build_bin_results(
    df_prepared: pd.DataFrame,
    bin_col: str,
    edges: np.ndarray,
    min_n_bin: int,
    min_n_day_bin: int,
    bootstrap_runs: int,
    alpha: float,
    seed: int,
    build_daily_panel: bool,
) -> BinResults:
    """
    Compute per-bin point estimates, cluster bootstrap CIs, and daily panels.

    Parameters
    ----------
    df_prepared : pd.DataFrame
        Prepared analysis frame from `prepare_analysis_frame`.
    bin_col : str
        Temporary bin column name to create.
    edges : np.ndarray
        Bin edges for eta.
    min_n_bin : int
        Minimum sample size for computing bin-level correlations.
    min_n_day_bin : int
        Minimum sample size for computing per-day correlations within a (Date, bin) cell.
    bootstrap_runs : int
        Number of bootstrap replicates (0 disables bootstrapping).
    alpha : float
        CI alpha.
    seed : int
        RNG seed.
    build_daily_panel : bool
        Whether to build and return the (Date, bin) panel.

    Returns
    -------
    results : BinResults
        Container with summary, panels, and optional bootstrap draws.

    Notes
    -----
    - Point estimates are computed on the full filtered sample.
    - Cluster CIs (if enabled) are based on a standard cluster bootstrap using
      sufficient statistics.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> demo = pd.DataFrame({\"ISIN\":[\"X\",\"X\"],\"Date\":[pd.Timestamp(\"2024-01-01\")]*2,\"__eta__\":[0.1,0.2],\"__imb__\":[0.2,0.1],\"__dir__\":[1,-1],\"__abs_imb__\":[0.2,0.1],\"__align__\":[0.2,-0.1]})
    >>> res = _build_bin_results(demo, bin_col=\"bin\", edges=np.array([0.1,0.2]), min_n_bin=1, min_n_day_bin=1, bootstrap_runs=0, alpha=0.05, seed=0, build_daily_panel=False)
    >>> isinstance(res.summary, pd.DataFrame)
    True
    """
    df = df_prepared.copy()
    df[bin_col] = assign_bins(df["__eta__"], edges=edges)
    df = df.dropna(subset=[bin_col]).copy()
    df[bin_col] = df[bin_col].astype(int)
    n_bins = int(edges.size - 1)

    point = compute_bin_summary(df, bin_col=bin_col, min_n=min_n_bin)

    # Base summary frame (full bin range).
    summary = pd.DataFrame({"bin": np.arange(n_bins, dtype=int)})
    summary = summary.merge(point, on="bin", how="left", sort=True)
    summary["eta_edge_left"] = edges[:-1]
    summary["eta_edge_right"] = edges[1:]

    # Pre-create CI columns for consistent downstream handling.
    for metric in ["mean_align", "mean_abs_imb", "corr_dir_imb"]:
        summary[f"ci_date_{metric}_lo"] = np.nan
        summary[f"ci_date_{metric}_hi"] = np.nan

    sums_date = pd.DataFrame()
    daily_panel = pd.DataFrame()
    stat_cols: List[str] = []
    boot_date: Optional[Dict[str, np.ndarray]] = None

    # Date-cluster panel (used for both bootstrap inference and time-stability diagnostics).
    if int(bootstrap_runs) > 0 or bool(build_daily_panel):
        sums_date, stat_cols = compute_cluster_bin_sums(df, cluster_cols=[COL_DATE], bin_col=bin_col)
        if build_daily_panel:
            daily_panel = compute_daily_panel_from_sums(
                sums_date,
                cluster_cols=[COL_DATE],
                bin_col=bin_col,
                min_n_corr=int(min_n_day_bin),
            )

    # Bootstrap CIs (optional).
    if int(bootstrap_runs) > 0 and not sums_date.empty:
        boot_date = bootstrap_cluster_curves(
            sums_date,
            cluster_cols=[COL_DATE],
            bin_col=bin_col,
            stat_cols=stat_cols,
            n_bins=n_bins,
            n_runs=int(bootstrap_runs),
            seed=int(seed),
        )
        lo, hi = percentile_ci(boot_date["rep_mean_align"], alpha=alpha)
        summary["ci_date_mean_align_lo"] = lo
        summary["ci_date_mean_align_hi"] = hi
        lo, hi = percentile_ci(boot_date["rep_mean_abs_imb"], alpha=alpha)
        summary["ci_date_mean_abs_imb_lo"] = lo
        summary["ci_date_mean_abs_imb_hi"] = hi
        lo, hi = percentile_ci(boot_date["rep_corr_dir_imb"], alpha=alpha)
        summary["ci_date_corr_dir_imb_lo"] = lo
        summary["ci_date_corr_dir_imb_hi"] = hi

    return BinResults(
        summary=summary,
        daily_panel=daily_panel,
        sums_date=sums_date,
        stat_cols=stat_cols,
        boot_date=boot_date,
        edges=edges,
    )


def _plotly_curve_date_ci(
    x_prop: np.ndarray,
    y_prop: np.ndarray,
    lo_prop: np.ndarray,
    hi_prop: np.ndarray,
    noise_lo_prop: np.ndarray,
    noise_hi_prop: np.ndarray,
    x_client: np.ndarray,
    y_client: np.ndarray,
    lo_client: np.ndarray,
    hi_client: np.ndarray,
    noise_lo_client: np.ndarray,
    noise_hi_client: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out_stem: str,
    output_dirs: PlotOutputDirs,
) -> None:
    """
    Plot a prop+client curve with Date-cluster bootstrap CI bands (Plotly).

    Parameters
    ----------
    x_prop, y_prop : np.ndarray
        Proprietary x/y arrays (typically eta bin centers and a metric).
    lo_prop, hi_prop : np.ndarray
        Date-cluster CI band for proprietary series.
    noise_lo_prop, noise_hi_prop : np.ndarray
        Zero-centered bootstrap noise band for proprietary series.
    x_client, y_client : np.ndarray
        Client x/y arrays.
    lo_client, hi_client : np.ndarray
        Date-cluster CI band for client series.
    noise_lo_client, noise_hi_client : np.ndarray
        Zero-centered bootstrap noise band for client series.
    x_label, y_label, title : str
        Axis labels and plot title.
    out_stem : str
        Output filename stem (without extension).
    output_dirs : PlotOutputDirs
        Canonical output directories for HTML/PNG figures.

    Returns
    -------
    None

    Notes
    -----
    This function requires plotly; call only when plotly is available.

    Examples
    --------
    >>> # Requires plotly installed; example omitted.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    for group_cfg in [
        (
            x_prop,
            y_prop,
            lo_prop,
            hi_prop,
            noise_lo_prop,
            noise_hi_prop,
            COLOR_BAND_PROPRIETARY,
            COLOR_PROPRIETARY,
            "Proprietary",
            1,
        ),
        (
            x_client,
            y_client,
            lo_client,
            hi_client,
            noise_lo_client,
            noise_hi_client,
            COLOR_BAND_CLIENT,
            COLOR_CLIENT,
            "Client",
            2,
        ),
    ]:
        x_vals, y_vals, lo_vals, hi_vals, nlo_vals, nhi_vals, ci_color, line_color, group_label, col = group_cfg
        order = np.argsort(x_vals)
        x_ord = np.asarray(x_vals, dtype=float)[order]
        y_ord = np.asarray(y_vals, dtype=float)[order]
        lo_ord = np.asarray(lo_vals, dtype=float)[order]
        hi_ord = np.asarray(hi_vals, dtype=float)[order]
        nlo_ord = np.asarray(nlo_vals, dtype=float)[order] if np.asarray(nlo_vals).size == x_ord.size else np.array([], dtype=float)
        nhi_ord = np.asarray(nhi_vals, dtype=float)[order] if np.asarray(nhi_vals).size == x_ord.size else np.array([], dtype=float)

        if nlo_ord.size == x_ord.size:
            mask_noise = np.isfinite(x_ord) & np.isfinite(nlo_ord) & np.isfinite(nhi_ord)
            if mask_noise.any():
                fig.add_trace(
                    go.Scatter(
                        x=x_ord[mask_noise],
                        y=nhi_ord[mask_noise],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_ord[mask_noise],
                        y=nlo_ord[mask_noise],
                        mode="lines",
                        fill="tonexty",
                        fillcolor=COLOR_NOISE_BAND,
                        line=dict(width=0),
                        name="Bootstrap noise region",
                        legendgroup="bootstrap_noise",
                        showlegend=(col == 1),
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )

        mask_ci = np.isfinite(x_ord) & np.isfinite(lo_ord) & np.isfinite(hi_ord)
        if mask_ci.any():
            fig.add_trace(
                go.Scatter(
                    x=x_ord[mask_ci],
                    y=hi_ord[mask_ci],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_ord[mask_ci],
                    y=lo_ord[mask_ci],
                    mode="lines",
                    fill="tonexty",
                    fillcolor=ci_color,
                    line=dict(width=0),
                    name=f"{group_label} date-cluster CI",
                    legendgroup=f"{group_label}_ci",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )

        mask_line = np.isfinite(x_ord) & np.isfinite(y_ord)
        if mask_line.any():
            fig.add_trace(
                go.Scatter(
                    x=x_ord[mask_line],
                    y=y_ord[mask_line],
                    mode="lines+markers",
                    name=group_label,
                    legendgroup=group_label,
                    line=dict(color=line_color),
                    marker=dict(color=line_color),
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

        fig.add_hline(y=0.0, line=dict(color="rgba(75,85,99,0.75)", width=1, dash="dot"), row=1, col=col)
        fig.update_xaxes(title_text=x_label, row=1, col=col)

    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_layout(
        title=title,
        height=520,
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1.0),
    )
    save_plotly_figure(
        fig,
        stem=out_stem,
        dirs=output_dirs,
        write_html=True,
        write_png=True,
        strict_png=False,
    )


def _mpl_curve_date_ci(
    x_prop: np.ndarray,
    y_prop: np.ndarray,
    lo_prop: np.ndarray,
    hi_prop: np.ndarray,
    noise_lo_prop: np.ndarray,
    noise_hi_prop: np.ndarray,
    x_client: np.ndarray,
    y_client: np.ndarray,
    lo_client: np.ndarray,
    hi_client: np.ndarray,
    noise_lo_client: np.ndarray,
    noise_hi_client: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out_png: Path,
) -> None:
    """
    Plot a prop+client curve with Date-cluster bootstrap CI bands (matplotlib).

    Parameters
    ----------
    x_prop, y_prop : np.ndarray
        Proprietary x/y arrays.
    lo_prop, hi_prop : np.ndarray
        Date-cluster CI band for prop.
    noise_lo_prop, noise_hi_prop : np.ndarray
        Zero-centered bootstrap noise band for prop.
    x_client, y_client : np.ndarray
        Client x/y arrays.
    lo_client, hi_client : np.ndarray
        Date-cluster CI band for client.
    noise_lo_client, noise_hi_client : np.ndarray
        Zero-centered bootstrap noise band for client.
    x_label, y_label, title : str
        Labels and plot title.
    out_png : Path
        Output PNG path.

    Returns
    -------
    None

    Notes
    -----
    This is a non-interactive (Agg) plot suitable for scripts and servers.

    Examples
    --------
    >>> # Requires matplotlib installed; example omitted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    plot_cfg = [
        (axes[0], x_prop, y_prop, lo_prop, hi_prop, noise_lo_prop, noise_hi_prop, "tab:blue", "Proprietary"),
        (axes[1], x_client, y_client, lo_client, hi_client, noise_lo_client, noise_hi_client, "tab:orange", "Client"),
    ]
    for ax, x_vals, y_vals, lo_vals, hi_vals, nlo_vals, nhi_vals, color, group_label in plot_cfg:
        order = np.argsort(x_vals)
        x_ord = np.asarray(x_vals, dtype=float)[order]
        y_ord = np.asarray(y_vals, dtype=float)[order]
        lo_ord = np.asarray(lo_vals, dtype=float)[order]
        hi_ord = np.asarray(hi_vals, dtype=float)[order]
        nlo_ord = np.asarray(nlo_vals, dtype=float)[order] if np.asarray(nlo_vals).size == x_ord.size else np.array([], dtype=float)
        nhi_ord = np.asarray(nhi_vals, dtype=float)[order] if np.asarray(nhi_vals).size == x_ord.size else np.array([], dtype=float)

        if nlo_ord.size == x_ord.size:
            mask_noise = np.isfinite(x_ord) & np.isfinite(nlo_ord) & np.isfinite(nhi_ord)
            if mask_noise.any():
                ax.fill_between(
                    x_ord[mask_noise],
                    nlo_ord[mask_noise],
                    nhi_ord[mask_noise],
                    color="#6b7280",
                    alpha=0.22,
                    linewidth=0,
                    label="Bootstrap noise region",
                )

        mask_ci = np.isfinite(x_ord) & np.isfinite(lo_ord) & np.isfinite(hi_ord)
        if mask_ci.any():
            ax.fill_between(x_ord[mask_ci], lo_ord[mask_ci], hi_ord[mask_ci], color=color, alpha=0.2, linewidth=0, label="Date-cluster CI")

        mask_line = np.isfinite(x_ord) & np.isfinite(y_ord)
        if mask_line.any():
            ax.plot(x_ord[mask_line], y_ord[mask_line], color=color, marker="o", linewidth=1.6, label=group_label)

        ax.axhline(0.0, color="#4b5563", linestyle=":", linewidth=1.0, alpha=0.9)
        ax.set_xlabel(x_label)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        ax.legend()

    axes[0].set_ylabel(y_label)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def within_date_permutation_delta_align(
    df_prepared: pd.DataFrame,
    edges: np.ndarray,
    seed: int,
    n_runs: int,
    date_col: str = COL_DATE,
) -> Dict[str, Any]:
    """
    Run a within-Date permutation placebo for the top-minus-bottom mean alignment.

    Parameters
    ----------
    df_prepared : pd.DataFrame
        Prepared analysis frame from `prepare_analysis_frame` (must include Date and "__eta__", "__imb__", "__dir__").
    edges : np.ndarray
        Eta bin edges (defines bins; fixed during permutations).
    seed : int
        RNG seed.
    n_runs : int
        Number of permutation replicates.
    date_col : str
        Date column name.

    Returns
    -------
    out : Dict[str, Any]
        Dict with:
        - "delta_obs": observed top-minus-bottom mean alignment
        - "delta_perm": np.ndarray of permutation deltas (length n_runs)
        - "p_two_sided": two-sided permutation p-value with +1 smoothing

    Notes
    -----
    Null: within a trading day, the assignment of directions to metaorders is
    exchangeable. We permute Direction within each Date and recompute the
    top-minus-bottom difference in mean alignment across eta bins.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> demo = pd.DataFrame({\"Date\":[pd.Timestamp(\"2024-01-01\")]*4,\"__eta__\":[0.1,0.2,0.3,0.4],\"__imb__\":[0.1,0.1,0.1,0.1],\"__dir__\":[1,1,-1,-1],\"__abs_imb__\":[0.1]*4,\"__align__\":[0.1,0.1,-0.1,-0.1]})
    >>> out = within_date_permutation_delta_align(demo, edges=np.array([0.1,0.25,0.4]), seed=0, n_runs=10)
    >>> \"p_two_sided\" in out
    True
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    validate_required_columns(df_prepared, [date_col, "__eta__", "__imb__", "__dir__"], label="permutation")

    df = df_prepared.copy()
    df["__bin__"] = assign_bins(df["__eta__"], edges=edges)
    df = df.dropna(subset=["__bin__"]).copy()
    if df.empty:
        return {"delta_obs": float("nan"), "delta_perm": np.array([], dtype=float), "p_two_sided": float("nan")}

    n_bins = int(edges.size - 1)
    bin_codes = df["__bin__"].astype(int).to_numpy()
    imb = df["__imb__"].to_numpy(dtype=float)
    direction = df["__dir__"].to_numpy(dtype=float)

    # Factorize Date and sort by date for contiguous segments.
    date_codes, _ = pd.factorize(df[date_col], sort=False)
    order = np.argsort(date_codes, kind="mergesort")
    date_codes = date_codes[order]
    bin_codes = bin_codes[order]
    imb = imb[order]
    direction = direction[order]

    # Segment boundaries for each Date block.
    boundaries = np.flatnonzero(np.diff(date_codes) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [date_codes.size]])

    def mean_align_by_bin(dir_vec: np.ndarray) -> np.ndarray:
        align = dir_vec * imb
        sum_align = np.bincount(bin_codes, weights=align, minlength=n_bins).astype(float)
        count = np.bincount(bin_codes, minlength=n_bins).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(count > 0, sum_align / count, np.nan)
        return mean

    mean_obs = mean_align_by_bin(direction)
    delta_obs = float(mean_obs[-1] - mean_obs[0]) if np.isfinite(mean_obs[0]) and np.isfinite(mean_obs[-1]) else float("nan")

    rng = np.random.default_rng(int(seed))
    deltas = np.full(int(n_runs), np.nan, dtype=float)
    dir_perm = direction.copy()
    for r in range(int(n_runs)):
        # Permute within each Date segment.
        for s, e in zip(starts, ends):
            if e - s > 1:
                dir_perm[s:e] = rng.permutation(direction[s:e])
            else:
                dir_perm[s:e] = direction[s:e]
        mean_rep = mean_align_by_bin(dir_perm)
        deltas[r] = mean_rep[-1] - mean_rep[0]

    # Two-sided p-value with +1 smoothing.
    if not np.isfinite(delta_obs):
        p_two = float("nan")
    else:
        finite = np.isfinite(deltas)
        if not finite.any():
            p_two = float("nan")
        else:
            more_extreme = np.abs(deltas[finite]) >= abs(delta_obs)
            p_two = float((more_extreme.sum() + 1) / (finite.sum() + 1))

    return {"delta_obs": delta_obs, "delta_perm": deltas, "p_two_sided": p_two}


def compute_2d_heatmap_table(
    df_prepared: pd.DataFrame,
    edges_eta: np.ndarray,
    qv_col: str = COL_QV,
    n_qv_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute a 2D table of mean alignment on a (Q/V, eta) quantile grid.

    Parameters
    ----------
    df_prepared : pd.DataFrame
        Prepared analysis frame (must include "__eta__", "__align__" and `qv_col`).
    edges_eta : np.ndarray
        Eta bin edges (fixed).
    qv_col : str
        Column name for Q/V.
    n_qv_bins : int
        Number of Q/V quantile bins.

    Returns
    -------
    table : pd.DataFrame
        Long-form table with columns:
        - qv_bin, eta_bin, n, qv_center, eta_center, mean_align

    Notes
    -----
    This is a diagnostic to separate participation effects from size effects.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> demo = pd.DataFrame({\"__eta__\":[0.1,0.2],\"__align__\":[0.1,0.0],\"Q/V\":[1e-4, 2e-4]})
    >>> t = compute_2d_heatmap_table(demo, edges_eta=np.array([0.1,0.2]), n_qv_bins=1)
    >>> \"mean_align\" in t.columns
    True
    """
    validate_required_columns(df_prepared, ["__eta__", "__align__", qv_col], label="heatmap_2d")
    df = df_prepared.copy()
    df["__qv__"] = pd.to_numeric(df[qv_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df[(df["__qv__"] > 0) & np.isfinite(df["__qv__"].to_numpy(dtype=float))].copy()
    if df.empty:
        return pd.DataFrame()

    edges_qv = _quantile_bin_edges(df["__qv__"], n_bins=int(n_qv_bins))
    df["eta_bin"] = assign_bins(df["__eta__"], edges=edges_eta)
    df["qv_bin"] = assign_bins(df["__qv__"], edges=edges_qv)
    df = df.dropna(subset=["eta_bin", "qv_bin"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["eta_bin"] = df["eta_bin"].astype(int)
    df["qv_bin"] = df["qv_bin"].astype(int)

    g = df.groupby(["qv_bin", "eta_bin"], dropna=False, sort=False)
    table = g.agg(
        n=("__align__", "size"),
        mean_align=("__align__", "mean"),
        eta_center=("__eta__", "median"),
        qv_center=("__qv__", "median"),
    ).reset_index()
    return table


def plotly_heatmap_align(
    table: pd.DataFrame,
    title: str,
    out_stem: str,
    output_dirs: PlotOutputDirs,
) -> None:
    """
    Plot a Plotly heatmap for mean alignment on a 2D grid.

    Parameters
    ----------
    table : pd.DataFrame
        Output of `compute_2d_heatmap_table`.
    title : str
        Plot title.
    out_stem : str
        Output filename stem (without extension).
    output_dirs : PlotOutputDirs
        Canonical output directories for HTML/PNG figures.

    Returns
    -------
    None

    Notes
    -----
    Requires plotly.

    Examples
    --------
    >>> # Requires plotly installed; example omitted.
    """
    import plotly.graph_objects as go

    if table.empty:
        return

    # Pivot to matrix.
    pivot = table.pivot(index="qv_bin", columns="eta_bin", values="mean_align")
    # Centers: take median within each bin (from table rows).
    eta_centers = table.groupby("eta_bin", sort=True)["eta_center"].median()
    qv_centers = table.groupby("qv_bin", sort=True)["qv_center"].median()

    z = pivot.to_numpy(dtype=float)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=eta_centers.to_numpy(dtype=float),
            y=qv_centers.to_numpy(dtype=float),
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title="mean align"),
            hovertemplate="eta=%{x:.3g}<br>Q/V=%{y:.3g}<br>mean_align=%{z:.3g}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title=r"$\eta$ ", yaxis_title="Q/V", height=600, width=850)
    save_plotly_figure(
        fig,
        stem=out_stem,
        dirs=output_dirs,
        write_html=True,
        write_png=True,
        strict_png=False,
    )

def _maybe_run_regression(
    df_prepared: pd.DataFrame,
    imb_col: str,
    cluster: str,
    seed: int,
    sample_frac: float,
) -> pd.DataFrame:
    """
    Run the interaction regression if statsmodels is available.

    Parameters
    ----------
    df_prepared : pd.DataFrame
        Prepared analysis frame; may include control columns (Q/V, Vt/V, Daily Vol, Q).
    imb_col : str
        Name of imbalance column in the original df (for labeling only).
    cluster : str
        Cluster unit for standard errors. Only "date" is supported.
    seed : int
        RNG seed for optional subsampling.
    sample_frac : float
        Fraction of rows to keep (1.0 keeps all).

    Returns
    -------
    results : pd.DataFrame
        One-row DataFrame with coefficient estimates and inference.

    Notes
    -----
    Regression model:
        Direction ~ imb + imb*centered_log_eta + controls
    We demean by Date before fitting to absorb day-level confounding without
    adding thousands of dummy variables.

    Examples
    --------
    >>> import pandas as pd
    >>> _ = _maybe_run_regression(pd.DataFrame(), imb_col=\"imbalance_local\", cluster=\"date\", seed=0, sample_frac=1.0)
    """
    try:
        import statsmodels.api as sm
    except Exception:
        return pd.DataFrame([{"status": "skipped", "reason": "statsmodels_not_available"}])

    df = df_prepared.copy()
    if df.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "empty_sample"}])

    if not (0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in (0,1].")
    if sample_frac < 1.0:
        rng = np.random.default_rng(int(seed))
        keep = rng.random(len(df)) < float(sample_frac)
        df = df.loc[keep].copy()
        if df.empty:
            return pd.DataFrame([{"status": "skipped", "reason": "empty_after_subsample"}])

    # Controls: keep what is available.
    controls = []
    if COL_QV in df.columns:
        df["__log_qv__"] = np.log(pd.to_numeric(df[COL_QV], errors="coerce").replace([np.inf, -np.inf], np.nan))
        controls.append("__log_qv__")
    if COL_VTV in df.columns:
        df["__log_vtv__"] = np.log(pd.to_numeric(df[COL_VTV], errors="coerce").replace([np.inf, -np.inf], np.nan))
        controls.append("__log_vtv__")
    if COL_Q in df.columns:
        df["__log_q__"] = np.log(pd.to_numeric(df[COL_Q], errors="coerce").replace([np.inf, -np.inf], np.nan))
        controls.append("__log_q__")
    if COL_DAILY_VOL in df.columns:
        df["__daily_vol__"] = pd.to_numeric(df[COL_DAILY_VOL], errors="coerce").replace([np.inf, -np.inf], np.nan)
        controls.append("__daily_vol__")

    # Core regressors.
    df["__imb__"] = pd.to_numeric(df["__imb__"], errors="coerce")
    df["__dir__"] = pd.to_numeric(df["__dir__"], errors="coerce")
    df["__log_eta_c__"] = df["__log_eta__"] - float(np.nanmean(df["__log_eta__"].to_numpy(dtype=float)))
    df["__imb_x_logeta__"] = df["__imb__"].to_numpy(dtype=float) * df["__log_eta_c__"].to_numpy(dtype=float)

    if cluster != "date":
        raise ValueError('cluster must be "date" (other clusterings are disabled).')

    needed = ["__dir__", "__imb__", "__imb_x_logeta__", COL_DATE] + controls
    df = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if df.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "missing_values_after_controls"}])

    # Date demeaning (fixed-effect removal).
    for col in ["__dir__", "__imb__", "__imb_x_logeta__"] + controls:
        df[col] = df[col] - df.groupby(COL_DATE, dropna=False)[col].transform("mean")

    y = df["__dir__"].to_numpy(dtype=float)
    X_cols = ["__imb__", "__imb_x_logeta__"] + controls
    X = df[X_cols].to_numpy(dtype=float)
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X)
    res = model.fit()

    # Cluster SEs
    clusters = df[COL_DATE]

    res_cl = res.get_robustcov_results(cov_type="cluster", groups=clusters)

    param_names = ["const"] + X_cols
    out: Dict[str, Any] = {
        "status": "ok",
        "imbalance_col": imb_col,
        "n_obs": int(df.shape[0]),
        "n_clusters": int(pd.Series(clusters).nunique(dropna=False)),
    }
    for name, coef, se, pval in zip(param_names, res_cl.params, res_cl.bse, res_cl.pvalues):
        out[f"coef_{name}"] = float(coef)
        out[f"se_{name}"] = float(se)
        out[f"p_{name}"] = float(pval)
    return pd.DataFrame([out])


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Returns
    -------
    parser : argparse.ArgumentParser
        Configured argument parser.

    Notes
    -----
    The script is intended to be run as a research workflow. Defaults are
    conservative and overrideable.

    Examples
    --------
    >>> p = build_arg_parser()
    >>> isinstance(p, argparse.ArgumentParser)
    True
    """
    p = argparse.ArgumentParser(description="Crowding vs participation rate analysis (metaorders).")

    # Config and paths.
    p.add_argument(
        "--config-path",
        type=str,
        default="config_ymls/metaorder_statistics.yml",
        help="YAML config path for defaults. Default: config_ymls/metaorder_statistics.yml.",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset subfolder under out_files/images. Default: YAML DATASET_NAME if set, else ftsemib.",
    )
    p.add_argument(
        "--prop-path",
        type=str,
        default=None,
        help=(
            "Path to proprietary metaorder parquet. Default: "
            "out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_member_proprietary.parquet."
        ),
    )
    p.add_argument(
        "--client-path",
        type=str,
        default=None,
        help=(
            "Path to client (non-proprietary) metaorder parquet. Default: "
            "out_files/{DATASET_NAME}/metaorders_info_sameday_filtered_member_non_proprietary.parquet."
        ),
    )
    p.add_argument(
        "--output-file-path",
        type=str,
        default=None,
        help="Output base folder. Default: YAML OUTPUT_FILE_PATH if set, else out_files/{DATASET_NAME}.",
    )
    p.add_argument(
        "--img-output-path",
        type=str,
        default=None,
        help="Image base folder. Default: YAML IMG_OUTPUT_PATH if set, else images/{DATASET_NAME}.",
    )
    p.add_argument(
        "--analysis-tag",
        type=str,
        default="crowding_vs_part_rate",
        help="Output subfolder name. Default: crowding_vs_part_rate.",
    )

    # Analysis toggles.
    p.add_argument(
        "--imbalance-kind",
        type=str,
        default="local,cross",
        help="Comma-separated: local,cross,all. Default: local,cross.",
    )
    p.add_argument(
        "--eta-bins",
        type=int,
        default=10,
        help="Number of participation-rate quantile bins. Default: 10.",
    )
    p.add_argument(
        "--eta-binning",
        type=str,
        default="pooled_quantiles",
        choices=["pooled_quantiles", "group_quantiles"],
        help="How to build eta bin edges. Default: pooled_quantiles.",
    )
    p.add_argument(
        "--eta-max",
        type=float,
        default=1.0,
        help="Max participation rate to keep. Default: 1.0.",
    )
    p.add_argument(
        "--min-n-bin",
        type=int,
        default=100,
        help="Min n in bin to compute correlations. Default: 100.",
    )
    p.add_argument(
        "--min-n-day-bin",
        type=int,
        default=50,
        help="Min n in a (Date,bin) cell to compute correlations in the daily panel. Default: 50.",
    )
    p.add_argument(
        "--bootstrap-runs",
        type=int,
        default=1000,
        help="Bootstrap replicates for CIs. Default: 1000.",
    )
    p.add_argument(
        "--cluster-ci",
        type=str,
        default="date",
        choices=["date", "none"],
        help='Cluster unit for CIs (only "date" is supported). Use "none" to skip CI computation. Default: date.',
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="CI alpha (0.05 => 95%% CI). Default: 0.05.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility. Default: 0.",
    )

    # Optional analyses.
    p.add_argument(
        "--run-regressions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run interaction regressions (statsmodels). Default: enabled.",
    )
    p.add_argument(
        "--reg-cluster",
        type=str,
        default="date",
        choices=["date"],
        help='Cluster unit for regression SEs (only "date" is supported). Default: date.',
    )
    p.add_argument(
        "--reg-sample-frac",
        type=float,
        default=1.0,
        help="Row subsample fraction for regressions. Default: 1.0.",
    )
    p.add_argument(
        "--permutation-runs",
        type=int,
        default=0,
        help="Within-Date permutation placebo replicates. Default: 0 (disabled).",
    )
    p.add_argument(
        "--run-2d",
        action="store_true",
        help="Run 2D conditioning heatmaps (Q/V x eta). Default: disabled.",
    )
    p.add_argument(
        "--qv-bins",
        type=int,
        default=10,
        help="Number of Q/V quantile bins for 2D heatmaps. Default: 10.",
    )

    # Outputs.
    p.add_argument(
        "--plots",
        type=str,
        default="plotly",
        choices=["plotly", "matplotlib", "both"],
        help="Plot backend. Default: plotly.",
    )
    p.add_argument(
        "--write-parquet",
        action="store_true",
        help="Write large tables as parquet in addition to CSV. Default: disabled.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print plan; do not write outputs. Default: disabled.",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="Run a small smoke check on a subset of dates. Default: disabled.",
    )
    p.add_argument(
        "--self-check-n-dates",
        type=int,
        default=20,
        help="Number of random dates for --self-check. Default: 20.",
    )

    return p
def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the crowding-vs-participation analysis end-to-end.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        CLI arguments (defaults to sys.argv[1:] when None).

    Returns
    -------
    code : int
        Process exit code (0 on success).

    Notes
    -----
    This function orchestrates data loading, metric computation, bootstrapping,
    optional regressions/plots, and writing outputs.

    Examples
    --------
    >>> # Example is illustrative; requires the parquet inputs present.
    >>> # main([\"--dataset-name\",\"ftsemib\",\"--dry-run\"])  # doctest: +SKIP
    """
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_yaml_defaults(_resolve_repo_path(args.config_path) if args.config_path else _DEFAULT_CONFIG_PATH)
    paths = resolve_paths(cfg, args)

    imbalance_kinds = [s.strip() for s in str(args.imbalance_kind).split(",") if s.strip()]
    for k in imbalance_kinds:
        if k not in {"local", "cross", "all"}:
            raise ValueError(f"Unknown imbalance kind: {k}")

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if bool(args.self_check) and bool(args.dry_run):
        print("[warning] --self-check ignored because --dry-run is enabled.")

    if bool(args.self_check) and (not bool(args.dry_run)):
        suffix = f"self_check_{run_id}"
        paths = ResolvedPaths(
            dataset_name=paths.dataset_name,
            prop_path=paths.prop_path,
            client_path=paths.client_path,
            out_dir=paths.out_dir / suffix,
            img_dir=paths.img_dir / suffix,
            config_path=paths.config_path,
        )

    _ensure_dirs(paths, dry_run=bool(args.dry_run))
    img_output_dirs = make_plot_output_dirs(paths.img_dir, use_subdirs=True)

    # Manifest first (for reproducibility).
    manifest = {
        "run_id": run_id,
        "timestamp": dt.datetime.now().isoformat(),
        "git_hash": _try_git_hash(),
        "dataset_name": paths.dataset_name,
        "prop_path": str(paths.prop_path),
        "client_path": str(paths.client_path),
        "out_dir": str(paths.out_dir),
        "img_dir": str(paths.img_dir),
        "html_dir": str(img_output_dirs.html_dir),
        "png_dir": str(img_output_dirs.png_dir),
        "args": vars(args),
    }
    if not args.dry_run:
        _write_json(paths.out_dir / "run_manifest.json", manifest)

    # Load inputs with minimal columns.
    cols = _select_columns_for_load(run_regressions=bool(args.run_regressions), run_2d=bool(args.run_2d))
    prop = _read_parquet_with_fallback(paths.prop_path, columns=cols + ["imbalance_local", "imbalance_client_env"])
    client = _read_parquet_with_fallback(paths.client_path, columns=cols + ["imbalance_local", "imbalance_prop_env"])

    prop = ensure_date_column(prop, label="prop")
    client = ensure_date_column(client, label="client")

    validate_required_columns(prop, [COL_ISIN, COL_DATE, COL_DIR, COL_Q, COL_ETA], label="prop")
    validate_required_columns(client, [COL_ISIN, COL_DATE, COL_DIR, COL_Q, COL_ETA], label="client")

    # Ensure Date is datetime64[ns]
    for df, label in [(prop, "prop"), (client, "client")]:
        if not np.issubdtype(df[COL_DATE].dtype, np.datetime64):
            df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
        if df[COL_DATE].isna().any():
            raise ValueError(f"[{label}] Found NaN Date after parsing.")

    if bool(args.self_check) and (not bool(args.dry_run)):
        uniq_dates = np.array(sorted(pd.unique(prop[COL_DATE])))
        if uniq_dates.size == 0:
            raise ValueError("[self-check] No dates found in proprietary dataset.")
        n_dates = int(min(int(args.self_check_n_dates), int(uniq_dates.size)))
        rng = np.random.default_rng(int(args.seed))
        chosen = rng.choice(uniq_dates, size=n_dates, replace=False)
        prop = prop[prop[COL_DATE].isin(chosen)].copy()
        client = client[client[COL_DATE].isin(chosen)].copy()
        print(
            f"[self-check] Subsampled to {n_dates} dates: "
            f"prop rows={len(prop):,}, client rows={len(client):,}."
        )

    # Compute missing imbalance columns if needed.
    if ("local" in imbalance_kinds) and ("imbalance_local" not in prop.columns):
        prop = compute_imbalance_local(prop, out_col="imbalance_local")
    if ("local" in imbalance_kinds) and ("imbalance_local" not in client.columns):
        client = compute_imbalance_local(client, out_col="imbalance_local")

    if "cross" in imbalance_kinds:
        if "imbalance_client_env" not in prop.columns:
            prop = attach_environment_imbalance(prop, client, group_cols=[COL_ISIN, COL_DATE], out_col="imbalance_client_env")
        if "imbalance_prop_env" not in client.columns:
            client = attach_environment_imbalance(client, prop, group_cols=[COL_ISIN, COL_DATE], out_col="imbalance_prop_env")

    if "all" in imbalance_kinds:
        # Compute all-others imbalance on concatenated prop+client, then split.
        prop_tmp = prop[[COL_ISIN, COL_DATE, COL_DIR, COL_Q]].copy()
        client_tmp = client[[COL_ISIN, COL_DATE, COL_DIR, COL_Q]].copy()
        all_df = pd.concat([prop_tmp, client_tmp], ignore_index=True)
        all_df = compute_imbalance_local(all_df, out_col="imbalance_all_others")

        imb_all = all_df["imbalance_all_others"].to_numpy(dtype=float)
        if imb_all.size != (len(prop) + len(client)):
            raise RuntimeError("Unexpected all-others imbalance length mismatch.")
        prop["imbalance_all_others"] = imb_all[: len(prop)]
        client["imbalance_all_others"] = imb_all[len(prop) :]

    # Filter frames for binning (pooled edges can depend on chosen imbalance; keep consistent filters).
    # We build bins once from the local-kind frame by default; if local not run, use the first kind.
    eta_max = float(args.eta_max)

    # Prepare per-kind analysis and outputs.
    plotly_ok = _try_import_plotly() is not None
    want_plotly = args.plots in {"plotly", "both"}
    want_mpl = args.plots in {"matplotlib", "both"}
    if want_plotly and not plotly_ok:
        print("[warning] plotly not available; falling back to matplotlib.")
        want_plotly = False
        want_mpl = True

    for imb_kind in imbalance_kinds:
        prop_imb_col = _get_imbalance_col(imb_kind, group="prop")
        client_imb_col = _get_imbalance_col(imb_kind, group="client")

        if prop_imb_col not in prop.columns:
            raise KeyError(f"[prop] Missing required imbalance column for {imb_kind}: {prop_imb_col}")
        if client_imb_col not in client.columns:
            raise KeyError(f"[client] Missing required imbalance column for {imb_kind}: {client_imb_col}")

        prop_prep = prepare_analysis_frame(prop, imb_col=prop_imb_col, eta_max=eta_max)
        client_prep = prepare_analysis_frame(client, imb_col=client_imb_col, eta_max=eta_max)

        if prop_prep.empty or client_prep.empty:
            print(f"[warning] Empty sample after filtering for {imb_kind}. Skipping.")
            continue

        bins = make_eta_bins(
            eta_prop=prop_prep["__eta__"],
            eta_client=client_prep["__eta__"],
            n_bins=int(args.eta_bins),
            mode=str(args.eta_binning),
        )
        edges_prop = bins["edges_prop"]
        edges_client = bins["edges_client"]

        dry_run = bool(args.dry_run)
        cluster_ci = "none" if dry_run else str(args.cluster_ci)
        bootstrap_runs = 0 if (dry_run or cluster_ci == "none") else int(args.bootstrap_runs)
        if bool(args.self_check) and (not dry_run):
            bootstrap_runs = min(bootstrap_runs, 50)

        prop_res = _build_bin_results(
            prop_prep,
            bin_col="__bin__",
            edges=edges_prop,
            min_n_bin=int(args.min_n_bin),
            min_n_day_bin=int(args.min_n_day_bin),
            bootstrap_runs=bootstrap_runs,
            alpha=float(args.alpha),
            seed=int(args.seed),
            build_daily_panel=not dry_run,
        )
        client_res = _build_bin_results(
            client_prep,
            bin_col="__bin__",
            edges=edges_client,
            min_n_bin=int(args.min_n_bin),
            min_n_day_bin=int(args.min_n_day_bin),
            bootstrap_runs=bootstrap_runs,
            alpha=float(args.alpha),
            seed=int(args.seed) + 17,
            build_daily_panel=not dry_run,
        )

        prop_summary = prop_res.summary.assign(group="prop", imbalance_kind=imb_kind, eta_binning=str(args.eta_binning))
        client_summary = client_res.summary.assign(group="client", imbalance_kind=imb_kind, eta_binning=str(args.eta_binning))

        if dry_run:
            print(f"[dry-run] {imb_kind}: prop rows={len(prop_prep):,}, client rows={len(client_prep):,}")
            print(f"[dry-run] {imb_kind}: prop bins={len(edges_prop)-1}, client bins={len(edges_client)-1}")
            print(f"[dry-run] {imb_kind}: cluster_ci={cluster_ci}, bootstrap_runs={bootstrap_runs}")
            continue

        # Write tables.
        prop_summary.to_csv(paths.out_dir / f"bin_summary_prop_{imb_kind}.csv", index=False)
        client_summary.to_csv(paths.out_dir / f"bin_summary_client_{imb_kind}.csv", index=False)

        if not prop_res.daily_panel.empty:
            prop_daily = prop_res.daily_panel.rename(columns={"__bin__": "bin"}).assign(group="prop", imbalance_kind=imb_kind)
            prop_daily.to_csv(paths.out_dir / f"daily_bin_panel_prop_{imb_kind}.csv", index=False)
            if args.write_parquet:
                prop_daily.to_parquet(paths.out_dir / f"daily_bin_panel_prop_{imb_kind}.parquet", index=False)

        if not client_res.daily_panel.empty:
            client_daily = client_res.daily_panel.rename(columns={"__bin__": "bin"}).assign(group="client", imbalance_kind=imb_kind)
            client_daily.to_csv(paths.out_dir / f"daily_bin_panel_client_{imb_kind}.csv", index=False)
            if args.write_parquet:
                client_daily.to_parquet(paths.out_dir / f"daily_bin_panel_client_{imb_kind}.parquet", index=False)

        # Effect sizes and trend tests (point + bootstrap CIs when available).
        eff_rows: List[Dict[str, Any]] = []
        for res_obj, summary_df, group in [
            (prop_res, prop_summary, "prop"),
            (client_res, client_summary, "client"),
        ]:
            x = summary_df["eta_center"].to_numpy(dtype=float)
            for metric in ["mean_align", "mean_abs_imb", "corr_dir_imb"]:
                y = summary_df[metric].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                point_spearman = float(_spearman_corr(x[mask], y[mask])) if mask.sum() >= 3 else float("nan")
                point_delta = float(y[-1] - y[0]) if (y.size >= 2 and np.isfinite(y[0]) and np.isfinite(y[-1])) else float("nan")

                # Date bootstrap
                if res_obj.boot_date is not None:
                    rep = res_obj.boot_date[f"rep_{metric}"]
                    delta = compute_effect_sizes_from_bootstrap(rep, alpha=float(args.alpha))
                    sp = bootstrap_spearman_ci(rep, x=x, alpha=float(args.alpha))
                    eff_rows.append(
                        {
                            "group": group,
                            "imbalance_kind": imb_kind,
                            "metric": metric,
                            "cluster": "date",
                            "bootstrap_runs": int(bootstrap_runs),
                            "point_top_minus_bottom": point_delta,
                            "point_spearman_eta": point_spearman,
                            "boot_delta": delta["delta"],
                            "boot_delta_lo": delta["lo"],
                            "boot_delta_hi": delta["hi"],
                            "boot_spearman": sp["spearman"],
                            "boot_spearman_lo": sp["lo"],
                            "boot_spearman_hi": sp["hi"],
                        }
                    )

                # Point-only fallback (when no bootstrap requested)
                if res_obj.boot_date is None:
                    eff_rows.append(
                        {
                            "group": group,
                            "imbalance_kind": imb_kind,
                            "metric": metric,
                            "cluster": "none",
                            "bootstrap_runs": 0,
                            "point_top_minus_bottom": point_delta,
                            "point_spearman_eta": point_spearman,
                            "boot_delta": float("nan"),
                            "boot_delta_lo": float("nan"),
                            "boot_delta_hi": float("nan"),
                            "boot_spearman": float("nan"),
                            "boot_spearman_lo": float("nan"),
                            "boot_spearman_hi": float("nan"),
                        }
                    )

        eff = pd.DataFrame(eff_rows)
        eff.to_csv(paths.out_dir / f"effect_sizes_{imb_kind}.csv", index=False)

        # Permutation placebo (optional): delta of mean_align only.
        if int(args.permutation_runs) > 0:
            for df_prepared, edges, group in [(prop_prep, edges_prop, "prop"), (client_prep, edges_client, "client")]:
                perm = within_date_permutation_delta_align(
                    df_prepared,
                    edges=edges,
                    seed=int(args.seed) + (0 if group == "prop" else 11),
                    n_runs=int(args.permutation_runs),
                )
                perm_path = paths.out_dir / f"permutation_delta_align_{group}_{imb_kind}.csv"
                pd.DataFrame({"delta_perm": perm["delta_perm"]}).to_csv(perm_path, index=False)
                # Lightweight summary append.
                perm_summary = pd.DataFrame(
                    [
                        {
                            "group": group,
                            "imbalance_kind": imb_kind,
                            "delta_obs": perm["delta_obs"],
                            "p_two_sided": perm["p_two_sided"],
                            "n_runs": int(args.permutation_runs),
                        }
                    ]
                )
                perm_summary.to_csv(paths.out_dir / f"permutation_summary_{group}_{imb_kind}.csv", index=False)

        # Plots: curves for mean_align, mean_abs_imb, corr_dir_imb.
        if want_plotly or want_mpl:
            for metric, ylab in [
                ("mean_align", "E[Direction * imbalance]"),
                ("mean_abs_imb", "E[|imbalance|]"),
                ("corr_dir_imb", "Corr(Direction, imbalance)"),
            ]:
                title = f"{imb_kind}: {ylab} vs participation rate"
                prop_noise_lo, prop_noise_hi = bootstrap_centered_noise_band(
                    None if prop_res.boot_date is None else prop_res.boot_date.get(f"rep_{metric}"),
                    alpha=float(args.alpha),
                )
                client_noise_lo, client_noise_hi = bootstrap_centered_noise_band(
                    None if client_res.boot_date is None else client_res.boot_date.get(f"rep_{metric}"),
                    alpha=float(args.alpha),
                )
                if want_plotly:
                    _plotly_curve_date_ci(
                        x_prop=prop_summary["eta_center"].to_numpy(dtype=float),
                        y_prop=prop_summary[metric].to_numpy(dtype=float),
                        lo_prop=prop_summary[f"ci_date_{metric}_lo"].to_numpy(dtype=float),
                        hi_prop=prop_summary[f"ci_date_{metric}_hi"].to_numpy(dtype=float),
                        noise_lo_prop=prop_noise_lo,
                        noise_hi_prop=prop_noise_hi,
                        x_client=client_summary["eta_center"].to_numpy(dtype=float),
                        y_client=client_summary[metric].to_numpy(dtype=float),
                        lo_client=client_summary[f"ci_date_{metric}_lo"].to_numpy(dtype=float),
                        hi_client=client_summary[f"ci_date_{metric}_hi"].to_numpy(dtype=float),
                        noise_lo_client=client_noise_lo,
                        noise_hi_client=client_noise_hi,
                        x_label=r"$\eta$",
                        y_label=ylab,
                        title=title,
                        out_stem=f"curve_{metric}_vs_eta_{imb_kind}",
                        output_dirs=img_output_dirs,
                    )
                if want_mpl:
                    _mpl_curve_date_ci(
                        x_prop=prop_summary["eta_center"].to_numpy(dtype=float),
                        y_prop=prop_summary[metric].to_numpy(dtype=float),
                        lo_prop=prop_summary[f"ci_date_{metric}_lo"].to_numpy(dtype=float),
                        hi_prop=prop_summary[f"ci_date_{metric}_hi"].to_numpy(dtype=float),
                        noise_lo_prop=prop_noise_lo,
                        noise_hi_prop=prop_noise_hi,
                        x_client=client_summary["eta_center"].to_numpy(dtype=float),
                        y_client=client_summary[metric].to_numpy(dtype=float),
                        lo_client=client_summary[f"ci_date_{metric}_lo"].to_numpy(dtype=float),
                        hi_client=client_summary[f"ci_date_{metric}_hi"].to_numpy(dtype=float),
                        noise_lo_client=client_noise_lo,
                        noise_hi_client=client_noise_hi,
                        x_label=r"$\eta$",
                        y_label=ylab,
                        title=title,
                        out_png=img_output_dirs.png_dir / f"curve_{metric}_vs_eta_{imb_kind}.png",
                    )

        # 2D conditioning heatmaps (optional).
        if bool(args.run_2d) and want_plotly:
            for df_prepared, edges, group in [(prop_prep, edges_prop, "prop"), (client_prep, edges_client, "client")]:
                if COL_QV not in df_prepared.columns:
                    print(f"[warning] Missing {COL_QV} for 2D heatmap; skip ({group}, {imb_kind}).")
                    continue
                table = compute_2d_heatmap_table(df_prepared, edges_eta=edges, n_qv_bins=int(args.qv_bins))
                if table.empty:
                    continue
                table.to_csv(paths.out_dir / f"heatmap_align_qv_eta_{group}_{imb_kind}.csv", index=False)
                plotly_heatmap_align(
                    table,
                    title=f"{group} {imb_kind}: mean alignment on (Q/V, eta) grid",
                    out_stem=f"heatmap_align_qv_eta_{group}_{imb_kind}",
                    output_dirs=img_output_dirs,
                )

        # Regressions (optional)
        if bool(args.run_regressions):
            prop_reg = _maybe_run_regression(
                prop_prep,
                imb_col=prop_imb_col,
                cluster=str(args.reg_cluster),
                seed=int(args.seed),
                sample_frac=float(args.reg_sample_frac),
            ).assign(group="prop", imbalance_kind=imb_kind, reg_cluster=str(args.reg_cluster))
            client_reg = _maybe_run_regression(
                client_prep,
                imb_col=client_imb_col,
                cluster=str(args.reg_cluster),
                seed=int(args.seed) + 3,
                sample_frac=float(args.reg_sample_frac),
            ).assign(group="client", imbalance_kind=imb_kind, reg_cluster=str(args.reg_cluster))
            reg = pd.concat([prop_reg, client_reg], ignore_index=True)
            reg.to_csv(paths.out_dir / f"regression_results_{imb_kind}.csv", index=False)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
