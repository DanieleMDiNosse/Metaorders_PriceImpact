#!/usr/bin/env python3
"""
Crowding-conditioned impact analysis for proprietary and client metaorders.

What this script does
---------------------
This workflow tests whether end-of-execution impact is larger when a metaorder
trades with stronger crowding. The main crowding variable is the aligned
leave-one-out imbalance

    c_i = Direction_i * imbalance_local_i

where `imbalance_local_i` is computed on the same `(ISIN, Date)` cell after
excluding metaorder `i` itself. The script supports both within-group and
pooled-all-metaorder crowding environments. It:

- loads the canonical proprietary/client metaorder parquets,
- applies the same impact-section filters (`Q/V > MIN_QV`, finite impact,
  finite participation, `Participation Rate < ETA_MAX`),
- splits each group into crowding quantiles,
- fits the repository's log-binned WLS power-law curve within each crowding
  bucket,
- fits an additional joint-bin regression on cell means over
  `(Participation Rate, Vt/V, |imbalance|)`,
- compares predicted impact at common benchmark sizes,
- uses a Date-cluster bootstrap to build percentile confidence intervals, and
- repeats the crowding split within broad participation-rate bins as a
  robustness check.

How to run
----------
1) Edit `config_ymls/crowding_impact_analysis.yml`, or set
   `CROWDING_IMPACT_CONFIG=/path/to/config.yml`.
2) Activate the repository conda environment.
3) Run:

    python scripts/run_analysis.py crowding impact

Outputs
-------
- Tables under `out_files/{DATASET_NAME}/crowding_impact/`
- Figures under `images/{DATASET_NAME}/crowding_impact/{html,png}/`
- Run manifest and log file in the analysis output folder
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Ensure repository-root imports work when running
# `python scripts/run_analysis.py crowding impact` from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.config import format_path_template, load_yaml_mapping, resolve_opt_repo_path, resolve_repo_path
from moimpact.impact_fits import (
    filter_metaorders_info_for_fits,
    fit_power_law_logbins_wls_new,
    power_law,
    weighted_r2,
    weights_from_sigma,
)
from moimpact.logging_utils import PrintTee, setup_file_logger
from moimpact.plot_style import apply_shared_plotly_style, load_plot_style, plotly_legend_layout
from moimpact.plotting import (
    COLOR_CLIENT,
    COLOR_NEUTRAL,
    COLOR_PROPRIETARY,
    PlotOutputDirs,
    ensure_plot_dirs,
    make_plot_output_dirs,
    save_plotly_figure as _base_save_plotly_figure,
)


_CONFIG_ENV_VAR = "CROWDING_IMPACT_CONFIG"
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config_ymls" / "crowding_impact_analysis.yml"
_DEFAULT_IMPACT_CONFIG_PATH = _REPO_ROOT / "config_ymls" / "metaorder_computation.yml"

COL_DATE = "Date"
COL_PERIOD = "Period"
COL_GROUP = "group"
COL_GROUP_LABEL = "group_label"
COL_ISIN = "ISIN"
COL_DIRECTION = "Direction"
COL_Q = "Q"
COL_QV = "Q/V"
COL_VTV = "Vt/V"
COL_ETA = "Participation Rate"
COL_IMPACT = "Impact"
COL_IMBALANCE = "imbalance_local"
COL_ALIGNED_CROWDING = "aligned_crowding"
COL_ALL_IMBALANCE = "imbalance_all"
COL_ABS_IMBALANCE = "abs_imbalance"
COL_ABS_IMPACT = "abs_impact"
COL_LOG_IMPACT = "log_impact"
COL_LOG_ETA = "log_eta"
COL_LOG_VTV = "log_vtv"
COL_PROP_DUMMY = "is_proprietary"
COL_CLIENT_DUMMY = "is_client"
COL_METAORDER_ID = "metaorder_id"
COL_CROWDING_BIN = "crowding_bin"
COL_CROWDING_LABEL = "crowding_label"
COL_ETA_BIN = "eta_bin"
COL_ETA_LABEL = "eta_label"
COL_ETA_REG_BIN = "eta_reg_bin"
COL_VTV_REG_BIN = "vtv_reg_bin"
COL_ABS_IMB_REG_BIN = "abs_imb_reg_bin"

GROUP_ORDER = ("client", "proprietary")
GROUP_DISPLAY = {"client": "Client", "proprietary": "Proprietary"}
POOLED_DUMMY_GROUPS = ("proprietary", "client")
CROWDING_COLORS = {
    0: "#6B7280",  # low
    1: "#D97706",  # mid
    2: "#047857",  # high
    3: "#7C3AED",
}
BENCHMARK_COLORS = ["#1D4ED8", "#DC2626", "#059669", "#B45309"]

PLOT_STYLE = load_plot_style()
try:
    apply_shared_plotly_style(PLOT_STYLE)
except Exception:
    pass


@dataclass(frozen=True)
class ResolvedPaths:
    """Concrete input/output paths used by the analysis run."""

    dataset_name: str
    prop_path: Path
    client_path: Path
    out_dir: Path
    img_dir: Path
    config_path: Path
    impact_config_path: Path
    log_path: Path


@dataclass(frozen=True)
class FitDefaults:
    """Impact-fit defaults inherited from the repository's impact workflow."""

    min_qv: float
    n_logbins: int
    min_count: int
    eta_max: float


@dataclass(frozen=True)
class GroupLoadResult:
    """Loaded one-group sample plus pre-crowding filter counts."""

    frame: pd.DataFrame
    raw_n: int
    fit_filtered_n: int
    eta_filtered_n: int
    imbalance_source: str


@dataclass(frozen=True)
class VariantOutputs:
    """Tables produced by one crowding-conditioned impact analysis variant."""

    labeled: pd.DataFrame
    cutpoints: pd.DataFrame
    sample_sizes: pd.DataFrame
    fit_summary: pd.DataFrame
    predictions: pd.DataFrame
    contrasts: pd.DataFrame
    group_differences: pd.DataFrame
    binned_curve_data: pd.DataFrame


@dataclass(frozen=True)
class JointRegressionOutputs:
    """Tables produced by the joint-bin `(η, V_t/V, |imb|)` regression."""

    cutpoints: pd.DataFrame
    cell_data: pd.DataFrame
    fit_summary: pd.DataFrame
    group_differences: pd.DataFrame


@dataclass(frozen=True)
class PooledLogRegressionOutputs:
    """Tables produced by the pooled cell-level log-mean-impact regression."""

    sample_counts: pd.DataFrame
    cutpoints: pd.DataFrame
    cell_data: pd.DataFrame
    fit_summary: pd.DataFrame
    coefficients: pd.DataFrame


@dataclass(frozen=True)
class DateBootstrapSampler:
    """Day-cluster bootstrap sampler built from a working metaorder table."""

    df: pd.DataFrame
    row_positions: np.ndarray
    date_codes: np.ndarray
    n_dates: int

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "DateBootstrapSampler":
        if COL_DATE not in df.columns:
            raise KeyError(f"Missing required column: {COL_DATE}")
        if df.empty:
            raise ValueError("Cannot build a bootstrap sampler from an empty DataFrame.")
        date_codes, uniques = pd.factorize(pd.to_datetime(df[COL_DATE], errors="coerce"), sort=False)
        n_dates = int(len(uniques))
        if n_dates < 2:
            raise ValueError("At least two trading dates are required for the Date-cluster bootstrap.")
        return cls(
            df=df.reset_index(drop=True),
            row_positions=np.arange(len(df), dtype=np.int64),
            date_codes=np.asarray(date_codes, dtype=np.int64),
            n_dates=n_dates,
        )

    def sample(self, rng: np.random.Generator) -> pd.DataFrame:
        """Draw one bootstrap sample by resampling trading days with replacement."""
        weights = rng.multinomial(self.n_dates, np.full(self.n_dates, 1.0 / self.n_dates))
        row_mult = weights[self.date_codes]
        take = np.repeat(self.row_positions, row_mult)
        return self.df.iloc[take].reset_index(drop=True)


def _export_plotly_figure(fig, *args, **kwargs):
    """Save a Plotly figure while keeping the top-level title outside the panel."""
    fig.update_layout(title=None)
    return _base_save_plotly_figure(fig, *args, **kwargs)


def _period_start_ns(period_value: Any) -> Optional[int]:
    """Extract the first timestamp from the repository's `Period` encoding."""
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
        stripped = period_value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            first = stripped[1:-1].split(",")[0].strip()
            if not first:
                return None
            try:
                return int(first)
            except Exception:
                return None
        try:
            return int(stripped)
        except Exception:
            return None
    return None


def _read_parquet_with_fallback(path: Path, columns: Optional[Sequence[str]]) -> pd.DataFrame:
    """Read a parquet file, retrying with the available schema when needed."""
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    if columns is None:
        return pd.read_parquet(path)
    wanted = list(columns)
    try:
        return pd.read_parquet(path, columns=wanted)
    except Exception:
        try:
            import pyarrow.parquet as pq  # type: ignore

            available = set(pq.ParquetFile(path).schema_arrow.names)
            subset = [col for col in wanted if col in available]
            if subset:
                return pd.read_parquet(path, columns=subset)
        except Exception:
            pass
        full = pd.read_parquet(path)
        keep = [col for col in wanted if col in full.columns]
        return full[keep].copy()


def _validate_required_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    """Raise a clear error when an input frame misses required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"[{label}] Missing required columns: {missing}")


def _ensure_date_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Ensure a normalized `Date` column exists, deriving it from `Period` if needed."""
    out = df.copy()
    if COL_DATE in out.columns:
        out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce").dt.normalize()
    else:
        if COL_PERIOD not in out.columns:
            raise KeyError(f"[{label}] Missing both {COL_DATE} and {COL_PERIOD}; cannot infer the trading date.")
        starts = out[COL_PERIOD].apply(_period_start_ns)
        out[COL_DATE] = pd.to_datetime(starts, errors="coerce").dt.normalize()
    if out[COL_DATE].isna().any():
        raise ValueError(f"[{label}] Failed to parse the trading date for some rows.")
    return out


def _compute_imbalance_local(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = (COL_ISIN, COL_DATE),
    side_col: str = COL_DIRECTION,
    vol_col: str = COL_Q,
    out_col: str = COL_IMBALANCE,
) -> pd.DataFrame:
    """Compute the within-group leave-one-out signed-volume imbalance."""
    _validate_required_columns(df, list(group_cols) + [side_col, vol_col], label="imbalance_local")
    out = df.copy()
    out["__q__"] = pd.to_numeric(out[vol_col], errors="coerce")
    out["__d__"] = pd.to_numeric(out[side_col], errors="coerce")
    out["__qd__"] = out["__q__"].to_numpy(dtype=float) * out["__d__"].to_numpy(dtype=float)
    grouped = out.groupby(list(group_cols), dropna=False, sort=False)
    total_q = grouped["__q__"].transform("sum")
    total_qd = grouped["__qd__"].transform("sum")
    denom = total_q - out["__q__"]
    numer = total_qd - out["__qd__"]
    out[out_col] = np.where(denom > 0, numer / denom, np.nan)
    return out.drop(columns=["__q__", "__d__", "__qd__"])


def _try_git_hash() -> Optional[str]:
    """Return the current repository commit hash when available."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True).strip()
    except Exception:
        return None
    return out or None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON file with deterministic indentation and key ordering."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _parse_float_list(value: str) -> Tuple[float, ...]:
    """Parse a comma-separated list of floats from the CLI."""
    parts = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not parts:
        raise ValueError("Expected at least one numeric value.")
    return tuple(float(piece) for piece in parts)


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_repo_path(_REPO_ROOT, value)


def _resolve_opt_repo_path(value: Optional[str | Path], default: Path) -> Path:
    return resolve_opt_repo_path(_REPO_ROOT, value, default)


def _format_path_template(template: str, context: Mapping[str, str]) -> str:
    return format_path_template(template, context)


def _resolve_bool(cli_value: Optional[bool], cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if key in cfg:
        return bool(cfg.get(key))
    return bool(default)


def _resolve_paths(cfg: Mapping[str, Any], args: argparse.Namespace) -> ResolvedPaths:
    dataset_name = str(args.dataset_name or cfg.get("DATASET_NAME") or "ftsemib")
    path_context = {"DATASET_NAME": dataset_name}

    out_base_cfg = str(args.output_file_path or cfg.get("OUTPUT_FILE_PATH") or "out_files/{DATASET_NAME}")
    img_base_cfg = str(args.img_output_path or cfg.get("IMG_OUTPUT_PATH") or "images/{DATASET_NAME}")
    analysis_tag = str(args.analysis_tag or cfg.get("ANALYSIS_TAG") or "crowding_impact")

    out_base = _resolve_repo_path(_format_path_template(out_base_cfg, path_context))
    img_base = _resolve_repo_path(_format_path_template(img_base_cfg, path_context))

    prop_default = out_base / "metaorders_info_sameday_filtered_member_proprietary.parquet"
    client_default = out_base / "metaorders_info_sameday_filtered_member_non_proprietary.parquet"

    prop_cfg = cfg.get("PROP_PATH")
    client_cfg = cfg.get("CLIENT_PATH")
    prop_path = _resolve_opt_repo_path(
        args.prop_path or (
            _format_path_template(str(prop_cfg), path_context) if prop_cfg is not None else None
        ),
        prop_default,
    )
    client_path = _resolve_opt_repo_path(
        args.client_path or (
            _format_path_template(str(client_cfg), path_context) if client_cfg is not None else None
        ),
        client_default,
    )

    config_path = _resolve_repo_path(args.config_path)
    impact_config_path = _resolve_repo_path(
        args.impact_config_path or cfg.get("IMPACT_CONFIG_PATH") or _DEFAULT_IMPACT_CONFIG_PATH
    )
    out_dir = out_base / analysis_tag
    img_dir = img_base / analysis_tag
    log_path = out_dir / "crowding_impact_analysis.log"
    return ResolvedPaths(
        dataset_name=dataset_name,
        prop_path=prop_path,
        client_path=client_path,
        out_dir=out_dir,
        img_dir=img_dir,
        config_path=config_path,
        impact_config_path=impact_config_path,
        log_path=log_path,
    )


def _load_fit_defaults(cfg: Mapping[str, Any], args: argparse.Namespace, impact_cfg: Mapping[str, Any]) -> FitDefaults:
    def _first_not_none(*values):
        for value in values:
            if value is not None:
                return value
        return None

    min_qv = float(_first_not_none(args.min_qv, cfg.get("MIN_QV"), impact_cfg.get("MIN_QV"), 1.0e-5))
    n_logbins = int(_first_not_none(args.n_logbins, cfg.get("N_LOGBINS"), impact_cfg.get("N_LOGBIN"), 30))
    min_count = int(_first_not_none(args.min_count, cfg.get("MIN_COUNT"), impact_cfg.get("MIN_COUNT"), 20))
    eta_max = float(
        _first_not_none(args.eta_max, cfg.get("ETA_MAX"), impact_cfg.get("MAX_PARTICIPATION_RATE"), 1.0)
    )
    return FitDefaults(min_qv=min_qv, n_logbins=n_logbins, min_count=min_count, eta_max=eta_max)


def _crowding_label_map(n_bins: int) -> Dict[int, str]:
    if n_bins == 2:
        return {0: "Low", 1: "High"}
    if n_bins == 3:
        return {0: "Low", 1: "Mid", 2: "High"}
    return {idx: f"Q{idx + 1}" for idx in range(n_bins)}


def _eta_label_map(n_bins: int) -> Dict[int, str]:
    if n_bins == 2:
        return {0: "Low η", 1: "High η"}
    return {idx: f"η Q{idx + 1}" for idx in range(n_bins)}


def _quantile_edges(values: pd.Series, n_bins: int, label: str) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError(f"{label}: no finite values available for quantile binning.")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = clean.quantile(quantiles, interpolation="linear").to_numpy(dtype=float)
    edges = np.unique(edges)
    if edges.size != n_bins + 1:
        raise ValueError(
            f"{label}: cannot form {n_bins} quantile bins because only {edges.size - 1} unique intervals are available."
        )
    return edges


def _assign_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    codes = pd.cut(clean, bins=edges, include_lowest=True, labels=False, right=True)
    return codes.astype("Int64")


def _add_quantile_bins(
    df: pd.DataFrame,
    *,
    value_col: str,
    by_cols: Sequence[str],
    n_bins: int,
    bin_col: str,
    label_col: str,
    label_map: Mapping[int, str],
    variant: str,
    dimension: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out[bin_col] = pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"), index=out.index)
    cut_rows: List[Dict[str, Any]] = []

    for key, sub in out.groupby(list(by_cols), sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = {col: key_tuple[idx] for idx, col in enumerate(by_cols)}
        edges = _quantile_edges(sub[value_col], n_bins=n_bins, label=f"{variant} / {dimension} / {key_map}")
        codes = _assign_bins(sub[value_col], edges)
        actual_bins = int(codes.dropna().nunique())
        if actual_bins != n_bins:
            raise ValueError(
                f"{variant} / {dimension} / {key_map}: expected {n_bins} populated bins, found {actual_bins}."
            )
        out.loc[sub.index, bin_col] = codes
        for idx in range(n_bins):
            cut_rows.append(
                {
                    "variant": variant,
                    "dimension": dimension,
                    **key_map,
                    bin_col: idx,
                    label_col: label_map[idx],
                    "left_edge": float(edges[idx]),
                    "right_edge": float(edges[idx + 1]),
                    "n_bins": int(n_bins),
                }
            )

    out[label_col] = out[bin_col].map(label_map)
    return out, pd.DataFrame(cut_rows)


def _log_bin_edges(values: pd.Series, n_bins: int, label: str) -> np.ndarray:
    """Build observed-range logarithmic bin edges for one strictly positive variable."""
    if n_bins < 1:
        raise ValueError(f"{label}: n_bins must be >= 1.")
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean > 0.0]
    if clean.empty:
        raise ValueError(f"{label}: no strictly positive values are available for log binning.")
    lo = float(clean.min())
    hi = float(clean.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"{label}: invalid positive range for log binning.")
    return np.logspace(np.log10(lo), np.log10(hi), int(n_bins) + 1)


def _linear_bin_edges(values: pd.Series, n_bins: int, label: str) -> np.ndarray:
    """Build observed-range equal-width bin edges for one finite bounded variable."""
    if n_bins < 1:
        raise ValueError(f"{label}: n_bins must be >= 1.")
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError(f"{label}: no finite values are available for linear binning.")
    lo = float(clean.min())
    hi = float(clean.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"{label}: invalid range for linear binning.")
    return np.linspace(lo, hi, int(n_bins) + 1)


def _fit_joint_bin_regression_wls(
    binned: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit log(mean impact) on log(η), log(V_t/V), and |imb| using SEM-weighted WLS."""
    required = ["center_eta", "center_vtv", "center_abs_imb", "mean_imp", "sem_imp"]
    _validate_required_columns(binned, required, label="joint_bin_regression")

    work = binned.copy()
    n_required = 4
    if len(work) < n_required:
        raise ValueError(
            f"Not enough retained joint bins for WLS (got {len(work)}; need at least {n_required})."
        )

    log_eta = np.log(work["center_eta"].to_numpy(dtype=float))
    log_vtv = np.log(work["center_vtv"].to_numpy(dtype=float))
    abs_imb = work["center_abs_imb"].to_numpy(dtype=float)
    log_y = np.log(work["mean_imp"].to_numpy(dtype=float))
    var_logy = (work["sem_imp"].to_numpy(dtype=float) / work["mean_imp"].to_numpy(dtype=float)) ** 2
    w = np.where(np.isfinite(var_logy) & (var_logy > 0), 1.0 / var_logy, 0.0)
    if np.count_nonzero(w > 0) < n_required:
        raise ValueError("Not enough positive WLS weights for the joint-bin regression.")

    design = np.vstack([np.ones_like(log_y), log_eta, log_vtv, abs_imb]).T
    sqrt_w = np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(design * sqrt_w[:, None], log_y * sqrt_w, rcond=None)

    fitted_log_y = design @ coef
    residuals = log_y - fitted_log_y
    rss = float(np.sum(w * residuals**2))
    dof = max(len(log_y) - design.shape[1], 1)
    s2 = rss / dof
    xtwx = design.T @ (w[:, None] * design)
    try:
        cov = s2 * np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        cov = s2 * np.linalg.pinv(xtwx)
    se = np.sqrt(np.diag(cov))

    log_y_hat = float(coef[0])
    beta_log_eta_hat = float(coef[1])
    beta_log_vtv_hat = float(coef[2])
    beta_abs_imb_hat = float(coef[3])
    log_y_se = float(se[0])
    beta_log_eta_se = float(se[1])
    beta_log_vtv_se = float(se[2])
    beta_abs_imb_se = float(se[3])
    y_hat = float(np.exp(log_y_hat))
    y_se = float(y_hat * log_y_se)
    fitted_mean_imp = np.exp(fitted_log_y)

    work["log_mean_imp"] = log_y
    work["fitted_log_mean_imp"] = fitted_log_y
    work["fitted_mean_imp"] = fitted_mean_imp
    work["residual_log_mean_imp"] = residuals
    work["wls_weight"] = w

    return work, {
        "status": "ok",
        "error": "",
        "log_Y_hat": log_y_hat,
        "log_Y_se": log_y_se,
        "Y_hat": y_hat,
        "Y_se": y_se,
        "beta_log_eta_hat": beta_log_eta_hat,
        "beta_log_eta_se": beta_log_eta_se,
        "beta_log_vtv_hat": beta_log_vtv_hat,
        "beta_log_vtv_se": beta_log_vtv_se,
        "beta_abs_imb_hat": beta_abs_imb_hat,
        "beta_abs_imb_se": beta_abs_imb_se,
        "r2_log": float(weighted_r2(log_y, fitted_log_y, w=w)),
        "r2_lin": float(weighted_r2(work["mean_imp"].to_numpy(dtype=float), fitted_mean_imp, w=weights_from_sigma(work["sem_imp"].to_numpy(dtype=float)))),
        "dof": float(dof),
        "n_cells_retained": int(len(work)),
    }


def _build_joint_regression_group_differences(fit_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare the joint-regression slopes between proprietary and client fits."""
    if fit_summary.empty:
        return pd.DataFrame()
    ok = fit_summary.loc[fit_summary["status"] == "ok"].copy()
    prop = ok.loc[ok[COL_GROUP] == "proprietary"]
    client = ok.loc[ok[COL_GROUP] == "client"]
    if prop.empty or client.empty:
        return pd.DataFrame()

    prop_row = prop.iloc[0]
    client_row = client.iloc[0]
    return pd.DataFrame(
        [
            {
                "comparison": "prop_minus_client",
                "prop_minus_client_beta_log_eta_hat": float(prop_row["beta_log_eta_hat"] - client_row["beta_log_eta_hat"]),
                "prop_minus_client_beta_log_vtv_hat": float(prop_row["beta_log_vtv_hat"] - client_row["beta_log_vtv_hat"]),
                "prop_minus_client_beta_abs_imb_hat": float(prop_row["beta_abs_imb_hat"] - client_row["beta_abs_imb_hat"]),
            }
        ]
    )


def _analytic_ci(estimate: float, se: float, *, dof: float, alpha: float) -> Tuple[float, float]:
    """Build a symmetric t-based interval from one estimate and one standard error."""
    if not np.isfinite(estimate) or not np.isfinite(se) or se < 0:
        return float("nan"), float("nan")
    if se == 0:
        return float(estimate), float(estimate)
    df = float(dof) if np.isfinite(dof) and dof > 0 else float("nan")
    tcrit = float(stats.t.ppf(1.0 - float(alpha) / 2.0, df=df)) if np.isfinite(df) else 1.96
    half_width = tcrit * float(se)
    return float(estimate - half_width), float(estimate + half_width)


def _welch_satterthwaite_dof(var_a: float, dof_a: float, var_b: float, dof_b: float) -> float:
    """Approximate the effective degrees of freedom for a difference of independent estimates."""
    if not (np.isfinite(var_a) and np.isfinite(var_b) and np.isfinite(dof_a) and np.isfinite(dof_b)):
        return float("nan")
    if var_a < 0 or var_b < 0 or dof_a <= 0 or dof_b <= 0:
        return float("nan")
    numer = (var_a + var_b) ** 2
    denom = 0.0
    if var_a > 0:
        denom += (var_a**2) / dof_a
    if var_b > 0:
        denom += (var_b**2) / dof_b
    if denom <= 0:
        return float("nan")
    return float(numer / denom)


def _build_joint_regression_analytic_plot_tables(
    fit_summary: pd.DataFrame,
    *,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct analytic SE-based interval tables for the joint-regression coefficient plots."""
    if fit_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    metrics = [
        ("beta_log_eta_hat", "beta_log_eta_se"),
        ("beta_log_vtv_hat", "beta_log_vtv_se"),
        ("beta_abs_imb_hat", "beta_abs_imb_se"),
    ]
    ok = fit_summary.loc[fit_summary["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()

    coef_rows: List[Dict[str, Any]] = []
    for _, row in ok.iterrows():
        dof = float(row.get("dof", np.nan))
        for metric, se_col in metrics:
            estimate = float(row.get(metric, np.nan))
            se = float(row.get(se_col, np.nan))
            ci_low, ci_high = _analytic_ci(estimate, se, dof=dof, alpha=alpha)
            coef_rows.append(
                {
                    COL_GROUP: row[COL_GROUP],
                    COL_GROUP_LABEL: row[COL_GROUP_LABEL],
                    "metric": metric,
                    "point_estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    diff_rows: List[Dict[str, Any]] = []
    prop = ok.loc[ok[COL_GROUP] == "proprietary"]
    client = ok.loc[ok[COL_GROUP] == "client"]
    if not prop.empty and not client.empty:
        prop_row = prop.iloc[0]
        client_row = client.iloc[0]
        for metric, se_col in metrics:
            prop_est = float(prop_row.get(metric, np.nan))
            client_est = float(client_row.get(metric, np.nan))
            prop_se = float(prop_row.get(se_col, np.nan))
            client_se = float(client_row.get(se_col, np.nan))
            diff_est = prop_est - client_est
            diff_se = float(np.sqrt(max(prop_se**2 + client_se**2, 0.0))) if np.isfinite(prop_se) and np.isfinite(client_se) else float("nan")
            diff_dof = _welch_satterthwaite_dof(
                prop_se**2,
                float(prop_row.get("dof", np.nan)),
                client_se**2,
                float(client_row.get("dof", np.nan)),
            )
            ci_low, ci_high = _analytic_ci(diff_est, diff_se, dof=diff_dof, alpha=alpha)
            diff_rows.append(
                {
                    "comparison": "prop_minus_client",
                    "metric": f"prop_minus_client_{metric}",
                    "point_estimate": diff_est,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    return pd.DataFrame(coef_rows), pd.DataFrame(diff_rows)


def _analyse_joint_bin_regression(
    working: pd.DataFrame,
    *,
    n_eta_bins: int,
    n_vtv_bins: int,
    n_abs_imb_bins: int,
    min_cell_count: int,
) -> JointRegressionOutputs:
    """Run separate client/proprietary WLS fits on joint `(η, V_t/V, |imb|)` cell means."""
    labeled = working.copy()
    for col in [COL_ETA_REG_BIN, COL_VTV_REG_BIN, COL_ABS_IMB_REG_BIN]:
        labeled[col] = pd.Series(pd.array([pd.NA] * len(labeled), dtype="Int64"), index=labeled.index)

    cut_rows: List[Dict[str, Any]] = []
    edge_specs = [
        (COL_ETA, COL_ETA_REG_BIN, int(n_eta_bins), "eta", True),
        (COL_VTV, COL_VTV_REG_BIN, int(n_vtv_bins), "vtv", True),
        (COL_ABS_IMBALANCE, COL_ABS_IMB_REG_BIN, int(n_abs_imb_bins), "abs_imb", False),
    ]

    for group_name, sub in labeled.groupby(COL_GROUP, dropna=False, sort=False):
        for value_col, bin_col, n_bins, dimension, use_log_center in edge_specs:
            edges = (
                _log_bin_edges(sub[value_col], n_bins=n_bins, label=f"{group_name}/{dimension}")
                if use_log_center
                else _linear_bin_edges(sub[value_col], n_bins=n_bins, label=f"{group_name}/{dimension}")
            )
            codes = _assign_bins(sub[value_col], edges)
            labeled.loc[sub.index, bin_col] = codes
            for idx in range(n_bins):
                left = float(edges[idx])
                right = float(edges[idx + 1])
                center = float(np.sqrt(left * right)) if use_log_center else float(0.5 * (left + right))
                cut_rows.append(
                    {
                        "variant": "joint_bin_regression",
                        COL_GROUP: group_name,
                        COL_GROUP_LABEL: GROUP_DISPLAY[str(group_name)],
                        "dimension": dimension,
                        "bin_col": bin_col,
                        bin_col: int(idx),
                        "left_edge": left,
                        "right_edge": right,
                        "center_value": center,
                        "n_bins": int(n_bins),
                    }
                )

    cutpoints = pd.DataFrame(cut_rows)
    labeled = labeled.dropna(subset=[COL_ETA_REG_BIN, COL_VTV_REG_BIN, COL_ABS_IMB_REG_BIN]).copy()
    for col in [COL_ETA_REG_BIN, COL_VTV_REG_BIN, COL_ABS_IMB_REG_BIN]:
        labeled[col] = labeled[col].astype(int)

    cell_data = (
        labeled.groupby([COL_GROUP, COL_ETA_REG_BIN, COL_VTV_REG_BIN, COL_ABS_IMB_REG_BIN], dropna=False, sort=True)
        .agg(
            group_label=(COL_GROUP_LABEL, "first"),
            mean_imp=(COL_IMPACT, "mean"),
            std_imp=(COL_IMPACT, lambda s: s.std(ddof=1)),
            count=(COL_IMPACT, "size"),
            n_dates=(COL_DATE, "nunique"),
            n_isins=(COL_ISIN, "nunique"),
            mean_eta_raw=(COL_ETA, "mean"),
            mean_vtv_raw=(COL_VTV, "mean"),
            mean_abs_imb_raw=(COL_ABS_IMBALANCE, "mean"),
        )
        .reset_index()
    )
    cell_data["sem_imp"] = cell_data["std_imp"] / np.sqrt(np.maximum(cell_data["count"].to_numpy(dtype=float), 1.0))

    center_specs = [
        ("eta", COL_ETA_REG_BIN, "center_eta"),
        ("vtv", COL_VTV_REG_BIN, "center_vtv"),
        ("abs_imb", COL_ABS_IMB_REG_BIN, "center_abs_imb"),
    ]
    for dimension, bin_col, out_col in center_specs:
        centers = cutpoints.loc[cutpoints["dimension"] == dimension, [COL_GROUP, bin_col, "center_value"]].rename(
            columns={"center_value": out_col}
        )
        cell_data = cell_data.merge(centers, on=[COL_GROUP, bin_col], how="left")

    retained_mask = (
        (cell_data["count"].to_numpy(dtype=float) >= float(min_cell_count))
        & np.isfinite(cell_data["mean_imp"].to_numpy(dtype=float))
        & (cell_data["mean_imp"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(cell_data["sem_imp"].to_numpy(dtype=float))
        & (cell_data["sem_imp"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(cell_data["center_eta"].to_numpy(dtype=float))
        & (cell_data["center_eta"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(cell_data["center_vtv"].to_numpy(dtype=float))
        & (cell_data["center_vtv"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(cell_data["center_abs_imb"].to_numpy(dtype=float))
    )
    cell_data["retained_for_fit"] = retained_mask

    fit_rows: List[Dict[str, Any]] = []
    fitted_groups: List[pd.DataFrame] = []
    cell_keys = [COL_GROUP, COL_ETA_REG_BIN, COL_VTV_REG_BIN, COL_ABS_IMB_REG_BIN]
    for group_name in GROUP_ORDER:
        group_cells = cell_data.loc[cell_data[COL_GROUP] == group_name].copy()
        retained = group_cells.loc[group_cells["retained_for_fit"]].copy()
        base_row = {
            "variant": "joint_bin_regression",
            COL_GROUP: group_name,
            COL_GROUP_LABEL: GROUP_DISPLAY[group_name],
            "n_cells_total": int(len(group_cells)),
            "n_cells_retained": int(retained["retained_for_fit"].sum()),
            "eta_bins": int(n_eta_bins),
            "vtv_bins": int(n_vtv_bins),
            "abs_imb_bins": int(n_abs_imb_bins),
            "min_cell_count": int(min_cell_count),
        }
        try:
            fitted_cells, fit_stats = _fit_joint_bin_regression_wls(retained)
        except Exception as exc:
            fit_rows.append(
                {
                    **base_row,
                    "status": "fit_failed",
                    "error": str(exc),
                    "log_Y_hat": np.nan,
                    "log_Y_se": np.nan,
                    "Y_hat": np.nan,
                    "Y_se": np.nan,
                    "beta_log_eta_hat": np.nan,
                    "beta_log_eta_se": np.nan,
                    "beta_log_vtv_hat": np.nan,
                    "beta_log_vtv_se": np.nan,
                    "beta_abs_imb_hat": np.nan,
                    "beta_abs_imb_se": np.nan,
                    "r2_log": np.nan,
                    "r2_lin": np.nan,
                    "dof": np.nan,
                }
            )
            fitted_groups.append(group_cells)
            continue

        fit_rows.append({**base_row, **fit_stats})
        fitted_groups.append(
            group_cells.merge(
                fitted_cells[
                    cell_keys
                    + [
                        "log_mean_imp",
                        "fitted_log_mean_imp",
                        "fitted_mean_imp",
                        "residual_log_mean_imp",
                        "wls_weight",
                    ]
                ],
                on=cell_keys,
                how="left",
            )
        )

    fit_summary = pd.DataFrame(fit_rows)
    cell_table = pd.concat(fitted_groups, ignore_index=True) if fitted_groups else pd.DataFrame()
    return JointRegressionOutputs(
        cutpoints=cutpoints,
        cell_data=cell_table,
        fit_summary=fit_summary,
        group_differences=_build_joint_regression_group_differences(fit_summary),
    )


def _term_label(term: str) -> str:
    """Return a human-readable label for one pooled-regression coefficient."""
    return {
        "const": "Constant",
        COL_LOG_ETA: "log(η)",
        COL_LOG_VTV: "log(V_t/V)",
        COL_ABS_IMBALANCE: "|imb|",
        COL_PROP_DUMMY: "1{proprietary}",
        COL_CLIENT_DUMMY: "1{client}",
    }.get(term, str(term))


def _pooled_dummy_term(dummy_group: str) -> str:
    """Map the pooled-regression dummy target to its column name."""
    normalized = str(dummy_group).strip().lower()
    if normalized == "proprietary":
        return COL_PROP_DUMMY
    if normalized == "client":
        return COL_CLIENT_DUMMY
    raise ValueError(f"dummy_group must be one of {POOLED_DUMMY_GROUPS}; got {dummy_group!r}.")


def _pooled_log_regression_stem(stem: str, dummy_group: str) -> str:
    """Return the file stem associated with one dummy specification."""
    normalized = str(dummy_group).strip().lower()
    _pooled_dummy_term(normalized)
    if normalized == "proprietary":
        return stem
    suffix = f"{normalized}_dummy"
    if stem.endswith("_coefficients"):
        return f"{stem[:-len('_coefficients')]}_{suffix}_coefficients"
    return f"{stem}_{suffix}"


def _pooled_regression_stage_counts(df: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    """Count pooled, client, and proprietary rows at one regression-filter stage."""
    rows: List[Dict[str, Any]] = [
        {"stage": stage, COL_GROUP: "all", COL_GROUP_LABEL: "All", "n_rows": int(len(df))}
    ]
    for group_name in GROUP_ORDER:
        rows.append(
            {
                "stage": stage,
                COL_GROUP: group_name,
                COL_GROUP_LABEL: GROUP_DISPLAY[group_name],
                "n_rows": int((df[COL_GROUP] == group_name).sum()) if COL_GROUP in df.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def _prepare_pooled_log_regression_sample(
    working: pd.DataFrame,
    *,
    impact_mode: str,
    n_eta_bins: int,
    n_vtv_bins: int,
    n_abs_imb_bins: int,
    min_cell_count: int,
    dummy_group: str = "proprietary",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Summary
    -------
    Build the pooled cell-level sample used by the log-mean-impact regression.

    Parameters
    ----------
    working : pd.DataFrame
        Final crowding-analysis sample after the canonical fit, participation,
        and crowding filters.
    impact_mode : str
        Pooled regression convention. The standard-fit-aligned choice is
        ``"signed_mean"``, which bins signed metaorder impacts first and only
        then keeps cells with positive mean impact.
    n_eta_bins : int
        Number of common log bins for participation rate.
    n_vtv_bins : int
        Number of common log bins for `V_t / V`.
    n_abs_imb_bins : int
        Number of common linear bins for `|imb|`.
    min_cell_count : int
        Minimum number of metaorders retained per pooled cell.
    dummy_group : str
        Group whose indicator is used later in the pooled regression summary.
        Accepted values are ``"proprietary"`` and ``"client"``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Regression-ready cell table, stage-count table, and pooled cutpoints.

    Notes
    -----
    This preparation mirrors the standard impact-fit logic used elsewhere in the
    repository: all metaorders contribute to cell means, and the positivity
    restriction enters only at the cell-mean stage through ``mean_imp > 0`` and
    ``sem_imp > 0``.

    Examples
    --------
    >>> demo = pd.DataFrame({
    ...     "Date": pd.to_datetime(["2024-01-02"] * 8),
    ...     "ISIN": ["A"] * 8,
    ...     "group": ["client"] * 4 + ["proprietary"] * 4,
    ...     "group_label": ["Client"] * 4 + ["Proprietary"] * 4,
    ...     "Impact": [0.2, 0.1, 0.3, 0.25, 0.24, 0.22, 0.32, 0.30],
    ...     "Participation Rate": [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2],
    ...     "Vt/V": [0.03, 0.03, 0.04, 0.04, 0.03, 0.03, 0.04, 0.04],
    ...     "abs_imbalance": [0.2, 0.2, 0.4, 0.4, 0.2, 0.2, 0.4, 0.4],
    ... })
    >>> sample, counts, cutpoints = _prepare_pooled_log_regression_sample(
    ...     demo,
    ...     impact_mode="signed_mean",
    ...     n_eta_bins=2,
    ...     n_vtv_bins=2,
    ...     n_abs_imb_bins=2,
    ...     min_cell_count=2,
    ... )
    >>> int(counts.loc[counts["stage"] == "retained_cells", "n_rows"].max()) >= 2
    True
    """
    mode = str(impact_mode).strip().lower()
    if mode != "signed_mean":
        raise ValueError("impact_mode must be `signed_mean` for the pooled standard-fit-aligned regression.")
    normalized_dummy_group = str(dummy_group).strip().lower()
    _pooled_dummy_term(normalized_dummy_group)

    required = [COL_DATE, COL_ISIN, COL_GROUP, COL_GROUP_LABEL, COL_IMPACT, COL_ETA, COL_VTV, COL_ABS_IMBALANCE]
    _validate_required_columns(working, required, label="pooled_log_regression")

    sample = working.copy()
    for col in [COL_IMPACT, COL_ETA, COL_VTV, COL_ABS_IMBALANCE]:
        sample[col] = pd.to_numeric(sample[col], errors="coerce")
    sample[COL_DATE] = pd.to_datetime(sample[COL_DATE], errors="coerce")

    count_tables: List[pd.DataFrame] = [_pooled_regression_stage_counts(sample, stage="working_sample")]

    impact_mask = np.isfinite(sample[COL_IMPACT].to_numpy(dtype=float))
    sample = sample.loc[impact_mask].copy()
    count_tables.append(_pooled_regression_stage_counts(sample, stage="after_finite_impact"))

    eta_mask = np.isfinite(sample[COL_ETA].to_numpy(dtype=float)) & (sample[COL_ETA].to_numpy(dtype=float) > 0.0)
    sample = sample.loc[eta_mask].copy()
    count_tables.append(_pooled_regression_stage_counts(sample, stage="after_positive_eta"))

    vtv_mask = np.isfinite(sample[COL_VTV].to_numpy(dtype=float)) & (sample[COL_VTV].to_numpy(dtype=float) > 0.0)
    sample = sample.loc[vtv_mask].copy()
    count_tables.append(_pooled_regression_stage_counts(sample, stage="after_positive_vtv"))

    abs_imb_mask = np.isfinite(sample[COL_ABS_IMBALANCE].to_numpy(dtype=float))
    sample = sample.loc[abs_imb_mask].copy()
    count_tables.append(_pooled_regression_stage_counts(sample, stage="after_finite_abs_imb"))

    sample = sample.loc[sample[COL_DATE].notna()].copy().reset_index(drop=True)
    count_tables.append(_pooled_regression_stage_counts(sample, stage="after_valid_date"))

    for col in [COL_ETA_REG_BIN, COL_VTV_REG_BIN, COL_ABS_IMB_REG_BIN]:
        sample[col] = pd.Series(pd.array([pd.NA] * len(sample), dtype="Int64"), index=sample.index)

    cut_rows: List[Dict[str, Any]] = []
    edge_specs = [
        (COL_ETA, COL_ETA_REG_BIN, int(n_eta_bins), "eta", True),
        (COL_VTV, COL_VTV_REG_BIN, int(n_vtv_bins), "vtv", True),
        (COL_ABS_IMBALANCE, COL_ABS_IMB_REG_BIN, int(n_abs_imb_bins), "abs_imb", False),
    ]
    for value_col, bin_col, n_bins, dimension, use_log_center in edge_specs:
        edges = (
            _log_bin_edges(sample[value_col], n_bins=n_bins, label=f"pooled/{dimension}")
            if use_log_center
            else _linear_bin_edges(sample[value_col], n_bins=n_bins, label=f"pooled/{dimension}")
        )
        sample[bin_col] = _assign_bins(sample[value_col], edges)
        for idx in range(n_bins):
            left = float(edges[idx])
            right = float(edges[idx + 1])
            center = float(np.sqrt(left * right)) if use_log_center else float(0.5 * (left + right))
            cut_rows.append(
                {
                    "variant": "pooled_log_regression",
                    "dummy_group": normalized_dummy_group,
                    "dimension": dimension,
                    "bin_col": bin_col,
                    bin_col: int(idx),
                    "left_edge": left,
                    "right_edge": right,
                    "center_value": center,
                    "n_bins": int(n_bins),
                }
            )

    cutpoints = pd.DataFrame(cut_rows)
    sample = sample.dropna(subset=[COL_ETA_REG_BIN, COL_VTV_REG_BIN, COL_ABS_IMB_REG_BIN]).copy()
    for col in [COL_ETA_REG_BIN, COL_VTV_REG_BIN, COL_ABS_IMB_REG_BIN]:
        sample[col] = sample[col].astype(int)
    count_tables.append(_pooled_regression_stage_counts(sample, stage="assigned_bin_rows"))

    cell_data = (
        sample.groupby([COL_GROUP, COL_ETA_REG_BIN, COL_VTV_REG_BIN, COL_ABS_IMB_REG_BIN], dropna=False, sort=True)
        .agg(
            group_label=(COL_GROUP_LABEL, "first"),
            mean_imp=(COL_IMPACT, "mean"),
            std_imp=(COL_IMPACT, lambda s: s.std(ddof=1)),
            count=(COL_IMPACT, "size"),
            n_dates=(COL_DATE, "nunique"),
            n_isins=(COL_ISIN, "nunique"),
            mean_eta_raw=(COL_ETA, "mean"),
            mean_vtv_raw=(COL_VTV, "mean"),
            mean_abs_imb_raw=(COL_ABS_IMBALANCE, "mean"),
        )
        .reset_index()
    )
    cell_data["sem_imp"] = cell_data["std_imp"] / np.sqrt(np.maximum(cell_data["count"].to_numpy(dtype=float), 1.0))
    cell_data[COL_PROP_DUMMY] = (cell_data[COL_GROUP] == "proprietary").astype(int)
    cell_data[COL_CLIENT_DUMMY] = (cell_data[COL_GROUP] == "client").astype(int)
    cell_data["dummy_group"] = normalized_dummy_group
    count_tables.append(_pooled_regression_stage_counts(cell_data, stage="pooled_cell_grid"))

    center_specs = [
        ("eta", COL_ETA_REG_BIN, "center_eta"),
        ("vtv", COL_VTV_REG_BIN, "center_vtv"),
        ("abs_imb", COL_ABS_IMB_REG_BIN, "center_abs_imb"),
    ]
    for dimension, bin_col, out_col in center_specs:
        centers = cutpoints.loc[cutpoints["dimension"] == dimension, [bin_col, "center_value"]].rename(
            columns={"center_value": out_col}
        )
        cell_data = cell_data.merge(centers, on=[bin_col], how="left")

    retained_mask = (
        (cell_data["count"].to_numpy(dtype=float) >= float(min_cell_count))
        & np.isfinite(cell_data["mean_imp"].to_numpy(dtype=float))
        & (cell_data["mean_imp"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(cell_data["sem_imp"].to_numpy(dtype=float))
        & (cell_data["sem_imp"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(cell_data["center_eta"].to_numpy(dtype=float))
        & (cell_data["center_eta"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(cell_data["center_vtv"].to_numpy(dtype=float))
        & (cell_data["center_vtv"].to_numpy(dtype=float) > 0.0)
        & np.isfinite(cell_data["center_abs_imb"].to_numpy(dtype=float))
    )
    cell_data["retained_for_fit"] = retained_mask
    count_tables.append(_pooled_regression_stage_counts(cell_data.loc[cell_data["retained_for_fit"]].copy(), stage="retained_cells"))
    stage_counts = pd.concat(count_tables, ignore_index=True)
    stage_counts["dummy_group"] = normalized_dummy_group

    return cell_data, stage_counts, cutpoints


def _fit_pooled_log_regression(
    cell_data: pd.DataFrame,
    *,
    alpha: float,
    impact_mode: str,
    dummy_group: str = "proprietary",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Summary
    -------
    Fit the pooled cell-level log-mean-impact regression with analytic WLS SEs.

    Parameters
    ----------
    cell_data : pd.DataFrame
        Regression-ready cell table returned by
        `_prepare_pooled_log_regression_sample`.
    alpha : float
        Two-sided significance level used for the reported confidence interval.
    impact_mode : str
        Impact transform label carried into the fit summary.
    dummy_group : str
        Group encoded by the dummy regressor. Accepted values are
        ``"proprietary"`` and ``"client"``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Fitted cell table, one-row fit summary, and a long coefficient table.

    Notes
    -----
    The fitted specification is

    ``log(E[I | cell]) = a + b_eta log(eta_cell) + b_v log(V_t/V)_cell + b_c |imb|_cell + b_d 1{dummy_group_cell} + eps_cell``

    with WLS weights given by the delta-method variance of
    ``log(mean_imp_cell)``. This matches the standard fit machinery used in
    `scripts/run_analysis.py metaorders compute`.

    Examples
    --------
    >>> demo = pd.DataFrame({
    ...     "group": ["client", "proprietary", "client", "proprietary", "client", "proprietary"],
    ...     "group_label": ["Client", "Proprietary"] * 3,
    ...     "center_eta": [0.08, 0.08, 0.16, 0.16, 0.32, 0.32],
    ...     "center_vtv": [0.02, 0.02, 0.05, 0.05, 0.11, 0.11],
    ...     "center_abs_imb": [0.16, 0.16, 0.28, 0.28, 0.40, 0.40],
    ...     "mean_imp": [0.04, 0.05, 0.08, 0.10, 0.15, 0.20],
    ...     "sem_imp": [0.005, 0.006, 0.008, 0.009, 0.010, 0.012],
    ...     "count": [30, 30, 30, 30, 30, 30],
    ...     "is_proprietary": [0, 1, 0, 1, 0, 1],
    ...     "retained_for_fit": [True] * 6,
    ... })
    >>> fitted, summary, coefs = _fit_pooled_log_regression(demo, alpha=0.05, impact_mode="signed_mean")
    >>> set(coefs["term"]) >= {"const", "log_eta", "log_vtv", "abs_imbalance", "is_proprietary"}
    True
    """
    normalized_dummy_group = str(dummy_group).strip().lower()
    dummy_term = _pooled_dummy_term(normalized_dummy_group)
    required = [COL_GROUP, dummy_term, "center_eta", "center_vtv", "center_abs_imb", "mean_imp", "sem_imp", "retained_for_fit"]
    _validate_required_columns(cell_data, required, label="pooled_log_regression_fit")

    work = cell_data.loc[cell_data["retained_for_fit"]].copy()
    n_required = 5
    if len(work) < n_required:
        raise ValueError(
            f"Not enough retained pooled cells for WLS (got {len(work)}; need at least {n_required})."
        )

    log_eta = np.log(work["center_eta"].to_numpy(dtype=float))
    log_vtv = np.log(work["center_vtv"].to_numpy(dtype=float))
    abs_imb = work["center_abs_imb"].to_numpy(dtype=float)
    dummy_values = work[dummy_term].to_numpy(dtype=float)
    log_y = np.log(work["mean_imp"].to_numpy(dtype=float))
    var_logy = (work["sem_imp"].to_numpy(dtype=float) / work["mean_imp"].to_numpy(dtype=float)) ** 2
    w = np.where(np.isfinite(var_logy) & (var_logy > 0), 1.0 / var_logy, 0.0)
    if np.count_nonzero(w > 0) < n_required:
        raise ValueError("Not enough positive WLS weights for the pooled regression.")

    design = np.vstack([np.ones_like(log_y), log_eta, log_vtv, abs_imb, dummy_values]).T
    param_names = ["const", COL_LOG_ETA, COL_LOG_VTV, COL_ABS_IMBALANCE, dummy_term]
    sqrt_w = np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(design * sqrt_w[:, None], log_y * sqrt_w, rcond=None)

    fitted_log_y = design @ coef
    residuals = log_y - fitted_log_y
    rss = float(np.sum(w * residuals**2))
    dof = max(len(log_y) - design.shape[1], 1)
    s2 = rss / dof
    xtwx = design.T @ (w[:, None] * design)
    try:
        cov = s2 * np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        cov = s2 * np.linalg.pinv(xtwx)
    analytic_se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    t_values = np.divide(coef, analytic_se, out=np.full_like(coef, np.nan), where=analytic_se > 0)
    df_t = max(int(dof), 1)
    p_values = 2.0 * stats.t.sf(np.abs(t_values), df=df_t)
    t_crit = float(stats.t.ppf(1.0 - float(alpha) / 2.0, df=float(dof)))
    conf_int = np.column_stack([coef - t_crit * analytic_se, coef + t_crit * analytic_se])

    tss = float(np.sum(w * np.square(log_y - float(np.average(log_y, weights=w)))))
    r2 = float(1.0 - rss / tss) if tss > 0 else float("nan")
    n_obs = int(len(work))
    n_params = int(design.shape[1])
    adj_r2 = float(1.0 - (1.0 - r2) * (n_obs - 1.0) / (n_obs - n_params)) if n_obs > n_params else float("nan")

    term_order = {name: idx for idx, name in enumerate(param_names)}
    coef_rows: List[Dict[str, Any]] = []
    coef_lookup: Dict[str, Dict[str, float]] = {}
    for idx, term in enumerate(param_names):
        row = {
            "term": term,
            "term_label": _term_label(term),
            "term_order": int(term_order[term]),
            "estimate": float(coef[idx]),
            "analytic_se": float(analytic_se[idx]),
            "analytic_pvalue": float(p_values[idx]),
            "analytic_ci_low": float(conf_int[idx, 0]),
            "analytic_ci_high": float(conf_int[idx, 1]),
        }
        coef_rows.append(row)
        coef_lookup[term] = row

    beta_dummy = coef_lookup[dummy_term]["estimate"]
    fitted_mean_imp = np.exp(fitted_log_y)
    work["log_mean_imp"] = log_y
    work["fitted_log_mean_imp"] = fitted_log_y
    work["fitted_mean_imp"] = fitted_mean_imp
    work["residual_log_mean_imp"] = residuals
    work["wls_weight"] = w

    fit_summary = pd.DataFrame(
        [
            {
                "status": "ok",
                "error": "",
                "impact_mode": str(impact_mode),
                "dummy_group": normalized_dummy_group,
                "dummy_term": dummy_term,
                "n_cells_total": int(len(cell_data)),
                "n_obs": int(n_obs),
                "n_groups": int(work[COL_GROUP].nunique()),
                "covariance": "wls(sem_imp)",
                "r2": float(r2),
                "adj_r2": float(adj_r2),
                "dof": float(dof),
                "beta_const_hat": float(coef_lookup["const"]["estimate"]),
                "beta_log_eta_hat": float(coef_lookup[COL_LOG_ETA]["estimate"]),
                "beta_log_vtv_hat": float(coef_lookup[COL_LOG_VTV]["estimate"]),
                "beta_abs_imb_hat": float(coef_lookup[COL_ABS_IMBALANCE]["estimate"]),
                "beta_dummy_hat": float(beta_dummy),
                "beta_dummy_analytic_se": float(coef_lookup[dummy_term]["analytic_se"]),
                "beta_dummy_analytic_pvalue": float(coef_lookup[dummy_term]["analytic_pvalue"]),
                "beta_dummy_analytic_ci_low": float(coef_lookup[dummy_term]["analytic_ci_low"]),
                "beta_dummy_analytic_ci_high": float(coef_lookup[dummy_term]["analytic_ci_high"]),
                "dummy_multiplier": float(np.exp(beta_dummy)),
                "dummy_pct_effect": float(100.0 * (np.exp(beta_dummy) - 1.0)),
            }
        ]
    )
    cell_keys = [COL_GROUP, COL_ETA_REG_BIN, COL_VTV_REG_BIN, COL_ABS_IMB_REG_BIN]
    merged_cell_data = cell_data.merge(
        work[
            cell_keys
            + [
                "log_mean_imp",
                "fitted_log_mean_imp",
                "fitted_mean_imp",
                "residual_log_mean_imp",
                "wls_weight",
            ]
        ],
        on=cell_keys,
        how="left",
    )
    merged_cell_data["dummy_group"] = normalized_dummy_group
    coefficients = pd.DataFrame(coef_rows)
    coefficients["dummy_group"] = normalized_dummy_group
    return merged_cell_data, fit_summary, coefficients


def _analyse_pooled_log_regression(
    working: pd.DataFrame,
    *,
    alpha: float,
    impact_mode: str,
    n_eta_bins: int,
    n_vtv_bins: int,
    n_abs_imb_bins: int,
    min_cell_count: int,
    dummy_group: str = "proprietary",
) -> PooledLogRegressionOutputs:
    """Run the pooled cell-level log-mean-impact regression."""
    normalized_dummy_group = str(dummy_group).strip().lower()
    dummy_term = _pooled_dummy_term(normalized_dummy_group)
    sample, sample_counts, cutpoints = _prepare_pooled_log_regression_sample(
        working,
        impact_mode=impact_mode,
        n_eta_bins=n_eta_bins,
        n_vtv_bins=n_vtv_bins,
        n_abs_imb_bins=n_abs_imb_bins,
        min_cell_count=min_cell_count,
        dummy_group=normalized_dummy_group,
    )
    try:
        cell_data, fit_summary, coefficients = _fit_pooled_log_regression(
            sample,
            alpha=alpha,
            impact_mode=impact_mode,
            dummy_group=normalized_dummy_group,
        )
    except Exception as exc:
        cell_data = sample.copy()
        cell_data["dummy_group"] = normalized_dummy_group
        fit_summary = pd.DataFrame(
            [
                {
                    "status": "fit_failed",
                    "error": str(exc),
                    "impact_mode": str(impact_mode),
                    "dummy_group": normalized_dummy_group,
                    "dummy_term": dummy_term,
                    "n_cells_total": int(len(sample)),
                    "n_obs": int(len(sample)),
                    "n_groups": int(sample[COL_GROUP].nunique()) if COL_GROUP in sample.columns else 0,
                    "covariance": "wls(sem_imp)",
                    "r2": np.nan,
                    "adj_r2": np.nan,
                    "dof": np.nan,
                    "beta_const_hat": np.nan,
                    "beta_log_eta_hat": np.nan,
                    "beta_log_vtv_hat": np.nan,
                    "beta_abs_imb_hat": np.nan,
                    "beta_dummy_hat": np.nan,
                    "beta_dummy_analytic_se": np.nan,
                    "beta_dummy_analytic_pvalue": np.nan,
                    "beta_dummy_analytic_ci_low": np.nan,
                    "beta_dummy_analytic_ci_high": np.nan,
                    "dummy_multiplier": np.nan,
                    "dummy_pct_effect": np.nan,
                }
            ]
        )
        coefficients = pd.DataFrame(columns=["dummy_group"])
    return PooledLogRegressionOutputs(
        sample_counts=sample_counts,
        cutpoints=cutpoints,
        cell_data=cell_data,
        fit_summary=fit_summary,
        coefficients=coefficients,
    )


def _fit_slice(
    sub: pd.DataFrame,
    *,
    n_logbins: int,
    min_count: int,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    try:
        binned, params = fit_power_law_logbins_wls_new(
            sub[["Q/V", "Impact"]].copy(),
            n_logbins=n_logbins,
            min_count=min_count,
        )
    except Exception as exc:
        return None, {
            "status": "fit_failed",
            "error": str(exc),
            "Y_hat": np.nan,
            "Y_se": np.nan,
            "gamma_hat": np.nan,
            "gamma_se": np.nan,
            "r2_log": np.nan,
            "r2_lin": np.nan,
            "n_bins_retained": 0,
        }

    y_hat, y_se, gamma_hat, gamma_se, r2_log, r2_lin, _, _ = params
    return binned, {
        "status": "ok",
        "error": "",
        "Y_hat": float(y_hat),
        "Y_se": float(y_se),
        "gamma_hat": float(gamma_hat),
        "gamma_se": float(gamma_se),
        "r2_log": float(r2_log),
        "r2_lin": float(r2_lin),
        "n_bins_retained": int(len(binned)),
    }


def _predict_from_fit(y_hat: float, gamma_hat: float, benchmark_phi: float) -> float:
    if not np.isfinite(y_hat) or not np.isfinite(gamma_hat) or benchmark_phi <= 0:
        return float("nan")
    return float(power_law(np.asarray([benchmark_phi], dtype=float), y_hat, gamma_hat)[0])


def _fit_variant(
    labeled: pd.DataFrame,
    *,
    variant: str,
    slice_cols: Sequence[str],
    benchmark_phis: Sequence[float],
    n_logbins: int,
    min_count: int,
    store_binned: bool,
) -> VariantOutputs:
    sample_sizes = (
        labeled.groupby(list(slice_cols), dropna=False, sort=True)
        .agg(
            n_metaorders=(COL_METAORDER_ID, "size"),
            n_dates=(COL_DATE, "nunique"),
            n_isins=(COL_ISIN, "nunique"),
            crowding_mean=(COL_ALIGNED_CROWDING, "mean"),
            crowding_min=(COL_ALIGNED_CROWDING, "min"),
            crowding_max=(COL_ALIGNED_CROWDING, "max"),
            eta_mean=(COL_ETA, "mean"),
        )
        .reset_index()
        .assign(variant=variant)
    )

    fit_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    binned_rows: List[pd.DataFrame] = []

    for key, sub in labeled.groupby(list(slice_cols), dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = {col: key_tuple[idx] for idx, col in enumerate(slice_cols)}
        base_row = {
            "variant": variant,
            **key_map,
            COL_GROUP_LABEL: sub[COL_GROUP_LABEL].iat[0],
            "n_metaorders": int(len(sub)),
            "n_dates": int(sub[COL_DATE].nunique()),
            "n_isins": int(sub[COL_ISIN].nunique()),
        }
        if COL_CROWDING_LABEL in sub.columns:
            base_row[COL_CROWDING_LABEL] = sub[COL_CROWDING_LABEL].iat[0]
        if COL_ETA_LABEL in sub.columns:
            base_row[COL_ETA_LABEL] = sub[COL_ETA_LABEL].iat[0]

        binned, fit_stats = _fit_slice(sub, n_logbins=n_logbins, min_count=min_count)
        fit_row = {**base_row, **fit_stats}
        fit_rows.append(fit_row)

        for benchmark_phi in benchmark_phis:
            pred_rows.append(
                {
                    **base_row,
                    "benchmark_phi": float(benchmark_phi),
                    "predicted_impact": _predict_from_fit(
                        fit_stats["Y_hat"],
                        fit_stats["gamma_hat"],
                        float(benchmark_phi),
                    ),
                    "status": fit_stats["status"],
                }
            )

        if binned is not None and store_binned:
            binned_rows.append(
                binned.assign(
                    variant=variant,
                    **key_map,
                    **{
                        COL_GROUP_LABEL: sub[COL_GROUP_LABEL].iat[0],
                        COL_CROWDING_LABEL: sub[COL_CROWDING_LABEL].iat[0],
                    },
                    **(
                        {COL_ETA_LABEL: sub[COL_ETA_LABEL].iat[0]}
                        if COL_ETA_LABEL in sub.columns
                        else {}
                    ),
                )
            )

    fit_summary = pd.DataFrame(fit_rows)
    predictions = pd.DataFrame(pred_rows)
    binned_curve_data = pd.concat(binned_rows, ignore_index=True) if binned_rows else pd.DataFrame()
    contrasts = _build_contrast_table(predictions, variant=variant)
    group_differences = _build_group_difference_table(contrasts, variant=variant)
    return VariantOutputs(
        labeled=labeled,
        cutpoints=pd.DataFrame(),
        sample_sizes=sample_sizes,
        fit_summary=fit_summary,
        predictions=predictions,
        contrasts=contrasts,
        group_differences=group_differences,
        binned_curve_data=binned_curve_data,
    )


def _build_contrast_table(predictions: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    context_cols = [COL_GROUP]
    if COL_ETA_BIN in predictions.columns:
        context_cols.append(COL_ETA_BIN)
    label_cols = [COL_GROUP_LABEL]
    if COL_CROWDING_LABEL in predictions.columns:
        label_cols.append(COL_CROWDING_LABEL)
    if COL_ETA_LABEL in predictions.columns:
        label_cols.append(COL_ETA_LABEL)

    contrast_rows: List[Dict[str, Any]] = []
    by_cols = context_cols + ["benchmark_phi"]
    for key, sub in predictions.groupby(by_cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = {col: key_tuple[idx] for idx, col in enumerate(by_cols)}
        pivot = sub.set_index(COL_CROWDING_BIN)["predicted_impact"].to_dict()
        if 0 not in pivot or (len(pivot) - 1) not in pivot:
            continue
        first = 0
        last = max(pivot.keys())
        base = {
            "variant": variant,
            **key_map,
            COL_GROUP_LABEL: sub[COL_GROUP_LABEL].iat[0],
        }
        if COL_ETA_LABEL in sub.columns:
            base[COL_ETA_LABEL] = sub[COL_ETA_LABEL].iat[0]
        contrast_rows.append(
            {
                **base,
                "contrast_name": "high_minus_low",
                "point_estimate": float(pivot[last] - pivot[first]),
            }
        )
        if 1 in pivot:
            contrast_rows.append(
                {
                    **base,
                    "contrast_name": "mid_minus_low",
                    "point_estimate": float(pivot[1] - pivot[first]),
                }
            )
            contrast_rows.append(
                {
                    **base,
                    "contrast_name": "high_minus_mid",
                    "point_estimate": float(pivot[last] - pivot[1]),
                }
            )
    return pd.DataFrame(contrast_rows)


def _build_group_difference_table(contrasts: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    if contrasts.empty:
        return pd.DataFrame()
    target = contrasts[contrasts["contrast_name"] == "high_minus_low"].copy()
    if target.empty:
        return pd.DataFrame()
    context_cols = ["benchmark_phi"]
    if COL_ETA_BIN in target.columns:
        context_cols.append(COL_ETA_BIN)

    rows: List[Dict[str, Any]] = []
    for key, sub in target.groupby(context_cols, dropna=False, sort=True):
        prop_row = sub.loc[sub[COL_GROUP] == "proprietary"]
        client_row = sub.loc[sub[COL_GROUP] == "client"]
        if prop_row.empty or client_row.empty:
            continue
        prop_val = float(prop_row["point_estimate"].iat[0])
        client_val = float(client_row["point_estimate"].iat[0])
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = {col: key_tuple[idx] for idx, col in enumerate(context_cols)}
        row = {
            "variant": variant,
            **key_map,
            "contrast_name": "high_minus_low",
            "prop_high_minus_low": prop_val,
            "client_high_minus_low": client_val,
            "prop_minus_client": prop_val - client_val,
        }
        if COL_ETA_LABEL in sub.columns and sub[COL_ETA_LABEL].notna().any():
            row[COL_ETA_LABEL] = sub.loc[sub[COL_ETA_LABEL].notna(), COL_ETA_LABEL].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _analyse_main_variant(
    working: pd.DataFrame,
    *,
    n_crowding_quantiles: int,
    benchmark_phis: Sequence[float],
    n_logbins: int,
    min_count: int,
    store_binned: bool,
) -> VariantOutputs:
    labeled, cutpoints = _add_quantile_bins(
        working,
        value_col=COL_ALIGNED_CROWDING,
        by_cols=[COL_GROUP],
        n_bins=n_crowding_quantiles,
        bin_col=COL_CROWDING_BIN,
        label_col=COL_CROWDING_LABEL,
        label_map=_crowding_label_map(n_crowding_quantiles),
        variant="main",
        dimension="crowding",
    )
    out = _fit_variant(
        labeled,
        variant="main",
        slice_cols=[COL_GROUP, COL_CROWDING_BIN],
        benchmark_phis=benchmark_phis,
        n_logbins=n_logbins,
        min_count=min_count,
        store_binned=store_binned,
    )
    return VariantOutputs(
        labeled=labeled,
        cutpoints=cutpoints,
        sample_sizes=out.sample_sizes,
        fit_summary=out.fit_summary,
        predictions=out.predictions,
        contrasts=out.contrasts,
        group_differences=out.group_differences,
        binned_curve_data=out.binned_curve_data,
    )


def _analyse_eta_variant(
    working: pd.DataFrame,
    *,
    n_eta_bins: int,
    n_crowding_quantiles: int,
    benchmark_phis: Sequence[float],
    n_logbins: int,
    min_count: int,
    store_binned: bool,
) -> VariantOutputs:
    with_eta, eta_cutpoints = _add_quantile_bins(
        working,
        value_col=COL_ETA,
        by_cols=[COL_GROUP],
        n_bins=n_eta_bins,
        bin_col=COL_ETA_BIN,
        label_col=COL_ETA_LABEL,
        label_map=_eta_label_map(n_eta_bins),
        variant="eta_robustness",
        dimension="eta",
    )
    labeled, crowding_cutpoints = _add_quantile_bins(
        with_eta,
        value_col=COL_ALIGNED_CROWDING,
        by_cols=[COL_GROUP, COL_ETA_BIN],
        n_bins=n_crowding_quantiles,
        bin_col=COL_CROWDING_BIN,
        label_col=COL_CROWDING_LABEL,
        label_map=_crowding_label_map(n_crowding_quantiles),
        variant="eta_robustness",
        dimension="crowding",
    )
    out = _fit_variant(
        labeled,
        variant="eta_robustness",
        slice_cols=[COL_GROUP, COL_ETA_BIN, COL_CROWDING_BIN],
        benchmark_phis=benchmark_phis,
        n_logbins=n_logbins,
        min_count=min_count,
        store_binned=store_binned,
    )
    cutpoints = pd.concat([eta_cutpoints, crowding_cutpoints], ignore_index=True)
    return VariantOutputs(
        labeled=labeled,
        cutpoints=cutpoints,
        sample_sizes=out.sample_sizes,
        fit_summary=out.fit_summary,
        predictions=out.predictions,
        contrasts=out.contrasts,
        group_differences=out.group_differences,
        binned_curve_data=out.binned_curve_data,
    )


def _bootstrap_variant(
    sampler: DateBootstrapSampler,
    *,
    analyse_fn,
    bootstrap_runs: int,
    seed: int,
    show_progress: bool,
) -> Dict[str, pd.DataFrame]:
    if bootstrap_runs < 1:
        return {"fit_summary": pd.DataFrame(), "predictions": pd.DataFrame(), "contrasts": pd.DataFrame(), "group_differences": pd.DataFrame()}

    rng = np.random.default_rng(int(seed))
    fit_rows: List[pd.DataFrame] = []
    pred_rows: List[pd.DataFrame] = []
    contrast_rows: List[pd.DataFrame] = []
    diff_rows: List[pd.DataFrame] = []
    skipped_runs = 0

    for rep in range(bootstrap_runs):
        boot = sampler.sample(rng)
        try:
            out = analyse_fn(boot, store_binned=False)
        except Exception as exc:
            skipped_runs += 1
            if show_progress:
                print(f"[Bootstrap] Skipped replicate {rep + 1}/{bootstrap_runs}: {exc}")
            continue

        if not out.fit_summary.empty:
            fit_rows.append(out.fit_summary.assign(replicate=rep))
        if not out.predictions.empty:
            pred_rows.append(out.predictions.assign(replicate=rep))
        if not out.contrasts.empty:
            contrast_rows.append(out.contrasts.assign(replicate=rep))
        if not out.group_differences.empty:
            diff_rows.append(out.group_differences.assign(replicate=rep))

        if show_progress and ((rep + 1) % 10 == 0 or rep == 0 or rep + 1 == bootstrap_runs):
            print(f"[Bootstrap] Completed {rep + 1}/{bootstrap_runs} Date-cluster replicates.")

    if skipped_runs > 0:
        print(f"[Bootstrap] Skipped {skipped_runs}/{bootstrap_runs} ill-posed Date-cluster replicates.")

    return {
        "fit_summary": pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame(),
        "predictions": pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame(),
        "contrasts": pd.concat(contrast_rows, ignore_index=True) if contrast_rows else pd.DataFrame(),
        "group_differences": pd.concat(diff_rows, ignore_index=True) if diff_rows else pd.DataFrame(),
    }


def _bootstrap_joint_regression(
    sampler: DateBootstrapSampler,
    *,
    analyse_fn,
    bootstrap_runs: int,
    seed: int,
    show_progress: bool,
) -> Dict[str, pd.DataFrame]:
    """Bootstrap the joint-bin regression with Date-cluster resampling."""
    if bootstrap_runs < 1:
        return {"fit_summary": pd.DataFrame(), "group_differences": pd.DataFrame()}

    rng = np.random.default_rng(int(seed))
    fit_rows: List[pd.DataFrame] = []
    diff_rows: List[pd.DataFrame] = []
    skipped_runs = 0

    for rep in range(bootstrap_runs):
        boot = sampler.sample(rng)
        try:
            out = analyse_fn(boot)
        except Exception as exc:
            skipped_runs += 1
            if show_progress:
                print(f"[Bootstrap joint regression] Skipped replicate {rep + 1}/{bootstrap_runs}: {exc}")
            continue

        if not out.fit_summary.empty:
            fit_rows.append(out.fit_summary.assign(replicate=rep))
        if not out.group_differences.empty:
            diff_rows.append(out.group_differences.assign(replicate=rep))

        if show_progress and ((rep + 1) % 10 == 0 or rep == 0 or rep + 1 == bootstrap_runs):
            print(f"[Bootstrap joint regression] Completed {rep + 1}/{bootstrap_runs} Date-cluster replicates.")

    if skipped_runs > 0:
        print(
            f"[Bootstrap joint regression] Skipped {skipped_runs}/{bootstrap_runs} "
            "ill-posed Date-cluster replicates."
        )

    return {
        "fit_summary": pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame(),
        "group_differences": pd.concat(diff_rows, ignore_index=True) if diff_rows else pd.DataFrame(),
    }


def _bootstrap_pooled_log_regression(
    sampler: DateBootstrapSampler,
    *,
    analyse_fn,
    bootstrap_runs: int,
    seed: int,
    show_progress: bool,
) -> Dict[str, pd.DataFrame]:
    """Bootstrap the pooled cell-level log-mean-impact regression with Date resampling."""
    if bootstrap_runs < 1:
        return {"coefficients": pd.DataFrame()}

    rng = np.random.default_rng(int(seed))
    coefficient_rows: List[pd.DataFrame] = []
    skipped_runs = 0

    for rep in range(bootstrap_runs):
        boot = sampler.sample(rng)
        try:
            out = analyse_fn(boot)
        except Exception as exc:
            skipped_runs += 1
            if show_progress:
                print(f"[Bootstrap pooled regression] Skipped replicate {rep + 1}/{bootstrap_runs}: {exc}")
            continue

        if not out.coefficients.empty:
            coefficient_rows.append(out.coefficients.assign(replicate=rep))
        else:
            skipped_runs += 1
            if show_progress:
                print(f"[Bootstrap pooled regression] Skipped replicate {rep + 1}/{bootstrap_runs}: no coefficients.")
            continue

        if show_progress and ((rep + 1) % 10 == 0 or rep == 0 or rep + 1 == bootstrap_runs):
            print(f"[Bootstrap pooled regression] Completed {rep + 1}/{bootstrap_runs} Date-cluster replicates.")

    if skipped_runs > 0:
        print(
            f"[Bootstrap pooled regression] Skipped {skipped_runs}/{bootstrap_runs} "
            "ill-posed Date-cluster replicates."
        )

    return {
        "coefficients": pd.concat(coefficient_rows, ignore_index=True) if coefficient_rows else pd.DataFrame(),
    }


def _bootstrap_standard_error(samples: np.ndarray) -> float:
    """Estimate the standard error from bootstrap replicate dispersion."""
    clean = np.asarray(samples, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return float("nan")
    return float(np.std(clean, ddof=1))


def _percentile_interval(samples: np.ndarray, alpha: float) -> Tuple[float, float, int]:
    clean = np.asarray(samples, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan"), 0
    lo = float(np.percentile(clean, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(clean, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi, int(clean.size)


def _bootstrap_ci_from_table(
    point_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    *,
    id_cols: Sequence[str],
    value_cols: Sequence[str],
    alpha: float,
) -> pd.DataFrame:
    if point_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    grouped_boot: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    if not boot_df.empty:
        grouped_boot = {key: sub for key, sub in boot_df.groupby(list(id_cols), dropna=False, sort=False)}

    for _, point_row in point_df.iterrows():
        key = tuple(point_row[col] for col in id_cols)
        boot_sub = grouped_boot.get(key)
        for value_col in value_cols:
            if boot_sub is None or value_col not in boot_sub.columns:
                lo = hi = float("nan")
                valid = 0
                se = float("nan")
            else:
                samples = boot_sub[value_col].to_numpy(dtype=float)
                lo, hi, valid = _percentile_interval(samples, alpha=alpha)
                se = _bootstrap_standard_error(samples)
            rows.append(
                {
                    **{col: point_row[col] for col in id_cols},
                    "metric": value_col,
                    "point_estimate": float(point_row[value_col]) if pd.notna(point_row[value_col]) else float("nan"),
                    "ci_low": lo,
                    "ci_high": hi,
                    "se": se,
                    "bootstrap_valid_runs": valid,
                }
            )
    return pd.DataFrame(rows)


def _cluster_ci_from_coefficient_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Convert analytic coefficient intervals into the common CI-table schema."""
    if coefficients.empty:
        return pd.DataFrame()
    if {"cluster_ci_low", "cluster_ci_high"}.issubset(coefficients.columns):
        lo_col = "cluster_ci_low"
        hi_col = "cluster_ci_high"
    elif {"analytic_ci_low", "analytic_ci_high"}.issubset(coefficients.columns):
        lo_col = "analytic_ci_low"
        hi_col = "analytic_ci_high"
    else:
        raise KeyError("Coefficient table must contain either cluster_* or analytic_* confidence-interval columns.")
    required = ["term", "term_label", "term_order", "estimate", lo_col, hi_col]
    _validate_required_columns(coefficients, required, label="coefficient_ci_fallback")
    rows: List[Dict[str, Any]] = []
    for _, row in coefficients.iterrows():
        rows.append(
            {
                "term": row["term"],
                "term_label": row["term_label"],
                "term_order": int(row["term_order"]),
                "metric": "estimate",
                "point_estimate": float(row["estimate"]),
                "ci_low": float(row[lo_col]),
                "ci_high": float(row[hi_col]),
                "bootstrap_valid_runs": 0,
            }
        )
    return pd.DataFrame(rows)


def _build_filter_counts(
    label: str,
    raw_n: int,
    filtered_n: int,
    eta_filtered_n: int,
    crowding_n: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"group": label, "stage": "loaded", "n_rows": int(raw_n)},
            {"group": label, "stage": "after_fit_filters", "n_rows": int(filtered_n)},
            {"group": label, "stage": "after_eta_filter", "n_rows": int(eta_filtered_n)},
            {"group": label, "stage": "after_crowding_filter", "n_rows": int(crowding_n)},
        ]
    )


def _load_group_frame(
    path: Path,
    *,
    label: str,
    group_name: str,
    fit_defaults: FitDefaults,
) -> GroupLoadResult:
    wanted_cols = [
        COL_ISIN,
        COL_DATE,
        COL_PERIOD,
        COL_DIRECTION,
        "Price Change",
        "Daily Vol",
        COL_Q,
        COL_QV,
        COL_VTV,
        COL_ETA,
        "Impact",
        COL_IMBALANCE,
    ]
    raw = _read_parquet_with_fallback(path, columns=wanted_cols)
    raw = _ensure_date_column(raw, label=label)
    raw_n = len(raw)

    filtered = filter_metaorders_info_for_fits(raw, min_qv=fit_defaults.min_qv)
    filtered = _ensure_date_column(filtered, label=label)
    filtered_n = len(filtered)

    if COL_IMBALANCE not in filtered.columns:
        filtered = _compute_imbalance_local(filtered)
        imbalance_source = "recomputed"
    else:
        imbalance_source = "reused"

    filtered[COL_ETA] = pd.to_numeric(filtered[COL_ETA], errors="coerce")
    eta_mask = np.isfinite(filtered[COL_ETA].to_numpy(dtype=float)) & (filtered[COL_ETA] > 0.0) & (
        filtered[COL_ETA] < fit_defaults.eta_max
    )
    filtered = filtered.loc[eta_mask].copy()
    eta_filtered_n = len(filtered)

    if COL_VTV not in filtered.columns:
        filtered[COL_VTV] = filtered[COL_QV] / filtered[COL_ETA]
    filtered[COL_VTV] = pd.to_numeric(filtered[COL_VTV], errors="coerce")
    filtered[COL_IMBALANCE] = pd.to_numeric(filtered[COL_IMBALANCE], errors="coerce")
    filtered = filtered.loc[np.isfinite(filtered[COL_VTV].to_numpy(dtype=float)) & (filtered[COL_VTV] > 0.0)].copy()

    filtered[COL_GROUP] = group_name
    filtered[COL_GROUP_LABEL] = GROUP_DISPLAY[group_name]
    filtered[COL_METAORDER_ID] = [f"{group_name}_{idx}" for idx in range(len(filtered))]
    filtered = filtered[
        [
            COL_METAORDER_ID,
            COL_DATE,
            COL_ISIN,
            COL_GROUP,
            COL_GROUP_LABEL,
            COL_DIRECTION,
            COL_Q,
            COL_QV,
            COL_VTV,
            COL_ETA,
            COL_IMPACT,
            COL_IMBALANCE,
        ]
    ].reset_index(drop=True)

    print(
        f"[Load] {GROUP_DISPLAY[group_name]} metaorders: "
        f"loaded={raw_n:,}, after_fit_filters={filtered_n:,}, "
        f"after_eta_filter={eta_filtered_n:,} "
        f"(imbalance {imbalance_source})."
    )
    return GroupLoadResult(
        frame=filtered,
        raw_n=raw_n,
        fit_filtered_n=filtered_n,
        eta_filtered_n=eta_filtered_n,
        imbalance_source=imbalance_source,
    )


def _prepare_working_sample(
    client_result: GroupLoadResult,
    prop_result: GroupLoadResult,
    *,
    crowding_scope: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summary
    -------
    Build the final working sample for the selected crowding definition.

    Parameters
    ----------
    client_result : GroupLoadResult
        Loaded client sample after the fit and eta filters.
    prop_result : GroupLoadResult
        Loaded proprietary sample after the fit and eta filters.
    crowding_scope : str
        Crowding environment definition. Supported values:
        - ``within_group``: same-group leave-one-out imbalance
        - ``all``: all-metaorder leave-one-out imbalance on the same stock-day

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Final working metaorder table plus a filter-count table.

    Notes
    -----
    The all-metaorder specification keeps client/proprietary groups separate in
    the reported fits, but it computes the crowding environment on the pooled
    sample over each `(ISIN, Date)` cell.

    Examples
    --------
    >>> # Example omitted; requires real GroupLoadResult inputs.
    """
    crowding_scope_norm = str(crowding_scope).strip().lower()
    if crowding_scope_norm not in {"within_group", "all"}:
        raise ValueError("crowding_scope must be one of: within_group, all")

    working = pd.concat([client_result.frame, prop_result.frame], ignore_index=True)
    selected_imbalance_col = COL_IMBALANCE
    imbalance_note = "within-group leave-one-out"
    if crowding_scope_norm == "all":
        working = _compute_imbalance_local(
            working,
            group_cols=[COL_ISIN, COL_DATE],
            side_col=COL_DIRECTION,
            vol_col=COL_Q,
            out_col=COL_ALL_IMBALANCE,
        )
        selected_imbalance_col = COL_ALL_IMBALANCE
        imbalance_note = "all-metaorder leave-one-out"

    working[selected_imbalance_col] = pd.to_numeric(working[selected_imbalance_col], errors="coerce")
    crowding_mask = np.isfinite(working[selected_imbalance_col].to_numpy(dtype=float))
    working = working.loc[crowding_mask].copy().reset_index(drop=True)
    working["crowding_scope"] = crowding_scope_norm
    working["crowding_definition"] = imbalance_note
    working["crowding_input_col"] = selected_imbalance_col
    working[COL_ALIGNED_CROWDING] = (
        pd.to_numeric(working[COL_DIRECTION], errors="coerce").to_numpy(dtype=float)
        * working[selected_imbalance_col].to_numpy(dtype=float)
    )
    working[COL_ABS_IMBALANCE] = np.abs(working[selected_imbalance_col].to_numpy(dtype=float))

    if not working.empty and (np.abs(working[COL_ALIGNED_CROWDING].to_numpy(dtype=float)) > 1.0 + 1.0e-9).any():
        raise ValueError("Found |aligned crowding| > 1. This suggests a malformed crowding definition.")

    count_rows: List[pd.DataFrame] = []
    for group_name, group_result in [("client", client_result), ("proprietary", prop_result)]:
        crowding_n = int((working[COL_GROUP] == group_name).sum())
        count_rows.append(
            _build_filter_counts(
                group_name,
                group_result.raw_n,
                group_result.fit_filtered_n,
                group_result.eta_filtered_n,
                crowding_n,
            )
        )
        print(
            f"[Crowding] {GROUP_DISPLAY[group_name]} sample under `{crowding_scope_norm}` crowding: "
            f"after_crowding_filter={crowding_n:,}."
        )

    return working, pd.concat(count_rows, ignore_index=True)


def _plot_main_curves(
    outputs: VariantOutputs,
    dirs: PlotOutputDirs,
    *,
    benchmark_phis: Sequence[float],
    write_html: bool,
    write_png: bool,
) -> None:
    if outputs.fit_summary.empty or outputs.binned_curve_data.empty:
        return
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Client", "Proprietary"],
        shared_xaxes="all",
        shared_yaxes="all",
    )
    _ = benchmark_phis  # Benchmarks are used in downstream comparison plots, not on the full-curve figure.

    for col_idx, group in enumerate(GROUP_ORDER, start=1):
        group_fit = outputs.fit_summary[(outputs.fit_summary[COL_GROUP] == group) & (outputs.fit_summary["status"] == "ok")]
        group_bins = outputs.binned_curve_data[outputs.binned_curve_data[COL_GROUP] == group]
        for crowding_bin in sorted(group_fit[COL_CROWDING_BIN].dropna().unique()):
            fit_row = group_fit.loc[group_fit[COL_CROWDING_BIN] == crowding_bin].iloc[0]
            binned = group_bins.loc[group_bins[COL_CROWDING_BIN] == crowding_bin].copy()
            if binned.empty:
                continue
            color = CROWDING_COLORS.get(int(crowding_bin), BENCHMARK_COLORS[int(crowding_bin) % len(BENCHMARK_COLORS)])
            label = str(fit_row[COL_CROWDING_LABEL])
            fig.add_trace(
                go.Scatter(
                    x=binned["center_QV"],
                    y=binned["mean_imp"],
                    mode="markers",
                    marker=dict(size=7, color=color),
                    error_y=dict(type="data", array=binned["sem_imp"], visible=True, color=COLOR_NEUTRAL),
                    name=label,
                    legendgroup=label,
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
            x_grid = np.logspace(
                np.log10(float(binned["center_QV"].min())),
                np.log10(float(binned["center_QV"].max())),
                200,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=power_law(x_grid, float(fit_row["Y_hat"]), float(fit_row["gamma_hat"])),
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{label} fit",
                    legendgroup=label,
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )

    fig.update_xaxes(type="log", title_text="φ")
    fig.update_yaxes(type="log", title_text="I/σ")
    legend = plotly_legend_layout(PLOT_STYLE)
    legend["title"] = {"text": "Crowding quantile"}
    fig.update_layout(height=560, width=1050, legend=legend)
    _export_plotly_figure(
        fig,
        stem="main_crowding_impact_curves",
        dirs=dirs,
        write_html=write_html,
        write_png=write_png,
        strict_png=False,
    )


def _plot_predicted_impacts(
    predictions: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    dirs: PlotOutputDirs,
    *,
    write_html: bool,
    write_png: bool,
) -> None:
    if predictions.empty:
        return

    summary_lookup: Dict[Tuple[Any, ...], pd.Series] = {}
    if not prediction_summary.empty:
        summary_lookup = {
            (
                row[COL_GROUP],
                row.get(COL_ETA_BIN),
                row[COL_CROWDING_BIN],
                row["benchmark_phi"],
            ): row
            for _, row in prediction_summary.loc[prediction_summary["metric"] == "predicted_impact"].iterrows()
        }

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Client", "Proprietary"],
        shared_xaxes="all",
        shared_yaxes="all",
    )
    for col_idx, group in enumerate(GROUP_ORDER, start=1):
        group_pred = predictions[predictions[COL_GROUP] == group].copy()
        if group_pred.empty:
            continue
        for phi_idx, benchmark_phi in enumerate(sorted(group_pred["benchmark_phi"].dropna().unique())):
            sub = group_pred[group_pred["benchmark_phi"] == benchmark_phi].sort_values(COL_CROWDING_BIN)
            x = sub[COL_CROWDING_LABEL].tolist()
            y = sub["predicted_impact"].to_numpy(dtype=float)
            error_plus: List[float] = []
            error_minus: List[float] = []
            for _, row in sub.iterrows():
                key = (row[COL_GROUP], row.get(COL_ETA_BIN), row[COL_CROWDING_BIN], row["benchmark_phi"])
                summary_row = summary_lookup.get(key)
                if summary_row is None or not np.isfinite(summary_row.get("se", np.nan)):
                    error_plus.append(np.nan)
                    error_minus.append(np.nan)
                else:
                    error_plus.append(float(summary_row["se"]))
                    error_minus.append(float(summary_row["se"]))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    marker=dict(size=8, color=BENCHMARK_COLORS[phi_idx % len(BENCHMARK_COLORS)]),
                    line=dict(color=BENCHMARK_COLORS[phi_idx % len(BENCHMARK_COLORS)], width=2),
                    error_y=dict(type="data", array=error_plus, arrayminus=error_minus, visible=True),
                    name=f"φ={benchmark_phi:.0e}",
                    legendgroup=f"phi_{phi_idx}",
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
    fig.update_xaxes(title_text="Crowding quantile")
    fig.update_yaxes(title_text="Predicted I/σ")
    fig.update_layout(height=520, width=1050, legend=plotly_legend_layout(PLOT_STYLE))
    _export_plotly_figure(
        fig,
        stem="predicted_impact_by_crowding_quantile",
        dirs=dirs,
        write_html=write_html,
        write_png=write_png,
        strict_png=False,
    )


def _plot_difference_figure(
    contrasts_ci: pd.DataFrame,
    group_differences_ci: pd.DataFrame,
    dirs: PlotOutputDirs,
    *,
    write_html: bool,
    write_png: bool,
) -> None:
    if contrasts_ci.empty and group_differences_ci.empty:
        return
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["High minus low crowding gap", "Proprietary minus client gap"],
        shared_xaxes="all",
        shared_yaxes="all",
    )

    target = contrasts_ci[(contrasts_ci["metric"] == "point_estimate") & (contrasts_ci["contrast_name"] == "high_minus_low")]
    for group, color in [("client", COLOR_CLIENT), ("proprietary", COLOR_PROPRIETARY)]:
        sub = target[target[COL_GROUP] == group].sort_values("benchmark_phi")
        if sub.empty:
            continue
        y = sub["point_estimate"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=sub["benchmark_phi"],
                y=y,
                mode="lines+markers",
                marker=dict(size=9, color=color),
                line=dict(color=color, width=2),
                error_y=dict(
                    type="data",
                    array=(sub["ci_high"] - sub["point_estimate"]).to_numpy(dtype=float),
                    arrayminus=(sub["point_estimate"] - sub["ci_low"]).to_numpy(dtype=float),
                    visible=True,
                ),
                name=GROUP_DISPLAY[group],
                legendgroup=group,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    if not group_differences_ci.empty:
        sub = group_differences_ci[group_differences_ci["metric"] == "prop_minus_client"].sort_values("benchmark_phi")
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub["benchmark_phi"],
                    y=sub["point_estimate"],
                    mode="lines+markers",
                    marker=dict(size=9, color="#4B5563"),
                    line=dict(color="#4B5563", width=2),
                    error_y=dict(
                        type="data",
                        array=(sub["ci_high"] - sub["point_estimate"]).to_numpy(dtype=float),
                        arrayminus=(sub["point_estimate"] - sub["ci_low"]).to_numpy(dtype=float),
                        visible=True,
                    ),
                    name="Prop minus client",
                    legendgroup="prop_minus_client",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.update_xaxes(type="log", title_text="Benchmark φ", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Benchmark φ", row=1, col=2)
    fig.update_yaxes(title_text="Gap in predicted I/σ", row=1, col=1)
    fig.update_yaxes(title_text="Gap difference", row=1, col=2)
    fig.update_layout(height=520, width=1080, legend=plotly_legend_layout(PLOT_STYLE))
    _export_plotly_figure(
        fig,
        stem="crowding_gap_differences",
        dirs=dirs,
        write_html=write_html,
        write_png=write_png,
        strict_png=False,
    )


def _plot_eta_robustness(
    outputs: VariantOutputs,
    dirs: PlotOutputDirs,
    *,
    write_html: bool,
    write_png: bool,
) -> None:
    if outputs.fit_summary.empty or outputs.binned_curve_data.empty:
        return

    eta_bins = sorted(outputs.fit_summary[COL_ETA_BIN].dropna().unique())
    if not eta_bins:
        return
    fig = make_subplots(
        rows=len(eta_bins),
        cols=2,
        subplot_titles=[
            f"{GROUP_DISPLAY[group]} / {_eta_label_map(len(eta_bins)).get(int(eta_bin), f'η {eta_bin}')}"
            for eta_bin in eta_bins
            for group in GROUP_ORDER
        ],
        shared_xaxes="all",
        shared_yaxes="all",
    )

    for row_idx, eta_bin in enumerate(eta_bins, start=1):
        for col_idx, group in enumerate(GROUP_ORDER, start=1):
            group_fit = outputs.fit_summary[
                (outputs.fit_summary[COL_GROUP] == group)
                & (outputs.fit_summary[COL_ETA_BIN] == eta_bin)
                & (outputs.fit_summary["status"] == "ok")
            ]
            group_bins = outputs.binned_curve_data[
                (outputs.binned_curve_data[COL_GROUP] == group)
                & (outputs.binned_curve_data[COL_ETA_BIN] == eta_bin)
            ]
            for crowding_bin in sorted(group_fit[COL_CROWDING_BIN].dropna().unique()):
                fit_row = group_fit.loc[group_fit[COL_CROWDING_BIN] == crowding_bin].iloc[0]
                binned = group_bins.loc[group_bins[COL_CROWDING_BIN] == crowding_bin].copy()
                if binned.empty:
                    continue
                color = CROWDING_COLORS.get(int(crowding_bin), BENCHMARK_COLORS[int(crowding_bin) % len(BENCHMARK_COLORS)])
                label = str(fit_row[COL_CROWDING_LABEL])
                fig.add_trace(
                    go.Scatter(
                        x=binned["center_QV"],
                        y=binned["mean_imp"],
                        mode="markers",
                        marker=dict(size=6, color=color),
                        error_y=dict(type="data", array=binned["sem_imp"], visible=True, color=COLOR_NEUTRAL),
                        name=label,
                        legendgroup=label,
                        showlegend=(row_idx == 1 and col_idx == 1),
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                x_grid = np.logspace(
                    np.log10(float(binned["center_QV"].min())),
                    np.log10(float(binned["center_QV"].max())),
                    200,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=power_law(x_grid, float(fit_row["Y_hat"]), float(fit_row["gamma_hat"])),
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=f"{label} fit",
                        legendgroup=label,
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

    for row_idx in range(1, len(eta_bins) + 1):
        fig.update_xaxes(type="log", title_text="φ", row=row_idx, col=1)
        fig.update_xaxes(type="log", title_text="φ", row=row_idx, col=2)
        fig.update_yaxes(type="log", title_text="I/σ", row=row_idx, col=1)
        fig.update_yaxes(type="log", title_text="I/σ", row=row_idx, col=2)
    fig.update_layout(height=max(480, 360 * len(eta_bins)), width=1080, legend=plotly_legend_layout(PLOT_STYLE))
    _export_plotly_figure(
        fig,
        stem="eta_robustness_crowding_impact_curves",
        dirs=dirs,
        write_html=write_html,
        write_png=write_png,
        strict_png=False,
    )


def _plot_joint_regression_coefficients(
    coefficient_ci: pd.DataFrame,
    difference_ci: pd.DataFrame,
    dirs: PlotOutputDirs,
    *,
    write_html: bool,
    write_png: bool,
) -> None:
    """Plot joint-bin regression slope estimates and prop-minus-client differences using analytic SE-based intervals."""
    metrics = {
        "beta_log_eta_hat": "log(η)",
        "beta_log_vtv_hat": "log(V_t/V)",
        "beta_abs_imb_hat": "|imb|",
    }
    slope_ci = (
        coefficient_ci.loc[coefficient_ci["metric"].isin(metrics)].copy()
        if "metric" in coefficient_ci.columns
        else pd.DataFrame()
    )
    diff_metrics = {
        "prop_minus_client_beta_log_eta_hat": "log(η)",
        "prop_minus_client_beta_log_vtv_hat": "log(V_t/V)",
        "prop_minus_client_beta_abs_imb_hat": "|imb|",
    }
    diff_ci = (
        difference_ci.loc[difference_ci["metric"].isin(diff_metrics)].copy()
        if "metric" in difference_ci.columns
        else pd.DataFrame()
    )
    if slope_ci.empty and diff_ci.empty:
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Group-specific slope estimates", "Proprietary minus client"],
        shared_xaxes="all",
        shared_yaxes="all",
    )
    for group, color in [("client", COLOR_CLIENT), ("proprietary", COLOR_PROPRIETARY)]:
        sub = slope_ci.loc[slope_ci[COL_GROUP] == group].copy()
        if sub.empty:
            continue
        sub["metric_label"] = sub["metric"].map(metrics)
        sub["metric_order"] = sub["metric"].map({key: idx for idx, key in enumerate(metrics)})
        sub = sub.sort_values("metric_order")
        fig.add_trace(
            go.Scatter(
                x=sub["metric_label"],
                y=sub["point_estimate"],
                mode="lines+markers",
                marker=dict(size=9, color=color),
                line=dict(color=color, width=2),
                error_y=dict(
                    type="data",
                    array=(sub["ci_high"] - sub["point_estimate"]).to_numpy(dtype=float),
                    arrayminus=(sub["point_estimate"] - sub["ci_low"]).to_numpy(dtype=float),
                    visible=True,
                ),
                name=GROUP_DISPLAY[group],
                legendgroup=group,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    if not diff_ci.empty:
        diff_ci["metric_label"] = diff_ci["metric"].map(diff_metrics)
        diff_ci["metric_order"] = diff_ci["metric"].map({key: idx for idx, key in enumerate(diff_metrics)})
        diff_ci = diff_ci.sort_values("metric_order")
        fig.add_trace(
            go.Scatter(
                x=diff_ci["metric_label"],
                y=diff_ci["point_estimate"],
                mode="lines+markers",
                marker=dict(size=9, color="#4B5563"),
                line=dict(color="#4B5563", width=2),
                error_y=dict(
                    type="data",
                    array=(diff_ci["ci_high"] - diff_ci["point_estimate"]).to_numpy(dtype=float),
                    arrayminus=(diff_ci["point_estimate"] - diff_ci["ci_low"]).to_numpy(dtype=float),
                    visible=True,
                ),
                name="Prop minus client",
                legendgroup="prop_minus_client_joint_regression",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=0.0, line=dict(color="#9CA3AF", width=1, dash="dash"), row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color="#9CA3AF", width=1, dash="dash"), row=1, col=2)
    fig.update_yaxes(title_text="Coefficient estimate", row=1, col=1)
    fig.update_yaxes(title_text="Slope difference", row=1, col=2)
    fig.update_layout(height=520, width=1080, legend=plotly_legend_layout(PLOT_STYLE))
    _export_plotly_figure(
        fig,
        stem="joint_bin_regression_coefficients",
        dirs=dirs,
        write_html=write_html,
        write_png=write_png,
        strict_png=False,
    )


def _plot_pooled_log_regression_coefficients(
    coefficients: pd.DataFrame,
    coefficient_ci: pd.DataFrame,
    dirs: PlotOutputDirs,
    *,
    write_html: bool,
    write_png: bool,
    stem: str = "pooled_log_regression_coefficients",
) -> None:
    """Plot pooled log-regression coefficients using bootstrap intervals when available."""
    if coefficients.empty:
        return

    required = ["term", "term_label", "term_order", "estimate"]
    _validate_required_columns(coefficients, required, label="pooled_log_regression_plot")
    target = coefficients.copy()
    target = target.loc[target["term"] != "const"].copy()
    if target.empty:
        return

    ci_target = pd.DataFrame()
    if not coefficient_ci.empty:
        _validate_required_columns(
            coefficient_ci,
            ["term", "term_label", "term_order", "metric", "ci_low", "ci_high"],
            label="pooled_log_regression_plot_ci",
        )
        ci_target = coefficient_ci.loc[coefficient_ci["metric"] == "estimate"].copy()
        ci_target = ci_target.loc[ci_target["term"] != "const"].copy()
        ci_target = ci_target[["term", "term_label", "term_order", "ci_low", "ci_high"]]
        target = target.merge(
            ci_target,
            on=["term", "term_label", "term_order"],
            how="left",
            validate="one_to_one",
        )

    # Fall back to analytic intervals only when bootstrap intervals are unavailable.
    if "ci_low" not in target.columns:
        target["ci_low"] = np.nan
    if "ci_high" not in target.columns:
        target["ci_high"] = np.nan
    if {"analytic_ci_low", "analytic_ci_high"}.issubset(target.columns):
        missing_bootstrap = target["ci_low"].isna() | target["ci_high"].isna()
        target.loc[missing_bootstrap, "ci_low"] = target.loc[missing_bootstrap, "analytic_ci_low"]
        target.loc[missing_bootstrap, "ci_high"] = target.loc[missing_bootstrap, "analytic_ci_high"]

    target["term_order"] = pd.to_numeric(target["term_order"], errors="coerce")
    target = target.sort_values("term_order", ascending=False)
    colors = [
        COLOR_PROPRIETARY if term in {COL_PROP_DUMMY, COL_CLIENT_DUMMY} else "#4B5563"
        for term in target["term"].tolist()
    ]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=target["estimate"],
                y=target["term_label"],
                mode="markers",
                marker=dict(size=10, color=colors),
                error_x=dict(
                    type="data",
                    array=(target["ci_high"] - target["estimate"]).to_numpy(dtype=float),
                    arrayminus=(target["estimate"] - target["ci_low"]).to_numpy(dtype=float),
                    visible=True,
                ),
                showlegend=False,
            )
        ]
    )
    fig.add_vline(x=0.0, line=dict(color="#9CA3AF", width=1, dash="dash"))
    fig.update_xaxes(title_text="Coefficient estimate")
    fig.update_yaxes(title_text="")
    fig.update_layout(height=420, width=900)
    _export_plotly_figure(
        fig,
        stem=stem,
        dirs=dirs,
        write_html=write_html,
        write_png=write_png,
        strict_png=False,
    )


def _status_positive(point: float, lo: float, hi: float) -> str:
    if not np.isfinite(point):
        return "insufficient"
    if np.isfinite(lo) and lo > 0:
        return "yes"
    if np.isfinite(hi) and hi < 0:
        return "no"
    return "mixed" if point > 0 else "no"


def _status_monotonic(low_mid: Mapping[str, Any], high_mid: Mapping[str, Any]) -> str:
    if any(not np.isfinite(float(row.get("point_estimate", np.nan))) for row in [low_mid, high_mid]):
        return "insufficient"
    if (
        np.isfinite(float(low_mid.get("ci_low", np.nan)))
        and float(low_mid["ci_low"]) > 0
        and np.isfinite(float(high_mid.get("ci_low", np.nan)))
        and float(high_mid["ci_low"]) > 0
    ):
        return "yes"
    if float(low_mid["point_estimate"]) > 0 and float(high_mid["point_estimate"]) > 0:
        return "mixed"
    return "no"


def _build_acceptance_summary(
    contrasts_ci_main: pd.DataFrame,
    group_diff_ci_main: pd.DataFrame,
    contrasts_ci_eta: pd.DataFrame,
    joint_regression_ci: Optional[pd.DataFrame] = None,
    joint_regression_diff_ci: Optional[pd.DataFrame] = None,
    pooled_log_regression_ci: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    target_main = contrasts_ci_main[
        (contrasts_ci_main["metric"] == "point_estimate") & (contrasts_ci_main["contrast_name"] == "high_minus_low")
    ]
    for group in GROUP_ORDER:
        for _, row in target_main.loc[target_main[COL_GROUP] == group].iterrows():
            rows.append(
                {
                    "question": (
                        "Does impact increase with aligned crowding within client flow?"
                        if group == "client"
                        else "Does impact increase with aligned crowding within proprietary flow?"
                    ),
                    COL_GROUP: group,
                    "benchmark_phi": float(row["benchmark_phi"]),
                    "status": _status_positive(float(row["point_estimate"]), float(row["ci_low"]), float(row["ci_high"])),
                    "point_estimate": float(row["point_estimate"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )

    monotonic_main = contrasts_ci_main[
        (contrasts_ci_main["metric"] == "point_estimate")
        & (contrasts_ci_main["contrast_name"].isin(["mid_minus_low", "high_minus_mid"]))
    ]
    for group in GROUP_ORDER:
        for benchmark_phi, sub in monotonic_main.loc[monotonic_main[COL_GROUP] == group].groupby("benchmark_phi", sort=True):
            mid_low = sub.loc[sub["contrast_name"] == "mid_minus_low"]
            high_mid = sub.loc[sub["contrast_name"] == "high_minus_mid"]
            if mid_low.empty or high_mid.empty:
                continue
            rows.append(
                {
                    "question": "Is the increase monotonic across low / mid / high crowding?",
                    COL_GROUP: group,
                    "benchmark_phi": float(benchmark_phi),
                    "status": _status_monotonic(mid_low.iloc[0].to_dict(), high_mid.iloc[0].to_dict()),
                    "point_estimate": float(mid_low["point_estimate"].iat[0] + high_mid["point_estimate"].iat[0]),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )

    prop_client = group_diff_ci_main[group_diff_ci_main["metric"] == "prop_minus_client"]
    for _, row in prop_client.iterrows():
        rows.append(
            {
                "question": "Is the increase larger for proprietary than for client?",
                COL_GROUP: "prop_minus_client",
                "benchmark_phi": float(row["benchmark_phi"]),
                "status": _status_positive(float(row["point_estimate"]), float(row["ci_low"]), float(row["ci_high"])),
                "point_estimate": float(row["point_estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
        )

    eta_target = contrasts_ci_eta[
        (contrasts_ci_eta["metric"] == "point_estimate") & (contrasts_ci_eta["contrast_name"] == "high_minus_low")
    ]
    for _, row in eta_target.iterrows():
        rows.append(
            {
                "question": "Does the conclusion survive conditioning on participation?",
                COL_GROUP: row[COL_GROUP],
                "benchmark_phi": float(row["benchmark_phi"]),
                COL_ETA_LABEL: row.get(COL_ETA_LABEL),
                "status": _status_positive(float(row["point_estimate"]), float(row["ci_low"]), float(row["ci_high"])),
                "point_estimate": float(row["point_estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
        )

    if joint_regression_ci is not None and not joint_regression_ci.empty:
        target = joint_regression_ci.loc[joint_regression_ci["metric"] == "beta_abs_imb_hat"].copy()
        for _, row in target.iterrows():
            rows.append(
                {
                    "question": "After controlling for η and V_t/V, is the |imb| slope positive?",
                    COL_GROUP: row[COL_GROUP],
                    "benchmark_phi": np.nan,
                    "status": _status_positive(float(row["point_estimate"]), float(row["ci_low"]), float(row["ci_high"])),
                    "point_estimate": float(row["point_estimate"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )

    if joint_regression_diff_ci is not None and not joint_regression_diff_ci.empty:
        target = joint_regression_diff_ci.loc[
            joint_regression_diff_ci["metric"] == "prop_minus_client_beta_abs_imb_hat"
        ].copy()
        for _, row in target.iterrows():
            rows.append(
                {
                    "question": "After controlling for η and V_t/V, is the |imb| slope larger for proprietary?",
                    COL_GROUP: "prop_minus_client",
                    "benchmark_phi": np.nan,
                    "status": _status_positive(float(row["point_estimate"]), float(row["ci_low"]), float(row["ci_high"])),
                    "point_estimate": float(row["point_estimate"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )

    if pooled_log_regression_ci is not None and not pooled_log_regression_ci.empty:
        target = pooled_log_regression_ci.loc[
            (pooled_log_regression_ci["metric"] == "estimate")
            & (pooled_log_regression_ci["term"] == COL_ABS_IMBALANCE)
        ].copy()
        for _, row in target.iterrows():
            rows.append(
                {
                    "question": "In the pooled log-mean-impact regression, is the |imb| slope positive?",
                    COL_GROUP: "pooled",
                    "benchmark_phi": np.nan,
                    "status": _status_positive(float(row["point_estimate"]), float(row["ci_low"]), float(row["ci_high"])),
                    "point_estimate": float(row["point_estimate"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )

        target = pooled_log_regression_ci.loc[
            (pooled_log_regression_ci["metric"] == "estimate")
            & (pooled_log_regression_ci["term"] == COL_PROP_DUMMY)
        ].copy()
        for _, row in target.iterrows():
            rows.append(
                {
                    "question": "In the pooled log-mean-impact regression, is the proprietary dummy positive?",
                    COL_GROUP: "pooled",
                    "benchmark_phi": np.nan,
                    "status": _status_positive(float(row["point_estimate"]), float(row["ci_low"]), float(row["ci_high"])),
                    "point_estimate": float(row["point_estimate"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )

    return pd.DataFrame(rows)


def _print_acceptance_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("[Summary] Acceptance summary unavailable.")
        return
    print("[Summary] Acceptance criteria:")
    for question, sub in summary.groupby("question", sort=False):
        pieces = []
        for _, row in sub.iterrows():
            benchmark_phi = row.get("benchmark_phi")
            label = f"φ={benchmark_phi:.0e}" if pd.notna(benchmark_phi) else ""
            if COL_ETA_LABEL in row and pd.notna(row.get(COL_ETA_LABEL)):
                label = f"{row[COL_ETA_LABEL]} / {label}" if label else str(row[COL_ETA_LABEL])
            if pd.notna(row.get(COL_GROUP)) and row.get(COL_GROUP) not in {"prop_minus_client", ""}:
                label = f"{row[COL_GROUP]} / {label}" if label else str(row[COL_GROUP])
            if not label:
                label = str(row.get(COL_GROUP, ""))
            pieces.append(f"{label}: {row['status']}")
        print(f"  - {question} {'; '.join(pieces)}")


def _write_tables(base_dir: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    for stem, table in tables.items():
        if table is None or table.empty:
            continue
        table.to_csv(base_dir / f"{stem}.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Summary
    -------
    Build the CLI parser for the crowding-conditioned impact workflow.

    Parameters
    ----------
    None
        Parser construction does not consume runtime arguments.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with YAML-aware defaults and output toggles.

    Notes
    -----
    The script defaults to repository-native paths, but every input/output path
    and key analysis knob can be overridden from the command line.

    Examples
    --------
    >>> parser = build_arg_parser()
    >>> isinstance(parser, argparse.ArgumentParser)
    True
    """
    parser = argparse.ArgumentParser(description="Crowding-conditioned impact analysis for metaorders.")
    config_default = os.environ.get(_CONFIG_ENV_VAR, str(_DEFAULT_CONFIG_PATH))
    parser.add_argument("--config-path", type=str, default=config_default, help="YAML config path.")
    parser.add_argument("--impact-config-path", type=str, default=None, help="Optional impact YAML used for fit defaults.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name used inside output path templates.")
    parser.add_argument("--prop-path", type=str, default=None, help="Path to the proprietary metaorder parquet.")
    parser.add_argument("--client-path", type=str, default=None, help="Path to the client metaorder parquet.")
    parser.add_argument("--output-file-path", type=str, default=None, help="Base output folder.")
    parser.add_argument("--img-output-path", type=str, default=None, help="Base image folder.")
    parser.add_argument("--analysis-tag", type=str, default=None, help="Output subfolder tag.")
    parser.add_argument(
        "--crowding-scope",
        type=str,
        default=None,
        help="Crowding environment: `within_group` or `all` (all-metaorder leave-one-out).",
    )
    parser.add_argument("--min-qv", type=float, default=None, help="Impact-fit lower bound for Q/V.")
    parser.add_argument("--eta-max", type=float, default=None, help="Strict upper bound for participation rate.")
    parser.add_argument("--n-logbins", type=int, default=None, help="Number of logarithmic Q/V bins.")
    parser.add_argument("--min-count", type=int, default=None, help="Minimum retained count per log bin.")
    parser.add_argument(
        "--n-crowding-quantiles",
        type=int,
        default=None,
        help="Number of aligned-crowding quantiles in the main specification.",
    )
    parser.add_argument(
        "--benchmark-phis",
        type=str,
        default=None,
        help="Comma-separated benchmark Q/V values, e.g. '1e-3,1e-2'.",
    )
    parser.add_argument("--bootstrap-runs", type=int, default=None, help="Number of Date-cluster bootstrap replicates.")
    parser.add_argument("--alpha", type=float, default=None, help="Significance level for percentile intervals.")
    parser.add_argument("--seed", type=int, default=None, help="Bootstrap RNG seed.")
    parser.add_argument("--eta-robustness-bins", type=int, default=None, help="Number of broad eta bins in the robustness analysis.")
    parser.add_argument(
        "--joint-regression-eta-bins",
        type=int,
        default=None,
        help="Number of log bins for η in the joint-bin regression.",
    )
    parser.add_argument(
        "--joint-regression-vtv-bins",
        type=int,
        default=None,
        help="Number of log bins for V_t/V in the joint-bin regression.",
    )
    parser.add_argument(
        "--joint-regression-abs-imb-bins",
        type=int,
        default=None,
        help="Number of linear bins for |imbalance| in the joint-bin regression.",
    )
    parser.add_argument(
        "--joint-regression-min-count",
        type=int,
        default=None,
        help="Minimum cell count retained in the joint-bin regression.",
    )
    parser.add_argument(
        "--no-joint-regression",
        dest="run_joint_regression",
        action="store_false",
        help="Skip the joint (η, V_t/V, |imb|) bin-level regression.",
    )
    parser.add_argument(
        "--run-joint-regression",
        dest="run_joint_regression",
        action="store_true",
        help="Force the joint (η, V_t/V, |imb|) bin-level regression.",
    )
    parser.set_defaults(run_joint_regression=None)
    parser.add_argument(
        "--pooled-regression-impact-mode",
        type=str,
        default=None,
        help="Pooled regression convention. Use `signed_mean` to match the standard impact-fit logic.",
    )
    parser.add_argument(
        "--no-pooled-log-regression",
        dest="run_pooled_log_regression",
        action="store_false",
        help="Skip the pooled cell-level log-mean-impact regression.",
    )
    parser.add_argument(
        "--run-pooled-log-regression",
        dest="run_pooled_log_regression",
        action="store_true",
        help="Force the pooled cell-level log-mean-impact regression.",
    )
    parser.set_defaults(run_pooled_log_regression=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and manifest, then exit.")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false", help="Disable bootstrap progress prints.")
    parser.add_argument("--show-progress", dest="show_progress", action="store_true", help="Enable bootstrap progress prints.")
    parser.set_defaults(show_progress=None)
    parser.add_argument("--no-write-html", dest="write_html", action="store_false", help="Skip HTML figure export.")
    parser.add_argument("--write-html", dest="write_html", action="store_true", help="Force HTML figure export.")
    parser.set_defaults(write_html=None)
    parser.add_argument("--no-write-png", dest="write_png", action="store_false", help="Skip PNG figure export.")
    parser.add_argument("--write-png", dest="write_png", action="store_true", help="Force PNG figure export.")
    parser.set_defaults(write_png=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Summary
    -------
    Run the crowding-conditioned impact analysis end to end.

    Parameters
    ----------
    argv : Optional[Sequence[str]], default=None
        Optional CLI argument list. When omitted, arguments are read from
        `sys.argv`.

    Returns
    -------
    int
        Process-style exit status (`0` on success).

    Notes
    -----
    The workflow writes CSV tables, Plotly figures, a run manifest, and a log
    file under the configured analysis output folder. It uses a Date-cluster
    bootstrap, so at least two trading dates are required.

    Examples
    --------
    >>> isinstance(main(["--dry-run"]), int)
    True
    """
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = _resolve_repo_path(args.config_path)
    cfg = load_yaml_mapping(config_path)
    paths = _resolve_paths(cfg, args)

    impact_cfg = load_yaml_mapping(paths.impact_config_path)
    fit_defaults = _load_fit_defaults(cfg, args, impact_cfg)

    n_crowding_quantiles = int(args.n_crowding_quantiles or cfg.get("N_CROWDING_QUANTILES", 3))
    crowding_scope = str(args.crowding_scope or cfg.get("CROWDING_SCOPE") or "within_group").strip().lower()
    benchmark_phis = (
        _parse_float_list(args.benchmark_phis)
        if args.benchmark_phis is not None
        else tuple(float(x) for x in cfg.get("BENCHMARK_PHIS", [1.0e-3, 1.0e-2]))
    )
    bootstrap_runs = int(args.bootstrap_runs if args.bootstrap_runs is not None else cfg.get("BOOTSTRAP_RUNS", 200))
    alpha = float(args.alpha if args.alpha is not None else cfg.get("ALPHA", 0.05))
    seed = int(args.seed if args.seed is not None else cfg.get("RANDOM_STATE", 0))
    n_eta_bins = int(args.eta_robustness_bins if args.eta_robustness_bins is not None else cfg.get("ETA_ROBUSTNESS_BINS", 2))
    joint_reg_eta_bins = int(
        args.joint_regression_eta_bins
        if args.joint_regression_eta_bins is not None
        else cfg.get("JOINT_REG_ETA_BINS", 6)
    )
    joint_reg_vtv_bins = int(
        args.joint_regression_vtv_bins
        if args.joint_regression_vtv_bins is not None
        else cfg.get("JOINT_REG_VTV_BINS", 6)
    )
    joint_reg_abs_imb_bins = int(
        args.joint_regression_abs_imb_bins
        if args.joint_regression_abs_imb_bins is not None
        else cfg.get("JOINT_REG_ABS_IMB_BINS", 4)
    )
    joint_reg_min_count = int(
        args.joint_regression_min_count
        if args.joint_regression_min_count is not None
        else cfg.get("JOINT_REG_MIN_CELL_COUNT", fit_defaults.min_count)
    )
    run_joint_regression = _resolve_bool(args.run_joint_regression, cfg, "RUN_JOINT_BIN_REGRESSION", True)
    pooled_regression_impact_mode = str(
        args.pooled_regression_impact_mode or cfg.get("POOLED_REGRESSION_IMPACT_MODE") or "signed_mean"
    ).strip()
    run_pooled_log_regression = _resolve_bool(
        args.run_pooled_log_regression,
        cfg,
        "RUN_POOLED_LOG_REGRESSION",
        True,
    )
    write_html = _resolve_bool(args.write_html, cfg, "WRITE_HTML", True)
    write_png = _resolve_bool(args.write_png, cfg, "WRITE_PNG", True)
    show_progress = _resolve_bool(args.show_progress, cfg, "SHOW_PROGRESS", True)

    manifest = {
        "run_timestamp": dt.datetime.now().isoformat(),
        "git_hash": _try_git_hash(),
        "dataset_name": paths.dataset_name,
        "config_path": str(paths.config_path),
        "impact_config_path": str(paths.impact_config_path),
        "prop_path": str(paths.prop_path),
        "client_path": str(paths.client_path),
        "out_dir": str(paths.out_dir),
        "img_dir": str(paths.img_dir),
        "benchmark_phis": [float(phi) for phi in benchmark_phis],
        "bootstrap_runs": bootstrap_runs,
        "alpha": alpha,
        "seed": seed,
        "crowding_scope": crowding_scope,
        "n_crowding_quantiles": n_crowding_quantiles,
        "eta_robustness_bins": n_eta_bins,
        "run_joint_bin_regression": run_joint_regression,
        "joint_regression": {
            "eta_bins": joint_reg_eta_bins,
            "vtv_bins": joint_reg_vtv_bins,
            "abs_imb_bins": joint_reg_abs_imb_bins,
            "min_cell_count": joint_reg_min_count,
        },
        "run_pooled_log_regression": run_pooled_log_regression,
        "pooled_log_regression": {
            "impact_mode": pooled_regression_impact_mode,
            "dummy_groups": list(POOLED_DUMMY_GROUPS),
            "eta_bins": joint_reg_eta_bins,
            "vtv_bins": joint_reg_vtv_bins,
            "abs_imb_bins": joint_reg_abs_imb_bins,
            "min_cell_count": joint_reg_min_count,
            "covariance": "wls(sem_imp)",
        },
        "fit_defaults": {
            "min_qv": fit_defaults.min_qv,
            "n_logbins": fit_defaults.n_logbins,
            "min_count": fit_defaults.min_count,
            "eta_max": fit_defaults.eta_max,
        },
        "write_html": write_html,
        "write_png": write_png,
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    img_dirs = make_plot_output_dirs(paths.img_dir, use_subdirs=True)
    ensure_plot_dirs(img_dirs)
    logger = setup_file_logger(Path(__file__).stem, paths.log_path, mode="w", reset_handlers=True)

    with PrintTee(logger):
        print("[Crowding impact] Run started.")
        _write_json(paths.out_dir / "run_manifest.json", manifest)

        prop_result = _load_group_frame(
            paths.prop_path,
            label="proprietary",
            group_name="proprietary",
            fit_defaults=fit_defaults,
        )
        client_result = _load_group_frame(
            paths.client_path,
            label="client",
            group_name="client",
            fit_defaults=fit_defaults,
        )
        working, filter_counts = _prepare_working_sample(
            client_result,
            prop_result,
            crowding_scope=crowding_scope,
        )
        print(
            f"[Crowding impact] Working sample size: {len(working):,} metaorders "
            f"across {working[COL_DATE].nunique():,} trading dates "
            f"using `{crowding_scope}` crowding."
        )

        main_outputs = _analyse_main_variant(
            working,
            n_crowding_quantiles=n_crowding_quantiles,
            benchmark_phis=benchmark_phis,
            n_logbins=fit_defaults.n_logbins,
            min_count=fit_defaults.min_count,
            store_binned=True,
        )
        eta_outputs = _analyse_eta_variant(
            working,
            n_eta_bins=n_eta_bins,
            n_crowding_quantiles=n_crowding_quantiles,
            benchmark_phis=benchmark_phis,
            n_logbins=fit_defaults.n_logbins,
            min_count=fit_defaults.min_count,
            store_binned=True,
        )
        joint_reg_outputs = (
            _analyse_joint_bin_regression(
                working,
                n_eta_bins=joint_reg_eta_bins,
                n_vtv_bins=joint_reg_vtv_bins,
                n_abs_imb_bins=joint_reg_abs_imb_bins,
                min_cell_count=joint_reg_min_count,
            )
            if run_joint_regression
            else JointRegressionOutputs(
                cutpoints=pd.DataFrame(),
                cell_data=pd.DataFrame(),
                fit_summary=pd.DataFrame(),
                group_differences=pd.DataFrame(),
            )
        )
        pooled_reg_outputs = (
            _analyse_pooled_log_regression(
                working,
                alpha=alpha,
                impact_mode=pooled_regression_impact_mode,
                n_eta_bins=joint_reg_eta_bins,
                n_vtv_bins=joint_reg_vtv_bins,
                n_abs_imb_bins=joint_reg_abs_imb_bins,
                min_cell_count=joint_reg_min_count,
                dummy_group="proprietary",
            )
            if run_pooled_log_regression
            else PooledLogRegressionOutputs(
                sample_counts=pd.DataFrame(),
                cutpoints=pd.DataFrame(),
                cell_data=pd.DataFrame(),
                fit_summary=pd.DataFrame(),
                coefficients=pd.DataFrame(),
            )
        )
        pooled_reg_client_outputs = (
            _analyse_pooled_log_regression(
                working,
                alpha=alpha,
                impact_mode=pooled_regression_impact_mode,
                n_eta_bins=joint_reg_eta_bins,
                n_vtv_bins=joint_reg_vtv_bins,
                n_abs_imb_bins=joint_reg_abs_imb_bins,
                min_cell_count=joint_reg_min_count,
                dummy_group="client",
            )
            if run_pooled_log_regression
            else PooledLogRegressionOutputs(
                sample_counts=pd.DataFrame(),
                cutpoints=pd.DataFrame(),
                cell_data=pd.DataFrame(),
                fit_summary=pd.DataFrame(),
                coefficients=pd.DataFrame(),
            )
        )

        sampler = DateBootstrapSampler.from_frame(working)
        main_boot = _bootstrap_variant(
            sampler,
            analyse_fn=lambda df, store_binned=False: _analyse_main_variant(
                df,
                n_crowding_quantiles=n_crowding_quantiles,
                benchmark_phis=benchmark_phis,
                n_logbins=fit_defaults.n_logbins,
                min_count=fit_defaults.min_count,
                store_binned=store_binned,
            ),
            bootstrap_runs=bootstrap_runs,
            seed=seed,
            show_progress=show_progress,
        )
        eta_boot = _bootstrap_variant(
            sampler,
            analyse_fn=lambda df, store_binned=False: _analyse_eta_variant(
                df,
                n_eta_bins=n_eta_bins,
                n_crowding_quantiles=n_crowding_quantiles,
                benchmark_phis=benchmark_phis,
                n_logbins=fit_defaults.n_logbins,
                min_count=fit_defaults.min_count,
                store_binned=store_binned,
            ),
            bootstrap_runs=bootstrap_runs,
            seed=seed + 1,
            show_progress=show_progress,
        )
        joint_reg_boot = (
            _bootstrap_joint_regression(
                sampler,
                analyse_fn=lambda df: _analyse_joint_bin_regression(
                    df,
                    n_eta_bins=joint_reg_eta_bins,
                    n_vtv_bins=joint_reg_vtv_bins,
                    n_abs_imb_bins=joint_reg_abs_imb_bins,
                    min_cell_count=joint_reg_min_count,
                ),
                bootstrap_runs=bootstrap_runs,
                seed=seed + 2,
                show_progress=show_progress,
            )
            if run_joint_regression
            else {"fit_summary": pd.DataFrame(), "group_differences": pd.DataFrame()}
        )
        pooled_reg_boot = (
            _bootstrap_pooled_log_regression(
                sampler,
                analyse_fn=lambda df: _analyse_pooled_log_regression(
                    df,
                    alpha=alpha,
                    impact_mode=pooled_regression_impact_mode,
                    n_eta_bins=joint_reg_eta_bins,
                    n_vtv_bins=joint_reg_vtv_bins,
                    n_abs_imb_bins=joint_reg_abs_imb_bins,
                    min_cell_count=joint_reg_min_count,
                    dummy_group="proprietary",
                ),
                bootstrap_runs=bootstrap_runs,
                seed=seed + 3,
                show_progress=show_progress,
            )
            if run_pooled_log_regression and not pooled_reg_outputs.coefficients.empty
            else {"coefficients": pd.DataFrame()}
        )
        pooled_reg_client_boot = (
            _bootstrap_pooled_log_regression(
                sampler,
                analyse_fn=lambda df: _analyse_pooled_log_regression(
                    df,
                    alpha=alpha,
                    impact_mode=pooled_regression_impact_mode,
                    n_eta_bins=joint_reg_eta_bins,
                    n_vtv_bins=joint_reg_vtv_bins,
                    n_abs_imb_bins=joint_reg_abs_imb_bins,
                    min_cell_count=joint_reg_min_count,
                    dummy_group="client",
                ),
                bootstrap_runs=bootstrap_runs,
                seed=seed + 4,
                show_progress=show_progress,
            )
            if run_pooled_log_regression and not pooled_reg_client_outputs.coefficients.empty
            else {"coefficients": pd.DataFrame()}
        )

        curve_ci_main = _bootstrap_ci_from_table(
            main_outputs.fit_summary,
            main_boot["fit_summary"],
            id_cols=[COL_GROUP, COL_GROUP_LABEL, COL_CROWDING_BIN, COL_CROWDING_LABEL],
            value_cols=["Y_hat", "gamma_hat"],
            alpha=alpha,
        )
        prediction_ci_main = _bootstrap_ci_from_table(
            main_outputs.predictions,
            main_boot["predictions"],
            id_cols=[COL_GROUP, COL_GROUP_LABEL, COL_CROWDING_BIN, COL_CROWDING_LABEL, "benchmark_phi"],
            value_cols=["predicted_impact"],
            alpha=alpha,
        )
        contrasts_ci_main = _bootstrap_ci_from_table(
            main_outputs.contrasts,
            main_boot["contrasts"],
            id_cols=[COL_GROUP, COL_GROUP_LABEL, "benchmark_phi", "contrast_name"],
            value_cols=["point_estimate"],
            alpha=alpha,
        )
        group_diff_ci_main = _bootstrap_ci_from_table(
            main_outputs.group_differences,
            main_boot["group_differences"],
            id_cols=["benchmark_phi", "contrast_name"],
            value_cols=["prop_high_minus_low", "client_high_minus_low", "prop_minus_client"],
            alpha=alpha,
        )

        curve_ci_eta = _bootstrap_ci_from_table(
            eta_outputs.fit_summary,
            eta_boot["fit_summary"],
            id_cols=[COL_GROUP, COL_GROUP_LABEL, COL_ETA_BIN, COL_ETA_LABEL, COL_CROWDING_BIN, COL_CROWDING_LABEL],
            value_cols=["Y_hat", "gamma_hat"],
            alpha=alpha,
        )
        prediction_ci_eta = _bootstrap_ci_from_table(
            eta_outputs.predictions,
            eta_boot["predictions"],
            id_cols=[COL_GROUP, COL_GROUP_LABEL, COL_ETA_BIN, COL_ETA_LABEL, COL_CROWDING_BIN, COL_CROWDING_LABEL, "benchmark_phi"],
            value_cols=["predicted_impact"],
            alpha=alpha,
        )
        contrasts_ci_eta = _bootstrap_ci_from_table(
            eta_outputs.contrasts,
            eta_boot["contrasts"],
            id_cols=[COL_GROUP, COL_GROUP_LABEL, COL_ETA_BIN, COL_ETA_LABEL, "benchmark_phi", "contrast_name"],
            value_cols=["point_estimate"],
            alpha=alpha,
        )
        group_diff_ci_eta = _bootstrap_ci_from_table(
            eta_outputs.group_differences,
            eta_boot["group_differences"],
            id_cols=["benchmark_phi", COL_ETA_BIN, COL_ETA_LABEL, "contrast_name"],
            value_cols=["prop_high_minus_low", "client_high_minus_low", "prop_minus_client"],
            alpha=alpha,
        )
        joint_reg_ci = _bootstrap_ci_from_table(
            joint_reg_outputs.fit_summary,
            joint_reg_boot["fit_summary"],
            id_cols=[COL_GROUP, COL_GROUP_LABEL],
            value_cols=[
                "Y_hat",
                "beta_log_eta_hat",
                "beta_log_vtv_hat",
                "beta_abs_imb_hat",
            ],
            alpha=alpha,
        )
        joint_reg_diff_ci = _bootstrap_ci_from_table(
            joint_reg_outputs.group_differences,
            joint_reg_boot["group_differences"],
            id_cols=["comparison"],
            value_cols=[
                "prop_minus_client_beta_log_eta_hat",
                "prop_minus_client_beta_log_vtv_hat",
                "prop_minus_client_beta_abs_imb_hat",
            ],
            alpha=alpha,
        )
        joint_reg_plot_ci, joint_reg_plot_diff_ci = _build_joint_regression_analytic_plot_tables(
            joint_reg_outputs.fit_summary,
            alpha=alpha,
        )
        pooled_reg_ci = _bootstrap_ci_from_table(
            pooled_reg_outputs.coefficients,
            pooled_reg_boot["coefficients"],
            id_cols=["term", "term_label", "term_order"],
            value_cols=["estimate"],
            alpha=alpha,
        )
        if pooled_reg_ci.empty and not pooled_reg_outputs.coefficients.empty:
            pooled_reg_ci = _cluster_ci_from_coefficient_table(pooled_reg_outputs.coefficients)
        pooled_reg_client_ci = _bootstrap_ci_from_table(
            pooled_reg_client_outputs.coefficients,
            pooled_reg_client_boot["coefficients"],
            id_cols=["term", "term_label", "term_order"],
            value_cols=["estimate"],
            alpha=alpha,
        )
        if pooled_reg_client_ci.empty and not pooled_reg_client_outputs.coefficients.empty:
            pooled_reg_client_ci = _cluster_ci_from_coefficient_table(pooled_reg_client_outputs.coefficients)

        acceptance_summary = _build_acceptance_summary(
            contrasts_ci_main=contrasts_ci_main,
            group_diff_ci_main=group_diff_ci_main,
            contrasts_ci_eta=contrasts_ci_eta,
            joint_regression_ci=joint_reg_ci,
            joint_regression_diff_ci=joint_reg_diff_ci,
            pooled_log_regression_ci=pooled_reg_ci,
        )
        _print_acceptance_summary(acceptance_summary)

        _write_tables(
            paths.out_dir,
            {
                "filter_counts": filter_counts,
                "sample_sizes_main": main_outputs.sample_sizes,
                "crowding_cutpoints_main": main_outputs.cutpoints,
                "fit_summary_main": main_outputs.fit_summary,
                "predicted_impacts_main": main_outputs.predictions,
                "binned_curve_data_main": main_outputs.binned_curve_data,
                "bootstrap_curve_parameters_main": curve_ci_main,
                "bootstrap_predicted_impacts_main": prediction_ci_main,
                "monotonic_contrasts_main": contrasts_ci_main,
                "group_difference_main": group_diff_ci_main,
                "sample_sizes_eta_robustness": eta_outputs.sample_sizes,
                "cutpoints_eta_robustness": eta_outputs.cutpoints,
                "fit_summary_eta_robustness": eta_outputs.fit_summary,
                "predicted_impacts_eta_robustness": eta_outputs.predictions,
                "binned_curve_data_eta_robustness": eta_outputs.binned_curve_data,
                "bootstrap_curve_parameters_eta_robustness": curve_ci_eta,
                "bootstrap_predicted_impacts_eta_robustness": prediction_ci_eta,
                "monotonic_contrasts_eta_robustness": contrasts_ci_eta,
                "group_difference_eta_robustness": group_diff_ci_eta,
                "joint_regression_cutpoints": joint_reg_outputs.cutpoints,
                "joint_regression_cell_data": joint_reg_outputs.cell_data,
                "joint_regression_fit_summary": joint_reg_outputs.fit_summary,
                "joint_regression_bootstrap_coefficients": joint_reg_ci,
                "joint_regression_group_difference": joint_reg_diff_ci,
                "pooled_log_regression_sample_counts": pooled_reg_outputs.sample_counts,
                "pooled_log_regression_cutpoints": pooled_reg_outputs.cutpoints,
                "pooled_log_regression_cell_data": pooled_reg_outputs.cell_data,
                "pooled_log_regression_fit_summary": pooled_reg_outputs.fit_summary,
                "pooled_log_regression_coefficients": pooled_reg_outputs.coefficients,
                "pooled_log_regression_bootstrap_coefficients": pooled_reg_ci,
                "pooled_log_regression_client_dummy_sample_counts": pooled_reg_client_outputs.sample_counts,
                "pooled_log_regression_client_dummy_cutpoints": pooled_reg_client_outputs.cutpoints,
                "pooled_log_regression_client_dummy_cell_data": pooled_reg_client_outputs.cell_data,
                "pooled_log_regression_client_dummy_fit_summary": pooled_reg_client_outputs.fit_summary,
                "pooled_log_regression_client_dummy_coefficients": pooled_reg_client_outputs.coefficients,
                "pooled_log_regression_client_dummy_bootstrap_coefficients": pooled_reg_client_ci,
                "acceptance_summary": acceptance_summary,
            },
        )

        _plot_main_curves(
            main_outputs,
            img_dirs,
            benchmark_phis=benchmark_phis,
            write_html=write_html,
            write_png=write_png,
        )
        _plot_predicted_impacts(
            main_outputs.predictions,
            prediction_ci_main,
            img_dirs,
            write_html=write_html,
            write_png=write_png,
        )
        _plot_difference_figure(
            contrasts_ci_main,
            group_diff_ci_main,
            img_dirs,
            write_html=write_html,
            write_png=write_png,
        )
        _plot_eta_robustness(
            eta_outputs,
            img_dirs,
            write_html=write_html,
            write_png=write_png,
        )
        _plot_joint_regression_coefficients(
            joint_reg_plot_ci,
            joint_reg_plot_diff_ci,
            img_dirs,
            write_html=write_html,
            write_png=write_png,
        )
        _plot_pooled_log_regression_coefficients(
            pooled_reg_outputs.coefficients,
            pooled_reg_ci,
            img_dirs,
            write_html=write_html,
            write_png=write_png,
            stem=_pooled_log_regression_stem("pooled_log_regression_coefficients", "proprietary"),
        )
        _plot_pooled_log_regression_coefficients(
            pooled_reg_client_outputs.coefficients,
            pooled_reg_client_ci,
            img_dirs,
            write_html=write_html,
            write_png=write_png,
            stem=_pooled_log_regression_stem("pooled_log_regression_coefficients", "client"),
        )

        print(f"[Crowding impact] Saved tables to {paths.out_dir}")
        print(f"[Crowding impact] Saved figures to {img_dirs.base_dir}")
        print(f"[Crowding impact] Saved run manifest to {paths.out_dir / 'run_manifest.json'}")
        print("[Crowding impact] Run completed.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
