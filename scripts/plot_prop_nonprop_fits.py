#!/usr/bin/env python3
"""
Compare proprietary and client metaorders with a YAML-configured workflow.

What this script does
---------------------
The script reloads the filtered proprietary and client metaorder tables and can
produce three comparison outputs:

- a combined power-law impact-fit overlay;
- a combined logarithmic impact-fit overlay;
- a day-cluster bootstrap test of the retention difference
  `R_prop - R_client`, where `R_g = mean(I(tau_end)) / mean(I(tau_start))`.

How to run
----------
1) Edit `config_ymls/plot_prop_nonprop_fits.yml`, or point
   `PLOT_PROP_NONPROP_FITS_CONFIG` to an alternate YAML file.
2) Run:

    python scripts/plot_prop_nonprop_fits.py

Outputs
-------
- Figures:
  `images/{DATASET_NAME}/prop_vs_nonprop/png/` and `.../html/`
- Tables:
  `out_files/{DATASET_NAME}/prop_vs_nonprop/`
- Log file:
  `out_files/{DATASET_NAME}/logs/plot_prop_nonprop_fits.log`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# Ensure repository-root imports (e.g., `moimpact`) work when running
# `python scripts/plot_prop_nonprop_fits.py` from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.config import cfg_require, format_path_template, load_yaml_mapping, resolve_repo_path
from moimpact.logging_utils import PrintTee, setup_file_logger
from moimpact.plot_style import (
    THEME_BG_COLOR,
    THEME_COLORWAY,
    THEME_FONT_FAMILY,
    THEME_GRID_COLOR,
    apply_plotly_style,
)
from moimpact.plotting import (
    COLOR_CLIENT,
    COLOR_NEUTRAL,
    COLOR_PROPRIETARY,
    PlotOutputDirs,
    ensure_plot_dirs,
    make_plot_output_dirs,
    save_plotly_figure,
)
from moimpact.stats.impact_paths import RetentionDifferenceBootstrapResult, bootstrap_retention_difference


_CONFIG_ENV_VAR = "PLOT_PROP_NONPROP_FITS_CONFIG"
_config_override = os.environ.get(_CONFIG_ENV_VAR)
if _config_override:
    _CONFIG_PATH = Path(_config_override).expanduser()
    if not _CONFIG_PATH.is_absolute():
        _CONFIG_PATH = (_REPO_ROOT / _CONFIG_PATH).resolve()
else:
    _CONFIG_PATH = _REPO_ROOT / "config_ymls" / "plot_prop_nonprop_fits.yml"
_CFG = load_yaml_mapping(_CONFIG_PATH)


def _cfg_require(key: str):
    return cfg_require(_CFG, key, _CONFIG_PATH)


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_repo_path(_REPO_ROOT, value)


def _format_path_template(template: str, context: Mapping[str, str]) -> str:
    return format_path_template(template, context)


def _parse_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(value)


def _parse_side_filter(value: object) -> Optional[int]:
    if value is None:
        return None
    side = str(value).strip().lower()
    if side in {"", "all", "none", "null"}:
        return None
    if side in {"buy", "b", "+1", "1"}:
        return 1
    if side in {"sell", "s", "-1"}:
        return -1
    raise ValueError("SIDE_FILTER must be one of: all/null, buy, or sell.")


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

DATASET_NAME = str(_CFG.get("DATASET_NAME") or "ftsemib")
COMPARISON_DIRNAME = str(_CFG.get("COMPARISON_DIRNAME") or "prop_vs_nonprop")
_PATH_CONTEXT = {"DATASET_NAME": DATASET_NAME}

OUTPUT_DIR = _resolve_repo_path(
    _format_path_template(str(_cfg_require("OUTPUT_FILE_PATH")), _PATH_CONTEXT)
)
IMG_BASE_DIR = _resolve_repo_path(
    _format_path_template(str(_cfg_require("IMG_OUTPUT_PATH")), _PATH_CONTEXT)
)
SUMMARY_OUTPUT_DIR = OUTPUT_DIR / COMPARISON_DIRNAME
IMG_DIR = IMG_BASE_DIR / COMPARISON_DIRNAME

PROPRIETARY_PATH = _resolve_repo_path(
    _format_path_template(str(_cfg_require("PROPRIETARY_PATH")), _PATH_CONTEXT)
)
CLIENT_PATH = _resolve_repo_path(
    _format_path_template(str(_cfg_require("CLIENT_PATH")), _PATH_CONTEXT)
)

RUN_POWER_LAW_OVERLAY = bool(_cfg_require("RUN_POWER_LAW_OVERLAY"))
FIT_OUTPUT_STEM = str(_cfg_require("FIT_OUTPUT_STEM"))
FIT_N_LOGBINS = int(_cfg_require("FIT_N_LOGBINS"))
FIT_MIN_COUNT = int(_cfg_require("FIT_MIN_COUNT"))
FIT_USE_MEDIAN = bool(_cfg_require("FIT_USE_MEDIAN"))
FIT_MIN_QV = float(_cfg_require("FIT_MIN_QV"))
FIT_MAX_PARTICIPATION_RATE = _parse_optional_float(_CFG.get("FIT_MAX_PARTICIPATION_RATE"))
FIT_MIN_DURATION_SECONDS = _parse_optional_float(_CFG.get("FIT_MIN_DURATION_SECONDS"))

RUN_RETENTION_BOOTSTRAP = bool(_cfg_require("RUN_RETENTION_BOOTSTRAP"))
RETENTION_OUTPUT_STEM = str(_cfg_require("RETENTION_OUTPUT_STEM"))
RETENTION_TAU_START = float(_cfg_require("RETENTION_TAU_START"))
RETENTION_TAU_END = float(_cfg_require("RETENTION_TAU_END"))
RETENTION_DURATION_MULTIPLIER = float(_cfg_require("RETENTION_DURATION_MULTIPLIER"))
RETENTION_BOOTSTRAP_RUNS = int(_cfg_require("RETENTION_BOOTSTRAP_RUNS"))
RETENTION_ALPHA = float(_cfg_require("RETENTION_ALPHA"))
RETENTION_RANDOM_STATE = (
    None if _CFG.get("RETENTION_RANDOM_STATE") is None else int(_CFG["RETENTION_RANDOM_STATE"])
)
RETENTION_MIN_QV = float(_cfg_require("RETENTION_MIN_QV"))
RETENTION_MAX_PARTICIPATION_RATE = _parse_optional_float(_CFG.get("RETENTION_MAX_PARTICIPATION_RATE"))
RETENTION_MIN_DURATION_SECONDS = _parse_optional_float(_CFG.get("RETENTION_MIN_DURATION_SECONDS"))
RETENTION_SIDE_FILTER = _parse_side_filter(_CFG.get("RETENTION_SIDE_FILTER"))


def _weights_from_sigma(sigma: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    weights = np.zeros_like(sigma, dtype=float)
    valid = np.isfinite(sigma) & (sigma > 0.0)
    weights[valid] = 1.0 / np.square(sigma[valid])
    return weights


def _weighted_r2(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.shape != yhat.shape or y.shape != w.shape:
        raise ValueError("y, yhat and w must have the same shape.")
    valid = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w) & (w > 0.0)
    if int(np.count_nonzero(valid)) < 3:
        return float("nan")
    y_valid = y[valid]
    yhat_valid = yhat[valid]
    w_valid = w[valid]
    ybar = float(np.average(y_valid, weights=w_valid))
    denom = float(np.sum(w_valid * np.square(y_valid - ybar)))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum(w_valid * np.square(y_valid - yhat_valid)) / denom)


def _power_law(qv: np.ndarray, Y: float, gamma: float) -> np.ndarray:
    return Y * np.power(qv, gamma)


def _logarithmic_impact(qv: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.log10(1.0 + b * qv)


def _fit_power_law_logbins_wls(
    metaorders: pd.DataFrame,
    *,
    n_logbins: int,
    min_count: int,
    use_median: bool,
) -> Tuple[pd.DataFrame, Tuple[float, float, float, float, float, float]]:
    sub = metaorders[(metaorders["Q/V"] > 0.0) & np.isfinite(metaorders["Impact"])].copy()
    if sub.empty:
        raise ValueError("No valid rows (Q/V > 0 and finite Impact).")

    x = sub["Q/V"].to_numpy(dtype=float)
    y = sub["Impact"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("Invalid Q/V range for log binning.")

    edges = np.logspace(np.log10(x_min), np.log10(x_max), int(n_logbins) + 1)
    bin_idx = np.digitize(x, edges) - 1
    mask = (bin_idx >= 0) & (bin_idx < int(n_logbins))
    x = x[mask]
    y = y[mask]
    bin_idx = bin_idx[mask]

    binned_raw = pd.DataFrame({"x": x, "y": y, "bin": bin_idx})
    agg = (
        binned_raw.groupby("bin", sort=True)["y"]
        .agg(mean_imp="mean", median_imp="median", std_imp=lambda s: s.std(ddof=1), count="size")
        .sort_index()
    )
    y_stat = agg["median_imp"] if use_median else agg["mean_imp"]
    y_std = agg["std_imp"].to_numpy(dtype=float)
    count = agg["count"].to_numpy(dtype=int)
    sem = y_std / np.sqrt(np.maximum(count, 1))
    bins_present = agg.index.to_numpy(dtype=int)
    left_edges = edges[bins_present]
    right_edges = edges[bins_present + 1]
    centers = np.sqrt(left_edges * right_edges)

    binned = (
        pd.DataFrame(
            {
                "center_QV": centers,
                "mean_imp": y_stat.to_numpy(dtype=float),
                "std_imp": y_std,
                "sem_imp": sem,
                "count": count,
            }
        )
        .sort_values("center_QV")
        .reset_index(drop=True)
    )
    binned = binned[
        (binned["count"] >= int(min_count))
        & np.isfinite(binned["mean_imp"])
        & np.isfinite(binned["sem_imp"])
        & (binned["sem_imp"] > 0.0)
        & (binned["mean_imp"] > 0.0)
    ].reset_index(drop=True)
    if len(binned) < 3:
        raise ValueError(f"Not enough valid bins after filtering (got {len(binned)}).")

    X = np.log(binned["center_QV"].to_numpy(dtype=float))
    Z = np.log(binned["mean_imp"].to_numpy(dtype=float))
    var_logy = np.square(binned["sem_imp"].to_numpy(dtype=float) / binned["mean_imp"].to_numpy(dtype=float))
    w = np.where(np.isfinite(var_logy) & (var_logy > 0.0), 1.0 / var_logy, 0.0)
    A = np.vstack([np.ones_like(X), X]).T
    Aw = A * np.sqrt(w)[:, None]
    Zw = Z * np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(Aw, Zw, rcond=None)
    a_hat, gamma_hat = coef
    Y_hat = float(np.exp(a_hat))
    residuals = Z - (a_hat + gamma_hat * X)
    rss = float(np.sum(w * residuals**2))
    dof = max(len(Z) - 2, 1)
    s2 = rss / dof
    XtWX = A.T @ (w[:, None] * A)
    cov = s2 * np.linalg.inv(XtWX)
    a_se, gamma_se = np.sqrt(np.diag(cov))
    Y_se = float(Y_hat * a_se)
    z_hat = a_hat + gamma_hat * X
    r2_log = _weighted_r2(Z, z_hat, w=w)
    y_hat = _power_law(binned["center_QV"].to_numpy(dtype=float), Y_hat, gamma_hat)
    w_lin = _weights_from_sigma(binned["sem_imp"].to_numpy(dtype=float))
    r2_lin = _weighted_r2(binned["mean_imp"].to_numpy(dtype=float), y_hat, w=w_lin)
    return binned, (Y_hat, Y_se, float(gamma_hat), float(gamma_se), r2_log, r2_lin)


def _fit_logarithmic_from_binned(
    binned: pd.DataFrame,
) -> Tuple[float, float, float, float, float]:
    if binned.empty:
        raise ValueError("No bins available for logarithmic fit.")

    x = binned["center_QV"].to_numpy(dtype=float)
    y = binned["mean_imp"].to_numpy(dtype=float)
    sigma = binned["sem_imp"].to_numpy(dtype=float)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite values encountered in binned data for logarithmic fit.")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Non-positive or non-finite SEM values encountered for logarithmic fit.")

    x_pos = x[x > 0.0]
    y_pos = y[y > 0.0]
    if x_pos.size > 0 and y_pos.size > 0:
        b0 = 1.0
        denom = np.log10(1.0 + b0 * float(np.max(x_pos)))
        a0 = float(np.max(y_pos) / denom) if denom > 0.0 else float(np.mean(y_pos))
    else:
        a0, b0 = 1.0, 1.0

    try:
        popt, pcov = curve_fit(
            _logarithmic_impact,
            x,
            y,
            p0=(a0, b0),
            sigma=sigma,
            absolute_sigma=True,
            maxfev=20000,
            bounds=((0.0, 0.0), (np.inf, np.inf)),
        )
    except Exception as exc:
        raise ValueError(f"Nonlinear logarithmic fit failed: {exc}") from exc

    a_hat = float(popt[0])
    b_hat = float(popt[1])
    if pcov is None or not np.all(np.isfinite(pcov)):
        a_se = float("nan")
        b_se = float("nan")
    else:
        a_se, b_se = np.sqrt(np.diag(pcov))
        a_se = float(a_se)
        b_se = float(b_se)

    y_hat = _logarithmic_impact(x, a_hat, b_hat)
    w_lin = _weights_from_sigma(sigma)
    r2_lin = _weighted_r2(y, y_hat, w=w_lin)
    return a_hat, a_se, b_hat, b_se, r2_lin


def _plot_fit(
    fig: go.Figure,
    binned: pd.DataFrame,
    params: Tuple[float, float, float, float, float, float],
    *,
    label_prefix: str,
    color: str,
    fit_dash: str,
) -> None:
    Y_hat, Y_se, gamma_hat, gamma_se, r2_log, _ = params
    fig.add_trace(
        go.Scatter(
            x=binned["center_QV"],
            y=binned["mean_imp"],
            mode="markers",
            marker=dict(size=7, color=color),
            error_y=dict(type="data", array=binned["sem_imp"], visible=True, color=COLOR_NEUTRAL),
            name=f"{label_prefix}: means +/- SEM",
            showlegend=False,
        )
    )
    x_grid = np.logspace(
        np.log10(float(binned["center_QV"].min())),
        np.log10(float(binned["center_QV"].max())),
        300,
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=_power_law(x_grid, Y_hat, gamma_hat),
            mode="lines",
            line=dict(color=color, width=2, dash=fit_dash),
            name=rf"$R^2 = {r2_log:.2f}$",
        )
    )


def _plot_logarithmic_fit(
    fig: go.Figure,
    binned: pd.DataFrame,
    params: Tuple[float, float, float, float, float],
    *,
    label_prefix: str,
    color: str,
    fit_dash: str,
) -> None:
    a_hat, a_se, b_hat, b_se, r2_lin = params
    fig.add_trace(
        go.Scatter(
            x=binned["center_QV"],
            y=binned["mean_imp"],
            mode="markers",
            marker=dict(size=7, color=color),
            error_y=dict(type="data", array=binned["sem_imp"], visible=True, color=COLOR_NEUTRAL),
            name=f"{label_prefix}: means +/- SEM",
            showlegend=False,
        )
    )
    x_grid = np.logspace(
        np.log10(float(binned["center_QV"].min())),
        np.log10(float(binned["center_QV"].max())),
        300,
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=_logarithmic_impact(x_grid, a_hat, b_hat),
            mode="lines",
            line=dict(color=color, width=2, dash=fit_dash),
            name=rf"$R^2 = {r2_lin:.2f}$",
        )
    )


def _derive_log_fit_output_stem(power_law_stem: str) -> str:
    if "power_law" in power_law_stem:
        return power_law_stem.replace("power_law", "logarithmic", 1)
    return f"{power_law_stem}_logarithmic"


def _period_duration_seconds(period: object) -> float:
    if period is None:
        return np.nan
    try:
        start, end = period
    except Exception:
        return np.nan

    try:
        start_val = int(start)
        end_val = int(end)
    except Exception:
        try:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
        except Exception:
            return np.nan
    else:
        start_ts = pd.Timestamp(start_val)
        end_ts = pd.Timestamp(end_val)

    try:
        return float((end_ts - start_ts).total_seconds())
    except Exception:
        return np.nan


def _prepare_metaorders(
    metaorders: pd.DataFrame,
    *,
    min_qv: float,
    max_participation_rate: Optional[float],
    min_duration_seconds: Optional[float],
    require_impact_paths: bool = False,
    side_filter: Optional[int] = None,
) -> pd.DataFrame:
    out = metaorders.copy()
    required = {"Q/V", "Price Change", "Direction", "Daily Vol"}
    if require_impact_paths:
        required.update({"Period", "partial_impact", "aftermath_impact"})
    missing = required.difference(out.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    numeric_cols = ["Q/V", "Impact", "Participation Rate", "Price Change", "Daily Vol", "Q", "Direction"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if min_duration_seconds is not None:
        out["DurationSeconds"] = out["Period"].apply(_period_duration_seconds)
        out = out[out["DurationSeconds"] >= float(min_duration_seconds)].copy()
        out.drop(columns=["DurationSeconds"], inplace=True)

    if side_filter is not None:
        out = out[out["Direction"].eq(float(side_filter))].copy()

    out = out[out["Q/V"] > float(min_qv)].copy()
    if max_participation_rate is not None and "Participation Rate" in out.columns:
        out = out[out["Participation Rate"] < float(max_participation_rate)].copy()

    out["Impact"] = pd.to_numeric(out["Price Change"] * out["Direction"] / out["Daily Vol"], errors="coerce")
    numeric_present = [col for col in numeric_cols if col in out.columns]
    if numeric_present:
        out[numeric_present] = out[numeric_present].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["Q/V", "Impact"]).reset_index(drop=True)
    return out


def _load_metaorders(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported extension for metaorders file: {path.name}")


def _build_retention_bootstrap_figure(
    result: RetentionDifferenceBootstrapResult,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=result.bootstrap_delta_retention,
            nbinsx=min(max(int(np.sqrt(max(result.n_bootstrap_valid, 1))), 15), 60),
            marker_color=COLOR_PROPRIETARY,
            opacity=0.8,
            name="Bootstrap delta draws",
            hovertemplate="Delta retention %{x:.4f}<br>Count %{y}<extra></extra>",
        )
    )
    fig.add_vrect(
        x0=result.delta_ci_low,
        x1=result.delta_ci_high,
        fillcolor="rgba(31, 119, 180, 0.14)",
        line_width=0,
        annotation_text=f"{100.0 * (1.0 - result.alpha):.0f}% CI",
        annotation_position="top left",
    )
    fig.add_vline(
        x=0.0,
        line=dict(color=COLOR_NEUTRAL, width=1.5, dash="dash"),
        annotation_text="Null = 0",
        annotation_position="top right",
    )
    fig.add_vline(
        x=result.delta_retention,
        line=dict(color=COLOR_CLIENT, width=2),
        annotation_text=f"Observed = {result.delta_retention:.4f}",
        annotation_position="top",
    )
    fig.update_layout(
        barmode="overlay",
        xaxis_title=rf"$R_{{prop}} - R_{{client}}$ with $R = \bar I({result.tau_end:g}) / \bar I({result.tau_start:g})$",
        yaxis_title="Bootstrap count",
        showlegend=False,
    )
    fig.add_annotation(
        x=0.01,
        y=0.98,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        text=(
            f"Prop retention={result.proprietary_retention:.4f}<br>"
            f"Client retention={result.client_retention:.4f}<br>"
            f"Delta={result.delta_retention:.4f}<br>"
            f"p={result.p_value:.4g}"
        ),
        font=dict(size=ANNOTATION_FONT_SIZE),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="rgba(0,0,0,0.12)",
        borderwidth=1,
    )
    return fig


def _save_retention_outputs(
    result: RetentionDifferenceBootstrapResult,
    *,
    output_dir: Path,
    output_stem: str,
) -> Tuple[Path, Path, Path]:
    summary_frame = pd.DataFrame([result.summary_dict()])
    draws_frame = pd.DataFrame(
        {
            "proprietary_retention_bootstrap": result.bootstrap_proprietary_retention,
            "client_retention_bootstrap": result.bootstrap_client_retention,
            "delta_retention_bootstrap": result.bootstrap_delta_retention,
        }
    )

    summary_csv_path = output_dir / f"{output_stem}_summary.csv"
    summary_parquet_path = output_dir / f"{output_stem}_summary.parquet"
    draws_parquet_path = output_dir / f"{output_stem}_draws.parquet"
    summary_frame.to_csv(summary_csv_path, index=False)
    summary_frame.to_parquet(summary_parquet_path, index=False)
    draws_frame.to_parquet(draws_parquet_path, index=False)
    return summary_csv_path, summary_parquet_path, draws_parquet_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Summary
    -------
    Run the YAML-configured proprietary-vs-client comparison workflow.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Unused placeholder kept for a stable script signature.

    Returns
    -------
    None

    Notes
    -----
    This script is fully configured via YAML. The `argv` argument is accepted
    only to keep the entrypoint easy to test and mirror the repository's other
    analysis scripts.
    """
    del argv

    log_path = OUTPUT_DIR / "logs" / "plot_prop_nonprop_fits.log"
    logger = setup_file_logger(Path(__file__).stem, log_path, mode="a")
    with PrintTee(logger):
        print("[Prop vs Non-Prop] Comparison run started...")
        print(
            "[Prop vs Non-Prop] Parameters — \n"
            f"  DATASET={DATASET_NAME}\n"
            f"  RUN_POWER_LAW_OVERLAY={RUN_POWER_LAW_OVERLAY}\n"
            f"  RUN_RETENTION_BOOTSTRAP={RUN_RETENTION_BOOTSTRAP}\n"
            f"  FIT_N_LOGBINS={FIT_N_LOGBINS}, FIT_MIN_COUNT={FIT_MIN_COUNT}, FIT_USE_MEDIAN={FIT_USE_MEDIAN}\n"
            f"  RETENTION_TAU_START={RETENTION_TAU_START}, RETENTION_TAU_END={RETENTION_TAU_END}, "
            f"RETENTION_BOOTSTRAP_RUNS={RETENTION_BOOTSTRAP_RUNS}, RETENTION_ALPHA={RETENTION_ALPHA}, "
            f"RETENTION_RANDOM_STATE={RETENTION_RANDOM_STATE}, RETENTION_SIDE_FILTER={RETENTION_SIDE_FILTER}"
        )
        print(
            "[Prop vs Non-Prop] Paths — \n"
            f"  PROPRIETARY_PATH={PROPRIETARY_PATH}\n"
            f"  CLIENT_PATH={CLIENT_PATH}\n"
            f"  SUMMARY_OUTPUT_DIR={SUMMARY_OUTPUT_DIR}\n"
            f"  IMG_DIR={IMG_DIR}"
        )

        if not PROPRIETARY_PATH.exists():
            raise FileNotFoundError(f"Missing proprietary metaorders file: {PROPRIETARY_PATH}")
        if not CLIENT_PATH.exists():
            raise FileNotFoundError(f"Missing client metaorders file: {CLIENT_PATH}")

        plot_dirs: PlotOutputDirs = make_plot_output_dirs(IMG_DIR, use_subdirs=True)
        ensure_plot_dirs(plot_dirs)
        SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"[Proprietary] Loading {PROPRIETARY_PATH}")
        proprietary_raw = _load_metaorders(PROPRIETARY_PATH)
        print(f"[Client] Loading {CLIENT_PATH}")
        client_raw = _load_metaorders(CLIENT_PATH)

        if RUN_POWER_LAW_OVERLAY:
            proprietary_fit = _prepare_metaorders(
                proprietary_raw,
                min_qv=FIT_MIN_QV,
                max_participation_rate=FIT_MAX_PARTICIPATION_RATE,
                min_duration_seconds=FIT_MIN_DURATION_SECONDS,
            )
            client_fit = _prepare_metaorders(
                client_raw,
                min_qv=FIT_MIN_QV,
                max_participation_rate=FIT_MAX_PARTICIPATION_RATE,
                min_duration_seconds=FIT_MIN_DURATION_SECONDS,
            )
            print(f"[Proprietary][Fit] Rows after filtering: {len(proprietary_fit):,}")
            print(f"[Client][Fit] Rows after filtering: {len(client_fit):,}")

            power_law_fits = []
            logarithmic_fits = []
            for label, frame, color, fit_dash in (
                ("Proprietary", proprietary_fit, COLOR_PROPRIETARY, "solid"),
                ("Client", client_fit, COLOR_CLIENT, "solid"),
            ):
                binned, params = _fit_power_law_logbins_wls(
                    frame,
                    n_logbins=FIT_N_LOGBINS,
                    min_count=FIT_MIN_COUNT,
                    use_median=FIT_USE_MEDIAN,
                )
                power_law_fits.append((label, binned, params, color, fit_dash))
                Y_hat, Y_se, gamma_hat, gamma_se, r2_log, r2_lin = params
                print(
                    f"[{label}][Fit] Y={Y_hat:.6g} +/- {Y_se:.3g} | "
                    f"gamma={gamma_hat:.6f} +/- {gamma_se:.3g} | "
                    f"R2_log={r2_log:.4f} | R2_lin={r2_lin:.4f} | bins={len(binned)}"
                )
                log_params = _fit_logarithmic_from_binned(binned)
                logarithmic_fits.append((label, binned, log_params, color, fit_dash))
                a_hat, a_se, b_hat, b_se, r2_lin_log = log_params
                print(
                    f"[{label}][Log Fit] a={a_hat:.6g} +/- {a_se:.3g} | "
                    f"b={b_hat:.6g} +/- {b_se:.3g} | "
                    f"R2_lin={r2_lin_log:.4f} | bins={len(binned)}"
                )

            fit_fig = go.Figure()
            for label, binned, params, color, fit_dash in power_law_fits:
                _plot_fit(
                    fit_fig,
                    binned,
                    params,
                    label_prefix=label,
                    color=color,
                    fit_dash=fit_dash,
                )
            fit_fig.update_xaxes(type="log", title_text="$\phi$", title_font=dict(size=LABEL_FONT_SIZE))
            fit_fig.update_yaxes(type="log", title_text=r"$I/\sigma$", title_font=dict(size=LABEL_FONT_SIZE))
            fit_fig.update_layout(
                showlegend=True,
                title=None,
            )
            fit_html_path, fit_png_path = save_plotly_figure(
                fit_fig,
                stem=FIT_OUTPUT_STEM,
                dirs=plot_dirs,
                write_html=True,
                write_png=True,
                strict_png=False,
            )
            print(
                "[Prop vs Non-Prop] Saved power-law overlay figure "
                f"to HTML={fit_html_path} PNG={fit_png_path}"
            )

            log_fit_fig = go.Figure()
            for label, binned, params, color, fit_dash in logarithmic_fits:
                _plot_logarithmic_fit(
                    log_fit_fig,
                    binned,
                    params,
                    label_prefix=label,
                    color=color,
                    fit_dash=fit_dash,
                )
            log_fit_fig.update_xaxes(type="log", title_text="$\phi$", title_font=dict(size=LABEL_FONT_SIZE))
            log_fit_fig.update_yaxes(type="log", title_text=r"$I/\sigma$", title_font=dict(size=LABEL_FONT_SIZE))
            log_fit_fig.update_layout(
                showlegend=True,
                title=None,
            )
            log_fit_output_stem = _derive_log_fit_output_stem(FIT_OUTPUT_STEM)
            log_fit_html_path, log_fit_png_path = save_plotly_figure(
                log_fit_fig,
                stem=log_fit_output_stem,
                dirs=plot_dirs,
                write_html=True,
                write_png=True,
                strict_png=False,
            )
            print(
                "[Prop vs Non-Prop] Saved logarithmic overlay figure "
                f"to HTML={log_fit_html_path} PNG={log_fit_png_path}"
            )

        if RUN_RETENTION_BOOTSTRAP:
            proprietary_retention = _prepare_metaorders(
                proprietary_raw,
                min_qv=RETENTION_MIN_QV,
                max_participation_rate=RETENTION_MAX_PARTICIPATION_RATE,
                min_duration_seconds=RETENTION_MIN_DURATION_SECONDS,
                require_impact_paths=True,
                side_filter=RETENTION_SIDE_FILTER,
            )
            client_retention = _prepare_metaorders(
                client_raw,
                min_qv=RETENTION_MIN_QV,
                max_participation_rate=RETENTION_MAX_PARTICIPATION_RATE,
                min_duration_seconds=RETENTION_MIN_DURATION_SECONDS,
                require_impact_paths=True,
                side_filter=RETENTION_SIDE_FILTER,
            )
            print(f"[Proprietary][Retention] Rows after filtering: {len(proprietary_retention):,}")
            print(f"[Client][Retention] Rows after filtering: {len(client_retention):,}")

            result = bootstrap_retention_difference(
                proprietary_retention,
                client_retention,
                tau_start=RETENTION_TAU_START,
                tau_end=RETENTION_TAU_END,
                duration_multiplier=RETENTION_DURATION_MULTIPLIER,
                alpha=RETENTION_ALPHA,
                n_bootstrap=RETENTION_BOOTSTRAP_RUNS,
                random_state=RETENTION_RANDOM_STATE,
            )
            print(
                "[Prop vs Non-Prop][Retention] "
                f"R_prop={result.proprietary_retention:.6f} "
                f"(CI [{result.proprietary_ci_low:.6f}, {result.proprietary_ci_high:.6f}]), "
                f"R_client={result.client_retention:.6f} "
                f"(CI [{result.client_ci_low:.6f}, {result.client_ci_high:.6f}]), "
                f"delta={result.delta_retention:.6f} "
                f"(CI [{result.delta_ci_low:.6f}, {result.delta_ci_high:.6f}]), "
                f"p={result.p_value:.6g}, valid_bootstrap_runs={result.n_bootstrap_valid}"
            )

            summary_csv_path, summary_parquet_path, draws_parquet_path = _save_retention_outputs(
                result,
                output_dir=SUMMARY_OUTPUT_DIR,
                output_stem=RETENTION_OUTPUT_STEM,
            )
            print(
                "[Prop vs Non-Prop][Retention] Saved summary outputs to "
                f"{summary_csv_path}, {summary_parquet_path}, and {draws_parquet_path}"
            )

            retention_fig = _build_retention_bootstrap_figure(result)
            retention_html_path, retention_png_path = save_plotly_figure(
                retention_fig,
                stem=RETENTION_OUTPUT_STEM,
                dirs=plot_dirs,
                write_html=True,
                write_png=True,
                strict_png=False,
            )
            print(
                "[Prop vs Non-Prop][Retention] Saved bootstrap figure "
                f"to HTML={retention_html_path} PNG={retention_png_path}"
            )

        if not RUN_POWER_LAW_OVERLAY and not RUN_RETENTION_BOOTSTRAP:
            print("[Prop vs Non-Prop] Both run flags are false; nothing to do.")


if __name__ == "__main__":
    main()
