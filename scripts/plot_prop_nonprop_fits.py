#!/usr/bin/env python3
"""
Plot power-law impact fits for proprietary vs non-proprietary metaorders on a single figure.

This mirrors the weighted log-binned fits used in `metaorder_computation.py`
and overlays both categories for quick comparison.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure repository-root imports (e.g., `moimpact`) work when running
# `python scripts/plot_prop_nonprop_fits.py` from the repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_REPO_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from moimpact.plot_style import (
    THEME_COLORWAY,
    apply_plotly_style,
    THEME_BG_COLOR,
    THEME_FONT_FAMILY,
    THEME_GRID_COLOR,
)
from moimpact.plotting import (
    COLOR_CLIENT,
    COLOR_NEUTRAL,
    COLOR_PROPRIETARY,
    PlotOutputDirs,
    make_plot_output_dirs,
    save_plotly_figure,
)

TICK_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 15
LEGEND_FONT_SIZE = 12
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

def _weights_from_sigma(sigma: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    w = np.zeros_like(sigma, dtype=float)
    ok = np.isfinite(sigma) & (sigma > 0)
    w[ok] = 1.0 / np.square(sigma[ok])
    return w


def _weighted_r2(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.shape != yhat.shape or y.shape != w.shape:
        raise ValueError("y, yhat and w must have the same shape.")
    valid = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w) & (w > 0)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    yv = y[valid]
    yhatv = yhat[valid]
    wv = w[valid]
    ybar = float(np.average(yv, weights=wv))
    denom = float(np.sum(wv * np.square(yv - ybar)))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum(wv * np.square(yv - yhatv)) / denom)


# ---------------------------------------------------------------------------
# Core fitting helpers (mirrors metaorder_computation.py)
# ---------------------------------------------------------------------------
def power_law(qv: np.ndarray, Y: float, gamma: float) -> np.ndarray:
    return Y * np.power(qv, gamma)


def fit_power_law_logbins_wls(
    subdf: pd.DataFrame,
    n_logbins: int = 30,
    min_count: int = 20,
    use_median: bool = False,
) -> Tuple[pd.DataFrame, Tuple[float, float, float, float, float, float]]:
    sub = subdf[(subdf["Q/V"] > 0) & np.isfinite(subdf["Impact"])].copy()
    if sub.empty:
        raise ValueError("No valid rows (Q/V>0 and finite Impact).")
    x = sub["Q/V"].to_numpy()
    y = sub["Impact"].to_numpy()
    x_min = x.min()
    x_max = x.max()
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("Invalid Q/V range for log binning.")
    edges = np.logspace(np.log10(x_min), np.log10(x_max), n_logbins + 1)
    bin_idx = np.digitize(x, edges) - 1
    mask = (bin_idx >= 0) & (bin_idx < n_logbins)
    x, y, bin_idx = x[mask], y[mask], bin_idx[mask]
    dfb = pd.DataFrame({"x": x, "y": y, "bin": bin_idx})
    agg = (
        dfb.groupby("bin")["y"]
        .agg(mean_imp="mean", median_imp="median", std_imp=lambda s: s.std(ddof=1), count="size")
        .sort_index()
    )
    y_stat = agg["median_imp"] if use_median else agg["mean_imp"]
    y_std = agg["std_imp"].to_numpy()
    n = agg["count"].to_numpy()
    sem = y_std / np.sqrt(np.maximum(n, 1))
    bins_present = agg.index.to_numpy()
    left_edges = edges[bins_present]
    right_edges = edges[bins_present + 1]
    x_center = np.sqrt(left_edges * right_edges)
    binned = (
        pd.DataFrame(
            {
                "center_QV": x_center,
                "mean_imp": y_stat.to_numpy(),
                "std_imp": y_std,
                "sem_imp": sem,
                "count": n,
            }
        )
        .sort_values("center_QV")
        .reset_index(drop=True)
    )
    binned = binned[
        (binned["count"] >= min_count)
        & np.isfinite(binned["mean_imp"])
        & np.isfinite(binned["sem_imp"])
        & (binned["sem_imp"] > 0)
        & (binned["mean_imp"] > 0)
    ]
    if len(binned) < 3:
        raise ValueError(f"Not enough valid bins after filtering (got {len(binned)}).")
    X = np.log(binned["center_QV"].to_numpy())
    Z = np.log(binned["mean_imp"].to_numpy())
    var_logy = (binned["sem_imp"].to_numpy() / binned["mean_imp"].to_numpy()) ** 2
    w = np.where(np.isfinite(var_logy) & (var_logy > 0), 1.0 / var_logy, 0.0)
    A = np.vstack([np.ones_like(X), X]).T
    Aw = A * np.sqrt(w)[:, None]
    Zw = Z * np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(Aw, Zw, rcond=None)
    a_hat, gamma_hat = coef
    Y_hat = float(np.exp(a_hat))
    res = Z - (a_hat + gamma_hat * X)
    RSS = np.sum(w * res**2)
    dof = max(len(Z) - 2, 1)
    s2 = RSS / dof
    XtWX = A.T @ (w[:, None] * A)
    cov = s2 * np.linalg.inv(XtWX)
    a_se, gamma_se = np.sqrt(np.diag(cov))
    Y_se = Y_hat * a_se
    Zhat = a_hat + gamma_hat * X
    R2_log = _weighted_r2(Z, Zhat, w=w)
    yhat = power_law(binned["center_QV"].to_numpy(), Y_hat, gamma_hat)
    w_lin = _weights_from_sigma(binned["sem_imp"].to_numpy())
    R2_lin = _weighted_r2(binned["mean_imp"].to_numpy(), yhat, w=w_lin)
    return binned, (Y_hat, Y_se, gamma_hat, gamma_se, R2_log, R2_lin)


def plot_fit(
    fig: go.Figure,
    binned: pd.DataFrame,
    params,
    label_prefix=None,
    label_size: int = 16,
    legend_size: int = 14,
    series_color: Optional[str] = None,
    fit_dash: str = "solid",
):
    Y, Y_err, gamma, gamma_err, R2_log, R2_lin = params
    if series_color is None:
        color_idx = len(fig.data) // 2
        series_color = THEME_COLORWAY[color_idx % len(THEME_COLORWAY)]
    fig.add_trace(
        go.Scatter(
            x=binned["center_QV"],
            y=binned["mean_imp"],
            mode="markers",
            marker=dict(size=7, color=series_color),
            error_y=dict(type="data", array=binned["sem_imp"], visible=True, color=COLOR_NEUTRAL),
            name="Bin means +/- SEM" if label_prefix is None else f"{label_prefix}: means +/- SEM",
        )
    )
    x_min, x_max = binned["center_QV"].min(), binned["center_QV"].max()
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=power_law(x_grid, Y, gamma),
            mode="lines",
            line=dict(color=series_color, width=2, dash=fit_dash),
            name=(
                rf'{" " if label_prefix is None else label_prefix + ": "}'
                rf"$I/\sigma = ({Y:.3g}\pm{Y_err:.2g})(Q/V)^{{{gamma:.3f}\pm{gamma_err:.3f}}}$"
            ),
        )
    )
    fig.update_xaxes(type="log", title_text="Q/V", title_font=dict(size=label_size))
    fig.update_yaxes(type="log", title_text=r"$I/\sigma$", title_font=dict(size=label_size))
    fig.update_layout(legend=dict(font=dict(size=legend_size)))


# ---------------------------------------------------------------------------
# Data preparation and CLI
# ---------------------------------------------------------------------------
def _period_duration_seconds(period) -> float:
    if period is None:
        return np.nan
    try:
        start, end = period
    except Exception:
        return np.nan
    delta = end - start
    try:
        return float(pd.to_timedelta(delta).total_seconds())
    except Exception:
        return np.nan


def prepare_metaorders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Period" not in df.columns:
        raise KeyError("Expected 'Period' column to compute durations for filtering.")
    df["DurationSeconds"] = df["Period"].apply(_period_duration_seconds)
    df = df[df["DurationSeconds"] >= 60].drop(columns=["DurationSeconds"]).reset_index(drop=True)
    df = df[df["Q/V"] > 1e-5]
    df["Impact"] = df["Price Change"] * df["Direction"] / df["Daily Vol"]
    numeric_cols = ["Q/V", "Impact", "Participation Rate", "Price Change", "Daily Vol", "Q"]
    numeric_present = [c for c in numeric_cols if c in df.columns]
    df[numeric_present] = df[numeric_present].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Q/V", "Impact"])
    return df


def load_metaorders(path: str) -> pd.DataFrame:
    path_obj = _resolve_repo_path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix == ".parquet":
        return pd.read_parquet(path_obj)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path_obj)
    raise ValueError(f"Unsupported extension for metaorders file: {path_obj.name}")


def _resolve_repo_path(path: str | Path) -> Path:
    """Resolve CLI paths relative to the repository root when not absolute."""
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = (_REPO_ROOT / path_obj).resolve()
    return path_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay proprietary vs non-proprietary power-law fits.")
    parser.add_argument(
        "--prop-path",
        default="out_files/metaorders_info_sameday_filtered_member_proprietary.parquet",
        help="Path to proprietary metaorders info file (.parquet, .csv, .pkl).",
    )
    parser.add_argument(
        "--nonprop-path",
        default="out_files/metaorders_info_sameday_filtered_member_non_proprietary.parquet",
        help="Path to non-proprietary metaorders info file (.parquet, .csv, .pkl).",
    )
    parser.add_argument(
        "--out",
        default="images/prop_vs_nonprop/power_law_prop_vs_nonprop.png",
        help="Output path for the combined figure.",
    )
    parser.add_argument("--n-logbins", type=int, default=30, help="Number of logarithmic bins for Q/V.")
    parser.add_argument("--min-count", type=int, default=20, help="Minimum observations per bin to keep it.")
    parser.add_argument(
        "--use-median",
        action="store_true",
        help="Use median impact instead of mean when aggregating bins.",
    )
    parser.add_argument(
        "--title",
        default="Power-law fits: proprietary vs non-proprietary",
        help="Title for the plot.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    out_path = _resolve_repo_path(args.out)
    output_dirs: PlotOutputDirs = make_plot_output_dirs(out_path.parent, use_subdirs=True)

    datasets = [
        ("Proprietary", args.prop_path, COLOR_PROPRIETARY, "solid"),
        ("Client", args.nonprop_path, COLOR_CLIENT, "dash"),
    ]
    fits = []

    for label, path, color, fit_dash in datasets:
        resolved_path = _resolve_repo_path(path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Missing dataset for {label}: {resolved_path}")
        print(f"[{label}] Loading {resolved_path}")
        df = load_metaorders(resolved_path)
        df = prepare_metaorders(df)
        print(f"[{label}] Rows after filtering: {len(df):,}")
        binned, params = fit_power_law_logbins_wls(
            df,
            n_logbins=args.n_logbins,
            min_count=args.min_count,
            use_median=args.use_median,
        )
        fits.append((label, binned, params, color, fit_dash))
        Y_hat, Y_se, gamma_hat, gamma_se, R2_log, R2_lin = params
        print(
            f"[{label}] Y={Y_hat:.6g} +/- {Y_se:.3g} | gamma={gamma_hat:.6f} +/- {gamma_se:.3g} "
            f"| R2_log={R2_log:.4f} | R2_lin={R2_lin:.4f} | bins={len(binned)}"
        )

    fig = go.Figure()
    for label, binned, params, color, fit_dash in fits:
        plot_fit(fig, binned, params, label_prefix=label, series_color=color, fit_dash=fit_dash)
    fig.update_layout(
        title=args.title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    html_path, png_path = save_plotly_figure(
        fig,
        stem=out_path.stem,
        dirs=output_dirs,
        write_html=True,
        write_png=True,
        strict_png=False,
    )
    print(f"Saved combined figure to HTML={html_path} PNG={png_path}")


if __name__ == "__main__":
    main()
