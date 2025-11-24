#!/usr/bin/env python3
"""
Plot power-law impact fits for proprietary vs non-proprietary metaorders on a single figure.

This mirrors the weighted log-binned fits used in `metaorder_computation.py`
and overlays both categories for quick comparison.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()


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
    Zbar = np.average(Z, weights=w)
    R2_log = 1.0 - np.sum(w * (Z - Zhat) ** 2) / np.sum(w * (Z - Zbar) ** 2)
    yhat = power_law(binned["center_QV"].to_numpy(), Y_hat, gamma_hat)
    ybar = np.mean(binned["mean_imp"].to_numpy())
    R2_lin = 1.0 - np.sum((binned["mean_imp"].to_numpy() - yhat) ** 2) / np.sum(
        (binned["mean_imp"].to_numpy() - ybar) ** 2
    )
    return binned, (Y_hat, Y_se, gamma_hat, gamma_se, R2_log, R2_lin)


def plot_fit(ax, binned: pd.DataFrame, params, label_prefix=None, label_size: int = 16, legend_size: int = 14):
    Y, Y_err, gamma, gamma_err, R2_log, R2_lin = params
    ax.errorbar(
        binned["center_QV"],
        binned["mean_imp"],
        yerr=binned["sem_imp"],
        fmt="o",
        alpha=0.8,
        ecolor="gray",
        label="Bin means +/- SEM" if label_prefix is None else f"{label_prefix}: means +/- SEM",
    )
    x_min, x_max = binned["center_QV"].min(), binned["center_QV"].max()
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    ax.plot(
        x_grid,
        power_law(x_grid, Y, gamma),
        label=(
            rf'{" " if label_prefix is None else label_prefix + ": "}'
            rf"$I/\sigma = ({Y:.3g}\pm{Y_err:.2g})(Q/V)^{{{gamma:.3f}\pm{gamma_err:.3f}}}$"
        ),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Q/V", fontsize=label_size)
    ax.set_ylabel(r"I / $\sigma$", fontsize=label_size)
    ax.tick_params(axis="both", which="both", labelsize=label_size)
    ax.legend(loc="best", fontsize=legend_size)


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
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix == ".parquet":
        return pd.read_parquet(path_obj)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path_obj)
    raise ValueError(f"Unsupported extension for metaorders file: {path_obj.name}")


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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 14,
        }
    )

    datasets = [
        ("Proprietary", args.prop_path),
        ("Non-proprietary", args.nonprop_path),
    ]
    fits = []

    for label, path in datasets:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset for {label}: {path}")
        print(f"[{label}] Loading {path}")
        df = load_metaorders(path)
        df = prepare_metaorders(df)
        print(f"[{label}] Rows after filtering: {len(df):,}")
        binned, params = fit_power_law_logbins_wls(
            df,
            n_logbins=args.n_logbins,
            min_count=args.min_count,
            use_median=args.use_median,
        )
        fits.append((label, binned, params))
        Y_hat, Y_se, gamma_hat, gamma_se, R2_log, R2_lin = params
        print(
            f"[{label}] Y={Y_hat:.6g} +/- {Y_se:.3g} | gamma={gamma_hat:.6f} +/- {gamma_se:.3g} "
            f"| R2_log={R2_log:.4f} | R2_lin={R2_lin:.4f} | bins={len(binned)}"
        )

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, binned, params in fits:
        plot_fit(ax, binned, params, label_prefix=label)
    ax.set_title(args.title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved combined figure to {out_path}")


if __name__ == "__main__":
    main()
