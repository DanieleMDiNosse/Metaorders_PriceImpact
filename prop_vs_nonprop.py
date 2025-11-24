#!/usr/bin/env python3
"""
Imbalance / crowding analysis for proprietary vs non-proprietary metaorders.

- Loads two files: proprietary and (optionally) non-proprietary.
- Computes per-(ISIN, Date) *local* imbalance excluding each metaorder itself.
- Computes correlations between Direction and local imbalance, with
  95% confidence intervals (Fisher's z).
- Reproduces the "global imbalance" correlation as a sanity check.
- Additionally computes *daily* correlations Corr(Direction, imbalance_local)
  with CIs for both groups, and (optionally) saves plots across days.

Usage (with defaults pointing to parquet files):
    python imbalance_analysis.py

Or explicitly:
    python imbalance_analysis.py \
        --prop out_files/metaorders_info_sameday_filtered_member_proprietary.parquet \
        --client out_files/metaorders_info_sameday_filtered_member_non_proprietary.parquet \
        --min-n 100
"""

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (good for scripts / servers)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Correlation with confidence interval
# ---------------------------------------------------------------------
def corr_with_ci(
    x: Iterable[float],
    y: Iterable[float],
    alpha: float = 0.05,
) -> Tuple[float, float, float, int]:
    """
    Compute Pearson correlation r and a (1 - alpha) CI using Fisher's z-transform.

    Returns
    -------
    r : float
        Pearson correlation.
    lo : float
        Lower bound of the CI.
    hi : float
        Upper bound of the CI.
    n : int
        Sample size used in the correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    n = x.size

    if n < 3:
        return float("nan"), float("nan"), float("nan"), n

    r = float(np.corrcoef(x, y)[0, 1])

    # Guard against |r|=1 which would blow up Fisher's transform
    r_clipped = max(min(r, 0.999999), -0.999999)

    # Fisher z-transform
    z = 0.5 * math.log((1.0 + r_clipped) / (1.0 - r_clipped))
    se = 1.0 / math.sqrt(n - 3)

    # z critical value for two-sided CI; alpha=0.05 -> ~1.96
    from scipy.stats import norm
    z_crit = norm.ppf(1.0 - alpha / 2.0)

    z_lo = z - z_crit * se
    z_hi = z + z_crit * se

    # Back-transform to r
    lo = math.tanh(z_lo)
    hi = math.tanh(z_hi)

    return r, lo, hi, n


def extract_date(period_list):
    if len(period_list) > 0:
        return pd.to_datetime(period_list[0]).date()
    return None


# ---------------------------------------------------------------------
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
    For each group (e.g. ISIN, Date) and each metaorder i in that group, compute:
        imbalance_i = sum_{j != i} Q_j * D_j / sum_{j != i} Q_j

    This is the signed volume imbalance of *other* metaorders on the same
    stock & day, in the spirit of Bucci / Brière.
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
    alpha: float = 0.05,
) -> None:
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
    r_loc, lo_loc, hi_loc, n_loc = corr_with_ci(x, y, alpha=alpha)

    if math.isnan(r_loc):
        print("\nCorr(Direction, local imbalance): not enough data (n < 3).")
    else:
        print(
            f"\nCorr({side_col}, {imb_col}) = {r_loc:.3f} "
            f"(95% CI [{lo_loc:.3f}, {hi_loc:.3f}], n={n_loc})"
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
    For each Date, compute Corr(Direction, imbalance_local) + CI.
    Returns a DataFrame with columns: Date, r, lo, hi, n.
    """
    rows = []
    for d, g in df.groupby(date_col, sort=True):
        x = g[side_col].to_numpy(dtype=float)
        y = g[imb_col].to_numpy(dtype=float)
        r, lo, hi, n = corr_with_ci(x, y, alpha=alpha)
        rows.append({"Date": d, "r": r, "lo": lo, "hi": hi, "n": n})

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
    Compute imbalance on (ISIN, Date) using only the source_df metaorders.
    This is used for cross-group crowding where the environment is the *other* group.
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
    """Attach environment imbalance (computed on environment_df) to each target row."""
    env_imb = compute_environment_imbalance(
        environment_df,
        group_cols=group_cols,
        side_col=side_col,
        vol_col=vol_col,
        new_col=new_col,
    )
    return target_df.merge(env_imb, on=list(group_cols), how="left", sort=False)


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
    Save two plots:
    - daily_crowding_daily_corr.png : daily correlations with CI bands
    - daily_crowding_rolling_{N}d.png: rolling mean of daily correlations
    """
    if smoothing_days < 1:
        raise ValueError("smoothing_days must be >= 1")

    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    # 1) Raw daily correlations with CI bands
    fig, ax = plt.subplots(figsize=(12, 5))

    # Proprietary
    ax.plot(
        daily_prop["Date"],
        daily_prop["r"],
        label=label_prop,
        linewidth=1.5,
    )
    ax.fill_between(
        daily_prop["Date"],
        daily_prop["lo"],
        daily_prop["hi"],
        alpha=0.2,
    )

    # Client
    ax.plot(
        daily_client["Date"],
        daily_client["r"],
        label=label_client,
        linestyle="--",
        linewidth=1.5,
    )
    ax.fill_between(
        daily_client["Date"],
        daily_client["lo"],
        daily_client["hi"],
        alpha=0.2,
    )

    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel("Daily correlation")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    out_path1 = out_prefix_path.parent / f"{out_prefix_path.name}_daily_corr.png"
    fig.savefig(out_path1, bbox_inches="tight")
    plt.close(fig)
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

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        daily_prop["Date"],
        daily_prop["r_roll"],
        label=f"Prop {smoothing_days}-day mean",
        linewidth=1.5,
    )
    ax.plot(
        daily_client["Date"],
        daily_client["r_roll"],
        label=f"Client {smoothing_days}-day mean",
        linewidth=1.5,
    )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel(f"{smoothing_days}-day rolling correlation")
    ax.set_xlabel("Date")
    resolved_smoothed_title = (
        smoothed_title
        if smoothed_title is not None
        else f"Smoothed daily crowding ({smoothing_days}-day rolling mean)"
    )
    ax.set_title(resolved_smoothed_title)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    out_path2 = out_prefix_path.parent / f"{out_prefix_path.name}_rolling_{smoothing_days}d.png"
    fig.savefig(out_path2, bbox_inches="tight")
    plt.close(fig)
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
    Compute daily Corr(Direction, imbalance_local) for prop and client,
    print summary, and optionally save plots using a configurable smoothing window.
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
        print(
            f"\nProp mean daily correlation (unfiltered): {daily_prop['r'].mean():.3f}"
        )
        print(
            f"Prop mean daily correlation (n >= {min_n}): {daily_prop_f['r'].mean():.3f}"
        )
    if not daily_client_f.empty:
        print(
            f"\nClient mean daily correlation (unfiltered): {daily_client['r'].mean():.3f}"
        )
        print(
            f"Client mean daily correlation (n >= {min_n}): {daily_client_f['r'].mean():.3f}"
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
    Prop vs client (and vice-versa) crowding:
    Corr(Direction_group, imbalance built from the other group on the same ISIN/day).
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

    daily_prop_env = daily_crowding_ts(prop_with_client, imb_col=prop_env_col, alpha=alpha)
    daily_client_env = daily_crowding_ts(client_with_prop, imb_col=client_env_col, alpha=alpha)

    daily_prop_f = daily_prop_env[daily_prop_env["n"] >= min_n].reset_index(drop=True)
    daily_client_f = daily_client_env[daily_client_env["n"] >= min_n].reset_index(drop=True)

    print(f"\nProp vs client days with n >= {min_n}: {len(daily_prop_f)} "
          f"(out of {len(daily_prop_env)})")
    print(f"Client vs prop days with n >= {min_n}: {len(daily_client_f)} "
          f"(out of {len(daily_client_env)})")

    if not daily_prop_f.empty:
        print(
            f"\nProp vs client mean daily correlation (unfiltered): {daily_prop_env['r'].mean():.3f}"
        )
        print(
            f"Prop vs client mean daily correlation (n >= {min_n}): {daily_prop_f['r'].mean():.3f}"
        )
    if not daily_client_f.empty:
        print(
            f"\nClient vs prop mean daily correlation (unfiltered): {daily_client_env['r'].mean():.3f}"
        )
        print(
            f"Client vs prop mean daily correlation (n >= {min_n}): {daily_client_f['r'].mean():.3f}"
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
    Corr(Direction_group, imbalance of all other metaorders on the same ISIN/day),
    where 'others' includes both prop and client and excludes the metaorder itself.
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


def compute_daily_metaorder_counts(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame with daily counts per group and their imbalance."""
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
    """Save a time-series plot and histogram of the daily count imbalance."""
    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    # Time-series plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        daily_metaorders.index,
        daily_metaorders["imbalance_counts"],
        color="tab:blue",
        linewidth=1.5,
    )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("(N_prop - N_client) / (N_prop + N_client)")
    ax.set_title("Daily metaorder count imbalance")
    fig.autofmt_xdate()
    plt.tight_layout()
    ts_path = out_prefix_path.parent / f"{out_prefix_path.name}_imbalance_timeseries.png"
    fig.savefig(ts_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Daily count plots] Saved imbalance time-series to: {ts_path}")

    # Histogram of imbalance distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        daily_metaorders["imbalance_counts"].dropna(),
        bins="auto",
        density=True,
        color="tab:blue",
        alpha=0.7,
    )
    ax.set_title("Distribution of daily count imbalance")
    ax.set_xlabel("Imbalance")
    ax.set_ylabel("Density")
    plt.tight_layout()
    hist_path = out_prefix_path.parent / f"{out_prefix_path.name}_imbalance_histogram.png"
    fig.savefig(hist_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Daily count plots] Saved imbalance histogram to: {hist_path}")


def run_daily_count_imbalance_analysis(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    out_prefix: str,
    make_plots: bool = True,
) -> None:
    """Compute and optionally plot daily count imbalance between prop and client flow."""
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Imbalance/crowding analysis for metaorders."
    )
    parser.add_argument(
        "--prop",
        required=False,
        help="Path to proprietary metaorders file.",
        default="out_files/metaorders_info_sameday_filtered_member_proprietary.parquet",
    )
    parser.add_argument(
        "--client",
        required=False,
        help="Path to non-proprietary (client) metaorders file.",
        default="out_files/metaorders_info_sameday_filtered_member_non_proprietary.parquet",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for confidence intervals (default: 0.05 -> 95%% CI).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=100,
        help="Minimum number of metaorders per day to include in daily analysis/plots.",
    )
    parser.add_argument(
        "--smoothing-days",
        type=int,
        default=5,
        help="Number of days used for smoothing the rolling correlation plot.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="/home/danielemdn/Documents/repositories/Metaorders_PriceImpact/images/prop_vs_nonprop",
        help="Directory where the plot files will be saved.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving daily crowding plots.",
    )
    return parser.parse_args()


def load_metaorders(path: str) -> pd.DataFrame:
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
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    required = ["ISIN", "Q", "Direction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    return df


def main() -> None:
    args = parse_args()

    # Load data
    metaorders_proprietary = load_metaorders(args.prop)
    metaorders_non_proprietary = load_metaorders(args.client)
    metaorders_proprietary["Group"] = "prop"
    metaorders_non_proprietary["Group"] = "client"

    # Add local daily imbalance
    metaorders_proprietary = add_daily_imbalance(metaorders_proprietary)
    metaorders_non_proprietary = add_daily_imbalance(metaorders_non_proprietary)

    # Global analysis on full sample
    analyze_flow(
        metaorders_proprietary,
        "Proprietary metaorders",
        alpha=args.alpha,
    )
    analyze_flow(
        metaorders_non_proprietary,
        "Non-proprietary (client) metaorders",
        alpha=args.alpha,
    )

    # Daily time-series analysis + plots
    run_daily_crowding_analysis(
        metaorders_proprietary,
        metaorders_non_proprietary,
        alpha=args.alpha,
        min_n=args.min_n,
        out_prefix=str(Path(args.plot_dir) / "daily_crowding"),
        make_plots=not args.no_plots,
        smoothing_days=args.smoothing_days,
    )

    # Cross-group crowding: prop vs client environments
    run_cross_group_crowding_analysis(
        metaorders_proprietary,
        metaorders_non_proprietary,
        alpha=args.alpha,
        min_n=args.min_n,
        out_prefix=str(Path(args.plot_dir) / "cross_crowding"),
        make_plots=not args.no_plots,
        smoothing_days=args.smoothing_days,
    )

    # Versus all others (prop + client) with self-exclusion
    run_all_vs_all_crowding_analysis(
        metaorders_proprietary,
        metaorders_non_proprietary,
        alpha=args.alpha,
        min_n=args.min_n,
        out_prefix=str(Path(args.plot_dir) / "all_vs_all_crowding"),
        make_plots=not args.no_plots,
        smoothing_days=args.smoothing_days,
    )

    run_daily_count_imbalance_analysis(
        metaorders_proprietary,
        metaorders_non_proprietary,
        out_prefix=str(Path(args.plot_dir) / "daily_counts"),
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
