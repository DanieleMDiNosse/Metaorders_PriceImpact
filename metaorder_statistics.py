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
- Optionally builds aggregate density plots from the raw metaorders_dict output
  (durations, volumes, inter-arrivals, participation).

Usage (with defaults pointing to parquet files):
    python imbalance_analysis.py

Or explicitly:
    python imbalance_analysis.py \
        --prop out_files/metaorders_info_sameday_filtered_member_proprietary.parquet \
        --client out_files/metaorders_info_sameday_filtered_member_non_proprietary.parquet \
        --min-n 100
"""

import datetime as dt
import gc
import math
import builtins
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (good for scripts / servers)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Sizes tuned for print-friendly plots
TICK_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 15
LEGEND_FONT_SIZE = 12
DEFAULT_FIGSIZE = (9, 5.5)

plt.rcParams.update({
    "font.size": TICK_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
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
    """Print to stdout and append the same message to the script log file."""
    sep = kwargs.get("sep", " ")
    message = sep.join(str(a) for a in args)
    logger.info(message)
    _original_print(*args, **kwargs)


builtins.print = log_print

# ---------------------------------------------------------------------
# Configuration (edit here to change inputs/plots/flags)
# ---------------------------------------------------------------------
PROP_PATH = Path("out_files/metaorders_info_sameday_filtered_member_proprietary.parquet")
CLIENT_PATH = Path("out_files/metaorders_info_sameday_filtered_member_non_proprietary.parquet")

ALPHA = 0.05 # significance level for confidence intervals
MIN_N = 100 # minimum number of metaorders per day to include in the analysis
SMOOTHING_DAYS = 5 # number of days to smooth the correlation
PLOT_DIR = Path("/home/danielemdn/Documents/repositories/Metaorders_PriceImpact/images/prop_vs_nonprop")

# Daily returns / imbalance-return scatter
ATTACH_DAILY_RETURNS = True  # compute and attach close-to-close daily returns per ISIN/date
PLOT_IMBALANCE_VS_RETURNS = True  # scatter plot imbalance_local vs daily returns
RETURNS_DATA_DIR = Path("/home/danielemdn/Documents/repositories/Metaorders_PriceImpact/data")
RETURNS_USE_MOT_DATA = False  # match the dataset filter used in metaorder_computation
RETURNS_TRADING_HOURS = ("09:30:00", "17:30:00")
DAILY_RETURN_COL = "Daily Return"

# Toggles for imbalance-specific analyses
ACF_IMBALANCE = True # whether to compute autocorrelation of imbalances
DISTRIBUTIONS_IMBALANCE = True # whether to plot distributions of imbalances

# Plotting parameters
ACF_MAX_LAG = 300 # maximum lag for autocorrelation
ACF_BOOTSTRAP_SAMPLES = 100 # number of bootstrap samples for autocorrelation confidence intervals
IMBALANCE_HIST_BINS = 50 # number of bins for imbalance histograms
PARTICIPATION_BINS = 100 # number of bins for participation histograms
ACF_OUTPUT_DIRNAME = "acf"  # subfolder inside PLOT_DIR for per-ISIN ACF plots

# Metaorder dictionary stats (raw metaorder indices, not the per-metaorder info parquet)
RUN_METAORDER_DICT_STATS = True
METAORDER_STATS_LEVEL = "member"
METAORDER_STATS_PROPRIETARY = False
METAORDER_STATS_PROPRIETARY_TAG = "proprietary" if METAORDER_STATS_PROPRIETARY else "non_proprietary"
METAORDER_STATS_USE_MOT_DATA = False
METAORDER_STATS_DATA_DIR = Path("/home/danielemdn/Documents/repositories/Metaorders_PriceImpact/data")
METAORDER_STATS_PARQUET_DIR = METAORDER_STATS_DATA_DIR
METAORDER_STATS_OUT_DIR = Path("out_files")
METAORDER_STATS_DICT_PATH = METAORDER_STATS_OUT_DIR / f"metaorders_dict_all_{METAORDER_STATS_LEVEL}_{METAORDER_STATS_PROPRIETARY_TAG}.pkl"
METAORDER_STATS_IMG_DIR = Path(f"images/{METAORDER_STATS_LEVEL}_{METAORDER_STATS_PROPRIETARY_TAG}")

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
# Daily returns helpers
# ---------------------------------------------------------------------
def _is_mot_name(name: str) -> bool:
    return "MOT" in name.upper()


def list_isin_parquet_paths(data_dir: Path, use_mot_data: bool = False) -> List[Path]:
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
        is_mot = _is_mot_name(name)
        if use_mot_data and not is_mot:
            continue
        if (not use_mot_data) and is_mot:
            continue
        paths.append(path)
    return sorted(paths)


def compute_daily_returns_for_path(parquet_path: Path, trading_hours: Tuple[str, str]) -> pd.Series:
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
    use_mot_data: bool = False,
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
) -> Dict[Tuple[str, dt.date], float]:
    isin_set = {str(isin) for isin in isins if pd.notna(isin)}
    if not isin_set:
        return {}

    paths = list_isin_parquet_paths(data_dir, use_mot_data=use_mot_data)
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
    out = metaorders.copy()

    if ("ISIN" not in out.columns) or ("Date" not in out.columns):
        placeholder = pd.Series(np.nan, index=out.index, dtype=float)
        existing = out[new_col] if new_col in out.columns else None
        changed = existing is None or not placeholder.equals(existing)
        out[new_col] = placeholder
        return out, changed

    def _to_date(val):
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
    """Plot a 1D PDF as a line using histogram density estimates."""
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


# ---------------------------------------------------------------------
# Metaorder dictionary statistics (durations, volumes, inter-arrivals)
# ---------------------------------------------------------------------
def list_metaorder_parquet_paths(data_dir: Path, use_mot_data: bool) -> List[Path]:
    if not data_dir.exists():
        print(f"[Metaorder stats] Parquet directory not found: {data_dir}")
        return []
    paths: List[Path] = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() != ".parquet":
            continue
        is_mot = _is_mot_name(path.name)
        if use_mot_data and not is_mot:
            continue
        if (not use_mot_data) and is_mot:
            continue
        paths.append(path)
    return paths


def load_trades_filtered_for_stats(
    path: Path,
    proprietary: Optional[bool],
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
) -> pd.DataFrame:
    """Load trades for stats, optionally filtering by proprietary flag."""
    trades = pd.read_parquet(path)
    if proprietary is True:
        trades = trades[trades["Trade Type Aggressive"] == "Dealing_on_own_account"].copy()
    elif proprietary is False:
        trades = trades[trades["Trade Type Aggressive"] != "Dealing_on_own_account"].copy()
    # proprietary=None -> no filter (full tape)
    start, end = trading_hours
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
    use_mot_data: bool,
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
) -> None:
    if not metaorders_dict_path.exists():
        print(f"[Metaorder stats] Metaorder dictionary not found: {metaorders_dict_path}")
        return

    img_dir.mkdir(parents=True, exist_ok=True)

    try:
        metaorders_dict_all = pickle.load(metaorders_dict_path.open("rb"))
    except Exception as exc:
        print(f"[Metaorder stats] Failed to load {metaorders_dict_path}: {exc}")
        return

    parquet_paths = list_metaorder_parquet_paths(parquet_dir, use_mot_data=use_mot_data)
    if not parquet_paths:
        print(f"[Metaorder stats] No parquet files found in {parquet_dir} (use_mot_data={use_mot_data}).")
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
        members, counts = zip(*sorted(counts_by_member.items(), key=lambda x: x[1], reverse=True))
        plt.figure(figsize=(12, 6.5))
        plt.bar(np.arange(len(counts)), counts)
        plt.xlabel("Member ID")
        plt.ylabel("Number of metaorders")
        plt.title("Metaorders per member")
        plt.xticks(rotation=90)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(img_dir / "metaorders_per_member_all.png")
        plt.close()

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
        plotted = plot_pdf_line(ax, data, bins=bins, color="tab:blue", logx=logx, logy=logy)
        if not plotted:
            tqdm.write(f"Skipping plot {title}: no valid numeric data")
            plt.close(fig)
            return
        mean_val = float(np.nanmean(data))
        median_val = float(np.nanmedian(data))
        # ax.axvline(mean_val, color="r", linestyle="dashed", linewidth=1, label=f"Mean: {mean_val:.2f}")
        # ax.axvline(median_val, color="g", linestyle="dashed", linewidth=1, label=f"Median: {median_val:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        fig.savefig(img_dir / filename)
        plt.close(fig)

    _pdf_plot(
        [d / 60 for d in durations],
        "Metaorder duration",
        "Minutes",
        "metaorder_duration_all.png",
        logx=True,
    )
    _pdf_plot([t / 60 for t in inter_arrivals], "Inter-arrival times", "Minutes", "interarrival_all.png", logx=True)
    _pdf_plot(meta_volumes, "Metaorder volumes", "Volume", "volumes_all.png", logx=True)
    _pdf_plot([q for q in q_over_v], "Q/V ", "Q/V", "q_over_v_all.png", logx=True)
    _pdf_plot(
        [r for r in participation_rates],
        "Participation rate",
        "Metaorder volume / volume during metaorder",
        "participation_rate_all.png",
        logx=True,
    )


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
    target_clean = target_df.drop(columns=[new_col], errors="ignore")
    return target_clean.merge(env_imb, on=list(group_cols), how="left", sort=False)


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
    fig, ax = plt.subplots(figsize=(12, 6.5))

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

    fig, ax = plt.subplots(figsize=(12, 6.5))
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
    """Save a time-series plot and PDF estimate of the daily count imbalance."""
    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    # Time-series plot
    fig, ax = plt.subplots(figsize=(12, 6.5))
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

    # PDF of imbalance distribution
    fig, ax = plt.subplots(figsize=(12, 7))
    plotted = plot_pdf_line(
        ax,
        daily_metaorders["imbalance_counts"].dropna(),
        bins="auto",
        color="tab:blue",
    )
    ax.set_title("Distribution of daily count imbalance")
    ax.set_xlabel("Imbalance")
    ax.set_ylabel("Density")
    if plotted:
        plt.tight_layout()
        hist_path = out_prefix_path.parent / f"{out_prefix_path.name}_imbalance_histogram.png"
        fig.savefig(hist_path, bbox_inches="tight")
        print(f"[Daily count plots] Saved imbalance density plot to: {hist_path}")
    else:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        plt.tight_layout()
        hist_path = out_prefix_path.parent / f"{out_prefix_path.name}_imbalance_histogram.png"
        fig.savefig(hist_path, bbox_inches="tight")
        print(f"[Daily count plots] Imbalance density plot skipped (no data).")
    plt.close(fig)


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


def compute_binned_abs_imbalance(
    df: pd.DataFrame,
    participation_col: str = "Participation Rate",
    imbalance_col: str = "imbalance_local",
    bins: int = 100,
) -> pd.DataFrame:
    """
    Return a DataFrame with mean absolute imbalance grouped by participation-rate bins.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if participation_col not in df.columns or imbalance_col not in df.columns:
        return pd.DataFrame()

    numeric = (
        df[[participation_col, imbalance_col]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if numeric.empty:
        return pd.DataFrame()

    numeric["abs_imbalance"] = numeric[imbalance_col].abs()
    pr_min = numeric[participation_col].min()
    pr_max = numeric[participation_col].max()

    if pr_min == pr_max:
        return pd.DataFrame(
            {
                "bin_center": [pr_min],
                "avg_abs_imbalance": [numeric["abs_imbalance"].mean()],
                "count": [len(numeric)],
            }
        )

    bin_edges = np.linspace(pr_min, pr_max, bins + 1)
    numeric["bin"] = pd.cut(
        numeric[participation_col],
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )
    grouped = numeric.groupby("bin", observed=False)
    agg = grouped["abs_imbalance"].agg(["mean", "size"]).reset_index()
    agg = agg.rename(columns={"mean": "avg_abs_imbalance", "size": "count"})
    agg["bin_center"] = agg["bin"].apply(
        lambda interval: (interval.left + interval.right) / 2.0 if isinstance(interval, pd.Interval) else np.nan
    )
    agg = agg.sort_values("bin_center").reset_index(drop=True)
    return agg[["bin_center", "avg_abs_imbalance", "count"]]


def plot_participation_vs_abs_imbalance(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    out_path: str | Path,
    bins: int = 100,
    participation_col: str = "Participation Rate",
    imbalance_col: str = "imbalance_local",
) -> None:
    """
    Plot mean absolute imbalance as a function of participation rate for prop vs client flow.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    curves = []
    for df, label in [
        (metaorders_proprietary, "Proprietary"),
        (metaorders_non_proprietary, "Client"),
    ]:
        summary = compute_binned_abs_imbalance(
            df,
            participation_col=participation_col,
            imbalance_col=imbalance_col,
            bins=bins,
        )
        if summary.empty:
            print(f"[Participation plot] Skipping {label.lower()} data (insufficient valid rows).")
            continue
        curves.append((summary, label))

    if not curves:
        print("\n[Participation plot] No data available to plot participation vs |imbalance|.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for summary, label in curves:
        ax.plot(
            summary["bin_center"],
            summary["avg_abs_imbalance"],
            label=label,
            linewidth=1.5,
        )

    ax.set_xlabel("Participation rate")
    ax.set_ylabel("Average |imbalance_local| per bin")
    ax.set_title("Average absolute imbalance vs participation rate")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Participation plot] Saved participation vs |imbalance| plot to: {out_path}")


def plot_imbalance_vs_daily_return(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    out_path: str | Path,
    imbalance_col: str = "imbalance_local",
    return_col: str = DAILY_RETURN_COL,
) -> None:
    """Scatter plot of imbalance vs daily returns for proprietary and client flow."""
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

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    for label, g in valid.groupby("Group"):
        ax.scatter(
            g[imbalance_col],
            g[return_col],
            s=12,
            alpha=0.5,
            label=label,
        )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Imbalance (others on same ISIN/day)")
    ax.set_ylabel("Daily log return (close-to-close)")
    ax.set_title("Imbalance vs daily returns")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Imbalance vs returns] Saved scatter plot to: {out_path}")


def plot_imbalance_distributions(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    plot_dir: str | Path,
    bins: int = 50,
) -> None:
    """Plot distributions of within-group and cross-group imbalances split by buy/sell."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    within_series = [
        (
            metaorders_proprietary.loc[metaorders_proprietary["Direction"] == 1, "imbalance_local"],
            "Prop buy",
        ),
        (
            metaorders_proprietary.loc[metaorders_proprietary["Direction"] == -1, "imbalance_local"],
            "Prop sell",
        ),
        (
            metaorders_non_proprietary.loc[metaorders_non_proprietary["Direction"] == 1, "imbalance_local"],
            "Client buy",
        ),
        (
            metaorders_non_proprietary.loc[metaorders_non_proprietary["Direction"] == -1, "imbalance_local"],
            "Client sell",
        ),
    ]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 14))
    within_ax, cross_ax = axes

    added_within = False
    for series, label in within_series:
        data = pd.to_numeric(series, errors="coerce").dropna()
        if data.empty:
            continue
        added_within = plot_pdf_line(within_ax, data, bins=bins, label=label) or added_within
    within_ax.set_title("Within-group imbalance distributions (PDF)")
    within_ax.set_xlabel("imbalance_local")
    within_ax.set_ylabel("Density")
    if added_within:
        within_ax.legend()
    else:
        within_ax.text(0.5, 0.5, "No within-group data", ha="center", va="center")

    added_cross = False
    has_cross_cols = ("imbalance_client_env" in metaorders_proprietary.columns) and (
        "imbalance_prop_env" in metaorders_non_proprietary.columns
    )
    if not has_cross_cols:
        cross_ax.text(0.5, 0.5, "Cross-group imbalance columns not found", ha="center", va="center")
        print("\n[Imbalance distribution] Cross-group imbalance columns not found; skipping cross density plot.")
    else:
        cross_series = [
            (
                metaorders_proprietary.loc[metaorders_proprietary["Direction"] == 1, "imbalance_client_env"],
                "Prop buy (client env)",
            ),
            (
                metaorders_proprietary.loc[metaorders_proprietary["Direction"] == -1, "imbalance_client_env"],
                "Prop sell (client env)",
            ),
            (
                metaorders_non_proprietary.loc[metaorders_non_proprietary["Direction"] == 1, "imbalance_prop_env"],
                "Client buy (prop env)",
            ),
            (
                metaorders_non_proprietary.loc[metaorders_non_proprietary["Direction"] == -1, "imbalance_prop_env"],
                "Client sell (prop env)",
            ),
        ]

        for series, label in cross_series:
            data = pd.to_numeric(series, errors="coerce").dropna()
            if data.empty:
                continue
            added_cross = plot_pdf_line(cross_ax, data, bins=bins, label=label) or added_cross

        cross_ax.set_title("Cross-group imbalance distributions (PDF)")
        cross_ax.set_xlabel("Environment imbalance")
        cross_ax.set_ylabel("Density")
        if added_cross:
            cross_ax.legend()
        else:
            cross_ax.text(0.5, 0.5, "No cross-group data", ha="center", va="center")

    plt.tight_layout()
    out_path = plot_dir / "imbalance_distribution.png"
    if added_within or added_cross:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"\n[Imbalance distribution] Saved combined density plot to: {out_path}")
    else:
        print("\n[Imbalance distribution] Skipped density plot (no data).")
    plt.close(fig)


def compute_autocorr_fft(series: Iterable[float], max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast ACF via FFT; returns lags and autocorrelation values up to max_lag."""
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
    """Bootstrap permutation noise band for the ACF (excludes lag 0)."""
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


def plot_direction_autocorrelation(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    out_path: str | Path,
    max_lag: int = 300,
    n_bootstrap: int = 100,
) -> None:
    """Plot autocorrelation of metaorder signs for proprietary vs client flow with noise bands."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lags_prop, acf_prop = compute_autocorr_fft(metaorders_proprietary["Direction"], max_lag)
    lags_client, acf_client = compute_autocorr_fft(metaorders_non_proprietary["Direction"], max_lag)

    if lags_prop.size == 0 and lags_client.size == 0:
        print("\n[ACF] No data available to plot autocorrelations.")
        return

    lower_prop, upper_prop = bootstrap_noise_band(metaorders_proprietary["Direction"], max_lag, n_bootstrap)
    lower_client, upper_client = bootstrap_noise_band(metaorders_non_proprietary["Direction"], max_lag, n_bootstrap)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True)
    configs = [
        ("Proprietary", lags_prop, acf_prop, lower_prop, upper_prop, axes[0]),
        ("Client", lags_client, acf_client, lower_client, upper_client, axes[1]),
    ]

    for title, lags, acf_vals, lower, upper, ax in configs:
        if lags.size <= 1:
            ax.set_visible(False)
            continue
        if lower.size and upper.size and lower.shape[0] == lags.size - 1:
            ax.fill_between(lags[1:], lower, upper, color="gray", alpha=0.3, label="95% noise band")
        ax.plot(lags[1:], acf_vals[1:], label="Empirical ACF")
        ax.set_xlim(0, max_lag)
        ax.set_xlabel("Lag")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"Autocorrelation of metaorder signs ({title.lower()})")
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend()

    axes[0].set_ylabel("Autocorrelation")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ACF] Saved metaorder sign autocorrelation plot to: {out_path}")


def plot_direction_autocorrelation_per_isin(
    metaorders_proprietary: pd.DataFrame,
    metaorders_non_proprietary: pd.DataFrame,
    out_dir: str | Path,
    max_lag: int = 300,
    n_bootstrap: int = 100,
) -> None:
    """Compute and save per-ISIN autocorrelation plots with bootstrap bands."""
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

        fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True)
        configs = [
            ("Proprietary", lags_prop, acf_prop, lower_prop, upper_prop, axes[0]),
            ("Client", lags_client, acf_client, lower_client, upper_client, axes[1]),
        ]

        for title, lags, acf_vals, lower, upper, ax in configs:
            if lags.size <= 1:
                ax.set_visible(False)
                continue
            if lower.size and upper.size and lower.shape[0] == lags.size - 1:
                ax.fill_between(lags[1:], lower, upper, color="gray", alpha=0.3, label="95% noise band")
            ax.plot(lags[1:], acf_vals[1:], label="Empirical ACF")
            ax.set_xlim(0, max_lag)
            ax.set_xlabel("Lag")
            # ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_title(f"{title} — ISIN {isin}")
            ax.grid(True, linestyle=":", linewidth=0.5)
            ax.legend()

        axes[0].set_ylabel("Autocorrelation")
        plt.tight_layout()
        out_path = out_dir / f"{isin}_acf.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def load_metaorders(path: str | Path) -> pd.DataFrame:
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
    if RUN_METAORDER_DICT_STATS:
        run_metaorder_dict_statistics(
            metaorders_dict_path=METAORDER_STATS_DICT_PATH,
            parquet_dir=METAORDER_STATS_PARQUET_DIR,
            img_dir=METAORDER_STATS_IMG_DIR,
            proprietary=METAORDER_STATS_PROPRIETARY,
            use_mot_data=METAORDER_STATS_USE_MOT_DATA,
            trading_hours=RETURNS_TRADING_HOURS,
        )

    # Load data
    metaorders_proprietary = load_metaorders(PROP_PATH)
    metaorders_non_proprietary = load_metaorders(CLIENT_PATH)
    metaorders_proprietary["Group"] = "prop"
    metaorders_non_proprietary["Group"] = "client"

    # Add within-group and cross-group imbalances, then persist if we added anything new
    prop_updated = False
    client_updated = False

    if "imbalance_local" in metaorders_proprietary.columns:
        print("\n[Prop vs Non-Prop] Daily imbalance already present in proprietary metaorders dataframe. Skipping within-group calculation.")
    else:
        metaorders_proprietary = add_daily_imbalance(metaorders_proprietary)
        prop_updated = True

    if "imbalance_local" in metaorders_non_proprietary.columns:
        print("\n[Prop vs Non-Prop] Daily imbalance already present in non-proprietary metaorders dataframe. Skipping within-group calculation.")
    else:
        metaorders_non_proprietary = add_daily_imbalance(metaorders_non_proprietary)
        client_updated = True

    prop_env_col = "imbalance_client_env"
    client_env_col = "imbalance_prop_env"

    if prop_env_col in metaorders_proprietary.columns:
        print("\n[Prop vs Non-Prop] Cross-group imbalance already present in proprietary metaorders dataframe. Skipping client-environment calculation.")
    else:
        metaorders_proprietary = attach_environment_imbalance(
            metaorders_proprietary,
            metaorders_non_proprietary,
            new_col=prop_env_col,
        )
        prop_updated = True

    if client_env_col in metaorders_non_proprietary.columns:
        print("\n[Prop vs Non-Prop] Cross-group imbalance already present in non-proprietary metaorders dataframe. Skipping proprietary-environment calculation.")
    else:
        metaorders_non_proprietary = attach_environment_imbalance(
            metaorders_non_proprietary,
            metaorders_proprietary,
            new_col=client_env_col,
        )
        client_updated = True

    if ATTACH_DAILY_RETURNS:
        all_isins: List[str] = sorted(
            set(metaorders_proprietary["ISIN"].astype(str)).union(metaorders_non_proprietary["ISIN"].astype(str))
        )
        daily_returns = build_daily_returns_lookup(
            RETURNS_DATA_DIR,
            isins=all_isins,
            use_mot_data=RETURNS_USE_MOT_DATA,
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

    participation_plot_path = PLOT_DIR / "participation_vs_abs_imbalance.png"
    plot_participation_vs_abs_imbalance(
        metaorders_proprietary,
        metaorders_non_proprietary,
        out_path=participation_plot_path,
        bins=PARTICIPATION_BINS,
    )

    if DISTRIBUTIONS_IMBALANCE:
        plot_imbalance_distributions(
            metaorders_proprietary,
            metaorders_non_proprietary,
            plot_dir=PLOT_DIR,
            bins=IMBALANCE_HIST_BINS,
        )

    if ACF_IMBALANCE:
        plot_direction_autocorrelation_per_isin(
            metaorders_proprietary,
            metaorders_non_proprietary,
            out_dir=PLOT_DIR / ACF_OUTPUT_DIRNAME,
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

    run_daily_count_imbalance_analysis(
        metaorders_proprietary,
        metaorders_non_proprietary,
        out_prefix=str(PLOT_DIR / "daily_counts"),
        make_plots=True,
    )


if __name__ == "__main__":
    main()
