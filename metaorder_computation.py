#!/usr/bin/env python
# coding: utf-8

import builtins
import logging
import os
import gc
import pickle
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

import seaborn as sns
sns.set_theme()

# Sizes tuned for paper-ready readability
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

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
LOG_PATH = Path(__file__).with_suffix(".log") if "__file__" in globals() else Path("metaorder_computation.log")
logger = logging.getLogger(Path(__file__).stem if "__file__" in globals() else "metaorder_computation")
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

# Local helpers
from utils import (
    agents_activity_sparse,
    build_trades_view,
    find_metaorders,
    map_trade_codes,
    preprocess_log_returns,
    realized_kernel_fast,
    realized_variance_fast,
    bipower_variation_fast,
    build_daily_cache,
)

sns.set_theme()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LEVEL = "member"  # "member" or "client"
PROPRIETARY = False  # True -> only proprietary trades (LEVEL must be member), False -> only non-proprietary trades (LEVEL must be member or client)
PROPRIETARY_TAG = "proprietary" if PROPRIETARY else "non_proprietary"
RECOMPUTE = True
N = None  # set to an integer to subsample ISIN files
USE_MOT_DATA = False  # False -> use files without "MOT" in the name, True -> only files containing "MOT"

PATH_DATA_FOLDER = "/home/danielemdn/Documents/repositories/Metaorders_PriceImpact/data"
PATH_NEW_DATA_FOLDER = "/home/danielemdn/Documents/repositories/Metaorders_PriceImpact/data"
OUT_DIR = "out_files"
IMG_DIR = f"images/{LEVEL}_{PROPRIETARY_TAG}"


os.makedirs(PATH_NEW_DATA_FOLDER, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(f"{IMG_DIR}/signature_plots", exist_ok=True)
warnings.filterwarnings(
    "ignore",
    message="Series.view is deprecated and will be removed in a future version",
    category=FutureWarning,
)

COLUMN_POSITIONS = {
    "Trade Time": 3,
    "ID Member": 1,
    "Total Amount Buy": 9,
    "Total Quantity Buy": 7,
    "Total Amount Sell": 10,
    "Total Quantity Sell": 8,
    "ID Client": 0,
}

# Section toggles
RUN_INTRO = True
RUN_METAORDER_COMPUTATION = False
RUN_SIGNATURE_PLOTS = False
RUN_SQL_FITS = True
RUN_WLS = True
RUN_IMPACT_PATH_PLOT = False
IMPACT_HORIZONS_MIN = (1, 3, 10, 30, 60)
SECONDS_FILTER = 120  # minimum metaorder duration (seconds)
MIN_QV = 1e-5  # minimum Q/V to keep a metaorder for fitting
COMPUTE_IMPACT_PATHS = False  # compute partial/aftermath impact vectors in the metaorders dataset
AFTERMATH_DURATION_MULTIPLIER = 2.0  # horizon multiple (x duration) for aftermath impact sampling
AFTERMATH_NUM_SAMPLES = 30  # number of evenly spaced samples in the aftermath window
MAX_GAP = pd.Timedelta(hours=1)
MIN_TRADES = 5
RESAMPLE_FREQ = "1000s"

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def pack_path(path: Optional[List[float]]) -> Optional[bytes]:
    """Convert a list of floats to packed float32 bytes (or None)."""
    if path is None:
        return None
    if len(path) == 0:
        return b""
    arr = np.asarray(path, dtype=np.float32)
    return arr.tobytes()


def unpack_path(blob: Optional[Union[bytes, bytearray, memoryview, List[float], np.ndarray]]) -> Optional[np.ndarray]:
    """
    Convert packed bytes (or legacy list/ndarray) back to a float32 numpy array.
    If the input is bytes-like, return a view using np.frombuffer.
    If the input is a list/array (older saved files), convert it with np.asarray.
    """
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return np.frombuffer(blob, dtype=np.float32)
    # Fallback for old saved data where the column might be a list or an array
    return np.asarray(blob, dtype=np.float32)


def _is_mot_name(name: str) -> bool:
    return "MOT" in name.upper()


def _dataset_filter(name: str) -> bool:
    is_mot = _is_mot_name(name)
    return is_mot if USE_MOT_DATA else not is_mot


def list_raw_csv_paths() -> List[str]:
    paths = []
    for p in os.listdir(PATH_DATA_FOLDER):
        if not p.endswith(".csv"):
            continue
        if p == "ALTRI_FTSEMIB.csv":
            continue
        if not _dataset_filter(p):
            continue
        paths.append(os.path.join(PATH_DATA_FOLDER, p))
    return sorted(paths)


def list_parquet_paths() -> List[str]:
    paths = []
    for p in os.listdir(PATH_NEW_DATA_FOLDER):
        if not p.endswith(".parquet"):
            continue
        if not _dataset_filter(p):
            continue
        paths.append(os.path.join(PATH_NEW_DATA_FOLDER, p))
    return sorted(paths)


def ensure_transforms():
    """Convert raw CSVs to parquet once for the selected dataset."""
    paths = list_raw_csv_paths()
    for path in tqdm(paths, desc="Transforming CSV->parquet", dynamic_ncols=True):
        new_path = os.path.join(
            PATH_NEW_DATA_FOLDER, f"{os.path.splitext(os.path.basename(path))[0]}.parquet"
        )
        if os.path.exists(new_path):
            continue
        tqdm.write(f"Transforming {path} -> {new_path}")
        df = pd.read_csv(path, sep=";")
        df_mapped = map_trade_codes(df)
        df_transformed = build_trades_view(df_mapped)
        df_transformed.to_parquet(new_path)


def load_trades_filtered(path: str, proprietary: bool) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if proprietary:
        df = df[df["Trade Type Aggressive"] == "Dealing_on_own_account"].copy()
    else:
        df = df[df["Trade Type Aggressive"] != "Dealing_on_own_account"].copy()
    df = df[
        (df["Trade Time"].dt.time >= pd.to_datetime("09:30:00").time())
        & (df["Trade Time"].dt.time <= pd.to_datetime("17:30:00").time())
    ].copy()
    df = df.reset_index(drop=True)
    df["__row_id__"] = np.arange(len(df), dtype=np.int64)
    df.sort_values(["Trade Time", "__row_id__"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["ISIN"] = os.path.splitext(os.path.basename(path))[0]
    return df


def _split_by_gap(
    meta_idx_list: List[int],
    times: np.ndarray,
    max_gap_ns: np.timedelta64,
) -> List[List[int]]:
    """Split a metaorder by gaps greater than max_gap_ns."""
    idx_arr = np.asarray(meta_idx_list, dtype=np.int64)
    ts = times[idx_arr]
    if ts.size < 2:
        return []
    diffs = ts[1:] - ts[:-1]
    split_idx = np.flatnonzero(diffs > max_gap_ns)
    parts = [idx_arr] if split_idx.size == 0 else np.split(idx_arr, split_idx + 1)
    return [p.tolist() for p in parts]


def compute_metaorders_per_isin(
    trades: pd.DataFrame,
    level: str,
    max_gap_ns: np.timedelta64,
    min_trades: int,
    min_duration_seconds: float,
    counts_acc: Optional[Dict[str, int]] = None,
) -> Dict[int, List[List[int]]]:
    trades_np = trades.to_numpy()
    positions = {
        "Trade Time": trades.columns.get_loc("Trade Time"),
        "ID Member": trades.columns.get_loc("ID Member"),
        "ID Client": trades.columns.get_loc("ID Client"),
        "Total Amount Buy": trades.columns.get_loc("Total Amount Buy"),
        "Total Quantity Buy": trades.columns.get_loc("Total Quantity Buy"),
        "Total Amount Sell": trades.columns.get_loc("Total Amount Sell"),
        "Total Quantity Sell": trades.columns.get_loc("Total Quantity Sell"),
    }

    indices_by_agent, act_by_agent = agents_activity_sparse(trades_np, positions, level=level)
    n_trades = len(trades_np)
    act_dense = np.zeros(n_trades, dtype=np.int8)
    times_arr = trades["Trade Time"].to_numpy()

    metaorders_dict: Dict[int, List[List[int]]] = {}
    raw_count = 0
    after_trades_count = 0
    after_duration_count = 0
    for aid in indices_by_agent.keys():
        idxs = indices_by_agent[aid]
        signs = act_by_agent[aid]
        if idxs.size == 0:
            continue
        act_dense[idxs] = signs
        _, meta_idxs, n_meta = find_metaorders(act_dense, min_child=2)
        if n_meta == 0:
            act_dense[idxs] = 0
            continue

        kept: List[List[int]] = []
        for meta_idx_list in meta_idxs:
            if len(meta_idx_list) < 2:
                continue
            t_start = pd.Timestamp(trades_np[meta_idx_list[0], positions["Trade Time"]])
            t_end = pd.Timestamp(trades_np[meta_idx_list[-1], positions["Trade Time"]])
            if t_start.date() != t_end.date():
                continue
            clients = np.unique(trades_np[meta_idx_list, positions["ID Client"]])
            if len(clients) > 1:
                continue
            segments = _split_by_gap(
                meta_idx_list,
                times_arr,
                max_gap_ns=max_gap_ns,
            )
            raw_count += len(segments)
            for segment in segments:
                if len(segment) < min_trades:
                    continue
                after_trades_count += 1
                duration_seconds = (times_arr[segment[-1]] - times_arr[segment[0]]) / np.timedelta64(1, "s")
                if duration_seconds < min_duration_seconds:
                    continue
                after_duration_count += 1
                kept.append(segment)
        if kept:
            metaorders_dict[aid] = kept
        act_dense[idxs] = 0
    if counts_acc is not None:
        counts_acc["raw"] = counts_acc.get("raw", 0) + raw_count
        counts_acc["after_trades"] = counts_acc.get("after_trades", 0) + after_trades_count
        counts_acc["after_duration"] = counts_acc.get("after_duration", 0) + after_duration_count
    return metaorders_dict


def _last_price_at_or_before(target_ns: np.int64, ts_ns: np.ndarray, prices: np.ndarray) -> float:
    """Return last traded price at or before target_ns, else NaN."""
    idx = np.searchsorted(ts_ns, target_ns, side="right") - 1
    if idx < 0 or idx >= len(prices):
        return np.nan
    return float(prices[idx])


def compute_metaorders_info(
    trades: pd.DataFrame,
    metaorders_dict: Dict[int, List[List[int]]],
    impact_horizons_min: Iterable[int] = IMPACT_HORIZONS_MIN,
    compute_paths: bool = COMPUTE_IMPACT_PATHS,
    aftermath_num_samples: int = AFTERMATH_NUM_SAMPLES,
    aftermath_duration_multiplier: float = AFTERMATH_DURATION_MULTIPLIER,
) -> List[Tuple]:
    tt: pd.Series = trades["Trade Time"]
    day_arr = tt.dt.date.values
    plc = trades["Price Last Contract"].to_numpy()
    pfc = trades["Price First Contract"].to_numpy()
    direction_arr = trades["Direction"].to_numpy()
    member_id_arr = trades["ID Member"].to_numpy()
    client_id_arr = trades["ID Client"].to_numpy()
    q_buy = trades["Total Quantity Buy"].to_numpy(dtype=float)
    q_sell = trades["Total Quantity Sell"].to_numpy(dtype=float)
    vol_arr = q_buy + q_sell
    csum_vol = np.cumsum(vol_arr)
    ts_ns = trades["Trade Time"].view("int64").to_numpy()
    price_last = trades["Price Last Contract"].to_numpy(dtype=float)
    horizon_ns = [np.int64(m * 60 * 1_000_000_000) for m in impact_horizons_min]

    daily_cache = build_daily_cache(trades, resample_freq=RESAMPLE_FREQ)

    rows: List[Tuple] = []
    for agent_id, meta_list in metaorders_dict.items():
        for idx_meta, idx_list in enumerate(meta_list):
            s = idx_list[0]
            e = idx_list[-1]
            start_ts = tt.iloc[s]
            end_ts = tt.iloc[e]
            metaorder_volume = float(vol_arr[np.asarray(idx_list, dtype=int)].sum())
            volume_during_metaorder = float(csum_vol[e] - (csum_vol[s - 1] if s > 0 else 0.0))
            direction = direction_arr[e]
            current_day = day_arr[s]
            daily_vol, daily_volume = daily_cache.get(current_day, (np.nan, 0.0))
            delta_p = float(np.log(plc[e]) - np.log(pfc[s]))
            qv = float(metaorder_volume / daily_volume) if daily_volume != 0 else np.nan
            eta = float(metaorder_volume / volume_during_metaorder) if volume_during_metaorder != 0 else np.inf
            n_child = len(idx_list)
            start_log_price = np.log(pfc[s]) if pfc[s] > 0 else np.nan
            valid_for_impacts = np.isfinite(start_log_price) and np.isfinite(daily_vol) and daily_vol != 0

            partial_impacts: Optional[List[float]] = None
            aftermath_impacts: Optional[List[float]] = None

            if compute_paths:
                partial_impacts = []
                for child_idx in idx_list:
                    p_child = price_last[child_idx]
                    if p_child > 0 and valid_for_impacts:
                        ret_child = np.log(p_child) - start_log_price
                        imp_child = ret_child * direction / daily_vol
                    else:
                        imp_child = np.nan
                    partial_impacts.append(float(imp_child) if np.isfinite(imp_child) else np.nan)

                duration_ns = ts_ns[e] - ts_ns[s]
                max_offset_ns = np.int64(max(aftermath_duration_multiplier * duration_ns, 0))
                samples = max(int(aftermath_num_samples), 0)
                aftermath_impacts = []
                if samples > 0:
                    if max_offset_ns > 0:
                        offsets = np.linspace(0, max_offset_ns, samples).astype(np.int64)
                    else:
                        offsets = np.zeros(samples, dtype=np.int64)
                    for offset in offsets:
                        target_ns = ts_ns[e] + offset
                        p_t = _last_price_at_or_before(target_ns, ts_ns, price_last)
                        if p_t > 0 and valid_for_impacts:
                            ret = np.log(p_t) - start_log_price
                            imp = ret * direction / daily_vol
                        else:
                            imp = np.nan
                        aftermath_impacts.append(float(imp) if np.isfinite(imp) else np.nan)
            impact_h_vals: List[float] = []
            for h_ns in horizon_ns:
                target_ns = ts_ns[e] + h_ns
                p_t = _last_price_at_or_before(target_ns, ts_ns, price_last)
                if p_t > 0 and valid_for_impacts:
                    ret = np.log(p_t) - start_log_price
                    imp = ret * direction / daily_vol
                else:
                    imp = np.nan
                impact_h_vals.append(float(imp) if np.isfinite(imp) else np.nan)
            partial_blob = pack_path(partial_impacts) if compute_paths else None
            aftermath_blob = pack_path(aftermath_impacts) if compute_paths else None
            rows.append(
                (
                    trades.iloc[0]["ISIN"],
                    agent_id,
                    client_id_arr[e],
                    direction,
                    delta_p,
                    daily_vol,
                    float(metaorder_volume),
                    qv,
                    eta,
                    n_child,
                    [start_ts, end_ts],
                    partial_blob,
                    aftermath_blob,
                    *impact_h_vals,
                )
            )
    return rows


def power_law(qv: np.ndarray, Y: float, gamma: float) -> np.ndarray:
    return Y * np.power(qv, gamma)


def _interpolate_impact_path(
    partial: Optional[Iterable[float]],
    aftermath: Optional[Iterable[float]],
    time_grid: np.ndarray,
    duration_multiplier: float,
) -> Optional[np.ndarray]:
    """Interpolate one metaorder impact path onto a common normalized grid."""
    times: List[float] = []
    values: List[float] = []

    if partial is not None:
        partial_arr = np.asarray(partial, dtype=float).ravel()
        if partial_arr.size > 0:
            times.extend(np.linspace(0.0, 1.0, partial_arr.size))
            values.extend(partial_arr.tolist())
    if aftermath is not None:
        aftermath_arr = np.asarray(aftermath, dtype=float).ravel()
        if aftermath_arr.size > 0:
            times.extend(1.0 + np.linspace(0.0, duration_multiplier, aftermath_arr.size))
            values.extend(aftermath_arr.tolist())

    if not times or not values:
        return None

    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    if mask.sum() < 2:
        return None

    t = t[mask]
    v = v[mask]
    order = np.argsort(t)
    t_sorted = t[order]
    v_sorted = v[order]

    # Drop duplicate timestamps to keep np.interp stable
    dedup_idx = np.concatenate(([0], np.where(np.diff(t_sorted) > 0)[0] + 1))
    t_unique = t_sorted[dedup_idx]
    v_unique = v_sorted[dedup_idx]
    if t_unique.size < 2:
        return None

    return np.interp(time_grid, t_unique, v_unique)


def fit_power_law_logbins_wls(
    subdf: pd.DataFrame,
    n_logbins: int = 30,
    min_count: int = 100,
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
    params = (Y_hat, Y_se, gamma_hat, gamma_se, R2_log, R2_lin)
    return binned, params


def power_law(x: np.ndarray, Y: float, gamma: float) -> np.ndarray:
    """Simple power-law function: y = Y * x^gamma."""
    return Y * np.power(x, gamma)


def logarithmic_impact(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Logarithmic impact function:

        I / sigma = a * log10(1 + b * Q/V)

    where `x` corresponds to Q/V.
    """
    return a * np.log10(1.0 + b * x)


def fit_power_law_logbins_wls_new(
    subdf: pd.DataFrame,
    n_logbins: int = 30,
    min_count: int = 100,
    use_median: bool = False,
    control_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Tuple[
    float, float, float, float, float, float,
    Optional[pd.Series], Optional[pd.Series]
]]:
    """
    Fit a power-law impact curve using log-binned WLS, optionally with controls.

    Parameters
    ----------
    subdf : DataFrame
        Must contain columns:
          - 'Q/V'     : participation/size proxy (phi)
          - 'Impact'  : (signed) I/sigma or similar
        And, if control_cols is not None: those columns as controls.
    n_logbins : int
        Number of logarithmic bins in Q/V.
    min_count : int
        Minimum number of metaorders per bin to keep the bin.
    use_median : bool
        If True, use the median impact per bin instead of the mean.
    control_cols : list of str or None
        Names of columns in subdf to be used as *bin-level controls*.
        For each control, the bin-level mean is added to the WLS as a regressor.
        If you want log-controls, precompute them in subdf before calling.

    Returns
    -------
    binned : DataFrame
        One row per retained bin, with:
          - center_QV, mean_imp, std_imp, sem_imp, count,
          - (and one column per control if control_cols is not None).
    params : tuple
        (Y_hat, Y_se, gamma_hat, gamma_se, R2_log, R2_lin,
         beta_controls, beta_controls_se)

        - Y_hat, gamma_hat : power-law prefactor and exponent
        - Y_se, gamma_se   : their standard errors
        - R2_log           : R^2 in log space
        - R2_lin           : R^2 in linear space
        - beta_controls    : pd.Series of control coefficients (or None)
        - beta_controls_se : pd.Series of control SEs (or None)
    """
    # 1) Filter valid rows
    sub = subdf[(subdf["Q/V"] > 0) & np.isfinite(subdf["Impact"])].copy()
    if control_cols is not None:
        for c in control_cols:
            sub = sub[np.isfinite(sub[c])]
    if sub.empty:
        raise ValueError("No valid rows (Q/V>0 and finite Impact/controls).")

    x = sub["Q/V"].to_numpy()
    y = sub["Impact"].to_numpy()

    x_min = x.min()
    x_max = x.max()
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("Invalid Q/V range for log binning.")

    # 2) Log-binning in Q/V
    edges = np.logspace(np.log10(x_min), np.log10(x_max), n_logbins + 1)
    bin_idx = np.digitize(x, edges) - 1  # bins 0..n_logbins-1

    mask = (bin_idx >= 0) & (bin_idx < n_logbins)
    sub = sub.iloc[mask].copy()
    sub["bin"] = bin_idx[mask]

    # 3) Aggregate impact stats per bin
    grp = sub.groupby("bin")
    agg_imp = grp["Impact"].agg(
        mean_imp="mean",
        median_imp="median",
        std_imp=lambda s: s.std(ddof=1),
        count="size",
    ).sort_index()

    # If controls requested, aggregate their bin means
    if control_cols is not None and len(control_cols) > 0:
        agg_ctrl = grp[control_cols].mean().sort_index()
        agg = agg_imp.join(agg_ctrl)
    else:
        agg = agg_imp

    # Choose mean or median as y-stat
    y_stat = agg["median_imp"] if use_median else agg["mean_imp"]
    y_std = agg["std_imp"].to_numpy()
    n = agg["count"].to_numpy()
    sem = y_std / np.sqrt(np.maximum(n, 1))

    bins_present = agg.index.to_numpy()
    left_edges = edges[bins_present]
    right_edges = edges[bins_present + 1]
    x_center = np.sqrt(left_edges * right_edges)

    # Build binned DataFrame
    cols = {
        "center_QV": x_center,
        "mean_imp": y_stat.to_numpy(),
        "std_imp": y_std,
        "sem_imp": sem,
        "count": n,
    }
    if control_cols is not None and len(control_cols) > 0:
        for c in control_cols:
            cols[c] = agg[c].to_numpy()

    binned = pd.DataFrame(cols).sort_values("center_QV").reset_index(drop=True)

    # 4) Filter bins
    cond = (
        (binned["count"] >= min_count)
        & np.isfinite(binned["mean_imp"])
        & np.isfinite(binned["sem_imp"])
        & (binned["sem_imp"] > 0)
        & (binned["mean_imp"] > 0)  # needed for log
    )
    if control_cols is not None:
        for c in control_cols:
            cond &= np.isfinite(binned[c])

    binned = binned[cond].reset_index(drop=True)

    if len(binned) < (2 + (len(control_cols) if control_cols else 0)):
        raise ValueError(
            f"Not enough valid bins after filtering (got {len(binned)}; "
            f"need at least {2 + (len(control_cols) if control_cols else 0)})."
        )

    # 5) Build WLS in log space
    X = np.log(binned["center_QV"].to_numpy())
    Z = np.log(binned["mean_imp"].to_numpy())

    # Var(log y) ≈ (sem / mean)^2  -> weights = 1 / var
    var_logy = (binned["sem_imp"].to_numpy() / binned["mean_imp"].to_numpy()) ** 2
    w = np.where(np.isfinite(var_logy) & (var_logy > 0), 1.0 / var_logy, 0.0)

    # Design matrix: [1, log(Q/V), controls...]
    A_cols = [np.ones_like(X), X]
    control_names = []
    if control_cols is not None and len(control_cols) > 0:
        for c in control_cols:
            A_cols.append(binned[c].to_numpy())
            control_names.append(c)

    A = np.vstack(A_cols).T  # shape (n_bins, 2 + n_controls)

    # Weighted least squares via normal equations
    sqrt_w = np.sqrt(w)
    Aw = A * sqrt_w[:, None]
    Zw = Z * sqrt_w

    coef, _, _, _ = np.linalg.lstsq(Aw, Zw, rcond=None)
    # coef[0] = a_hat, coef[1] = gamma_hat, coef[2:] = beta_controls

    a_hat = float(coef[0])
    gamma_hat = float(coef[1])
    beta_controls = None
    beta_controls_se = None

    if len(control_names) > 0:
        beta_controls = pd.Series(coef[2:], index=control_names)

    Y_hat = float(np.exp(a_hat))

    # 6) Covariance matrix and SEs
    res = Z - (A @ coef)
    RSS = np.sum(w * res**2)
    dof = max(len(Z) - A.shape[1], 1)
    s2 = RSS / dof
    XtWX = A.T @ (w[:, None] * A)
    cov = s2 * np.linalg.inv(XtWX)
    se_all = np.sqrt(np.diag(cov))

    a_se = float(se_all[0])
    gamma_se = float(se_all[1])
    Y_se = float(Y_hat * a_se)  # delta-method for exp(a_hat)

    if len(control_names) > 0:
        beta_controls_se = pd.Series(se_all[2:], index=control_names)

    # 7) R^2 in log space
    Zhat = A @ coef
    Zbar = np.average(Z, weights=w)
    R2_log = 1.0 - np.sum(w * (Z - Zhat) ** 2) / np.sum(w * (Z - Zbar) ** 2)

    # 8) R^2 in linear space (only for the power-law part)
    yhat = power_law(binned["center_QV"].to_numpy(), Y_hat, gamma_hat)
    ybar = np.mean(binned["mean_imp"].to_numpy())
    R2_lin = 1.0 - np.sum((binned["mean_imp"].to_numpy() - yhat) ** 2) / np.sum(
        (binned["mean_imp"].to_numpy() - ybar) ** 2
    )

    params = (Y_hat, Y_se, gamma_hat, gamma_se, R2_log, R2_lin, beta_controls, beta_controls_se)

    return binned, params


def fit_logarithmic_from_binned(
    binned: pd.DataFrame,
) -> Tuple[float, float, float, float, float]:
    """
    Fit a logarithmic impact curve on already log-binned data via WLS:

        mean_imp_k ≈ a * log10(1 + b * center_QV_k)

    using bin-level standard errors as weights.

    Returns
    -------
    (a_hat, a_se, b_hat, b_se, R2_lin)
    """
    if binned.empty:
        raise ValueError("No bins available for logarithmic fit.")

    x = binned["center_QV"].to_numpy()
    y = binned["mean_imp"].to_numpy()
    sem = binned["sem_imp"].to_numpy()

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite values encountered in binned data for logarithmic fit.")

    sigma = sem
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Non-positive or non-finite SEM values encountered for logarithmic fit.")

    # Initial guesses: b0 > 0, a0 chosen to roughly match the largest point
    x_pos = x[x > 0]
    y_pos = y[y > 0]
    if x_pos.size >= 1 and y_pos.size >= 1:
        b0 = 1.0
        try:
            a0 = float(y_pos.max() / np.log10(1.0 + b0 * x_pos.max()))
        except ZeroDivisionError:
            a0 = float(y_pos.mean())
    else:
        a0, b0 = 1.0, 1.0

    def _log_model(xx, a, b):
        return logarithmic_impact(xx, a, b)

    try:
        popt, pcov = curve_fit(
            _log_model,
            x,
            y,
            p0=(a0, b0),
            sigma=sigma,
            absolute_sigma=True,
            maxfev=20000,
            bounds=(
                (0.0, 0.0),   # a >= 0, b >= 0
                (np.inf, np.inf),
            ),
        )
    except Exception as e:
        raise ValueError(f"Nonlinear logarithmic fit failed: {e}") from e

    a_hat = float(popt[0])
    b_hat = float(popt[1])

    if pcov is None or not np.all(np.isfinite(pcov)):
        a_se = float("nan")
        b_se = float("nan")
    else:
        perr = np.sqrt(np.diag(pcov))
        a_se = float(perr[0])
        b_se = float(perr[1])

    # Goodness-of-fit in linear space
    yhat = _log_model(x, a_hat, b_hat)
    ybar = np.mean(y)
    denom = np.sum((y - ybar) ** 2)
    if denom <= 0:
        R2_lin = float("nan")
    else:
        R2_lin = 1.0 - np.sum((y - yhat) ** 2) / denom

    return a_hat, a_se, b_hat, b_se, R2_lin


def plot_fit(
    ax,
    binned: pd.DataFrame,
    params,
    label_prefix=None,
    label_size: int = 16,
    legend_size: int = 14,
    log_params: Optional[Tuple[float, float, float, float, float]] = None,
):
    Y, Y_err, gamma, gamma_err, R2_log, R2_lin, beta_controls, beta_controls_se = params
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

    # Power-law fit line
    ax.plot(
        x_grid,
        power_law(x_grid, Y, gamma),
        label=(
            rf'{" " if label_prefix is None else label_prefix + ": "}'
            rf"$I/\sigma = ({Y:.3g}\pm{Y_err:.2g})(Q/V)^{{{gamma:.3f}\pm{gamma_err:.3f}}}$"
        ),
    )

    # Optional logarithmic fit overlay: I/sigma = a * log10(1 + b * Q/V)
    if log_params is not None:
        a_hat, a_se, b_hat, b_se, _ = log_params
        ax.plot(
            x_grid,
            logarithmic_impact(x_grid, a_hat, b_hat),
            label=(
                rf'{" " if label_prefix is None else label_prefix + ": "}'
                rf"$I/\sigma = ({a_hat:.3g}\pm{a_se:.2g})\log_{{10}}(1 + ({b_hat:.3g}\pm{b_se:.2g})\,Q/V)$"
            ),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Q/V", fontsize=label_size)
    ax.set_ylabel(r"I / $\sigma$", fontsize=label_size)
    ax.tick_params(axis="both", which="both", labelsize=label_size)
    ax.legend(loc="best", fontsize=legend_size)


def plot_normalized_impact_path(
    df: pd.DataFrame,
    out_path: str,
    duration_multiplier: float = AFTERMATH_DURATION_MULTIPLIER,
    n_grid: int = 300,
):
    """
    Build and plot the average normalized impact path (execution + aftermath).
    t=1 marks the end of the metaorder, aftermath spans up to (1 + duration_multiplier).
    """
    grid = np.linspace(0.0, 1.0 + duration_multiplier, n_grid)
    paths = []
    iterator = tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Normalized impact paths",
        dynamic_ncols=True,
    )
    for row in iterator:
        partial_raw = getattr(row, "partial_impact", None)
        aftermath_raw = getattr(row, "aftermath_impact", None)

        partial = unpack_path(partial_raw)
        aftermath = unpack_path(aftermath_raw)

        path = _interpolate_impact_path(partial, aftermath, grid, duration_multiplier)
        if path is not None:
            paths.append(path)

    if not paths:
        tqdm.write("No valid impact paths to plot.")
        return

    arr = np.vstack(paths)
    mean_path = np.nanmean(arr, axis=0)
    p25 = np.nanpercentile(arr, 25, axis=0)
    p75 = np.nanpercentile(arr, 75, axis=0)

    plt.figure(figsize=(12, 7))
    plt.plot(grid, mean_path, label="Mean impact path", color="black")
    # plt.fill_between(grid, p25, p75, color="gray", alpha=0.25, label="IQR")
    plt.axvline(1.0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Normalized time")
    plt.ylabel("Impact")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    tqdm.write(f"Saved normalized impact path plot to {out_path}")


# ---------------------------------------------------------------------------
# Intro / transforms
# ---------------------------------------------------------------------------
if RUN_INTRO:
    print("[Intro] Ensuring parquet transforms are up to date...")
    print(
        "[Intro] Parameters — "
        f"LEVEL={LEVEL}, PROPRIETARY={PROPRIETARY}, USE_MOT_DATA={USE_MOT_DATA}, "
        f"IMPACT_HORIZONS_MIN={IMPACT_HORIZONS_MIN}, RECOMPUTE={RECOMPUTE}, N={N}, "
        f"PATH_DATA_FOLDER={PATH_DATA_FOLDER}, PATH_NEW_DATA_FOLDER={PATH_NEW_DATA_FOLDER}"
    )
    ensure_transforms()
    dfs_path_new = list_parquet_paths()
    dataset_label = "MOT" if USE_MOT_DATA else "non-MOT"
    print(f"[Intro] Parquet files available ({dataset_label}): {len(dfs_path_new)}")


# ---------------------------------------------------------------------------
# Metaorder computation (aggregated across ISIN)
# ---------------------------------------------------------------------------
if RUN_METAORDER_COMPUTATION:
    dataset_label = "MOT" if USE_MOT_DATA else "non-MOT"
    print(f"[Metaorders] Computing metaorders across all ISINs ({dataset_label})...")
    dfs_path_new = list_parquet_paths()
    if N is not None:
        dfs_path_new = dfs_path_new[:N]

    filtered_path = os.path.join(OUT_DIR, f"metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl")

    if os.path.exists(filtered_path) and not RECOMPUTE:
        print(f"Loading {filtered_path}")
        metaorders_dict_all = pickle.load(open(filtered_path, "rb"))
    else:
        try:
            max_gap_np = MAX_GAP.to_numpy()
        except AttributeError:
            max_gap_np = np.timedelta64(int(MAX_GAP.value), "ns")

        filtered_dict: Dict[str, Dict[int, List[List[int]]]] = {}
        metaorder_counts = {"raw": 0, "after_trades": 0, "after_duration": 0}
        for path in tqdm(dfs_path_new, desc="Metaorders per ISIN (filtered)", dynamic_ncols=True):
            isin = os.path.splitext(os.path.basename(path))[0]
            trades = load_trades_filtered(path, PROPRIETARY)
            filtered_dict[isin] = compute_metaorders_per_isin(
                trades,
                LEVEL,
                max_gap_ns=max_gap_np,
                min_trades=MIN_TRADES,
                min_duration_seconds=SECONDS_FILTER,
                counts_acc=metaorder_counts,
            )
            del trades
            gc.collect()
        metaorders_dict_all = filtered_dict
        pickle.dump(metaorders_dict_all, open(filtered_path, "wb"))
        tqdm.write(f"Saved {filtered_path}")
        print(
            "[Metaorders][ALL] raw (gap<{MAX_GAP}): "
            f"{metaorder_counts['raw']} -> after min trades ({MIN_TRADES}): "
            f"{metaorder_counts['after_trades']} -> after duration ({SECONDS_FILTER}s): "
            f"{metaorder_counts['after_duration']}"
        )


# ---------------------------------------------------------------------------
# Signature plots (aggregate)
# ---------------------------------------------------------------------------
if RUN_SIGNATURE_PLOTS:
    print("[Signature] Generating volatility signature plots per ISIN...")
    dfs_path_new = list_parquet_paths()
    if N is not None:
        dfs_path_new = dfs_path_new[:N]
    for path in tqdm(dfs_path_new, desc="Signature plots", dynamic_ncols=True):
        isin = os.path.splitext(os.path.basename(path))[0]
        trades = load_trades_filtered(path, PROPRIETARY)
        prices = trades[["Trade Time", "Price Last Contract"]]
        intervals_sec = list(range(1, 2000, 20))
        mean_rv, se_rv = [], []
        mean_bpv, se_bpv = [], []
        mean_rk, se_rk = [], []
        for sec in intervals_sec:
            delta = f"{sec}s"
            log_returns = preprocess_log_returns(prices.copy(), delta)
            rvs = np.array(realized_variance_fast(log_returns))
            bpvs = np.array(bipower_variation_fast(log_returns))
            rks = np.array(realized_kernel_fast(log_returns))

            def _mean_se(arr):
                valid = ~np.isnan(arr)
                if not np.any(valid):
                    return np.nan, np.nan
                arr = arr[valid]
                return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1) / np.sqrt(len(arr)))

            m, s = _mean_se(rvs); mean_rv.append(m); se_rv.append(s)
            m, s = _mean_se(bpvs); mean_bpv.append(m); se_bpv.append(s)
            m, s = _mean_se(rks); mean_rk.append(m); se_rk.append(s)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
        axs[0].errorbar(intervals_sec, mean_rv, yerr=2 * np.array(se_rv), fmt="o", label="RV +/- 2SE")
        axs[0].set_title("Realized Variance")
        axs[0].set_xlabel("Delta (sec)")
        axs[0].grid(True, ls="--")
        axs[0].legend()

        axs[1].errorbar(intervals_sec, mean_bpv, yerr=2 * np.array(se_bpv), fmt="o", label="BPV +/- 2SE")
        axs[1].set_title("Bipower Variation")
        axs[1].set_xlabel("Delta (sec)")
        axs[1].grid(True, ls="--")
        axs[1].legend()

        axs[2].errorbar(intervals_sec, mean_rk, yerr=2 * np.array(se_rk), fmt="o", label="RK +/- 2SE")
        axs[2].set_title("Realized Kernel")
        axs[2].set_xlabel("Delta (sec)")
        axs[2].grid(True, ls="--")
        axs[2].legend()

        plt.suptitle(f"Volatility signature plot ({isin})")
        plt.tight_layout()
        plt.savefig(os.path.join(f"{IMG_DIR}/signature_plots", f"signature_plot_{isin}.png"))
        plt.close()


# ---------------------------------------------------------------------------
# SQL Fits: build metaorders_info aggregated
# ---------------------------------------------------------------------------
if RUN_SQL_FITS:
    print("[SQL Fits] Building metaorders info dataframe...")
    dfs_path_new = list_parquet_paths()
    metaorders_dict_all = pickle.load(open(os.path.join(OUT_DIR, f"metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl"), "rb"))
    metaorders_info_records: List[Tuple] = []
    impact_cols = [f"Impact_{m}m" for m in IMPACT_HORIZONS_MIN]
    for path in tqdm(dfs_path_new, desc="Building metaorders info", dynamic_ncols=True):
        isin = os.path.splitext(os.path.basename(path))[0]
        trades = load_trades_filtered(path, PROPRIETARY)
        metaorders_dict = metaorders_dict_all.get(isin, {})
        metaorders_info_records.extend(compute_metaorders_info(trades, metaorders_dict))
        del trades
        gc.collect()
    metaorders_info_df_sameday = pd.DataFrame(
        metaorders_info_records,
        columns=(
            "ISIN",
            "Member",
            "Client",
            "Direction",
            "Price Change",
            "Daily Vol",
            "Q",
            "Q/V",
            "Participation Rate",
            "N Child",
            "Period",
            "partial_impact",
            "aftermath_impact",
            *impact_cols,
        ),
    )
    info_path_filtered = os.path.join(OUT_DIR, f"metaorders_info_sameday_filtered_{LEVEL}_{PROPRIETARY_TAG}.parquet")
    info_path_unfiltered = os.path.join(OUT_DIR, f"metaorders_info_sameday_{LEVEL}_{PROPRIETARY_TAG}.parquet")

    print("[SQL Fits] Saving unfiltered metaorders info dataframe...")
    metaorders_info_df_sameday.to_parquet(info_path_unfiltered, index=False)
    tqdm.write(f"Saved {info_path_unfiltered}")

    print(f"[SQL Fits] Applying filters (Q/V > {MIN_QV}) and computing Impact...")
    metaorders_info_df_sameday_filtered = metaorders_info_df_sameday[
        metaorders_info_df_sameday["Q/V"] > MIN_QV
    ].copy()
    metaorders_info_df_sameday_filtered["Impact"] = (
        metaorders_info_df_sameday_filtered["Price Change"]
        * metaorders_info_df_sameday_filtered["Direction"]
        / metaorders_info_df_sameday_filtered["Daily Vol"]
    )
    numeric_cols = [c for c in ["Q/V", "Impact", "Participation Rate", "Price Change", "Daily Vol", "Q"] if c in metaorders_info_df_sameday_filtered.columns]
    if numeric_cols:
        metaorders_info_df_sameday_filtered[numeric_cols] = metaorders_info_df_sameday_filtered[numeric_cols].replace([np.inf, -np.inf], np.nan)
    metaorders_info_df_sameday_filtered = metaorders_info_df_sameday_filtered.dropna(subset=["Q/V", "Impact"]).reset_index(drop=True)
    print(
        f"[SQL Fits] Metaorders before Q/V filter: {len(metaorders_info_df_sameday)} | "
        f"after Q/V filter: {len(metaorders_info_df_sameday_filtered)}"
    )
    metaorders_info_df_sameday_filtered.to_parquet(info_path_filtered, index=False)
    tqdm.write(f"Saved {info_path_filtered}")


# ---------------------------------------------------------------------------
# WLS fits (aggregated)
# ---------------------------------------------------------------------------
if RUN_WLS:
    print("[WLS] Running weighted least-squares impact fits...")
    info_path_filtered = os.path.join(OUT_DIR, f"metaorders_info_sameday_filtered_{LEVEL}_{PROPRIETARY_TAG}.parquet")
    info_path_unfiltered = os.path.join(OUT_DIR, f"metaorders_info_sameday_{LEVEL}_{PROPRIETARY_TAG}.parquet")
    if os.path.exists(info_path_filtered):
        df = pd.read_parquet(info_path_filtered)
    elif os.path.exists(info_path_unfiltered):
        print("[WLS] Filtered parquet missing; applying unified filters to unfiltered file...")
        raw_df = pd.read_parquet(info_path_unfiltered)
        df = raw_df[raw_df["Q/V"] > MIN_QV].copy()
        df["Impact"] = df["Price Change"] * df["Direction"] / df["Daily Vol"]
        numeric_cols = [c for c in ["Q/V", "Impact", "Participation Rate", "Price Change", "Daily Vol", "Q"] if c in df.columns]
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["Q/V", "Impact"]).reset_index(drop=True)
        print(
            f"[WLS] Metaorders before Q/V filter: {len(raw_df)} | "
            f"after Q/V filter: {len(df)}"
        )
        df.to_parquet(info_path_filtered, index=False)
        print(f"[WLS] Saved filtered parquet to {info_path_filtered}")
    else:
        raise FileNotFoundError(
            "Missing metaorders info parquet: run the SQL fits section to generate the filtered dataset."
        )

    if RUN_IMPACT_PATH_PLOT:
        out_path = os.path.join(IMG_DIR, f"normalized_impact_path_{LEVEL}_{PROPRIETARY_TAG}.png")
        plot_normalized_impact_path(df, out_path, duration_multiplier=AFTERMATH_DURATION_MULTIPLIER)

    LABEL_SIZE = 16
    LEGEND_SIZE = 14
    TITLE_SIZE = 18
    plt.rcParams.update(
        {
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )
    n_logbins = 30
    min_count = 20
    # binned_all, params_all = fit_power_law_logbins_wls(df, n_logbins=n_logbins, min_count=min_count, use_median=False)
    binned_all, params_all = fit_power_law_logbins_wls_new(df, n_logbins=n_logbins, min_count=min_count, use_median=False, control_cols=None)
    Y_hat, Y_se, gamma_hat, gamma_se, R2_log, R2_lin, beta_controls, beta_controls_se = params_all
    log_params_all = fit_logarithmic_from_binned(binned_all)
    a_hat, a_se, b_hat, b_se, R2_lin_log = log_params_all
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_fit(ax, binned_all, params_all, log_params=log_params_all)
    ax.set_title("Impact fits (power-law and logarithmic, aggregated)", fontsize=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"power_law_fit_overall_{LEVEL}.png"), dpi=300)
    plt.close()
    print("--- Overall (All) ---")
    print(f"Power law: Y = {Y_hat:.6g} +/- {Y_se:.3g}")
    print(f"Power law: gamma = {gamma_hat:.6f} +/- {gamma_se:.3g}")
    print(f"Power law: R^2_log = {R2_log:.4f} | R^2_lin = {R2_lin:.4f}")
    print(f"Logarithmic: a = {a_hat:.6g} +/- {a_se:.3g} | b = {b_hat:.6g} +/- {b_se:.3g} | R^2_lin = {R2_lin_log:.4f}")
    print(f"Bins used: {len(binned_all)} (min_count >= {min_count})")

    PR_CANDIDATES = "Participation Rate"
    pr_nbins = 2
    labels = [f"Q{j+1}" for j in range(pr_nbins)]
    df["PR_bin"] = pd.qcut(df[PR_CANDIDATES], q=pr_nbins, labels=labels, duplicates="drop")
    fig, ax = plt.subplots(figsize=(12, 8))
    fits_by_pr = {}
    for label in df["PR_bin"].dropna().unique():
        sub = df[df["PR_bin"] == label]
        try:
            binned_sub, params_sub = fit_power_law_logbins_wls_new(
                sub, n_logbins=n_logbins, min_count=min_count, use_median=False, control_cols=None
            )
        except Exception as e:
            print(f"[{label}] skipped: {e}")
            continue
        # Only power-law fits when conditioning on participation rate
        plot_fit(ax, binned_sub, params_sub, label_prefix=f"PR {label}")
        fits_by_pr[str(label)] = params_sub
    ax.set_title("Power-law fits conditioned on participation rate (aggregated)", fontsize=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"power_law_fits_by_participation_rate_{LEVEL}.png"), dpi=300)
    plt.close()
    print("--- Conditioned on Participation Rate (power-law only) ---")
    for k, power_params in fits_by_pr.items():
        Y, Y_se, gamma, gamma_se, R2_log, R2_lin, beta_controls, beta_controls_se = power_params
        print(f"[PR {k}] Y = {Y:.6g} +/- {Y_se:.3g} | gamma = {gamma:.6f} +/- {gamma_se:.3g} |")
        print(f"[PR {k}] R^2_log = {R2_log:.4f} | R^2_lin = {R2_lin:.4f}")
        print(f"[PR {k}] Beta controls: {beta_controls}")
        print(f"[PR {k}] Beta controls SE: {beta_controls_se}")
