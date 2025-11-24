#!/usr/bin/env python
# coding: utf-8

import os
import gc
import pickle
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

import seaborn as sns

sns.set_theme()

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
RUN_INTRO = False
RUN_METAORDER_COMPUTATION = False
RUN_SOME_STATISTICS_ABOUT_METAORDERS = False
RUN_SIGNATURE_PLOTS = False
RUN_SQL_FITS = True
RUN_WLS = True
IMPACT_HORIZONS_MIN = (1, 3, 10, 30, 60)
SECONDS_FILTER = 60  # minimum metaorder duration (seconds)
MIN_QV = 1e-5  # minimum Q/V to keep a metaorder for fitting

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
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
    for path in list_raw_csv_paths():
        new_path = os.path.join(
            PATH_NEW_DATA_FOLDER, f"{os.path.splitext(os.path.basename(path))[0]}.parquet"
        )
        if os.path.exists(new_path):
            continue
        print(f"Transforming {path} -> {new_path}")
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


def compute_metaorders_per_isin(trades: pd.DataFrame, level: str) -> Dict[int, List[List[int]]]:
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

    metaorders_dict: Dict[int, List[List[int]]] = {}
    for aid in tqdm(indices_by_agent.keys(), desc="Agents"):
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
            kept.append(meta_idx_list)
        if kept:
            metaorders_dict[aid] = kept
        act_dense[idxs] = 0
    return metaorders_dict


def split_metaorders_by_gap(metaorders: Dict[int, List[List[int]]], times: np.ndarray, max_gap_ns: np.int64, min_trades: int) -> Dict[int, List[List[int]]]:
    new_dict: Dict[int, List[List[int]]] = {}
    for member_id, meta_list in metaorders.items():
        split_list: List[List[int]] = []
        for meta in meta_list:
            if len(meta) < 2:
                split_list.append(meta)
                continue
            idx_arr = np.asarray(meta, dtype=np.int64)
            ts = times[idx_arr]
            diffs = ts[1:] - ts[:-1]
            split_idx = np.flatnonzero(diffs > max_gap_ns)
            if split_idx.size == 0:
                parts = [idx_arr]
            else:
                parts = np.split(idx_arr, split_idx + 1)
            for part in parts:
                if part.size >= min_trades:
                    split_list.append(part.tolist())
        new_dict[member_id] = split_list
    return new_dict


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

    daily_cache = build_daily_cache(trades)

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
            impact_h_vals: List[float] = []
            for h_ns in horizon_ns:
                target_ns = ts_ns[e] + h_ns
                p_t = _last_price_at_or_before(target_ns, ts_ns, price_last)
                if (
                    p_t > 0
                    and np.isfinite(start_log_price)
                    and np.isfinite(daily_vol)
                    and daily_vol != 0
                ):
                    ret = np.log(p_t) - start_log_price
                    imp = ret * direction / daily_vol
                else:
                    imp = np.nan
                impact_h_vals.append(float(imp) if np.isfinite(imp) else np.nan)
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
                    *impact_h_vals,
                )
            )
    return rows


def _period_duration_seconds(period) -> float:
    """Return duration in seconds for a (start, end) tuple; NaN on malformed input."""
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


def apply_metaorder_filters(
    df: pd.DataFrame,
    seconds_filter: float = SECONDS_FILTER,
    min_qv: float = MIN_QV,
) -> pd.DataFrame:
    """
    Apply all metaorder-level filters once, after metaorders are computed.
    - Drop metaorders shorter than `seconds_filter`.
    - Require Q/V > `min_qv`.
    - Compute Impact = Price Change * Direction / Daily Vol.
    - Drop rows with non-finite key fields.
    """
    if "Period" not in df.columns:
        raise KeyError("Expected 'Period' column to compute durations for filtering.")
    out = df.copy()
    out["DurationSeconds"] = out["Period"].apply(_period_duration_seconds)
    out = out[out["DurationSeconds"] >= seconds_filter].drop(columns=["DurationSeconds"])
    out = out[out["Q/V"] > min_qv]

    if {"Price Change", "Direction", "Daily Vol"} - set(out.columns):
        missing = {"Price Change", "Direction", "Daily Vol"} - set(out.columns)
        raise KeyError(f"Missing columns required to compute Impact: {missing}")
    out["Impact"] = out["Price Change"] * out["Direction"] / out["Daily Vol"]

    numeric_cols = [c for c in ["Q/V", "Impact", "Participation Rate", "Price Change", "Daily Vol", "Q"] if c in out.columns]
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    out = out.dropna(subset=["Q/V", "Impact"]).reset_index(drop=True)
    return out


def power_law(qv: np.ndarray, Y: float, gamma: float) -> np.ndarray:
    return Y * np.power(qv, gamma)


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

from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd


def power_law(x: np.ndarray, Y: float, gamma: float) -> np.ndarray:
    """Simple power-law function: y = Y * x^gamma."""
    return Y * np.power(x, gamma)


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



def plot_fit(ax, binned: pd.DataFrame, params, label_prefix=None, label_size: int = 16, legend_size: int = 14):
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

    unfiltered_path = os.path.join(OUT_DIR, f"metaorders_dict_all_nofilter_{LEVEL}_{PROPRIETARY_TAG}.pkl")
    filtered_path = os.path.join(OUT_DIR, f"metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl")

    if os.path.exists(unfiltered_path) and not RECOMPUTE:
        print(f"Loading {unfiltered_path}")
        metaorders_dict_all = pickle.load(open(unfiltered_path, "rb"))
    else:
        metaorders_dict_all: Dict[str, Dict[int, List[List[int]]]] = {}
        for i, path in enumerate(dfs_path_new):
            isin = os.path.splitext(os.path.basename(path))[0]
            print(f"({i+1}/{len(dfs_path_new)}) Computing metaorders for {isin}")
            trades = load_trades_filtered(path, PROPRIETARY)
            metaorders_dict_all[isin] = compute_metaorders_per_isin(trades, LEVEL)
            del trades
            gc.collect()
        pickle.dump(metaorders_dict_all, open(unfiltered_path, "wb"))
        print(f"Saved {unfiltered_path}")

    MAX_GAP = pd.Timedelta(hours=1)
    MIN_TRADES = 2

    if os.path.exists(filtered_path) and not RECOMPUTE:
        print(f"Loading {filtered_path}")
        metaorders_dict_all = pickle.load(open(filtered_path, "rb"))
    else:
        S_NS = np.int64(1_000_000_000)
        M_NS = np.int64(60) * S_NS
        H_NS = np.int64(60) * M_NS
        DAY_NS = np.int64(24) * H_NS
        START_NS = np.int64(9) * H_NS + np.int64(30) * M_NS
        END_NS = np.int64(17) * H_NS + np.int64(30) * M_NS
        try:
            max_gap_np = MAX_GAP.to_numpy()
        except AttributeError:
            max_gap_np = np.timedelta64(int(MAX_GAP.value), "ns")

        filtered_dict: Dict[str, Dict[int, List[List[int]]]] = {}
        for i, path in enumerate(dfs_path_new):
            isin = os.path.splitext(os.path.basename(path))[0]
            print(f"({i+1}/{len(dfs_path_new)}) Filtering metaorders for {isin}")
            trades = load_trades_filtered(path, PROPRIETARY)
            tt = trades["Trade Time"]
            if tt.dt.tz is None:
                ns_since_midnight = (tt.view("i8") % DAY_NS).astype(np.int64)
            else:
                ns_since_midnight = (
                    tt.dt.hour.astype(np.int64) * H_NS
                    + tt.dt.minute.astype(np.int64) * M_NS
                    + tt.dt.second.astype(np.int64) * S_NS
                    + tt.dt.microsecond.astype(np.int64) * np.int64(1_000)
                    + tt.dt.nanosecond.astype(np.int64)
                )
            in_hours = (ns_since_midnight >= START_NS) & (ns_since_midnight <= END_NS)
            trades = trades.loc[in_hours].copy()
            times_arr = trades["Trade Time"].to_numpy()
            unfiltered_isin = metaorders_dict_all.get(isin, {})
            filtered_dict[isin] = split_metaorders_by_gap(unfiltered_isin, times_arr, max_gap_np, MIN_TRADES)
            del trades
            gc.collect()
        metaorders_dict_all = filtered_dict
        pickle.dump(metaorders_dict_all, open(filtered_path, "wb"))
        print(f"Saved {filtered_path}")


# ---------------------------------------------------------------------------
# Aggregated statistics and plots
# ---------------------------------------------------------------------------
if RUN_SOME_STATISTICS_ABOUT_METAORDERS:
    print("[Stats] Building aggregated statistics and histograms...")
    metaorders_dict_all = pickle.load(open(os.path.join(OUT_DIR, f"metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl"), "rb"))
    dfs_path_new = list_parquet_paths()
    total_metaorders = 0
    counts_by_member: Dict[int, int] = {}
    durations: List[float] = []
    inter_arrivals: List[float] = []
    meta_volumes: List[float] = []
    q_over_v: List[float] = []
    participation_rates: List[float] = []

    for path in tqdm(dfs_path_new, desc="ISINs"):
        isin = os.path.splitext(os.path.basename(path))[0]
        trades = load_trades_filtered(path, PROPRIETARY)
        
        # Optimization: Pre-compute numpy arrays
        times = trades["Trade Time"].to_numpy()
        q_buy = trades["Total Quantity Buy"].to_numpy()
        q_sell = trades["Total Quantity Sell"].to_numpy()
        
        # Pre-calculate daily volumes
        daily_vols = trades.groupby(trades["Trade Time"].dt.date)[["Total Quantity Buy", "Total Quantity Sell"]].sum().sum(axis=1).to_dict()

        metaorders_dict = metaorders_dict_all.get(isin, {})
        total_metaorders += sum(len(v) for v in metaorders_dict.values())
        # counts
        for member, metas in metaorders_dict.items():
            counts_by_member[member] = counts_by_member.get(member, 0) + len(metas)
        # durations and inter-arrivals, volumes, q/v, participation
        for metas in metaorders_dict.values():
            for meta in metas:
                if not meta:
                    continue
                start_idx, end_idx = meta[0], meta[-1]
                
                start_time_np = times[start_idx]
                end_time_np = times[end_idx]
                
                dur_seconds = (end_time_np - start_time_np) / np.timedelta64(1, 's')
                durations.append(float(dur_seconds))
                
                meta_indices = np.array(meta)
                vols = float(q_buy[meta_indices].sum() + q_sell[meta_indices].sum())
                meta_volumes.append(vols)
                
                start_date = pd.Timestamp(start_time_np).date()
                day_volume = daily_vols.get(start_date, 0.0)
                
                if day_volume != 0:
                    q_over_v.append(float(vols / day_volume))
                
                slice_volume = float(q_buy[start_idx : end_idx + 1].sum() + q_sell[start_idx : end_idx + 1].sum())
                if slice_volume != 0:
                    participation_rates.append(float(vols / slice_volume))
                
                if len(meta) > 1:
                    meta_times = times[meta_indices]
                    diffs = (meta_times[1:] - meta_times[:-1]) / np.timedelta64(1, 's')
                    inter_arrivals.extend(diffs.tolist())
        del trades
        gc.collect()

    print(f"Total metaorders (all ISINs): {total_metaorders}")
    if counts_by_member:
        members, counts = zip(*sorted(counts_by_member.items(), key=lambda x: x[1], reverse=True))
        plt.figure(figsize=(10, 4))
        plt.bar([str(m) for m in members], counts)
        plt.xlabel("Member ID")
        plt.ylabel("Number of metaorders")
        plt.title("Metaorders per member (aggregated)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, "metaorders_per_member_all.png"))
        plt.close()

    def _hist(data: List[float], title: str, xlabel: str, filename: str, logy: bool = True):
        if not data:
            print(f"Skipping plot {title}: no data")
            return
        plt.figure(figsize=(8, 3))
        plt.hist(data, bins=50, density=True, alpha=0.6, color="b")
        plt.axvline(np.mean(data), color="r", linestyle="dashed", linewidth=1, label=f"Mean: {np.mean(data):.2f}")
        plt.axvline(np.median(data), color="g", linestyle="dashed", linewidth=1, label=f"Median: {np.median(data):.2f}")
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        if logy:
            plt.yscale("log")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, filename))
        plt.close()

    _hist([d / 60 for d in durations], "Metaorder duration (all ISINs)", "Minutes", "metaorder_duration_all.png")
    _hist([t / 60 for t in inter_arrivals], "Inter-arrival times (all ISINs)", "Minutes", "interarrival_all.png")
    _hist(meta_volumes, "Metaorder volumes (all ISINs)", "Volume", "volumes_all.png")
    _hist([100 * q for q in q_over_v], "Q/V (all ISINs)", "Q/V (%)", "q_over_v_all.png")
    _hist(
        [100 * r for r in participation_rates],
        "Participation rate (all ISINs)",
        "Metaorder volume / volume during metaorder (%)",
        "participation_rate_all.png",
    )


# ---------------------------------------------------------------------------
# Signature plots (aggregate)
# ---------------------------------------------------------------------------
if RUN_SIGNATURE_PLOTS:
    print("[Signature] Generating volatility signature plots per ISIN...")
    dfs_path_new = list_parquet_paths()
    if N is not None:
        dfs_path_new = dfs_path_new[:N]
    for path in dfs_path_new:
        isin = os.path.splitext(os.path.basename(path))[0]
        print(f"Computing signature plot for {isin}")
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

        fig, axs = plt.subplots(1, 3, figsize=(18, 4), tight_layout=True)
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
    for i, path in enumerate(dfs_path_new):
        isin = os.path.splitext(os.path.basename(path))[0]
        print(f"({i+1}/{len(dfs_path_new)}) Building metaorders_info for {isin}")
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
            *impact_cols,
        ),
    )
    info_path_filtered = os.path.join(OUT_DIR, f"metaorders_info_sameday_filtered_{LEVEL}_{PROPRIETARY_TAG}.parquet")
    info_path_unfiltered = os.path.join(OUT_DIR, f"metaorders_info_sameday_{LEVEL}_{PROPRIETARY_TAG}.parquet")

    print("[SQL Fits] Saving unfiltered metaorders info dataframe...")
    metaorders_info_df_sameday.to_parquet(info_path_unfiltered, index=False)
    print(f"Saved {info_path_unfiltered}")

    print(
        f"[SQL Fits] Applying unified filters (duration >= {SECONDS_FILTER}s, Q/V > {MIN_QV}) "
        "and computing Impact..."
    )
    metaorders_info_df_sameday_filtered = apply_metaorder_filters(
        metaorders_info_df_sameday, seconds_filter=SECONDS_FILTER, min_qv=MIN_QV
    )
    metaorders_info_df_sameday_filtered.to_parquet(info_path_filtered, index=False)
    print(f"Saved {info_path_filtered}")


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
        df = apply_metaorder_filters(raw_df, seconds_filter=SECONDS_FILTER, min_qv=MIN_QV)
        df.to_parquet(info_path_filtered, index=False)
        print(f"[WLS] Saved filtered parquet to {info_path_filtered}")
    else:
        raise FileNotFoundError(
            "Missing metaorders info parquet: run the SQL fits section to generate the filtered dataset."
        )

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
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_fit(ax, binned_all, params_all)
    ax.set_title("Power-law fit (aggregated)", fontsize=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"power_law_fit_overall_{LEVEL}.png"), dpi=300)
    plt.close()
    print("--- Overall (All) ---")
    print(f"Y = {Y_hat:.6g} +/- {Y_se:.3g}")
    print(f"gamma = {gamma_hat:.6f} +/- {gamma_se:.3g}")
    print(f"R^2_log = {R2_log:.4f} | R^2_lin = {R2_lin:.4f}")
    print(f"Bins used: {len(binned_all)} (min_count >= {min_count})")

    PR_CANDIDATES = "Participation Rate"
    pr_nbins = 2
    labels = [f"Q{j+1}" for j in range(pr_nbins)]
    df["PR_bin"] = pd.qcut(df[PR_CANDIDATES], q=pr_nbins, labels=labels, duplicates="drop")
    fig, ax = plt.subplots(figsize=(9, 6))
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
        plot_fit(ax, binned_sub, params_sub, label_prefix=f"PR {label}")
        fits_by_pr[str(label)] = params_sub
    ax.set_title("Power-law fits conditioned on participation rate (aggregated)", fontsize=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"power_law_fits_by_participation_rate_{LEVEL}.png"), dpi=300)
    plt.close()
    print("--- Conditioned on Participation Rate ---")
    for k, (Y, Y_se, gamma, gamma_se, R2_log, R2_lin, beta_controls, beta_controls_se) in fits_by_pr.items():
        print(f"[PR {k}]  Y = {Y:.6g} +/- {Y_se:.3g} | gamma = {gamma:.6f} +/- {gamma_se:.3g} |")
        print(f"Beta controls: {beta_controls}")
        print(f"Beta controls SE: {beta_controls_se}")
