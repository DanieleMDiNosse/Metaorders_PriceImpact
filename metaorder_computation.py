#!/usr/bin/env python
# coding: utf-8

import builtins
import logging
import os
import sys
import gc
import pickle
import yaml
from typing import Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path
from bisect import bisect_right

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as student_t
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D plots

import seaborn as sns
sns.set_theme()

# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def _weights_from_sigma(sigma: np.ndarray) -> np.ndarray:
    """Convert per-observation sigma (std/SEM) into WLS weights 1/sigma^2."""
    sigma = np.asarray(sigma, dtype=float)
    w = np.zeros_like(sigma, dtype=float)
    ok = np.isfinite(sigma) & (sigma > 0)
    w[ok] = 1.0 / np.square(sigma[ok])
    return w


def _weighted_r2(y: np.ndarray, yhat: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """
    Weighted coefficient of determination:
        R^2 = 1 - sum_i w_i (y_i - yhat_i)^2 / sum_i w_i (y_i - ybar_w)^2
    Falls back to unweighted when w is None.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError(f"y and yhat must have the same shape (got {y.shape} vs {yhat.shape}).")

    if w is None:
        w_arr = np.ones_like(y, dtype=float)
    else:
        w_arr = np.asarray(w, dtype=float)
        if w_arr.shape != y.shape:
            raise ValueError(f"w must have the same shape as y (got {w_arr.shape} vs {y.shape}).")

    valid = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w_arr) & (w_arr > 0)
    if np.count_nonzero(valid) < 3:
        return float("nan")

    yv = y[valid]
    yhatv = yhat[valid]
    wv = w_arr[valid]

    ybar = float(np.average(yv, weights=wv))
    denom = float(np.sum(wv * np.square(yv - ybar)))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum(wv * np.square(yv - yhatv)) / denom)


# ---------------------------------------------------------------------------
# Configuration loader (YAML)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_CONFIG_PATH = _SCRIPT_DIR / "config_ymls" / "metaorder_computation.yml"
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config file: {_CONFIG_PATH}")

_CFG = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
if not isinstance(_CFG, dict):
    raise TypeError(f"Config must be a mapping (YAML dict): {_CONFIG_PATH}")


def _cfg_require(key: str):
    if key not in _CFG:
        raise KeyError(f"Missing required key '{key}' in {_CONFIG_PATH}")
    return _CFG[key]


def _resolve_repo_path(value: Union[str, Path]) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (_SCRIPT_DIR / path).resolve()
    return str(path)


def _resolve_opt_repo_path(value: Optional[Union[str, Path]], default: Union[str, Path]) -> str:
    """Resolve a path, falling back to `default` when the config value is None."""
    if value is None:
        return _resolve_repo_path(default)
    return _resolve_repo_path(value)


# Sizes tuned for paper-ready readability (loaded from YAML)
TICK_FONT_SIZE = int(_cfg_require("TICK_FONT_SIZE"))
LABEL_FONT_SIZE = int(_cfg_require("LABEL_FONT_SIZE"))
TITLE_FONT_SIZE = int(_cfg_require("TITLE_FONT_SIZE"))
LEGEND_FONT_SIZE = int(_cfg_require("LEGEND_FONT_SIZE"))
DEFAULT_FIGSIZE = tuple(_cfg_require("DEFAULT_FIGSIZE"))

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
    # Windows consoles / default locale encodings may not support some unicode
    # symbols used in prints (e.g., η). Use a tolerant error handler so logging
    # never crashes the pipeline.
    _file_handler = logging.FileHandler(LOG_PATH, mode="a", errors="backslashreplace")
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)

_original_print = builtins.print


def log_print(*args, **kwargs):
    """Print to stdout and append the same message to the script log file."""
    sep = kwargs.get("sep", " ")
    message = sep.join(str(a) for a in args)
    try:
        logger.info(message)
    except Exception:
        # Logging should never prevent the computation from running.
        pass

    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fall back to an ASCII-safe representation for legacy encodings.
        stream = kwargs.get("file", sys.stdout)
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe = message.encode(encoding, errors="backslashreplace").decode(encoding, errors="ignore")
        _original_print(safe, file=stream, end=kwargs.get("end", "\n"), flush=kwargs.get("flush", False))


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

# ---------------------------------------------------------------------------
# Config (loaded from YAML)
# ---------------------------------------------------------------------------
LEVEL = str(_cfg_require("LEVEL"))  # "member" or "client"
PROPRIETARY = bool(_cfg_require("PROPRIETARY"))
RECOMPUTE = bool(_cfg_require("RECOMPUTE"))
TRADING_HOURS = tuple(_cfg_require("TRADING_HOURS"))

# How to choose the daily volume used in Q/V:
#   - "same_day"  : use volume on the metaorder day (default, original behavior)
#   - "prev_day"  : use volume of the previous trading day (if available)
#   - "avg_5d"    : use the average daily volume over the last up to 5 trading days
Q_V_DENOMINATOR_MODE = str(_cfg_require("Q_V_DENOMINATOR_MODE"))

# How to choose the daily volatility used in impact normalization:
#   - "same_day" : use volatility on the metaorder day (legacy behavior)
#   - "prev_day" : use volatility of the previous trading day (if available)
#   - "avg_5d"   : use the average daily volatility over the last up to 5 trading days
DAILY_VOL_MODE = str(_cfg_require("DAILY_VOL_MODE"))

DATASET_NAME = str(_CFG.get("DATASET_NAME") or "ftsemib")
DATA_ROOT = _resolve_repo_path(_CFG.get("DATA_ROOT") or "data")
PATH_DATA_FOLDER = _resolve_opt_repo_path(
    _CFG.get("PATH_DATA_FOLDER"), Path(DATA_ROOT) / DATASET_NAME
)
PATH_NEW_DATA_FOLDER = _resolve_opt_repo_path(
    _CFG.get("PATH_NEW_DATA_FOLDER"), Path(DATA_ROOT) / DATASET_NAME
)
OUT_ROOT = _resolve_repo_path(_CFG.get("OUT_ROOT") or "out_files")
OUT_DIR = _resolve_opt_repo_path(_CFG.get("OUT_DIR"), Path(OUT_ROOT) / DATASET_NAME)
PROPRIETARY_TAG = "proprietary" if PROPRIETARY else "non_proprietary"
IMG_ROOT = _resolve_repo_path(_CFG.get("IMG_ROOT") or "images")
_img_dir_override = _CFG.get("IMG_DIR")
IMG_DIR = (
    _resolve_repo_path(_img_dir_override)
    if _img_dir_override
    else _resolve_repo_path(Path(IMG_ROOT) / DATASET_NAME / f"{LEVEL}_{PROPRIETARY_TAG}")
)


os.makedirs(PATH_NEW_DATA_FOLDER, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(f"{IMG_DIR}/signature_plots", exist_ok=True)

# Section toggles
RUN_INTRO = bool(_cfg_require("RUN_INTRO"))
RUN_METAORDER_COMPUTATION = bool(_cfg_require("RUN_METAORDER_COMPUTATION"))
RUN_SIGNATURE_PLOTS = bool(_cfg_require("RUN_SIGNATURE_PLOTS"))
RUN_SQL_FITS = bool(_cfg_require("RUN_SQL_FITS"))
RUN_WLS = bool(_cfg_require("RUN_WLS"))
RUN_IMPACT_PATH_PLOT = bool(_cfg_require("RUN_IMPACT_PATH_PLOT"))
IMPACT_HORIZONS_MIN = tuple(int(x) for x in _cfg_require("IMPACT_HORIZONS_MIN"))
SECONDS_FILTER = int(_cfg_require("SECONDS_FILTER"))
MIN_QV = float(_cfg_require("MIN_QV"))
COMPUTE_IMPACT_PATHS = bool(_cfg_require("COMPUTE_IMPACT_PATHS"))
AFTERMATH_DURATION_MULTIPLIER = float(_cfg_require("AFTERMATH_DURATION_MULTIPLIER"))
AFTERMATH_NUM_SAMPLES = int(_cfg_require("AFTERMATH_NUM_SAMPLES"))
MAX_GAP = pd.Timedelta(str(_cfg_require("MAX_GAP")))
MIN_TRADES = int(_cfg_require("MIN_TRADES"))
RESAMPLE_FREQ = str(_cfg_require("RESAMPLE_FREQ"))
N_LOGBIN = int(_cfg_require("N_LOGBIN"))
MIN_COUNT = int(_cfg_require("MIN_COUNT"))
MIN_COUNT_SURFACE = int(_cfg_require("MIN_COUNT_SURFACE"))
MAX_PARTICIPATION_RATE = float(_cfg_require("MAX_PARTICIPATION_RATE"))
N_PR_BINS_SURFACE = int(_cfg_require("N_PR_BINS_SURFACE"))
N_SIGNATURE_PLOTS: Optional[int] = (
    None if _CFG.get("N_SIGNATURE_PLOTS") is None else int(_CFG["N_SIGNATURE_PLOTS"])
)

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


def list_raw_csv_paths() -> List[str]:
    paths = []
    for p in os.listdir(PATH_DATA_FOLDER):
        if not p.endswith(".csv"):
            continue
        if p == "ALTRI_FTSEMIB.csv":
            continue
        paths.append(os.path.join(PATH_DATA_FOLDER, p))
    return sorted(paths)


def list_parquet_paths() -> List[str]:
    paths = []
    for p in os.listdir(PATH_NEW_DATA_FOLDER):
        if not p.endswith(".parquet"):
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
        # if os.path.exists(new_path):
        #     continue
        tqdm.write(f"Transforming {path} -> {new_path}")
        df = pd.read_csv(path, sep=";")
        if len(df.columns) == 1:
            df = pd.read_csv(path)
        df_mapped = map_trade_codes(df)
        df_transformed = build_trades_view(df_mapped)
        df_transformed.to_parquet(new_path)


def load_trades_base(path: str, trading_hours: Tuple[str, str] = TRADING_HOURS) -> pd.DataFrame:
    """Load one ISIN parquet, apply trading-hours filter, sort, and tag ISIN."""
    df = pd.read_parquet(path)
    start, end = trading_hours
    df = df[
        (df["Trade Time"].dt.time >= pd.to_datetime(start).time())
        & (df["Trade Time"].dt.time <= pd.to_datetime(end).time())
    ].copy()
    df = df.reset_index(drop=True)
    df["__row_id__"] = np.arange(len(df), dtype=np.int64)
    df.sort_values(["Trade Time", "__row_id__"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["ISIN"] = os.path.splitext(os.path.basename(path))[0]
    return df


def filter_trades_by_group(trades: pd.DataFrame, proprietary: bool) -> pd.DataFrame:
    """Return a view filtered to prop or non-prop trades, preserving order."""
    if proprietary:
        mask = trades["Trade Type Aggressive"] == "Dealing_on_own_account"
    else:
        mask = trades["Trade Type Aggressive"] != "Dealing_on_own_account"
    out = trades.loc[mask].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def load_trades_full(path: str, trading_hours: Tuple[str, str] = TRADING_HOURS) -> pd.DataFrame:
    """Load full (unfiltered) tape for the ISIN within trading hours."""
    return load_trades_base(path, trading_hours=trading_hours)


def load_trades_filtered(path: str, proprietary: bool, trading_hours: Tuple[str, str] = TRADING_HOURS) -> pd.DataFrame:
    """Load filtered tape for metaorder detection (prop vs non-prop)."""
    trades = load_trades_base(path, trading_hours=trading_hours)
    return filter_trades_by_group(trades, proprietary)


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


def _volume_over_window(
    ts_ns: np.ndarray,
    csum_vol: np.ndarray,
    start_ns: np.int64,
    end_ns: np.int64,
) -> float:
    """
    Sum volume between [start_ns, end_ns] using cumulative volume array built on
    the *full* tape.
    """
    start_idx = np.searchsorted(ts_ns, start_ns, side="left")
    end_idx = np.searchsorted(ts_ns, end_ns, side="right") - 1
    if end_idx < start_idx or end_idx < 0 or start_idx >= ts_ns.size:
        return 0.0
    prev = csum_vol[start_idx - 1] if start_idx > 0 else 0.0
    return float(csum_vol[end_idx] - prev)


def _select_daily_metric(
    daily_cache: Dict[pd.Timestamp, Tuple[float, float]],
    daily_cache_days: List[pd.Timestamp],
    current_day,
    mode: str,
    value_index: int,
) -> float:
    """
    Pick a daily metric (volatility or volume) from the cache according to `mode`.
    value_index: 0 -> volatility, 1 -> volume.
    """
    default_entry = (np.nan, 0.0)

    if mode == "same_day":
        return daily_cache.get(current_day, default_entry)[value_index]

    if not daily_cache_days:
        return np.nan

    # Position of the last available trading day not after current_day
    idx = bisect_right(daily_cache_days, current_day) - 1
    if idx < 0:
        return np.nan

    if mode == "prev_day":
        if idx == 0:
            return np.nan
        ref_day = daily_cache_days[idx - 1]
        return daily_cache.get(ref_day, default_entry)[value_index]

    if mode == "avg_5d":
        start_idx = max(0, idx - 4)
        window_days = daily_cache_days[start_idx : idx + 1]
        vals = [daily_cache.get(d, default_entry)[value_index] for d in window_days]
        return float(np.mean(vals)) if len(vals) > 0 else np.nan

    # Fallback to same-day metric if mode is unrecognized
    return daily_cache.get(current_day, default_entry)[value_index]


def compute_metaorders_info(
    trades: pd.DataFrame,
    metaorders_dict: Dict[int, List[List[int]]],
    trades_full: pd.DataFrame,
    impact_horizons_min: Iterable[int] = IMPACT_HORIZONS_MIN,
    compute_paths: bool = COMPUTE_IMPACT_PATHS,
    aftermath_num_samples: int = AFTERMATH_NUM_SAMPLES,
    aftermath_duration_multiplier: float = AFTERMATH_DURATION_MULTIPLIER,
) -> List[Tuple]:
    if trades_full is None:
        raise ValueError("trades_full must be provided (full tape needed for denominators and prices).")
    if not metaorders_dict or trades.empty:
        return []
    full_trades = trades_full
    isin = trades["ISIN"].iat[0] if "ISIN" in trades.columns else ""

    tt_meta: pd.Series = trades["Trade Time"]
    day_arr = tt_meta.dt.date.values
    plc = trades["Price Last Contract"].to_numpy(dtype=float)
    pfc = trades["Price First Contract"].to_numpy(dtype=float)
    direction_arr = trades["Direction"].to_numpy()
    client_id_arr = trades["ID Client"].to_numpy()
    vol_arr = trades[["Total Quantity Buy", "Total Quantity Sell"]].to_numpy(dtype=float).sum(axis=1)
    ts_meta_ns = trades["Trade Time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)

    ts_full_ns = full_trades["Trade Time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    price_last_full = full_trades["Price Last Contract"].to_numpy(dtype=float)
    full_vol_arr = (
        full_trades[["Total Quantity Buy", "Total Quantity Sell"]].to_numpy(dtype=float).sum(axis=1)
    )
    csum_vol_full = np.cumsum(full_vol_arr)

    horizon_ns = [np.int64(m * 60 * 1_000_000_000) for m in impact_horizons_min]

    daily_cache = build_daily_cache(full_trades, resample_freq=RESAMPLE_FREQ)
    # Sorted list of days available in the cache (trading days)
    daily_cache_days = sorted(daily_cache.keys())

    rows: List[Tuple] = []
    for agent_id, meta_list in metaorders_dict.items():
        for idx_list in meta_list:
            s = idx_list[0]
            e = idx_list[-1]
            start_ts = tt_meta.iloc[s]
            end_ts = tt_meta.iloc[e]
            metaorder_volume = float(vol_arr[np.asarray(idx_list, dtype=int)].sum())
            start_ns = np.int64(pd.Timestamp(start_ts).value)
            end_ns = np.int64(pd.Timestamp(end_ts).value)
            volume_during_metaorder = _volume_over_window(ts_full_ns, csum_vol_full, start_ns, end_ns)
            direction = direction_arr[e]
            current_day = day_arr[s]
            # Select daily volatility for impact normalization and volume for Q/V
            daily_volatility = _select_daily_metric(
                daily_cache, daily_cache_days, current_day, DAILY_VOL_MODE, value_index=0
            )
            denom_volume = _select_daily_metric(
                daily_cache, daily_cache_days, current_day, Q_V_DENOMINATOR_MODE, value_index=1
            )

            delta_p = float(np.log(plc[e]) - np.log(pfc[s]))
            if denom_volume is not None and np.isfinite(denom_volume) and denom_volume > 0:
                qv = float(metaorder_volume / denom_volume)
                flow_ratio = float(volume_during_metaorder / denom_volume)
            else:
                qv = np.nan
                flow_ratio = np.nan
            eta = float(metaorder_volume / volume_during_metaorder) if volume_during_metaorder != 0 else np.inf
            n_child = len(idx_list)
            start_log_price = np.log(pfc[s]) if pfc[s] > 0 else np.nan
            valid_for_impacts = (
                np.isfinite(start_log_price) and np.isfinite(daily_volatility) and daily_volatility != 0
            )

            partial_impacts: Optional[List[float]] = None
            aftermath_impacts: Optional[List[float]] = None

            if compute_paths:
                partial_impacts = []
                for child_idx in idx_list:
                    p_child = plc[child_idx]
                    if p_child > 0 and valid_for_impacts:
                        ret_child = np.log(p_child) - start_log_price
                        imp_child = ret_child * direction / daily_volatility
                    else:
                        imp_child = np.nan
                    partial_impacts.append(float(imp_child) if np.isfinite(imp_child) else np.nan)

                duration_ns = ts_meta_ns[e] - ts_meta_ns[s]
                max_offset_ns = np.int64(max(aftermath_duration_multiplier * duration_ns, 0))
                samples = max(int(aftermath_num_samples), 0)
                aftermath_impacts = []
                if samples > 0:
                    if max_offset_ns > 0:
                        offsets = np.linspace(0, max_offset_ns, samples).astype(np.int64)
                    else:
                        offsets = np.zeros(samples, dtype=np.int64)
                    for offset in offsets:
                        target_ns = ts_meta_ns[e] + offset
                        p_t = _last_price_at_or_before(target_ns, ts_full_ns, price_last_full)
                        if p_t > 0 and valid_for_impacts:
                            ret = np.log(p_t) - start_log_price
                            imp = ret * direction / daily_volatility
                        else:
                            imp = np.nan
                        aftermath_impacts.append(float(imp) if np.isfinite(imp) else np.nan)
            impact_h_vals: List[float] = []
            for h_ns in horizon_ns:
                target_ns = ts_meta_ns[e] + h_ns
                p_t = _last_price_at_or_before(target_ns, ts_full_ns, price_last_full)
                if p_t > 0 and valid_for_impacts:
                    ret = np.log(p_t) - start_log_price
                    imp = ret * direction / daily_volatility
                else:
                    imp = np.nan
                impact_h_vals.append(float(imp) if np.isfinite(imp) else np.nan)
            partial_blob = pack_path(partial_impacts) if compute_paths else None
            aftermath_blob = pack_path(aftermath_impacts) if compute_paths else None
            rows.append(
                (
                    isin,
                    agent_id,
                    client_id_arr[e],
                    direction,
                    delta_p,
                    daily_volatility,
                    float(metaorder_volume),
                    qv,
                    eta,
                    flow_ratio,
                    n_child,
                    # Store period endpoints as epoch-nanoseconds (Python ints) so fastparquet can
                    # JSON-encode the list deterministically (Timestamp / numpy scalar are not JSON-serializable).
                    [int(start_ns), int(end_ns)],
                    partial_blob,
                    aftermath_blob,
                    *impact_h_vals,
                )
            )
    return rows


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


def _prepare_impact_surface_bins(
    df: pd.DataFrame,
    n_qv_bins: int,
    n_pr_bins: int,
    qv_col: str,
    impact_col: str,
    pr_col: str,
    *,
    context: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shared preprocessing for 2D impact surfaces:
    - validate required columns
    - filter invalid rows
    - compute log-spaced bin edges and bin indices

    Returns (qv_edges, pr_edges, qv_idx, pr_idx, impact_values).
    """
    required_cols = {qv_col, impact_col, pr_col}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise ValueError(f"Missing required columns for surface computation: {missing}")

    sub = df[[qv_col, impact_col, pr_col]].copy()
    sub[qv_col] = pd.to_numeric(sub[qv_col], errors="coerce")
    sub[impact_col] = pd.to_numeric(sub[impact_col], errors="coerce")
    sub[pr_col] = pd.to_numeric(sub[pr_col], errors="coerce")
    sub = sub[
        (sub[qv_col] > MIN_QV)
        & np.isfinite(sub[impact_col])
        & np.isfinite(sub[pr_col])
        & (sub[pr_col] > 0)
        & (sub[pr_col] < MAX_PARTICIPATION_RATE)
    ]
    if sub.empty:
        raise ValueError(
            f"No valid rows ({qv_col}>{MIN_QV}, finite Impact and Participation Rate)."
        )

    qv = sub[qv_col].to_numpy()
    pr = sub[pr_col].to_numpy()
    imp = sub[impact_col].to_numpy()

    qv_min = qv.min()
    qv_max = qv.max()
    if not np.isfinite(qv_min) or not np.isfinite(qv_max) or qv_max <= qv_min:
        raise ValueError(f"Invalid {qv_col} range for log binning ({context}).")

    pr_min = pr.min()
    pr_max = pr.max()
    if not np.isfinite(pr_min) or not np.isfinite(pr_max) or pr_max <= pr_min:
        raise ValueError(f"Invalid Participation Rate range for log binning ({context}).")

    qv_edges = np.logspace(np.log10(qv_min), np.log10(qv_max), n_qv_bins + 1)
    pr_edges = np.logspace(np.log10(pr_min), np.log10(pr_max), n_pr_bins + 1)

    qv_idx = np.digitize(qv, qv_edges) - 1
    pr_idx = np.digitize(pr, pr_edges) - 1

    mask = (
        (qv_idx >= 0) & (qv_idx < n_qv_bins) &
        (pr_idx >= 0) & (pr_idx < n_pr_bins)
    )
    return qv_edges, pr_edges, qv_idx[mask], pr_idx[mask], imp[mask]


def compute_impact_surface(
    df: pd.DataFrame,
    n_qv_bins: int,
    n_pr_bins: int,
    min_count: int,
    qv_col: str = "Q/V",
    impact_col: str = "Impact",
    pr_col: str = "Participation Rate",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a 2D surface of mean impact as a function of a volume ratio (`qv_col`) and participation rate.

    Both axes are log-binned.

    Returns
    -------
    qv_edges : np.ndarray
        Bin edges for the volume ratio (log-spaced).
    pr_edges : np.ndarray
        Bin edges for participation rate (log-spaced).
    impact_grid : np.ndarray
        Array of shape (n_pr_bins, n_qv_bins) with mean impact per 2D bin
        (NaN where the bin has fewer than `min_count` observations).
    count_grid : np.ndarray
        Array of shape (n_pr_bins, n_qv_bins) with counts per 2D bin.
    """
    qv_edges, pr_edges, qv_idx, pr_idx, imp = _prepare_impact_surface_bins(
        df,
        n_qv_bins=n_qv_bins,
        n_pr_bins=n_pr_bins,
        qv_col=qv_col,
        impact_col=impact_col,
        pr_col=pr_col,
        context="surface",
    )

    df_bins = pd.DataFrame({"qv_bin": qv_idx, "pr_bin": pr_idx, "Impact": imp})
    agg = (
        df_bins.groupby(["qv_bin", "pr_bin"])["Impact"]
        .agg(["mean", "count"])
        .reset_index()
    )

    impact_grid = np.full((n_pr_bins, n_qv_bins), np.nan, dtype=float)
    count_grid = np.zeros((n_pr_bins, n_qv_bins), dtype=int)

    for row in agg.itertuples(index=False):
        q_bin = int(row.qv_bin)
        p_bin = int(row.pr_bin)
        cnt = int(row.count)
        if cnt >= min_count:
            impact_grid[p_bin, q_bin] = float(row.mean)
            count_grid[p_bin, q_bin] = cnt

    return qv_edges, pr_edges, impact_grid, count_grid


def compute_impact_surface_stats(
    df: pd.DataFrame,
    n_qv_bins: int,
    n_pr_bins: int,
    min_count: int,
    qv_col: str = "Q/V",
    impact_col: str = "Impact",
    pr_col: str = "Participation Rate",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D log-binned statistics of impact vs (qv_col, participation rate).

    Returns
    -------
    qv_edges, pr_edges : np.ndarray
        Log-spaced bin edges for qv_col and participation rate.
    mean_grid : np.ndarray
        Mean impact per 2D bin (NaN where count < min_count).
    sem_grid : np.ndarray
        Standard error of the mean per 2D bin (NaN where count < min_count).
    count_grid : np.ndarray
        Raw counts per 2D bin (filled for all populated bins).
    """
    qv_edges, pr_edges, qv_idx, pr_idx, imp = _prepare_impact_surface_bins(
        df,
        n_qv_bins=n_qv_bins,
        n_pr_bins=n_pr_bins,
        qv_col=qv_col,
        impact_col=impact_col,
        pr_col=pr_col,
        context="surface stats",
    )

    df_bins = pd.DataFrame({"qv_bin": qv_idx, "pr_bin": pr_idx, "Impact": imp})
    agg = (
        df_bins.groupby(["qv_bin", "pr_bin"])["Impact"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    mean_grid = np.full((n_pr_bins, n_qv_bins), np.nan, dtype=float)
    sem_grid = np.full((n_pr_bins, n_qv_bins), np.nan, dtype=float)
    count_grid = np.zeros((n_pr_bins, n_qv_bins), dtype=int)

    for row in agg.itertuples(index=False):
        q_bin = int(row.qv_bin)
        p_bin = int(row.pr_bin)
        cnt = int(row.count)
        count_grid[p_bin, q_bin] = cnt
        if cnt < min_count:
            continue
        mean = float(row.mean)
        std = float(row.std) if np.isfinite(row.std) else float("nan")
        sem = std / np.sqrt(cnt) if np.isfinite(std) and cnt > 0 else float("nan")
        mean_grid[p_bin, q_bin] = mean
        sem_grid[p_bin, q_bin] = sem

    return qv_edges, pr_edges, mean_grid, sem_grid, count_grid


def fit_bivariate_power_law_eta_f_wls(
    df: pd.DataFrame,
    n_qv_bins: int,
    n_pr_bins: int,
    min_count: int,
    qv_col: str = "Vt/V",
    pr_col: str = "Participation Rate",
    impact_col: str = "Impact",
    ci_level: float = 0.95,
    surface_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fit the bivariate power law via 2D log-binned WLS:

        I/sigma = C * eta^delta * F^gamma,  where F = Vt/V

    Following the same spirit as the 1D WLS fit: estimate bin means and SEM,
    fit in log space, and weight by 1/Var(log(mean)) ≈ 1/(SEM/mean)^2.

    Returns
    -------
    binned : DataFrame
        One row per retained 2D bin with centers, mean/SEM impact, count, weights.
    params : dict
        Keys: C_hat, C_se, C_ci, delta_hat, delta_se, delta_ci, gamma_hat, gamma_se,
        gamma_ci, R2_log, R2_lin, dof, ci_level.
        Here R2_log is weighted in log space; R2_lin is weighted in linear space (1/SEM^2).
    """
    if surface_stats is None:
        qv_edges, pr_edges, mean_grid, sem_grid, count_grid = compute_impact_surface_stats(
            df,
            n_qv_bins=n_qv_bins,
            n_pr_bins=n_pr_bins,
            min_count=min_count,
            qv_col=qv_col,
            impact_col=impact_col,
            pr_col=pr_col,
        )
    else:
        qv_edges, pr_edges, mean_grid, sem_grid, count_grid = surface_stats

    qv_centers = np.sqrt(qv_edges[:-1] * qv_edges[1:])
    pr_centers = np.sqrt(pr_edges[:-1] * pr_edges[1:])

    rows = []
    for p in range(n_pr_bins):
        for q in range(n_qv_bins):
            cnt = int(count_grid[p, q])
            if cnt < min_count:
                continue
            mean = float(mean_grid[p, q])
            sem = float(sem_grid[p, q])
            if not (np.isfinite(mean) and np.isfinite(sem) and mean > 0 and sem > 0):
                continue
            rows.append(
                {
                    "center_eta": float(pr_centers[p]),
                    "center_QV": float(qv_centers[q]),
                    "mean_imp": mean,
                    "sem_imp": sem,
                    "count": cnt,
                }
            )
    binned = pd.DataFrame(rows)
    if len(binned) < 5:
        raise ValueError(f"Not enough valid 2D bins for bivariate WLS (got {len(binned)}).")

    log_eta = np.log(binned["center_eta"].to_numpy())
    log_qv = np.log(binned["center_QV"].to_numpy())
    log_y = np.log(binned["mean_imp"].to_numpy())

    var_logy = (binned["sem_imp"].to_numpy() / binned["mean_imp"].to_numpy()) ** 2
    w = np.where(np.isfinite(var_logy) & (var_logy > 0), 1.0 / var_logy, 0.0)
    if np.count_nonzero(w > 0) < 5:
        raise ValueError("Not enough positive weights for bivariate WLS.")

    A = np.vstack([np.ones_like(log_y), log_eta, log_qv]).T
    sqrt_w = np.sqrt(w)
    Aw = A * sqrt_w[:, None]
    Zw = log_y * sqrt_w

    coef, _, _, _ = np.linalg.lstsq(Aw, Zw, rcond=None)
    a_hat = float(coef[0])
    delta_hat = float(coef[1])
    gamma_hat = float(coef[2])
    C_hat = float(np.exp(a_hat))

    res = log_y - (A @ coef)
    RSS = float(np.sum(w * res**2))
    dof = max(len(log_y) - A.shape[1], 1)
    s2 = RSS / dof
    XtWX = A.T @ (w[:, None] * A)
    try:
        cov = s2 * np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = s2 * np.linalg.pinv(XtWX)
    se_all = np.sqrt(np.diag(cov))

    a_se = float(se_all[0])
    delta_se = float(se_all[1])
    gamma_se = float(se_all[2])
    C_se = float(C_hat * a_se)

    if not (0 < ci_level < 1):
        raise ValueError("ci_level must be in (0, 1).")
    tcrit = float(student_t.ppf(0.5 + ci_level / 2.0, dof)) if dof > 0 else 1.96
    a_ci = tcrit * a_se
    delta_ci = tcrit * delta_se
    gamma_ci = tcrit * gamma_se
    C_ci_low = float(np.exp(a_hat - a_ci))
    C_ci_high = float(np.exp(a_hat + a_ci))
    C_ci = float((C_ci_high - C_ci_low) / 2.0)

    log_y_hat = A @ coef
    R2_log = _weighted_r2(log_y, log_y_hat, w=w)

    y_hat = C_hat * np.power(binned["center_eta"].to_numpy(), delta_hat) * np.power(binned["center_QV"].to_numpy(), gamma_hat)
    y = binned["mean_imp"].to_numpy()
    w_lin = _weights_from_sigma(binned["sem_imp"].to_numpy())
    R2_lin = _weighted_r2(y, y_hat, w=w_lin)

    binned["w"] = w
    params = {
        "C_hat": C_hat,
        "C_se": C_se,
        "C_ci": C_ci,
        "C_ci_low": C_ci_low,
        "C_ci_high": C_ci_high,
        "delta_hat": delta_hat,
        "delta_se": delta_se,
        "delta_ci": float(delta_ci),
        "gamma_hat": gamma_hat,
        "gamma_se": gamma_se,
        "gamma_ci": float(gamma_ci),
        "R2_log": R2_log,
        "R2_lin": R2_lin,
        "dof": float(dof),
        "ci_level": float(ci_level),
    }
    return binned, params


def bivariate_logarithmic_impact(
    eta: np.ndarray, flow: np.ndarray, a: float, b: float, c: float
) -> np.ndarray:
    """
    Bivariate logarithmic impact surface:

        I / sigma = a * log10(1 + b * eta) * log10(1 + c * F)
        where F = Vt/V (market volume during the metaorder over daily volume)
    """
    return a * np.log10(1.0 + b * eta) * np.log10(1.0 + c * flow)


def fit_bivariate_logarithmic_eta_f_wls(
    df: pd.DataFrame,
    n_qv_bins: int,
    n_pr_bins: int,
    min_count: int,
    qv_col: str = "Vt/V",
    pr_col: str = "Participation Rate",
    impact_col: str = "Impact",
    ci_level: float = 0.95,
    surface_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fit the bivariate logarithmic surface on 2D log-binned means via SEM-weighted NLS:

        I/sigma = a * log10(1 + b * eta) * log10(1 + c * F),  where F = Vt/V

    Following the same spirit as the univariate logarithmic fit: work on bin means and
    use their SEM as observation uncertainty (sigma) in weighted nonlinear least squares.

    Returns
    -------
    binned : DataFrame
        One row per retained 2D bin with centers, mean/SEM impact, count.
    params : dict
        Keys: a_hat, a_se, a_ci, b_hat, b_se, b_ci, c_hat, c_se, c_ci,
        R2_log, R2_lin, dof, ci_level.
        Here R2_lin is weighted in linear space (1/SEM^2).
    """
    if surface_stats is None:
        qv_edges, pr_edges, mean_grid, sem_grid, count_grid = compute_impact_surface_stats(
            df,
            n_qv_bins=n_qv_bins,
            n_pr_bins=n_pr_bins,
            min_count=min_count,
            qv_col=qv_col,
            impact_col=impact_col,
            pr_col=pr_col,
        )
    else:
        qv_edges, pr_edges, mean_grid, sem_grid, count_grid = surface_stats

    qv_centers = np.sqrt(qv_edges[:-1] * qv_edges[1:])
    pr_centers = np.sqrt(pr_edges[:-1] * pr_edges[1:])

    rows = []
    for p in range(n_pr_bins):
        for q in range(n_qv_bins):
            cnt = int(count_grid[p, q])
            if cnt < min_count:
                continue
            mean = float(mean_grid[p, q])
            sem = float(sem_grid[p, q])
            if not (np.isfinite(mean) and np.isfinite(sem) and mean > 0 and sem > 0):
                continue
            rows.append(
                {
                    "center_eta": float(pr_centers[p]),
                    "center_QV": float(qv_centers[q]),
                    "mean_imp": mean,
                    "sem_imp": sem,
                    "count": cnt,
                }
            )
    binned = pd.DataFrame(rows)
    if len(binned) < 5:
        raise ValueError(f"Not enough valid 2D bins for bivariate logarithmic fit (got {len(binned)}).")

    eta = binned["center_eta"].to_numpy()
    qv = binned["center_QV"].to_numpy()
    y = binned["mean_imp"].to_numpy()
    sigma = binned["sem_imp"].to_numpy()

    if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(qv)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite values encountered in binned data for bivariate logarithmic fit.")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Non-positive or non-finite SEM values encountered for bivariate logarithmic fit.")

    # Initial guesses: scale b and c by inverse of typical ranges for numerical stability
    eta_max = float(np.max(eta))
    qv_max = float(np.max(qv))
    b0 = 1.0 / eta_max if eta_max > 0 else 1.0
    c0 = 1.0 / qv_max if qv_max > 0 else 1.0
    denom0 = np.log10(1.0 + b0 * eta) * np.log10(1.0 + c0 * qv)
    mask0 = np.isfinite(denom0) & (denom0 > 0)
    if np.any(mask0):
        a0 = float(np.nanmedian(y[mask0] / denom0[mask0]))
    else:
        a0 = float(np.nanmean(y))
    if not np.isfinite(a0) or a0 <= 0:
        a0 = 1.0

    def _model(X, a, b, c):
        eta_x, qv_x = X
        return bivariate_logarithmic_impact(eta_x, qv_x, a, b, c)

    try:
        popt, pcov = curve_fit(
            _model,
            (eta, qv),
            y,
            p0=(a0, b0, c0),
            sigma=sigma,
            absolute_sigma=True,
            maxfev=50000,
            bounds=((0.0, 0.0, 0.0), (np.inf, np.inf, np.inf)),
        )
    except Exception as e:
        raise ValueError(f"Bivariate logarithmic fit failed: {e}") from e

    a_hat = float(popt[0])
    b_hat = float(popt[1])
    c_hat = float(popt[2])

    if pcov is None or not np.all(np.isfinite(pcov)):
        a_se = float("nan")
        b_se = float("nan")
        c_se = float("nan")
    else:
        perr = np.sqrt(np.diag(pcov))
        a_se = float(perr[0])
        b_se = float(perr[1])
        c_se = float(perr[2])

    dof = max(len(y) - 3, 1)
    if not (0 < ci_level < 1):
        raise ValueError("ci_level must be in (0, 1).")
    tcrit = float(student_t.ppf(0.5 + ci_level / 2.0, dof)) if dof > 0 else 1.96
    a_ci = float(tcrit * a_se) if np.isfinite(a_se) else float("nan")
    b_ci = float(tcrit * b_se) if np.isfinite(b_se) else float("nan")
    c_ci = float(tcrit * c_se) if np.isfinite(c_se) else float("nan")

    yhat = _model((eta, qv), a_hat, b_hat, c_hat)

    valid = np.isfinite(yhat) & (yhat > 0) & np.isfinite(y) & (y > 0)
    if np.count_nonzero(valid) < 3:
        R2_log = float("nan")
    else:
        Z = np.log(y[valid])
        Zhat = np.log(yhat[valid])
        var_logy = (sigma[valid] / y[valid]) ** 2
        w = np.where(np.isfinite(var_logy) & (var_logy > 0), 1.0 / var_logy, 0.0)
        if np.count_nonzero(w > 0) < 3:
            R2_log = float("nan")
        else:
            R2_log = _weighted_r2(Z, Zhat, w=w)

    w_lin = _weights_from_sigma(sigma)
    R2_lin = _weighted_r2(y, yhat, w=w_lin)

    params = {
        "a_hat": a_hat,
        "a_se": a_se,
        "a_ci": a_ci,
        "b_hat": b_hat,
        "b_se": b_se,
        "b_ci": b_ci,
        "c_hat": c_hat,
        "c_se": c_se,
        "c_ci": c_ci,
        "R2_log": R2_log,
        "R2_lin": R2_lin,
        "dof": float(dof),
        "ci_level": float(ci_level),
    }
    return binned, params


def plot_bivariate_fit_surfaces_3d(
    df: pd.DataFrame,
    out_prefix: str,
    n_qv_bins: int,
    n_pr_bins: int,
    min_count: int,
    qv_col: str = "Vt/V",
) -> None:
    """
    Plot two fitted surfaces (power-law and logarithmic, using F = Vt/V) in the same 3D plot,
    and overlay the empirical mean surface with low alpha.
    """
    try:
        qv_edges, pr_edges, mean_grid, sem_grid, count_grid = compute_impact_surface_stats(
            df,
            n_qv_bins=n_qv_bins,
            n_pr_bins=n_pr_bins,
            min_count=min_count,
            qv_col=qv_col,
        )
    except Exception as e:
        print(f"[Bivariate Fits] Skipping bivariate surfaces: {e}")
        return

    surface_stats = (qv_edges, pr_edges, mean_grid, sem_grid, count_grid)

    params_pow = None
    try:
        _, params_pow = fit_bivariate_power_law_eta_f_wls(
            df,
            n_qv_bins=n_qv_bins,
            n_pr_bins=n_pr_bins,
            min_count=min_count,
            qv_col=qv_col,
            surface_stats=surface_stats,
        )
    except Exception as e:
        print(f"[Bivariate Fits] Power-law fit skipped: {e}")

    params_log = None
    try:
        _, params_log = fit_bivariate_logarithmic_eta_f_wls(
            df,
            n_qv_bins=n_qv_bins,
            n_pr_bins=n_pr_bins,
            min_count=min_count,
            qv_col=qv_col,
            surface_stats=surface_stats,
        )
    except Exception as e:
        print(f"[Bivariate Fits] Logarithmic fit skipped: {e}")

    if params_pow is None and params_log is None:
        print("[Bivariate Fits] No bivariate fit succeeded; skipping plot.")
        return

    if params_pow is not None:
        ci_pct = int(round(100.0 * float(params_pow["ci_level"])))
        print("[Bivariate Fits] Power-law: I/sigma = C * eta^delta * F^gamma")
        print(
            f"[Bivariate Fits]   C = {params_pow['C_hat']:.6g} +/- {params_pow['C_ci']:.3g} ({ci_pct}% CI) | "
            f"delta = {params_pow['delta_hat']:.6f} +/- {params_pow['delta_ci']:.3g} | "
            f"gamma = {params_pow['gamma_hat']:.6f} +/- {params_pow['gamma_ci']:.3g}"
        )
        print(
            f"[Bivariate Fits]   R^2_log = {params_pow['R2_log']:.4f} | R^2_lin = {params_pow['R2_lin']:.4f}"
        )

    if params_log is not None:
        ci_pct = int(round(100.0 * float(params_log["ci_level"])))
        print("[Bivariate Fits] Logarithmic: I/sigma = a*log10(1+b*eta)*log10(1+c*F)")
        print(
            f"[Bivariate Fits]   a = {params_log['a_hat']:.6g} +/- {params_log['a_ci']:.3g} ({ci_pct}% CI) | "
            f"b = {params_log['b_hat']:.6g} +/- {params_log['b_ci']:.3g} | "
            f"c = {params_log['c_hat']:.6g} +/- {params_log['c_ci']:.3g}"
        )
        print(
            f"[Bivariate Fits]   R^2_lin = {params_log['R2_lin']:.4f}"
        )

    qv_centers = np.sqrt(qv_edges[:-1] * qv_edges[1:])
    pr_centers = np.sqrt(pr_edges[:-1] * pr_edges[1:])
    QV_grid, PR_grid = np.meshgrid(qv_centers, pr_centers)

    domain_mask = (count_grid >= min_count) & np.isfinite(mean_grid) & (mean_grid > 0)
    empirical_for_plot = np.full_like(mean_grid, np.nan, dtype=float)
    empirical_for_plot[domain_mask] = mean_grid[domain_mask]

    if np.count_nonzero(np.isfinite(empirical_for_plot) & (empirical_for_plot > 0)) < 3:
        print("[Bivariate Fits] Not enough populated positive bins to plot bivariate surfaces.")
        return

    pow_for_plot = None
    if params_pow is not None:
        C_hat = float(params_pow["C_hat"])
        delta_hat = float(params_pow["delta_hat"])
        gamma_hat = float(params_pow["gamma_hat"])
        pow_grid = C_hat * np.power(PR_grid, delta_hat) * np.power(QV_grid, gamma_hat)
        pow_for_plot = np.full_like(pow_grid, np.nan, dtype=float)
        pow_mask = domain_mask & np.isfinite(pow_grid) & (pow_grid > 0)
        pow_for_plot[pow_mask] = pow_grid[pow_mask]

    log_for_plot = None
    if params_log is not None:
        a_hat = float(params_log["a_hat"])
        b_hat = float(params_log["b_hat"])
        c_hat = float(params_log["c_hat"])
        log_grid = bivariate_logarithmic_impact(PR_grid, QV_grid, a_hat, b_hat, c_hat)
        log_for_plot = np.full_like(log_grid, np.nan, dtype=float)
        log_mask = domain_mask & np.isfinite(log_grid) & (log_grid > 0)
        log_for_plot[log_mask] = log_grid[log_mask]

    # Residual grids
    pow_residual = None
    if pow_for_plot is not None:
        pow_residual = np.full_like(empirical_for_plot, np.nan, dtype=float)
        pow_residual[domain_mask] = empirical_for_plot[domain_mask] - pow_for_plot[domain_mask]

    log_residual = None
    if log_for_plot is not None:
        log_residual = np.full_like(empirical_for_plot, np.nan, dtype=float)
        log_residual[domain_mask] = empirical_for_plot[domain_mask] - log_for_plot[domain_mask]

    qv_edges_log = np.log10(qv_edges)
    pr_edges_log = np.log10(pr_edges)

    def _sym_vmax(arr: Optional[np.ndarray]) -> Optional[float]:
        if arr is None:
            return None
        finite = np.isfinite(arr)
        if not np.any(finite):
            return None
        vmax = float(np.nanmax(np.abs(arr[finite])))
        return vmax if vmax > 0 else None

    vmax_pow = _sym_vmax(pow_residual)
    vmax_log = _sym_vmax(log_residual)

    summary_lines = []
    if params_pow is not None:
        ci_pct = int(round(100.0 * float(params_pow["ci_level"])))
        summary_lines.append(
            f"Power: C={params_pow['C_hat']:.3g}±{params_pow['C_ci']:.2g}, "
            f"δ={params_pow['delta_hat']:.3f}±{params_pow['delta_ci']:.2g}, "
            f"γ={params_pow['gamma_hat']:.3f}±{params_pow['gamma_ci']:.2g} ({ci_pct}% CI), "
            f"R²={params_pow['R2_log']:.3f}"
        )
    if params_log is not None:
        ci_pct = int(round(100.0 * float(params_log["ci_level"])))
        summary_lines.append(
            f"Log: a={params_log['a_hat']:.3g}±{params_log['a_ci']:.2g}, "
            f"b={params_log['b_hat']:.3g}±{params_log['b_ci']:.2g}, "
            f"c={params_log['c_hat']:.3g}±{params_log['c_ci']:.2g} ({ci_pct}% CI), "
            f"R²={params_log['R2_lin']:.3f}"
        )

    # Interactive HTML 3D plot with both fits, empirical surface, and stacked residuals (no PNG saved)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig_html = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "scene", "rowspan": 2}, {"type": "heatmap"}], [None, {"type": "heatmap"}]],
            column_widths=[0.65, 0.35],
            horizontal_spacing=0.07,
            vertical_spacing=0.08,
        )

        # Empirical surface (grey, low alpha)
        fig_html.add_trace(
            go.Surface(
                x=QV_grid,
                y=PR_grid,
                z=empirical_for_plot,
                colorscale=[[0.0, "rgba(140,140,140,1.0)"], [1.0, "rgba(140,140,140,1.0)"]],
                opacity=0.25,
                showscale=False,
                name="Empirical mean surface",
            ),
            row=1,
            col=1,
        )

        if pow_for_plot is not None:
            fig_html.add_trace(
                go.Surface(
                    x=QV_grid,
                    y=PR_grid,
                    z=pow_for_plot,
                    colorscale=[[0.0, "rgba(31,119,180,1.0)"], [1.0, "rgba(31,119,180,1.0)"]],
                    opacity=0.65,
                    showscale=False,
                    name="Power-law fit surface",
                ),
                row=1,
                col=1,
            )

        if log_for_plot is not None:
            fig_html.add_trace(
                go.Surface(
                    x=QV_grid,
                    y=PR_grid,
                    z=log_for_plot,
                    colorscale=[[0.0, "rgba(255,127,14,1.0)"], [1.0, "rgba(255,127,14,1.0)"]],
                    opacity=0.65,
                    showscale=False,
                    name="Logarithmic fit surface",
                ),
                row=1,
                col=1,
            )

        # Power-law residuals (top-right)
        if pow_residual is not None and vmax_pow is not None:
            fig_html.add_trace(
                go.Heatmap(
                    x=qv_edges,
                    y=pr_edges,
                    z=pow_residual,
                    colorscale="RdBu",
                    zmid=0.0,
                    zmin=-vmax_pow,
                    zmax=vmax_pow,
                    coloraxis="coloraxis1",
                    hovertemplate="log10(F)=%{x:.3f}<br>log10(η)=%{y:.3f}<br>res=%{z:.4f}<extra></extra>",
                ),
                row=1,
                col=2,
            )
        else:
            fig_html.add_annotation(
                text="Power-law residuals unavailable",
                xref="x2 domain",
                yref="y2 domain",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Logarithmic residuals (bottom-right)
        if log_residual is not None and vmax_log is not None:
            fig_html.add_trace(
                go.Heatmap(
                    x=qv_edges,
                    y=pr_edges,
                    z=log_residual,
                    colorscale="RdBu",
                    zmid=0.0,
                    zmin=-vmax_log,
                    zmax=vmax_log,
                    coloraxis="coloraxis2",
                    hovertemplate="log10(F)=%{x:.3f}<br>log10(η)=%{y:.3f}<br>res=%{z:.4f}<extra></extra>",
                ),
                row=2,
                col=2,
            )
        else:
            fig_html.add_annotation(
                text="Logarithmic residuals unavailable",
                xref="x3 domain",
                yref="y3 domain",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        title_parts = [
            "Bivariate fits with residuals",
            # r"Power: $\hat{I}/\sigma = C\,\eta^{\delta}F^{\gamma}$; "
            # r"Log: $\hat{I}/\sigma = a\,\log_{10}(1+b\eta)\,\log_{10}(1+c\,F)$",
        ]
        if summary_lines:
            title_parts.append("<br>".join(summary_lines))

            fig_html.update_layout(
                scene=dict(
                    xaxis_title="F",
                    yaxis_title="η",
                    zaxis_title="Mean normalized impact",
                    xaxis=dict(type="log"),
                    yaxis=dict(type="log"),
                    zaxis=dict(type="log"),
                ),
                scene_camera=dict(eye=dict(x=1.3, y=1.3, z=0.8)),
                title="<br>".join(title_parts),
                coloraxis1=dict(colorbar=dict(title="Power law", y=0.78, len=0.35)),
                coloraxis2=dict(colorbar=dict(title="Logarithmic", y=0.22, len=0.35)),
                # In this mixed-type subplot layout, the first cartesian subplot is the
                # top-right heatmap (row=1,col=2) => xaxis/yaxis, and the second is the
                # bottom-right heatmap (row=2,col=2) => xaxis2/yaxis2.
                xaxis=dict(title="F", type="log"),
                yaxis=dict(title="η", type="log"),
                xaxis2=dict(title="F", type="log"),
                yaxis2=dict(title="η", type="log"),
                showlegend=True,
            )
        out_path_html = os.path.join(
            IMG_DIR, f"{out_prefix}_3d_surface_bivariate_fits_{LEVEL}_{PROPRIETARY_TAG}.html"
        )
        fig_html.write_html(out_path_html, include_plotlyjs="cdn")
        print(f"[Bivariate Fits] Saved interactive 3D HTML plot to {out_path_html}")
    except ImportError:
        print("[Bivariate Fits] plotly is not installed; skipping interactive HTML plot.")


def plot_impact_surface_and_heatmap(
    df: pd.DataFrame,
    out_prefix: str,
    n_qv_bins: int,
    n_pr_bins: int,
    min_count: int,
) -> None:
    """
    Produce a 3D surface plot and a 2D heatmap of mean normalized impact
    as a function of Q/V and participation rate.
    """
    try:
        qv_edges, pr_edges, impact_grid, count_grid = compute_impact_surface(
            df,
            n_qv_bins=n_qv_bins,
            n_pr_bins=n_pr_bins,
            min_count=min_count,
        )
    except Exception as e:
        print(f"[Surface] Skipping surface/heatmap plot: {e}")
        return

    # Require strictly positive mean impacts for log-scaling on the z-axis
    valid_mask = np.isfinite(impact_grid) & (impact_grid > 0)
    if valid_mask.sum() < 3:
        print("[Surface] Not enough populated 2D bins to plot surface/heatmap.")
        return

    qv_centers = np.sqrt(qv_edges[:-1] * qv_edges[1:])
    pr_centers = np.sqrt(pr_edges[:-1] * pr_edges[1:])
    QV_grid, PR_grid = np.meshgrid(qv_centers, pr_centers)

    # Use linear Q/V and participation values, with log-scaled axes
    impact_for_plot = np.full_like(impact_grid, np.nan, dtype=float)
    impact_for_plot[valid_mask] = impact_grid[valid_mask]

    # Determine shared color normalization in log space based on positive impacts
    positive_impacts = impact_for_plot[valid_mask]
    if positive_impacts.size == 0:
        print("[Surface] No positive impacts to plot in surface/heatmap.")
        return
    norm = LogNorm(vmin=positive_impacts.min(), vmax=positive_impacts.max())

    # 3D surface: Q/V and η on log-scaled axes,
    # with impact on a log-scaled z-axis and log-scaled colormap
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(111, projection="3d")
    surf = ax3d.plot_surface(
        QV_grid,
        PR_grid,
        impact_for_plot,
        cmap="viridis",
        norm=norm,
        linewidth=0,
        antialiased=True,
    )
    # Log scales on the Q/V, participation, and impact axes (like SQL fits)
    try:
        ax3d.set_xscale("log")
        ax3d.set_yscale("log")
        ax3d.set_zscale("log")
    except Exception:
        # If 3D log scales are unavailable, fall back silently to linear.
        pass
    ax3d.set_xlabel("Q/V")
    ax3d.set_ylabel(r"$\eta$")
    ax3d.set_zlabel("Mean normalized impact")
    fig.colorbar(surf, ax=ax3d, shrink=0.7, label="Mean normalized impact (log scale)")
    ax3d.set_title(r"Impact surface: mean normalized impact vs Q/V and $\eta$")
    plt.tight_layout()
    out_path_3d = os.path.join(IMG_DIR, f"{out_prefix}_3d_surface_{LEVEL}_{PROPRIETARY_TAG}.png")
    plt.savefig(out_path_3d, dpi=300)
    plt.close(fig)

    # 2D heatmap with log-scaled Q/V and participation axes
    # and a log-scaled colorbar for impact
    fig, ax = plt.subplots(figsize=(12, 8))
    hm = ax.pcolormesh(
        qv_edges,
        pr_edges,
        impact_for_plot,
        shading="auto",
        cmap="viridis",
        norm=norm,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Q/V")
    ax.set_ylabel(r"$\eta$")
    cbar = fig.colorbar(hm, ax=ax)
    cbar.set_label("Mean normalized impact (log scale)")
    # ax.set_title(r"Impact heatmap: mean normalized impact vs Q/V and $\eta$")
    plt.tight_layout()
    out_path_hm = os.path.join(IMG_DIR, f"{out_prefix}_heatmap_{LEVEL}_{PROPRIETARY_TAG}.png")
    plt.savefig(out_path_hm, dpi=300)
    plt.close(fig)

    print(
        f"[Surface] Saved 3D surface to {out_path_3d} "
        f"and heatmap to {out_path_hm} "
        f"(min_count per 2D bin = {min_count})"
    )

    # Optional interactive HTML 3D surface (browser-rotatable)
    try:
        import plotly.graph_objects as go

        # Plotly does not support LogNorm directly, so we color by log10(impact)
        # while keeping the z-values in linear space (the axis is log-scaled).
        impact_color = np.full_like(impact_for_plot, np.nan, dtype=float)
        impact_color[valid_mask] = np.log10(impact_for_plot[valid_mask])
        cmin = float(np.log10(positive_impacts.min()))
        cmax = float(np.log10(positive_impacts.max()))
        exp_min = int(np.floor(cmin))
        exp_max = int(np.ceil(cmax))
        tickvals = np.arange(exp_min, exp_max + 1)
        ticktext = [f"{10.0 ** e:.3g}" for e in tickvals]

        fig_html = go.Figure(
            data=[
                go.Surface(
                    x=QV_grid,
                    y=PR_grid,
                    z=impact_for_plot,
                    surfacecolor=impact_color,
                    colorscale="Viridis",
                    cmin=cmin,
                    cmax=cmax,
                    colorbar={
                        "title": "Mean normalized impact (log scale)",
                        "tickmode": "array",
                        "tickvals": tickvals,
                        "ticktext": ticktext,
                    },
                )
            ]
        )
        fig_html.update_layout(
            scene=dict(
                xaxis_title="Q/V",
                yaxis_title="η",
                zaxis_title="Mean normalized impact",
                xaxis=dict(type="log"),
                yaxis=dict(type="log"),
                zaxis=dict(type="log"),
            ),
            title="Impact surface: mean normalized impact vs Q/V and η",
        )
        out_path_html = os.path.join(
            IMG_DIR, f"{out_prefix}_3d_surface_{LEVEL}_{PROPRIETARY_TAG}.html"
        )
        fig_html.write_html(out_path_html, include_plotlyjs="cdn")
        print(f"[Surface] Saved interactive 3D HTML surface to {out_path_html}")
    except ImportError:
        print("[Surface] plotly is not installed; skipping interactive HTML 3D surface.")


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
        - R2_log           : weighted R^2 in log space (delta-method weights)
        - R2_lin           : weighted R^2 in linear space (1/SEM^2)
        - beta_controls    : pd.Series of control coefficients (or None)
        - beta_controls_se : pd.Series of control SEs (or None)
    """
    # 1) Filter valid rows
    qv = pd.to_numeric(subdf["Q/V"], errors="coerce")
    impact = pd.to_numeric(subdf["Impact"], errors="coerce")
    mask = (qv > 0) & np.isfinite(impact)

    controls_numeric: Dict[str, pd.Series] = {}
    if control_cols is not None:
        for c in control_cols:
            ctrl = pd.to_numeric(subdf[c], errors="coerce")
            controls_numeric[c] = ctrl
            mask &= np.isfinite(ctrl)

    sub = subdf.loc[mask].copy()
    sub["Q/V"] = qv.loc[mask].astype(float)
    sub["Impact"] = impact.loc[mask].astype(float)
    for c, ctrl in controls_numeric.items():
        sub[c] = ctrl.loc[mask].astype(float)
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

    Zhat = A @ coef
    R2_log = _weighted_r2(Z, Zhat, w=w)

    # 8) R^2 in linear space (only for the power-law part; weighted by 1/SEM^2)
    yhat = power_law(binned["center_QV"].to_numpy(), Y_hat, gamma_hat)
    w_lin = _weights_from_sigma(binned["sem_imp"].to_numpy())
    R2_lin = _weighted_r2(binned["mean_imp"].to_numpy(), yhat, w=w_lin)

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
        where R2_lin is weighted in linear space (1/SEM^2).
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

    # Goodness-of-fit in linear space (weighted by 1/SEM^2)
    yhat = _log_model(x, a_hat, b_hat)
    w_lin = _weights_from_sigma(sigma)
    R2_lin = _weighted_r2(y, yhat, w=w_lin)

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


def filter_metaorders_info_for_fits(df: pd.DataFrame, min_qv: float = MIN_QV) -> pd.DataFrame:
    """Apply the unified Q/V filter, compute `Impact`, and sanitize numeric columns."""
    required = {"Q/V", "Price Change", "Direction", "Daily Vol"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for filtering: {sorted(missing)}")

    out = df.copy()
    for col in ["Q/V", "Vt/V", "Participation Rate", "Price Change", "Daily Vol", "Q", "Direction"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out["Q/V"] > min_qv].copy()
    out["Impact"] = pd.to_numeric(
        out["Price Change"] * out["Direction"] / out["Daily Vol"], errors="coerce"
    )

    numeric_cols = [
        c
        for c in ["Q/V", "Vt/V", "Impact", "Participation Rate", "Price Change", "Daily Vol", "Q"]
        if c in out.columns
    ]
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return out.dropna(subset=["Q/V", "Impact"]).reset_index(drop=True)


def run_wls_fits_and_surfaces(df: pd.DataFrame) -> None:
    """Run aggregated WLS fits + downstream plots; safe to call only when df is non-empty."""
    label_size = 16
    legend_size = 14
    title_size = 18
    plt.rcParams.update(
        {
            "axes.labelsize": label_size,
            "axes.titlesize": title_size,
            "legend.fontsize": legend_size,
        }
    )

    try:
        binned_all, params_all = fit_power_law_logbins_wls_new(
            df,
            n_logbins=N_LOGBIN,
            min_count=MIN_COUNT,
            use_median=False,
            control_cols=None,
        )
    except Exception as e:
        print(f"[WLS] Skipping WLS fits and surfaces: {e}")
        return

    Y_hat, Y_se, gamma_hat, gamma_se, R2_log, R2_lin, beta_controls, beta_controls_se = params_all
    log_params_all = fit_logarithmic_from_binned(binned_all)
    a_hat, a_se, b_hat, b_se, R2_lin_log = log_params_all

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_fit(ax, binned_all, params_all, log_params=log_params_all)
    ax.set_title("Impact fits (power-law and logarithmic, aggregated)", fontsize=title_size)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"power_law_fit_overall_{LEVEL}.png"), dpi=300)
    plt.close()

    print("--- Overall (All) ---")
    print(f"Power law: Y = {Y_hat:.6g} +/- {Y_se:.3g}")
    print(f"Power law: gamma = {gamma_hat:.6f} +/- {gamma_se:.3g}")
    print(f"Power law: R^2_log = {R2_log:.4f} | R^2_lin = {R2_lin:.4f}")
    print(
        f"Logarithmic: a = {a_hat:.6g} +/- {a_se:.3g} | "
        f"b = {b_hat:.6g} +/- {b_se:.3g} | R^2_lin = {R2_lin_log:.4f}"
    )
    print(f"Bins used: {len(binned_all)} (min_count >= {MIN_COUNT})")

    pr_col = "Participation Rate"
    if pr_col in df.columns and df[pr_col].notna().sum() >= 2:
        pr_nbins = 2
        labels = [f"Q{j+1}" for j in range(pr_nbins)]
        try:
            df_pr = df.copy()
            df_pr["PR_bin"] = pd.qcut(df_pr[pr_col], q=pr_nbins, labels=labels, duplicates="drop")
        except Exception as e:
            print(f"[WLS] Conditioned-on-η fits skipped: {e}")
            df_pr = None
        if df_pr is not None:
            fig, ax = plt.subplots(figsize=(12, 8))
            fits_by_pr = {}
            for label in df_pr["PR_bin"].dropna().unique():
                sub = df_pr[df_pr["PR_bin"] == label]
                try:
                    binned_sub, params_sub = fit_power_law_logbins_wls_new(
                        sub,
                        n_logbins=N_LOGBIN,
                        min_count=MIN_COUNT,
                        use_median=False,
                        control_cols=None,
                    )
                except Exception as e:
                    print(f"[{label}] skipped: {e}")
                    continue
                plot_fit(ax, binned_sub, params_sub, label_prefix=rf"$\eta$ {label}")
                fits_by_pr[str(label)] = params_sub
            ax.set_title(r"Power-law fits conditioned on $\eta$ (aggregated)", fontsize=title_size)
            plt.tight_layout()
            plt.savefig(os.path.join(IMG_DIR, f"power_law_fits_by_participation_rate_{LEVEL}.png"), dpi=300)
            plt.close()

            print("--- Conditioned on η (power-law only) ---")
            for k, power_params in fits_by_pr.items():
                Y, Y_se, gamma, gamma_se, R2_log, R2_lin, beta_controls, beta_controls_se = power_params
                print(f"[η {k}] Y = {Y:.6g} +/- {Y_se:.3g} | gamma = {gamma:.6f} +/- {gamma_se:.3g} |")
                print(f"[η {k}] R^2_log = {R2_log:.4f} | R^2_lin = {R2_lin:.4f}")
                print(f"[η {k}] Beta controls: {beta_controls}")
                print(f"[η {k}] Beta controls SE: {beta_controls_se}")

    # -----------------------------------------------------------------------
    # 3D surface / heatmap of mean impact vs Q/V and participation rate
    # -----------------------------------------------------------------------
    try:
        plot_impact_surface_and_heatmap(
            df,
            out_prefix="impact_surface_qv_participation",
            n_qv_bins=N_LOGBIN,
            n_pr_bins=N_PR_BINS_SURFACE,
            min_count=MIN_COUNT_SURFACE,
        )
    except Exception as e:
        print(f"[Surface] skipped: {e}")

    # -----------------------------------------------------------------------
    # Bivariate fits: power-law and logarithmic vs empirical (3D + residuals)
    # -----------------------------------------------------------------------
    try:
        plot_bivariate_fit_surfaces_3d(
            df,
            out_prefix="impact_surface_qv_participation",
            n_qv_bins=N_LOGBIN,
            n_pr_bins=N_PR_BINS_SURFACE,
            min_count=MIN_COUNT_SURFACE,
        )
    except Exception as e:
        print(f"[Bivariate Fits] skipped: {e}")


# ---------------------------------------------------------------------------
# Intro / transforms
# ---------------------------------------------------------------------------
if RUN_INTRO:
    print("[Intro] Ensuring parquet transforms are up to date...")
    print(
        "[Intro] Parameters — "
        f"LEVEL={LEVEL}, PROPRIETARY={PROPRIETARY}, DATASET={DATASET_NAME}, "
        f"IMPACT_HORIZONS_MIN={IMPACT_HORIZONS_MIN}, RECOMPUTE={RECOMPUTE}"
    )
    ensure_transforms()
    dfs_path_new = list_parquet_paths()
    print(f"[Intro] Parquet files available for dataset '{DATASET_NAME}': {len(dfs_path_new)}")


# ---------------------------------------------------------------------------
# Metaorder computation (aggregated across ISIN)
# ---------------------------------------------------------------------------
if RUN_METAORDER_COMPUTATION:
    print(f"[Metaorders] Computing metaorders across all ISINs (dataset={DATASET_NAME})...")
    dfs_path_new = list_parquet_paths()

    filtered_path = os.path.join(OUT_DIR, f"metaorders_dict_all_{LEVEL}_{PROPRIETARY_TAG}.pkl")

    metaorders_dict_all: Optional[Dict[str, Dict[int, List[List[int]]]]] = None
    if os.path.exists(filtered_path) and not RECOMPUTE:
        print(f"Loading {filtered_path}")
        loaded = pickle.load(open(filtered_path, "rb"))
        if isinstance(loaded, dict) and len(loaded) == 0 and len(dfs_path_new) > 0:
            print(
                f"[Metaorders] Cached metaorders dict is empty at {filtered_path}; "
                "recomputing (set RECOMPUTE=true to force this explicitly)."
            )
        else:
            metaorders_dict_all = loaded

    if metaorders_dict_all is None:
        try:
            max_gap_np = MAX_GAP.to_numpy()
        except AttributeError:
            print(f"Warning: MAX_GAP {MAX_GAP} could not be converted to numpy timedelta64; assuming nanoseconds.")
            max_gap_np = np.timedelta64(int(MAX_GAP.value), "ns")

        filtered_dict: Dict[str, Dict[int, List[List[int]]]] = {}
        metaorder_counts = {"raw": 0, "after_trades": 0, "after_duration": 0}
        for path in tqdm(dfs_path_new, desc="Metaorders per ISIN (filtered)", dynamic_ncols=True):
            isin = os.path.splitext(os.path.basename(path))[0]
            trades = load_trades_filtered(path, proprietary=PROPRIETARY, trading_hours=TRADING_HOURS)
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
            f"[Metaorders][ALL] raw (gap<{MAX_GAP}): "
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
    if N_SIGNATURE_PLOTS is not None:
        dfs_path_new = dfs_path_new[:N_SIGNATURE_PLOTS]
    for path in tqdm(dfs_path_new, desc="Signature plots", dynamic_ncols=True):
        isin = os.path.splitext(os.path.basename(path))[0]
        trades = load_trades_full(path, trading_hours=TRADING_HOURS)
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
        trades_full = load_trades_full(path, trading_hours=TRADING_HOURS)
        trades = filter_trades_by_group(trades_full, PROPRIETARY)
        metaorders_dict = metaorders_dict_all.get(isin, {})
        metaorders_info_records.extend(
            compute_metaorders_info(trades, metaorders_dict, trades_full=trades_full)
        )
        del trades
        del trades_full
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
            "Vt/V",
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
    metaorders_info_df_sameday_filtered = filter_metaorders_info_for_fits(metaorders_info_df_sameday, min_qv=MIN_QV)
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
        df = filter_metaorders_info_for_fits(raw_df, min_qv=MIN_QV)
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

    # Coerce key numeric columns to avoid dtype-driven failures (e.g., object columns in old parquets).
    df = df.copy()
    for col in [
        "Q/V",
        "Vt/V",
        "Participation Rate",
        "Price Change",
        "Daily Vol",
        "Direction",
        "Q",
        "Impact",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Impact" not in df.columns and {"Price Change", "Direction", "Daily Vol"}.issubset(df.columns):
        df["Impact"] = pd.to_numeric(df["Price Change"] * df["Direction"] / df["Daily Vol"], errors="coerce")
    # Only sanitize numeric columns to avoid comparing arrays/bytes objects in replace
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Apply participation-rate filter for all subsequent fits and surfaces
    if "Participation Rate" in df.columns:
        before_pr = len(df)
        df = df[df["Participation Rate"] < MAX_PARTICIPATION_RATE].reset_index(drop=True)
        print(
            f"[WLS] After participation-rate filter (PR < {MAX_PARTICIPATION_RATE}): "
            f"{before_pr} -> {len(df)} metaorders"
        )

    if RUN_IMPACT_PATH_PLOT:
        out_path = os.path.join(IMG_DIR, f"normalized_impact_path_{LEVEL}_{PROPRIETARY_TAG}.png")
        plot_normalized_impact_path(df, out_path, duration_multiplier=AFTERMATH_DURATION_MULTIPLIER)

    if df.empty:
        print("[WLS] No metaorders available after filters; skipping WLS fits and surfaces.")
    else:
        run_wls_fits_and_surfaces(df)
