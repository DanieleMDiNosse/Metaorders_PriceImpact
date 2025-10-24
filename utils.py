from typing import Dict, Tuple, List
from collections import defaultdict
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import gc


def map_trade_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace numeric codes in trading-related columns with descriptive string labels.
    Spaces in labels are replaced by underscores.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing columns:
        EXECUTION_PHASE, TRADE_CANCELLATION_INDICATOR,
        PASSIVE_ORDER_INDICATOR_BUY, PASSIVE_ORDER_INDICATOR_SELL,
        TRADING_CAPACITY_BUY, TRADING_CAPACITY_SELL
    
    Returns
    -------
    pd.DataFrame
        New dataframe with mapped categorical string values.
    """
    
    # Define mappings
    execution_phase_map = {
        1: "Continuous_Trading_Phase",
        2: "Uncrossing_Phase",
        3: "Trading_At_Last_Phase",
        4: "Continuous_Uncrossing_Phase",
        5: "IPO"
    }

    trade_cancellation_map = {
        0: "False",
        1: "True"
    }

    passive_order_map = {
        0: "Aggressive",
        1: "Passive"
    }

    trading_capacity_map = {
        1: "Dealing_on_own_account",
        2: "Matched_principal",
        3: "Any_other_capacity"
    }

    # Copy to avoid mutating original DataFrame
    df_mapped = df.copy()

    # Apply mappings if columns exist
    mappings = {
        "EXECUTION_PHASE": execution_phase_map,
        "TRADE_CANCELLATION_INDICATOR": trade_cancellation_map,
        "PASSIVE_ORDER_INDICATOR_BUY": passive_order_map,
        "PASSIVE_ORDER_INDICATOR_SELL": passive_order_map,
        "TRADING_CAPACITY_BUY": trading_capacity_map,
        "TRADING_CAPACITY_SELL": trading_capacity_map
    }

    for col, mapping in mappings.items():
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].map(mapping)

    return df_mapped

def build_trades_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the target trades dataframe with the requested columns and logic.
    Preserves nanosecond precision for 'Trade Time' when combining TRADING_DAY (DD/MM/YYYY)
    and TRADETIME ('H:MM:SS.fffffffff' with up to 9 fractional digits).
    """
    required: List[str] = [
        'MIC','TRADING_DAY','TRADETIME','TRADED_QUANTITY','TRADED_PRICE','TRADED_AMOUNT',
        'COD_BUY','CLIENT_IDENTIFIC_SHORT_CODE_BUY','PASSIVE_ORDER_INDICATOR_BUY',
        'COD_SELL','CLIENT_IDENTIFIC_SHORT_CODE_SELL',
        'TRADING_CAPACITY_BUY','TRADING_CAPACITY_SELL'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = pd.DataFrame(index=df.index).copy()

    # Direction: BUY aggressive => +1, else -1
    buy_aggr = df['PASSIVE_ORDER_INDICATOR_BUY'].astype(str).str.strip().eq('Aggressive')
    out['Direction'] = np.where(buy_aggr, 1, -1)

    # IDs and counterparties
    out['ID Client'] = np.where(buy_aggr, df['CLIENT_IDENTIFIC_SHORT_CODE_BUY'],
                                df['CLIENT_IDENTIFIC_SHORT_CODE_SELL'])
    out['ID Member'] = np.where(buy_aggr, df['COD_BUY'], df['COD_SELL'])
    out['ID Counterpart Client'] = np.where(buy_aggr, df['CLIENT_IDENTIFIC_SHORT_CODE_SELL'],
                                            df['CLIENT_IDENTIFIC_SHORT_CODE_BUY'])
    out['ID Counterpart Member'] = np.where(buy_aggr, df['COD_SELL'], df['COD_BUY'])

    # Quantities / amounts
    qty = df['TRADED_QUANTITY']; amt = df['TRADED_AMOUNT']
    out['Total Quantity Buy']  = np.where(buy_aggr, qty, 0)
    out['Total Quantity Sell'] = np.where(buy_aggr, 0,   qty)
    out['Total Amount Buy']    = np.where(buy_aggr, amt, 0)
    out['Total Amount Sell']   = np.where(buy_aggr, 0,   amt)

    # Capacities
    out['Trade Type Aggressive'] = np.where(buy_aggr, df['TRADING_CAPACITY_BUY'],
                                            df['TRADING_CAPACITY_SELL'])
    out['Trade Type Passive']    = np.where(buy_aggr, df['TRADING_CAPACITY_SELL'],
                                            df['TRADING_CAPACITY_BUY'])

    # Trading venue and price
    out['Trading Venue'] = df['MIC']
    out['Price'] = df['TRADED_PRICE']
    out['Price First Contract'] = df['TRADED_PRICE']
    out['Price Last Contract']  = df['TRADED_PRICE']

    # --- Trade Time (ns-precise)
    # # TRADING_DAY is DD/MM/YYYY in your sample
    # day = pd.to_datetime(df['TRADING_DAY'], format='%d/%m/%Y', errors='coerce').dt.normalize()

    # # Normalize TRADETIME: accept '.' or ',' as decimal separator; pad single-digit hour '9:' -> '09:'
    # s = df['TRADETIME'].astype(str).str.strip().str.replace(",", ".", regex=False)
    # s = s.str.replace(r'^(\d):', r'0\1:', regex=True)

    # # Parse time-of-day with to_timedelta (preserves up to 9 fractional digits)
    # td = pd.to_timedelta(s, errors='coerce')

    out['Trade Time'] = df['TRADING_DAY'] + ' ' + df['TRADETIME']
    out['Trade Time'] = pd.to_datetime(out['Trade Time'], dayfirst=True, errors='raise')

    # 1) Aggregate by timestamp + identifiers (+ direction/venue for safety)
    group_keys = ['Trade Time', 'ID Client', 'ID Member', 'Trading Venue', 'Direction']

    g = (
        out.groupby(group_keys, as_index=False, dropna=False)
        .agg(**{
            'Total Quantity Buy':    ('Total Quantity Buy', 'sum'),
            'Total Quantity Sell':   ('Total Quantity Sell', 'sum'),
            'Total Amount Buy':      ('Total Amount Buy', 'sum'),
            'Total Amount Sell':     ('Total Amount Sell', 'sum'),
            'Trade Type Aggressive': ('Trade Type Aggressive', 'first'),
            'Trade Type Passive':    ('Trade Type Passive', 'first'),
            # helpers for price-first/last logic
            'price_min':             ('Price', 'min'),
            'price_max':             ('Price', 'max'),
        })
    )

    # 2) First/last per group based on Direction
    g['Price First Contract'] = np.where(g['Direction'].eq(1), g['price_min'], g['price_max'])
    g['Price Last Contract']  = np.where(g['Direction'].eq(1), g['price_max'], g['price_min'])

    # 3) Drop helpers and continue as before
    g = g.drop(columns=['price_min', 'price_max'])
    out = g

    # Final column order
    correct_column_order = [
        'ID Client', 'ID Member', 'Trading Venue', 'Trade Time', 'Direction', 'Price First Contract', 'Price Last Contract',
        'Total Quantity Buy', 'Total Quantity Sell', 'Total Amount Buy', 'Total Amount Sell',
        # 'ID Counterpart Client', 'ID Counterpart Member',
        'Trade Type Aggressive' 
        # 'Trade Type Passive'
    ]

    # Ensure numeric dtypes
    for c in ['Total Quantity Buy','Total Quantity Sell','Total Amount Buy','Total Amount Sell','Price First Contract', 'Price Last Contract','Direction']:
        out[c] = pd.to_numeric(out[c], errors='coerce')

    return out[correct_column_order]

# def agents_activity(trades, column_positions: Dict[str, int]):
#     '''This function creates three dictionaries where the keys are the agents IDs and the values are (i) her activity (ii) volume trades (iii) her invetory value
#     at each event occurred in the trades dataset.
    
#     Parameters
#     ----------
#     trades: np.ndarray
#         The array with the trades data.
#     column_positions: dict
#         The dictionary with the column positions.
        
#     Returns
#     -------
#     agents_act: dict
#         The dictionary with the agents' activity
#     agents_val: dict
#         The dictionary with the agents' inventory value
#     agents_vol: dict
#         The dictionary with the agents' volume
#     '''
    
#     trade_times = pd.to_datetime(trades[:, column_positions['Trade Time']])
#     sort_idx = np.argsort(trade_times)
#     trades, trade_times = trades[sort_idx], trade_times[sort_idx]

#     agent_ids = np.unique(trades[:, column_positions['ID Member']])

#     # full-length arrays
#     agents_val  = {aid: np.zeros(len(trades), dtype=np.float32)          for aid in agent_ids}
#     agents_act  = {aid: np.zeros(len(trades), dtype=np.int8)             for aid in agent_ids}
#     agents_vol  = {aid: np.zeros(len(trades), dtype=np.float32)          for aid in agent_ids}
#     agents_time = {aid: np.full(len(trades), np.datetime64('NaT', 'ns')) for aid in agent_ids}

#     for i in range(len(trades)):
#         aid           = trades[i, column_positions['ID Member']]
#         buy_vol       = trades[i, column_positions['Total Quantity Buy']]
#         sell_vol  = trades[i, column_positions['Total Quantity Sell']]
#         buy_amount    = trades[i, column_positions['Total Amount Buy']]
#         sell_amount = trades[i, column_positions['Total Amount Sell']]

#         if buy_amount > 0:
#             agents_act[aid][i] = 1
#             agents_val[aid][i] = buy_amount
#             agents_vol[aid][i] = buy_vol
#         else:
#             agents_act[aid][i] = -1
#             agents_val[aid][i] = -sell_amount
#             agents_vol[aid][i] = -sell_vol

#         agents_time[aid][i] = trade_times[i].isoformat(timespec='nanoseconds')

#     return agents_act, agents_val, agents_vol, agents_time

def find_metaorders(activity: np.ndarray, min_child: int = 2):
    """
    Detect same-sign consecutive runs (ignoring zeros) in `activity`.
    Returns:
      - metaorders_indexes: list of tuples of indices in the *reduced* sequence (nonzero-only)
      - metaorders_original_indexes: list of lists of indices in the original `activity`
      - n_metaorders: number of kept metaorders
    """
    activity = np.asarray(activity)
    nz = np.flatnonzero(activity)  # positions of non-zeros in original sequence
    if nz.size == 0:
        return [], [], 0

    signs = activity[nz].astype(np.int8, copy=False)
    if signs.size == 0:
        return [], [], 0

    # boundaries of constant-sign runs in the reduced sequence
    change_pos = np.flatnonzero(np.diff(signs) != 0)  # positions where sign changes (in reduced indexing)
    starts = np.concatenate(([0], change_pos + 1))
    ends   = np.concatenate((change_pos + 1, [signs.size]))  # exclusive

    lengths = ends - starts
    keep_idx = np.flatnonzero(lengths >= min_child)
    if keep_idx.size == 0:
        return [], [], 0

    # Build outputs
    metaorders_indexes = [tuple(range(starts[i], ends[i])) for i in keep_idx]
    metaorders_original_indexes = [nz[starts[i]:ends[i]].tolist() for i in keep_idx]
    return metaorders_indexes, metaorders_original_indexes, len(metaorders_original_indexes)


# ------------------------------
# Build sparse activity: only store the indices the agent touched and the sign there
# ------------------------------
def agents_activity_sparse(
    trades: np.ndarray,
    column_positions: Dict[str, int],
    level: str = "broker",
) -> Tuple[Dict[object, np.ndarray], Dict[object, np.ndarray]]:
    """
    Returns per-agent:
      - indices_by_agent[aid]: int64 array of trade indices where the agent traded
      - act_by_agent[aid]: int8 array of +1/-1 for those indices (aligned to indices_by_agent)
    """
    idcol = column_positions["ID Client"] if level == "client" else column_positions["ID Member"]
    abuy  = column_positions["Total Amount Buy"]
    asell = column_positions["Total Amount Sell"]

    idxs = defaultdict(list)
    acts = defaultdict(list)

    n = len(trades)
    for i in range(n):
        aid = trades[i, idcol]

        buy_amount  = trades[i, abuy]
        sell_amount = trades[i, asell]

        # Infer direction (dataset typically has one side > 0)
        # If both sides are zero (unlikely), skip.
        if buy_amount > 0:
            direction = 1
        elif sell_amount > 0:
            direction = -1
        else:
            continue  # nothing to record

        idxs[aid].append(i)
        acts[aid].append(direction)

    indices_by_agent = {aid: np.asarray(v, dtype=np.int64) for aid, v in idxs.items()}
    act_by_agent     = {aid: np.asarray(v, dtype=np.int8)  for aid, v in acts.items()}
    return indices_by_agent, act_by_agent

def high_low_volatility(prices: np.ndarray) -> float:
    """
    Volatility estimator: difference between the highest and lowest price.
    Accepts a 1D array or Series of prices.
    """
    if len(prices) == 0:
        return np.nan
    return np.nanmax(prices) - np.nanmin(prices)

def preprocess_log_returns(prices: pd.DataFrame, delta: str) -> Dict[pd.Timestamp, np.ndarray]:
    prices = prices.set_index('Trade Time')[['Price Last Contract']].dropna()
    prices.index = pd.to_datetime(prices.index)
    grouped = prices.groupby(prices.index.date)

    log_returns = {}
    for date, group in grouped:
        resampled = group['Price Last Contract'].resample(delta).last().dropna()
        if len(resampled) > 1:
            log_ret = np.diff(np.log(resampled.to_numpy()))
            if len(log_ret) > 0:
                log_returns[date] = log_ret
    return log_returns


def realized_variance_fast(log_returns_dict):
    return [np.sqrt(np.sum(r**2)) for r in log_returns_dict.values()]

def bipower_variation_fast(log_returns_dict):
    factor = np.pi / 2
    return [np.sqrt(factor * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))) if len(r) > 2 else np.nan
            for r in log_returns_dict.values()]

def realized_kernel_fast(log_returns, kernel='parzen', bandwidth=None):
    """
    Compute realized kernel:
    - If `log_returns` is a dict of arrays (multiple days): return list of RK sqrt.
    - If `log_returns` is a single pd.Series or np.array: return single RK sqrt.
    """
    # Vectorized Parzen kernel
    def parzen_vector(u):
        u = np.abs(u)
        k = np.zeros_like(u)
        mask1 = u <= 0.5
        mask2 = (u > 0.5) & (u <= 1.0)
        k[mask1] = 1 - 6*u[mask1]**2 + 6*u[mask1]**3
        k[mask2] = 2 * (1 - u[mask2])**3
        return k

    def compute_rk(r):
        n = len(r)
        if n < 3:
            return np.nan
        H = int(np.floor(n ** (2/3))) if bandwidth is None else bandwidth
        h_vals = np.arange(-H, H+1)
        weights = parzen_vector(h_vals / H) if kernel == 'parzen' else np.ones_like(h_vals)

        rk = 0.0
        for h, w in zip(h_vals, weights):
            if h >= 0:
                rk += w * np.dot(r[h:], r[:n - h])
            else:
                rk += w * np.dot(r[:n + h], r[-h:])
        return np.sqrt(rk)

    # Case 1: dictionary (multiple days)
    if isinstance(log_returns, dict):
        return [compute_rk(np.asarray(r)) for r in log_returns.values()]
    
    # Case 2: single Series or array
    elif isinstance(log_returns, (pd.Series, np.ndarray, list)):
        return compute_rk(np.asarray(log_returns))

    else:
        raise TypeError("Input must be a dict of arrays or a single Series/array of returns.")

def sum_contiguous(csum: np.ndarray, start: int, end_exclusive: int) -> float:
    """Sum of csum[start:end_exclusive] in O(1). Handles start==0 gracefully."""
    if end_exclusive <= 0:
        return 0.0
    if start <= 0:
        return float(csum[end_exclusive - 1])
    return float(csum[end_exclusive - 1] - csum[start - 1])

def build_daily_cache(trades_members: pd.DataFrame):
    """
    Precompute per-day:
      - daily volatility via realized_kernel_fast on 120s resampled PLC
      - daily total volume (buy + sell)
    Returns dict: day(date) -> (daily_vol, daily_volume)

    Fixes:
      - Handle duplicate timestamps before resampling (keep LAST tick per timestamp).
      - Use a closed-open day window [d, d+1).
    """
    # Ensure dtype + sort once
    # if not np.issubdtype(trades_members["Trade Time"].dtype, np.datetime64):
    if not np.issubdtype(np.asarray(trades_members["Trade Time"]).dtype, np.datetime64):
        trades_members = trades_members.copy()
        trades_members["Trade Time"] = pd.to_datetime(trades_members["Trade Time"], errors="raise")
    trades_members = trades_members.sort_values("Trade Time", kind="mergesort")

    days = sorted(trades_members["Trade Time"].dt.date.unique())
    cache = {}

    for d in days:
        day_start = pd.Timestamp(d)
        day_end   = day_start + pd.Timedelta(days=1)

        # Closed-open slice [d, d+1)
        day_df = trades_members[(trades_members["Trade Time"] >= day_start) &
                                (trades_members["Trade Time"] <  day_end)]

        if day_df.empty:
            cache[d] = (np.nan, 0.0)
            continue

        # Daily volume (buy + sell)
        daily_volume = float(
            day_df[["Total Quantity Buy", "Total Quantity Sell"]].to_numpy().sum()
        )

        # Price series with UNIQUE timestamps: keep the last trade per exact timestamp
        plc_series = (
            day_df.loc[:, ["Trade Time", "Price Last Contract"]]
                  .dropna(subset=["Trade Time"])
                  .sort_values("Trade Time", kind="mergesort")
                  .drop_duplicates(subset="Trade Time", keep="last")
                  .set_index("Trade Time")["Price Last Contract"]
        )

        if plc_series.empty:
            cache[d] = (np.nan, daily_volume)
            continue

        # 120s resample by last observed, forward-fill across gaps after the first tick
        resampled = plc_series.resample("120s").last().ffill().dropna()

        if resampled.size < 2:
            daily_vol = np.nan
        else:
            returns = np.diff(np.log(resampled.to_numpy()))
            rk_var = realized_kernel_fast(returns)  # provided elsewhere
            daily_vol = float(np.sqrt(rk_var))

        cache[d] = (daily_vol, daily_volume)

    return cache