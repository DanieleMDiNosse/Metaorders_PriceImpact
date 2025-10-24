from __future__ import annotations

import argparse
import gc
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

from utils import agents_activity_sparse, build_trades_view, find_metaorders, map_trade_codes

PROPRIETARY_LABEL = "Dealing_on_own_account"


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    level: str
    data_dir: Path
    cache_dir: Path
    output_dir: Path
    pickle_suffix: str = ""
    trade_filter: str = "exclude_proprietary"
    min_children: int = 2
    max_gap_minutes: int = 60
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00")


DEFAULT_DATA_DIR = Path("C:/Users/User01/Documents/di_nosse/FTSEMIB_MOT")
DEFAULT_CACHE_DIR = Path("C:/Users/User01/Documents/di_nosse/FTSEMIB_MOT_NEW")
DEFAULT_OUTPUT_DIR = Path("out_files")


CONFIGS: Dict[str, PipelineConfig] = {
    "client": PipelineConfig(
        name="client",
        level="client",
        data_dir=DEFAULT_DATA_DIR,
        cache_dir=DEFAULT_CACHE_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
    ),
    "broker": PipelineConfig(
        name="broker",
        level="member",
        data_dir=DEFAULT_DATA_DIR,
        cache_dir=DEFAULT_CACHE_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
    ),
    "proprietary": PipelineConfig(
        name="proprietary",
        level="member",
        data_dir=DEFAULT_DATA_DIR,
        cache_dir=DEFAULT_CACHE_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        trade_filter="only_proprietary",
        pickle_suffix="_proprietary",
    ),
}


class MetaordersPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.data_dir = config.data_dir
        self.cache_dir = config.cache_dir
        self.output_dir = config.output_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def metaorders_nofilter_path(self) -> Path:
        return self.output_dir / f"metaorders_dict_all_nofilter_{self.config.level}{self.config.pickle_suffix}.pkl"

    @property
    def metaorders_filtered_path(self) -> Path:
        return self.output_dir / f"metaorders_dict_all_{self.config.level}{self.config.pickle_suffix}.pkl"

    @property
    def metaorders_info_path(self) -> Path:
        return self.output_dir / f"metaorders_info_sameday_{self.config.level}{self.config.pickle_suffix}.pkl"

    def run(
        self,
        *,
        skip_conversion: bool = False,
        rebuild_metaorders: bool = False,
        recompute_info: bool = False,
        skip_plots: bool = False,
        quantile_bins: int = 30,
    ) -> pd.DataFrame:
        csv_paths, isins = self._list_csv_sources()
        if not skip_conversion:
            self._convert_csv_to_parquet(csv_paths)
        parquet_paths = self._list_parquet_sources()
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet files found in {self.cache_dir}.")

        metaorders_nofilter = self._load_or_build_metaorders(
            parquet_paths,
            isins,
            rebuild=rebuild_metaorders,
        )
        metaorders_filtered = self._load_or_filter_metaorders(
            parquet_paths,
            metaorders_nofilter,
            rebuild=rebuild_metaorders,
        )
        self._validate_metaorders(parquet_paths, metaorders_filtered)

        metaorders_info = self._load_or_compute_info(
            parquet_paths,
            metaorders_filtered,
            recompute=recompute_info,
        )
        info_df = self._metaorders_info_dataframe(metaorders_info)
        if not info_df.empty:
            info_df.sort_values(["ISIN", "Member", "Client"], inplace=True)
            parquet_out = self.metaorders_info_path.with_suffix(".parquet")
            info_df.to_parquet(parquet_out, index=False)
            if not skip_plots:
                self._impact_plots(info_df, quantile_bins=quantile_bins)
        else:
            print("Metaorders info dataframe is empty; skipping plots.")
        return info_df

    def _list_csv_sources(self) -> Tuple[List[Path], List[str]]:
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} not found.")
            return [], []
        csv_paths = [
            path
            for path in self.data_dir.iterdir()
            if path.suffix.lower() == ".csv" and path.name not in {"ALTRI_FTSEMIB.csv", "MOT.csv"}
        ]
        isins = [path.stem for path in csv_paths]
        return csv_paths, isins

    def _list_parquet_sources(self) -> List[Path]:
        return sorted(self.cache_dir.glob("*.parquet"))

    def _convert_csv_to_parquet(self, csv_paths: Sequence[Path]) -> None:
        for csv_path in csv_paths:
            target = self.cache_dir / f"{csv_path.stem}.parquet"
            if target.exists():
                continue
            print(f"Converting {csv_path.name} -> {target.name}")
            df_raw = pd.read_csv(csv_path, sep=";")
            df_mapped = map_trade_codes(df_raw)
            df_trades = build_trades_view(df_mapped)
            bad_mask = (df_trades["Total Quantity Buy"] > 0) & (df_trades["Total Quantity Sell"] > 0)
            if bad_mask.any():
                print(f"Warning: found {bad_mask.sum()} rows with both buy and sell quantities > 0 in {csv_path.name}.")
            df_trades.to_parquet(target)

    def _load_or_build_metaorders(
        self,
        parquet_paths: Sequence[Path],
        isins: Sequence[str],
        *,
        rebuild: bool,
    ) -> Dict[str, Dict[object, List[List[int]]]]:
        if self.metaorders_nofilter_path.exists() and not rebuild:
            print(f"Loading cached metaorders (no filter) from {self.metaorders_nofilter_path}.")
            return pickle.loads(self.metaorders_nofilter_path.read_bytes())

        print("Building metaorders without gap filtering...")
        metaorders_dict_all: Dict[str, Dict[object, List[List[int]]]] = {isin: {} for isin in isins}
        for idx, parquet_path in enumerate(parquet_paths, start=1):
            trades = self._load_filtered_trades(parquet_path)
            isin = parquet_path.stem
            print(f"({idx}/{len(parquet_paths)}) Processing {isin}: {len(trades)} trades after filtering.")
            if trades.empty:
                continue
            column_positions = self._column_positions(trades)
            trades_np = trades.to_numpy()
            indices_by_agent, act_by_agent = agents_activity_sparse(trades_np, column_positions, level=self.config.level)
            activity_buffer = np.zeros(len(trades), dtype=np.int8)
            metaorders_by_agent: Dict[object, List[List[int]]] = {}
            for agent_id in tqdm(indices_by_agent.keys(), desc="    Agents", leave=False):
                idxs = indices_by_agent[agent_id]
                signs = act_by_agent[agent_id]
                if idxs.size == 0:
                    continue
                activity_buffer[idxs] = signs
                _, meta_idxs, n_meta = find_metaorders(activity_buffer, min_child=self.config.min_children)
                if n_meta:
                    kept: List[List[int]] = []
                    for meta_idx_list in meta_idxs:
                        if len(meta_idx_list) < self.config.min_children:
                            continue
                        t_start = pd.Timestamp(trades_np[meta_idx_list[0], column_positions["Trade Time"]])
                        t_end = pd.Timestamp(trades_np[meta_idx_list[-1], column_positions["Trade Time"]])
                        if t_start.date() != t_end.date():
                            continue
                        clients = np.unique(trades_np[meta_idx_list, column_positions["ID Client"]])
                        if len(clients) > 1:
                            continue
                        kept.append([int(i) for i in meta_idx_list])
                    if kept:
                        metaorders_by_agent[agent_id] = kept
                activity_buffer[idxs] = 0
            metaorders_dict_all[isin] = metaorders_by_agent
            total_meta = sum(len(v) for v in metaorders_by_agent.values())
            print(f"    Stored {total_meta} metaorders for {isin}.")
            del trades, trades_np, indices_by_agent, act_by_agent, activity_buffer
            gc.collect()
        self.metaorders_nofilter_path.write_bytes(pickle.dumps(metaorders_dict_all))
        return metaorders_dict_all

    def _load_or_filter_metaorders(
        self,
        parquet_paths: Sequence[Path],
        metaorders_nofilter: Dict[str, Dict[object, List[List[int]]]],
        *,
        rebuild: bool,
    ) -> Dict[str, Dict[object, List[List[int]]]]:
        if self.metaorders_filtered_path.exists() and not rebuild:
            print(f"Loading filtered metaorders from {self.metaorders_filtered_path}.")
            return pickle.loads(self.metaorders_filtered_path.read_bytes())

        print("Applying gap filter to metaorders...")
        max_gap = pd.Timedelta(minutes=self.config.max_gap_minutes)
        filtered: Dict[str, Dict[object, List[List[int]]]] = {}
        for idx, parquet_path in enumerate(parquet_paths, start=1):
            isin = parquet_path.stem
            trades = self._load_filtered_trades(parquet_path)
            meta_by_agent = metaorders_nofilter.get(isin, {})
            new_meta: Dict[object, List[List[int]]] = {}
            for agent_id, metaorders in meta_by_agent.items():
                splitted: List[List[int]] = []
                for meta in metaorders:
                    if len(meta) < self.config.min_children:
                        splitted.append(list(meta))
                        continue
                    times = trades.loc[meta, "Trade Time"].to_numpy()
                    gaps = pd.Series(times[1:] - times[:-1])
                    split_idx = np.where(gaps > max_gap)[0] + 1
                    if split_idx.size == 0:
                        parts = [np.array(meta)]
                    else:
                        parts = np.split(np.array(meta), split_idx)
                    for part in parts:
                        if len(part) >= self.config.min_children:
                            splitted.append([int(i) for i in part])
                if splitted:
                    new_meta[agent_id] = splitted
            filtered[isin] = new_meta
            total_meta = sum(len(v) for v in new_meta.values())
            print(f"({idx}/{len(parquet_paths)}) {isin}: {total_meta} metaorders after gap filter.")
        self.metaorders_filtered_path.write_bytes(pickle.dumps(filtered))
        return filtered

    def _validate_metaorders(
        self,
        parquet_paths: Sequence[Path],
        metaorders_filtered: Mapping[str, Mapping[object, Sequence[Sequence[int]]]],
    ) -> None:
        print("Validating metaorders consistency...")
        issues = 0
        for parquet_path in parquet_paths:
            isin = parquet_path.stem
            trades = self._load_filtered_trades(parquet_path)
            for agent_id, metaorders in metaorders_filtered.get(isin, {}).items():
                for meta in metaorders:
                    idx = np.asarray(meta, dtype=int)
                    subset = trades.iloc[idx]
                    if not (subset["Direction"] == subset["Direction"].iloc[0]).all():
                        print(f"Direction mismatch detected for ISIN {isin}, agent {agent_id}.")
                        issues += 1
                    id_col = "ID Client" if self.config.level == "client" else "ID Member"
                    if not (subset[id_col] == subset[id_col].iloc[0]).all():
                        print(f"{id_col} mismatch detected for ISIN {isin}, agent {agent_id}.")
                        issues += 1
        if issues == 0:
            print("Metaorders consistency checks passed.")

    def _load_or_compute_info(
        self,
        parquet_paths: Sequence[Path],
        metaorders_filtered: Mapping[str, Mapping[object, Sequence[Sequence[int]]]],
        *,
        recompute: bool,
    ) -> Dict[str, Dict[object, Dict[int, Tuple]]]:
        if self.metaorders_info_path.exists() and not recompute:
            print(f"Loading cached metaorders info from {self.metaorders_info_path}.")
            return pickle.loads(self.metaorders_info_path.read_bytes())

        print("Computing metaorders info...")
        metaorders_info: Dict[str, Dict[object, Dict[int, Tuple]]] = {}
        for idx, parquet_path in enumerate(parquet_paths, start=1):
            isin = parquet_path.stem
            trades = self._load_filtered_trades(parquet_path)
            meta_by_agent = metaorders_filtered.get(isin, {})
            print(f"({idx}/{len(parquet_paths)}) {isin}: computing daily stats for {len(meta_by_agent)} agents.")
            if not meta_by_agent:
                continue
            trades = trades.copy()
            trades["__row_id__"] = np.arange(len(trades), dtype=np.int64)
            trades.sort_values(["Trade Time", "__row_id__"], kind="mergesort", inplace=True)
            trades.reset_index(drop=True, inplace=True)
            day_arr = trades["Trade Time"].dt.date.values
            price_last = trades["Price Last Contract"].to_numpy()
            price_first = trades["Price First Contract"].to_numpy()
            direction = trades["Direction"].to_numpy()
            member_id_arr = trades["ID Member"].to_numpy()
            client_id_arr = trades["ID Client"].to_numpy()
            vol = trades["Total Quantity Buy"].to_numpy(dtype=float) + trades["Total Quantity Sell"].to_numpy(dtype=float)
            csum_vol = np.cumsum(vol)
            daily_cache = _build_daily_cache(trades)
            info_for_isin: Dict[object, Dict[int, Tuple]] = {}
            for agent_id in tqdm(meta_by_agent.keys(), desc="    Agents", leave=False):
                meta_list = meta_by_agent[agent_id]
                agent_store: Dict[int, Tuple] = {}
                for meta_idx, idx_list in enumerate(meta_list):
                    idx_array = np.asarray(idx_list, dtype=int)
                    start_idx = idx_array[0]
                    end_idx = idx_array[-1]
                    start_ts = trades.at[start_idx, "Trade Time"]
                    end_ts = trades.at[end_idx, "Trade Time"]
                    meta_volume = float(vol[idx_array].sum())
                    volume_window = _sum_contiguous(csum_vol, start_idx, end_idx + 1)
                    direction_sign = int(direction[end_idx])
                    current_day = day_arr[start_idx]
                    daily_vol, daily_volume = daily_cache.get(current_day, (np.nan, 0.0))
                    delta_price = float(np.log(price_last[end_idx]) - np.log(price_first[start_idx]))
                    q_over_v = float(meta_volume / daily_volume) if daily_volume else np.nan
                    participation = float(meta_volume / volume_window) if volume_window else np.inf
                    agent_store[meta_idx] = (
                        isin,
                        int(member_id_arr[end_idx]),
                        client_id_arr[end_idx],
                        direction_sign,
                        delta_price,
                        daily_vol,
                        meta_volume,
                        q_over_v,
                        participation,
                        len(idx_array),
                        [start_ts, end_ts],
                    )
                if agent_store:
                    info_for_isin[agent_id] = agent_store
            if info_for_isin:
                metaorders_info[isin] = info_for_isin
        self.metaorders_info_path.write_bytes(pickle.dumps(metaorders_info))
        return metaorders_info

    def _metaorders_info_dataframe(
        self,
        metaorders_info: Mapping[str, Mapping[object, Mapping[int, Tuple]]],
    ) -> pd.DataFrame:
        rows: List[Tuple] = []
        for isin_dict in metaorders_info.values():
            for agent_dict in isin_dict.values():
                rows.extend(agent_dict.values())
        columns = (
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
        )
        if not rows:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(rows, columns=columns)
        df["Impact"] = df["Price Change"] * df["Direction"] / df["Daily Vol"]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["Q/V", "Impact"], inplace=True)
        return df

    def _impact_plots(self, df: pd.DataFrame, *, quantile_bins: int) -> None:
        df = df[df["Q/V"] > 0].copy()
        if df.empty:
            print("No positive Q/V values available for plotting.")
            return
        output_prefix = self.output_dir / f"impact_vs_qv_{self.config.name}"
        prt_median = df["Participation Rate"].median()
        subsets = {
            "low_participation": df[df["Participation Rate"] < prt_median],
            "high_participation": df[df["Participation Rate"] >= prt_median],
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, subset in subsets.items():
            if subset.empty:
                continue
            stats, params = _fit_power_law_quantile(subset, q_bins=quantile_bins)
            ax.errorbar(
                stats["mean_QV"],
                stats["mean_imp"],
                yerr=stats["sem_imp"],
                fmt="o" if label == "low_participation" else "s",
                label=label.replace("_", " ").title(),
                alpha=0.7,
            )
            qv_range = np.logspace(np.log10(stats["mean_QV"].min()), np.log10(stats["mean_QV"].max()), 200)
            ax.plot(qv_range, _power_law(qv_range, params[0], params[2]), "--", lw=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Q/V")
        ax.set_ylabel("Impact / sigma")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()
        fig_path = output_prefix.with_name(output_prefix.name + "_by_participation.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        stats, params = _fit_power_law_quantile(df, q_bins=quantile_bins)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(stats["mean_QV"], stats["mean_imp"], yerr=stats["sem_imp"], fmt="o", alpha=0.7)
        qv_range = np.logspace(np.log10(stats["mean_QV"].min()), np.log10(stats["mean_QV"].max()), 200)
        ax.plot(qv_range, _power_law(qv_range, params[0], params[2]), color="red", lw=2)
        ax.set_xlabel("Q/V (mean per bin)")
        ax.set_ylabel("Impact / sigma (mean per bin)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.set_title(f"Impact scaling ({self.config.name})")
        fig.tight_layout()
        fig_path = output_prefix.with_name(output_prefix.name + "_overall.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"Saved impact plots to {fig_path.parent}.")

    def _load_filtered_trades(self, parquet_path: Path) -> pd.DataFrame:
        trades = pd.read_parquet(parquet_path)
        if trades.empty:
            return trades
        trades = trades.copy()
        trades["Trade Time"] = pd.to_datetime(trades["Trade Time"])
        start, end = self.config.trading_hours
        mask_hours = trades["Trade Time"].dt.time.between(pd.to_datetime(start).time(), pd.to_datetime(end).time())
        trades = trades.loc[mask_hours]
        if self.config.trade_filter == "exclude_proprietary":
            trades = trades[trades["Trade Type Aggressive"] != PROPRIETARY_LABEL]
        elif self.config.trade_filter == "only_proprietary":
            trades = trades[trades["Trade Type Aggressive"] == PROPRIETARY_LABEL]
        trades.reset_index(drop=True, inplace=True)
        return trades

    @staticmethod
    def _column_positions(trades: pd.DataFrame) -> Dict[str, int]:
        return {col: trades.columns.get_loc(col) for col in trades.columns}


def _sum_contiguous(cumsum: np.ndarray, start: int, end_exclusive: int) -> float:
    if end_exclusive <= 0:
        return 0.0
    if start <= 0:
        return float(cumsum[end_exclusive - 1])
    return float(cumsum[end_exclusive - 1] - cumsum[start - 1])


def _build_daily_cache(trades: pd.DataFrame) -> Dict[pd.Timestamp, Tuple[float, float]]:
    trades = trades.copy()
    trades["Trade Time"] = pd.to_datetime(trades["Trade Time"])
    trades.sort_values("Trade Time", kind="mergesort", inplace=True)
    cache: Dict[pd.Timestamp, Tuple[float, float]] = {}
    for day, day_df in trades.groupby(trades["Trade Time"].dt.date, sort=True):
        daily_volume = float(day_df[["Total Quantity Buy", "Total Quantity Sell"]].to_numpy().sum())
        price_series = (
            day_df[["Trade Time", "Price Last Contract"]]
            .dropna(subset=["Trade Time"])
            .sort_values("Trade Time", kind="mergesort")
            .drop_duplicates(subset="Trade Time", keep="last")
            .set_index("Trade Time")["Price Last Contract"]
        )
        if price_series.empty:
            cache[day] = (np.nan, daily_volume)
            continue
        resampled = price_series.resample("120s").last().ffill().dropna()
        if resampled.size < 2:
            daily_vol = np.nan
        else:
            returns = np.diff(np.log(resampled.values))
            daily_vol = float(realized_kernel_fast(returns))
        cache[day] = (daily_vol, daily_volume)
    return cache


def preprocess_log_returns(prices: pd.DataFrame, delta: str) -> Dict[pd.Timestamp, np.ndarray]:
    prices = prices.dropna(subset=["Trade Time", "Price Last Contract"]).copy()
    prices["Trade Time"] = pd.to_datetime(prices["Trade Time"])
    prices.set_index("Trade Time", inplace=True)
    grouped = prices.groupby(prices.index.date)
    log_returns: Dict[pd.Timestamp, np.ndarray] = {}
    for day, group in grouped:
        resampled = group["Price Last Contract"].resample(delta).last().dropna()
        if len(resampled) > 1:
            diffs = np.diff(np.log(resampled.values))
            if diffs.size:
                log_returns[day] = diffs
    return log_returns


def realized_variance_fast(log_returns: Mapping[pd.Timestamp, np.ndarray]) -> List[float]:
    return [float(np.sqrt(np.sum(ret ** 2))) for ret in log_returns.values()]


def bipower_variation_fast(log_returns: Mapping[pd.Timestamp, np.ndarray]) -> List[float]:
    out: List[float] = []
    for ret in log_returns.values():
        if ret.size <= 2:
            out.append(np.nan)
            continue
        out.append(float(np.sqrt(np.pi / 2 * np.sum(np.abs(ret[1:]) * np.abs(ret[:-1])))))
    return out


def realized_kernel_fast(log_returns, kernel: str = "parzen", bandwidth: int | None = None) -> float | List[float]:
    def parzen(u: np.ndarray) -> np.ndarray:
        u = np.abs(u)
        weights = np.zeros_like(u, dtype=float)
        mask1 = u <= 0.5
        mask2 = (u > 0.5) & (u <= 1.0)
        weights[mask1] = 1 - 6 * u[mask1] ** 2 + 6 * u[mask1] ** 3
        weights[mask2] = 2 * (1 - u[mask2]) ** 3
        return weights

    def compute_single(ret: np.ndarray) -> float:
        n = ret.size
        if n < 3:
            return float("nan")
        h = int(np.floor(n ** (2 / 3))) if bandwidth is None else bandwidth
        lags = np.arange(-h, h + 1)
        weights = parzen(lags / h) if kernel == "parzen" else np.ones_like(lags, dtype=float)
        rk = 0.0
        for lag, weight in zip(lags, weights):
            if lag >= 0:
                rk += weight * np.dot(ret[lag:], ret[: n - lag])
            else:
                rk += weight * np.dot(ret[: n + lag], ret[-lag:])
        return float(np.sqrt(rk))

    if isinstance(log_returns, dict):
        return [compute_single(np.asarray(ret)) for ret in log_returns.values()]
    return compute_single(np.asarray(log_returns))


def _power_law(qv: np.ndarray, y_coef: float, gamma: float) -> np.ndarray:
    return y_coef * np.power(qv, gamma)


def _fit_power_law_quantile(
    df: pd.DataFrame,
    *,
    q_bins: int,
    initial_guess: Tuple[float, float] = (1.0, 0.5),
) -> Tuple[pd.DataFrame, Tuple[float, float, float, float, float]]:
    df = df[df["Q/V"] > 0].copy()
    if df.empty:
        raise ValueError("No positive Q/V values available for fitting.")
    df["bin"], _ = pd.qcut(df["Q/V"], q=q_bins, labels=False, retbins=True, duplicates="drop")
    grouped = (
        df.groupby("bin")
        .agg(
            mean_QV=("Q/V", "mean"),
            mean_imp=("Impact", "mean"),
            std_imp=("Impact", "std"),
            count=("Impact", "size"),
        )
        .query("count > 1")
    )
    grouped["sem_imp"] = grouped["std_imp"] / np.sqrt(grouped["count"])
    grouped.replace([np.inf, -np.inf], np.nan, inplace=True)
    grouped.dropna(subset=["sem_imp"], inplace=True)
    popt, pcov = curve_fit(
        _power_law,
        grouped["mean_QV"].values,
        grouped["mean_imp"].values,
        sigma=grouped["sem_imp"].values,
        absolute_sigma=True,
        p0=initial_guess,
    )
    y_hat, gamma_hat = popt
    y_err, gamma_err = np.sqrt(np.diag(pcov))
    y_true = grouped["mean_imp"].values
    y_pred = _power_law(grouped["mean_QV"].values, y_hat, gamma_hat)
    r_squared = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return grouped, (y_hat, y_err, gamma_hat, gamma_err, r_squared)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the metaorders pipeline for a given trade perspective.")
    parser.add_argument("mode", choices=CONFIGS.keys(), help="Perspective to analyse.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing raw CSV files.")
    parser.add_argument("--cache-dir", type=Path, help="Directory containing parquet cache files.")
    parser.add_argument("--output-dir", type=Path, help="Directory where intermediate artefacts are stored.")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip CSV to parquet conversion step.")
    parser.add_argument("--rebuild-metaorders", action="store_true", help="Recompute metaorders from scratch.")
    parser.add_argument("--recompute-info", action="store_true", help="Recompute metaorders info even if cached.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip impact plots.")
    parser.add_argument("--quantile-bins", type=int, default=30, help="Number of quantile bins for impact fits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = CONFIGS[args.mode]
    config = base_config
    if args.data_dir:
        config = replace(config, data_dir=args.data_dir)
    if args.cache_dir:
        config = replace(config, cache_dir=args.cache_dir)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    pipeline = MetaordersPipeline(config)
    info_df = pipeline.run(
        skip_conversion=args.skip_conversion,
        rebuild_metaorders=args.rebuild_metaorders,
        recompute_info=args.recompute_info,
        skip_plots=args.skip_plots,
        quantile_bins=args.quantile_bins,
    )
    print(f"Finished pipeline for {config.name}: {len(info_df)} metaorders analysed.")


if __name__ == "__main__":
    main()
