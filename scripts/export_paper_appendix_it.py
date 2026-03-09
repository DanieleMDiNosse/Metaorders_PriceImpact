#!/usr/bin/env python3
"""
Export appendix artifacts for Italian executing-member metaorders.

This script computes the distributional summaries for the subsample of metaorders
executed by Italian members (member nationality tag = ``it``) and writes:

- a CSV summary under ``out_files/{DATASET_NAME}/paper_appendix/``
- a LaTeX table under ``paper/tables/`` (to be included in the manuscript appendix)

The definitions match those used in ``scripts/metaorder_statistics.py``:
- metaorders are read from the metaorder-index dictionaries (pickle)
- trade-level quantities are taken from the per-ISIN tapes in ``data/parquet``
- durations and inter-arrivals are computed from child-trade timestamps
- Q is the sum of aggressive buy+sell quantities within the metaorder
- Q/V uses the full (unfiltered-by-capacity) daily volume
- participation rate uses the full (unfiltered-by-capacity) volume over the execution window

Notes
-----
- The inter-arrival distribution is defined over gaps between consecutive child
  trades, so its sample size is the number of gaps (not the number of metaorders).
- The script assumes that the nationality filter used to create the input
  metaorder dictionaries matches the `member_nationality` argument so that the
  stored indices align with the filtered trade tape ordering.

Examples
--------
From the repository root:

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate main
    python scripts/export_paper_appendix_it.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


AGGRESSIVE_MEMBER_NATIONALITY_COL = "Aggressive Member Nationality"


@dataclass(frozen=True)
class DistSummary:
    """Summary statistics for a 1D sample."""

    n: int
    mean: float
    median: float
    p25: float
    p75: float

    @property
    def iqr(self) -> float:
        """Interquartile range (p75 - p25)."""

        return float(self.p75 - self.p25)


@dataclass(frozen=True)
class MetaorderStats:
    """Distribution summaries + key counts for a metaorder group."""

    n_metaorders: int
    n_buy: int
    n_sell: int
    n_tie: int
    duration_sec: DistSummary
    interarrival_sec: DistSummary
    volume_q: DistSummary
    q_over_v: DistSummary
    participation_rate: DistSummary


def _summarize(values: Iterable[float]) -> DistSummary:
    """
    Summary
    -------
    Compute mean/median/quantiles for a list-like of floats.

    Parameters
    ----------
    values
        Sample values.

    Returns
    -------
    DistSummary
        Summary statistics. For an empty sample, fields are NaN and n=0.

    Notes
    -----
    Uses `np.quantile` for (0.25, 0.75). No winsorization is applied.
    """

    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        nan = float("nan")
        return DistSummary(n=0, mean=nan, median=nan, p25=nan, p75=nan)

    p25, p75 = np.quantile(arr, [0.25, 0.75])
    return DistSummary(
        n=int(arr.size),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p25=float(p25),
        p75=float(p75),
    )


def load_trades_filtered_for_stats(
    path: Path,
    proprietary: Optional[bool],
    trading_hours: Tuple[str, str],
    member_nationality: Optional[str],
) -> pd.DataFrame:
    """
    Summary
    -------
    Load a per-ISIN trade tape and apply the basic filters used for the paper's
    metaorder statistics.

    Parameters
    ----------
    path
        Parquet path for one ISIN trade tape.
    proprietary
        If True, keep only proprietary aggressive trades.
        If False, keep only non-proprietary aggressive trades.
        If None, keep all aggressive trades (no capacity filter).
    trading_hours
        Inclusive filter on `Trade Time` as ("HH:MM:SS", "HH:MM:SS").
    member_nationality
        Optional filter on `Aggressive Member Nationality` (e.g. "it" or "foreign").

    Returns
    -------
    pd.DataFrame
        Filtered tape, stably sorted by time, with a `__row_id__` helper column.

    Notes
    -----
    Stable sorting is required because the metaorder dictionaries store integer
    indices referring to the ordering used at construction time.
    """

    trades = pd.read_parquet(path)
    if proprietary is True:
        trades = trades[trades["Trade Type Aggressive"] == "Dealing_on_own_account"].copy()
    elif proprietary is False:
        trades = trades[trades["Trade Type Aggressive"] != "Dealing_on_own_account"].copy()

    if member_nationality is not None:
        nat = trades[AGGRESSIVE_MEMBER_NATIONALITY_COL].astype("string").str.strip().str.lower()
        trades = trades.loc[nat.eq(member_nationality).fillna(False)].copy()

    start, end = trading_hours
    trades = trades[
        (trades["Trade Time"].dt.time >= pd.to_datetime(start).time())
        & (trades["Trade Time"].dt.time <= pd.to_datetime(end).time())
    ].copy()

    trades = trades.reset_index(drop=True)
    trades["__row_id__"] = np.arange(len(trades), dtype=np.int64)
    trades.sort_values(["Trade Time", "__row_id__"], kind="mergesort", inplace=True)
    trades.reset_index(drop=True, inplace=True)
    return trades


def _volume_over_window(
    ts_ns: np.ndarray, csum_vol: np.ndarray, start_ns: np.int64, end_ns: np.int64
) -> float:
    """
    Summary
    -------
    Compute total traded volume between two timestamps using a prefix sum.

    Parameters
    ----------
    ts_ns
        Sorted timestamps in nanoseconds.
    csum_vol
        Cumulative sum of volume aligned with `ts_ns`.
    start_ns, end_ns
        Window endpoints in nanoseconds.

    Returns
    -------
    float
        Total volume in the inclusive window.

    Notes
    -----
    The implementation mirrors the helper in `scripts/metaorder_statistics.py`.
    """

    start_idx = np.searchsorted(ts_ns, start_ns, side="left")
    end_idx = np.searchsorted(ts_ns, end_ns, side="right") - 1
    if end_idx < start_idx or end_idx < 0 or start_idx >= ts_ns.size:
        return 0.0
    prev = csum_vol[start_idx - 1] if start_idx > 0 else 0.0
    return float(csum_vol[end_idx] - prev)


def compute_metaorder_stats(
    *,
    metaorders_dict_path: Path,
    parquet_dir: Path,
    proprietary: bool,
    trading_hours: Tuple[str, str],
    member_nationality: str,
) -> MetaorderStats:
    """
    Summary
    -------
    Compute metaorder-level distribution summaries for a given capacity.

    Parameters
    ----------
    metaorders_dict_path
        Pickle path to the `metaorders_dict_all_*` object.
    parquet_dir
        Directory with per-ISIN trade tapes (`*.parquet`).
    proprietary
        Whether to filter the trade tape to proprietary (True) or non-proprietary (False).
    trading_hours
        Inclusive filter on `Trade Time`.
    member_nationality
        Member nationality filter, expected to match the one used to build the dictionary.

    Returns
    -------
    MetaorderStats
        Counts + distribution summaries (duration, inter-arrival, Q, Q/V, eta).

    Notes
    -----
    - Duration and inter-arrivals are computed on the child-trade timestamps.
    - Q/V and participation rate are computed using the full (capacity-unfiltered)
      tape to define the reference volumes.
    """

    with metaorders_dict_path.open("rb") as f:
        metaorders_dict_all = pickle.load(f)

    durations_sec: List[float] = []
    inter_arrivals_sec: List[float] = []
    meta_volumes: List[float] = []
    q_over_v: List[float] = []
    participation_rates: List[float] = []

    n_buy = 0
    n_sell = 0
    n_tie = 0

    parquet_paths = sorted(parquet_dir.glob("*.parquet"))
    for tape_path in parquet_paths:
        isin = tape_path.stem
        metaorders_dict = metaorders_dict_all.get(isin)
        if not metaorders_dict:
            continue

        trades_full = load_trades_filtered_for_stats(
            tape_path,
            proprietary=None,
            trading_hours=trading_hours,
            member_nationality=None,
        )
        trades = load_trades_filtered_for_stats(
            tape_path,
            proprietary=proprietary,
            trading_hours=trading_hours,
            member_nationality=member_nationality,
        )

        times = trades["Trade Time"].to_numpy()
        q_buy = trades["Total Quantity Buy"].to_numpy(dtype=float)
        q_sell = trades["Total Quantity Sell"].to_numpy(dtype=float)

        times_full_ns = trades_full["Trade Time"].to_numpy("int64")
        vol_full = (
            trades_full["Total Quantity Buy"].to_numpy(dtype=float)
            + trades_full["Total Quantity Sell"].to_numpy(dtype=float)
        )
        csum_vol_full = np.cumsum(vol_full)

        daily_vols: Dict[dt.date, float] = (
            trades_full.groupby(trades_full["Trade Time"].dt.date)[["Total Quantity Buy", "Total Quantity Sell"]]
            .sum()
            .sum(axis=1)
            .to_dict()
        )

        for metas in metaorders_dict.values():
            for meta in metas:
                if not meta:
                    continue

                start_idx, end_idx = meta[0], meta[-1]
                meta_indices = np.asarray(meta, dtype=np.int64)

                start_time_np = times[start_idx]
                end_time_np = times[end_idx]
                dur_seconds = (end_time_np - start_time_np) / np.timedelta64(1, "s")
                durations_sec.append(float(dur_seconds))

                buy_qty = float(q_buy[meta_indices].sum())
                sell_qty = float(q_sell[meta_indices].sum())
                vol_q = buy_qty + sell_qty
                meta_volumes.append(vol_q)

                # Dominant aggressive-side quantity within the metaorder.
                if buy_qty > sell_qty:
                    n_buy += 1
                elif sell_qty > buy_qty:
                    n_sell += 1
                else:
                    n_tie += 1

                start_date = pd.Timestamp(start_time_np).date()
                day_volume = float(daily_vols.get(start_date, 0.0))
                if day_volume != 0.0:
                    q_over_v.append(float(vol_q / day_volume))

                start_ns = np.int64(pd.Timestamp(start_time_np).value)
                end_ns = np.int64(pd.Timestamp(end_time_np).value)
                slice_volume = _volume_over_window(times_full_ns, csum_vol_full, start_ns, end_ns)
                if slice_volume != 0.0:
                    participation_rates.append(float(vol_q / slice_volume))

                if meta_indices.size > 1:
                    meta_times = times[meta_indices]
                    diffs = (meta_times[1:] - meta_times[:-1]) / np.timedelta64(1, "s")
                    inter_arrivals_sec.extend([float(x) for x in diffs])

    n_metaorders = int(n_buy + n_sell + n_tie)
    return MetaorderStats(
        n_metaorders=n_metaorders,
        n_buy=int(n_buy),
        n_sell=int(n_sell),
        n_tie=int(n_tie),
        duration_sec=_summarize(durations_sec),
        interarrival_sec=_summarize(inter_arrivals_sec),
        volume_q=_summarize(meta_volumes),
        q_over_v=_summarize(q_over_v),
        participation_rate=_summarize(participation_rates),
    )


def _format_triplet(mean: float, median: float, iqr: float, decimals: int) -> str:
    """
    Summary
    -------
    Format `mean / median / IQR` with a fixed number of decimals.

    Parameters
    ----------
    mean, median, iqr
        Summary stats.
    decimals
        Decimal digits.

    Returns
    -------
    str
        Formatted string.
    """

    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} / {fmt.format(median)} / {fmt.format(iqr)}"


def write_latex_table(
    *,
    out_path: Path,
    stats_prop: MetaorderStats,
    stats_client: MetaorderStats,
    generated_by: str,
    summary_csv_path: Path,
) -> None:
    """
    Summary
    -------
    Write the appendix LaTeX table for Italian-member metaorder statistics.

    Parameters
    ----------
    out_path
        Output `.tex` path under `paper/`.
    stats_prop
        Proprietary (IT) stats.
    stats_client
        Client (IT) stats.
    generated_by
        Script identifier to store in a comment header.
    summary_csv_path
        Path to the CSV summary (traceability reference).

    Returns
    -------
    None
    """

    # Convert seconds -> minutes for duration and inter-arrival tables (as in paper figures).
    dur_prop_min = (stats_prop.duration_sec.mean / 60.0, stats_prop.duration_sec.median / 60.0, stats_prop.duration_sec.iqr / 60.0)
    dur_client_min = (stats_client.duration_sec.mean / 60.0, stats_client.duration_sec.median / 60.0, stats_client.duration_sec.iqr / 60.0)

    ia_prop_min = (stats_prop.interarrival_sec.mean / 60.0, stats_prop.interarrival_sec.median / 60.0, stats_prop.interarrival_sec.iqr / 60.0)
    ia_client_min = (stats_client.interarrival_sec.mean / 60.0, stats_client.interarrival_sec.median / 60.0, stats_client.interarrival_sec.iqr / 60.0)

    q_prop = (stats_prop.volume_q.mean, stats_prop.volume_q.median, stats_prop.volume_q.iqr)
    q_client = (stats_client.volume_q.mean, stats_client.volume_q.median, stats_client.volume_q.iqr)

    qv_prop = (stats_prop.q_over_v.mean, stats_prop.q_over_v.median, stats_prop.q_over_v.iqr)
    qv_client = (stats_client.q_over_v.mean, stats_client.q_over_v.median, stats_client.q_over_v.iqr)

    eta_prop = (
        stats_prop.participation_rate.mean,
        stats_prop.participation_rate.median,
        stats_prop.participation_rate.iqr,
    )
    eta_client = (
        stats_client.participation_rate.mean,
        stats_client.participation_rate.median,
        stats_client.participation_rate.iqr,
    )

    latex = f"""% AUTO-GENERATED FILE. DO NOT EDIT BY HAND.
% Generated by: {generated_by}
% Generated on: {dt.datetime.now().isoformat(timespec='seconds')}
% Source summary CSV: {summary_csv_path.as_posix()}
\n\\begin{{table}}[t]
  \\centering
  \\small
  \\begin{{tabular}}{{lcc}}
    \\toprule
    Quantity & Proprietary (IT) & Client (IT) \\\\
    \\midrule
    Metaorders (N) & {stats_prop.n_metaorders} & {stats_client.n_metaorders} \\\\
    Buys / sells & {stats_prop.n_buy} / {stats_prop.n_sell} & {stats_client.n_buy} / {stats_client.n_sell} \\\\
    Duration $T$ (min): mean / median / IQR & {_format_triplet(*dur_prop_min, decimals=2)} & {_format_triplet(*dur_client_min, decimals=2)} \\\\
    Inter-arrival $\\Delta t$ (min): mean / median / IQR & {_format_triplet(*ia_prop_min, decimals=2)} (N={stats_prop.interarrival_sec.n}) & {_format_triplet(*ia_client_min, decimals=2)} (N={stats_client.interarrival_sec.n}) \\\\
    Volume $Q$ (shares): mean / median / IQR & {_format_triplet(*q_prop, decimals=0)} & {_format_triplet(*q_client, decimals=0)} \\\\
    Relative size $Q/V$: mean / median / IQR & {_format_triplet(*qv_prop, decimals=4)} & {_format_triplet(*qv_client, decimals=4)} \\\\
    Participation rate $\\eta$: mean / median / IQR & {_format_triplet(*eta_prop, decimals=4)} & {_format_triplet(*eta_client, decimals=4)} \\\\
    \\bottomrule
  \\end{{tabular}}
  \\caption{{Summary statistics for member-level metaorders executed by Italian members (member nationality tag = IT). Values are mean / median / interquartile range (IQR). Inter-arrival times are computed across child-trade gaps (sample size N is the number of gaps).}}
  \\label{{tab:appendix_it_metaorder_stats}}
\\end{{table}}
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")


def build_summary_frame(stats: MetaorderStats, flow: str) -> pd.DataFrame:
    """
    Summary
    -------
    Convert a `MetaorderStats` object to a tidy DataFrame.

    Parameters
    ----------
    stats
        Computed statistics.
    flow
        Flow tag (e.g. "proprietary_it" or "client_it").

    Returns
    -------
    pd.DataFrame
        Tidy table with one row per metric.
    """

    rows: List[dict] = []
    rows.append({"flow": flow, "metric": "metaorders", "unit": "count", "n": stats.n_metaorders})
    rows.append({"flow": flow, "metric": "buy_metaorders", "unit": "count", "n": stats.n_buy})
    rows.append({"flow": flow, "metric": "sell_metaorders", "unit": "count", "n": stats.n_sell})
    rows.append({"flow": flow, "metric": "tie_metaorders", "unit": "count", "n": stats.n_tie})

    def add_dist(metric: str, unit: str, dist: DistSummary) -> None:
        rows.append(
            {
                "flow": flow,
                "metric": metric,
                "unit": unit,
                "n": dist.n,
                "mean": dist.mean,
                "median": dist.median,
                "p25": dist.p25,
                "p75": dist.p75,
                "iqr": dist.iqr,
            }
        )

    add_dist("duration_sec", "s", stats.duration_sec)
    add_dist("interarrival_sec", "s", stats.interarrival_sec)
    add_dist("volume_q", "shares", stats.volume_q)
    add_dist("q_over_v", "ratio", stats.q_over_v)
    add_dist("participation_rate", "ratio", stats.participation_rate)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-name", default="ftsemib")
    p.add_argument("--member-nationality", default="it", choices=["it", "foreign"])
    p.add_argument("--parquet-dir", default="data/parquet")
    p.add_argument("--trading-hours-start", default="09:30:00")
    p.add_argument("--trading-hours-end", default="17:30:00")
    p.add_argument("--paper-table-path", default="paper/tables/appendix_it_metaorder_stats.tex")
    return p.parse_args()


def main() -> None:
    """Entry point."""

    args = parse_args()
    dataset_name = str(args.dataset_name)
    member_nationality = str(args.member_nationality).strip().lower()
    trading_hours = (str(args.trading_hours_start), str(args.trading_hours_end))

    parquet_dir = Path(args.parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    out_dir = Path("out_files") / dataset_name / "paper_appendix"
    out_dir.mkdir(parents=True, exist_ok=True)

    metaorders_prop_path = (
        Path("out_files")
        / dataset_name
        / f"metaorders_dict_all_member_proprietary_member_nationality_{member_nationality}.pkl"
    )
    metaorders_client_path = (
        Path("out_files")
        / dataset_name
        / f"metaorders_dict_all_member_non_proprietary_member_nationality_{member_nationality}.pkl"
    )
    if not metaorders_prop_path.exists():
        raise FileNotFoundError(f"Metaorder dict not found: {metaorders_prop_path}")
    if not metaorders_client_path.exists():
        raise FileNotFoundError(f"Metaorder dict not found: {metaorders_client_path}")

    stats_prop = compute_metaorder_stats(
        metaorders_dict_path=metaorders_prop_path,
        parquet_dir=parquet_dir,
        proprietary=True,
        trading_hours=trading_hours,
        member_nationality=member_nationality,
    )
    stats_client = compute_metaorder_stats(
        metaorders_dict_path=metaorders_client_path,
        parquet_dir=parquet_dir,
        proprietary=False,
        trading_hours=trading_hours,
        member_nationality=member_nationality,
    )

    summary = pd.concat(
        [
            build_summary_frame(stats_prop, flow=f"proprietary_{member_nationality}"),
            build_summary_frame(stats_client, flow=f"client_{member_nationality}"),
        ],
        ignore_index=True,
    )
    summary_csv_path = out_dir / "appendix_it_metaorder_stats_summary.csv"
    summary.to_csv(summary_csv_path, index=False)

    manifest = {
        "generated_on": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "member_nationality": member_nationality,
        "trading_hours": {"start": trading_hours[0], "end": trading_hours[1]},
        "inputs": {
            "metaorders_dict_proprietary": metaorders_prop_path.as_posix(),
            "metaorders_dict_client": metaorders_client_path.as_posix(),
            "parquet_dir": parquet_dir.as_posix(),
        },
        "outputs": {
            "summary_csv": summary_csv_path.as_posix(),
            "paper_table_tex": str(args.paper_table_path),
        },
        "result_counts": {
            "proprietary": {
                "metaorders": stats_prop.n_metaorders,
                "buy": stats_prop.n_buy,
                "sell": stats_prop.n_sell,
                "tie": stats_prop.n_tie,
            },
            "client": {
                "metaorders": stats_client.n_metaorders,
                "buy": stats_client.n_buy,
                "sell": stats_client.n_sell,
                "tie": stats_client.n_tie,
            },
        },
    }
    (out_dir / "appendix_it_metaorder_stats_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    write_latex_table(
        out_path=Path(args.paper_table_path),
        stats_prop=stats_prop,
        stats_client=stats_client,
        generated_by="scripts/export_paper_appendix_it.py",
        summary_csv_path=summary_csv_path,
    )


if __name__ == "__main__":
    main()

