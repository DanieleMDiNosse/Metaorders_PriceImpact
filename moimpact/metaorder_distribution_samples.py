"""
Canonical trade filtering and sample extraction for metaorder summary/distribution scripts.
"""

from __future__ import annotations

import gc
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

AGGRESSIVE_MEMBER_NATIONALITY_COL = "Aggressive Member Nationality"


@dataclass(frozen=True)
class MetaorderDistributionSamples:
    """Container for the aggregated samples used across metaorder summary and distribution figures."""

    total_metaorders: int
    counts_by_member: Dict[object, int]
    durations_minutes: np.ndarray
    inter_arrivals_minutes: np.ndarray
    meta_volumes: np.ndarray
    q_over_v: np.ndarray
    participation_rates: np.ndarray
    n_buy_metaorders: int
    n_sell_metaorders: int
    n_tie_metaorders: int
    trade_volume_by_nationality: Dict[str, float]
    unknown_trade_volume: float
    nationality_counts: Dict[str, int]
    metaorder_volume_by_nationality: Dict[str, float]
    unknown_nationality_metaorders: int
    unknown_metaorder_volume: float
    mixed_nationality_metaorders: int
    mixed_metaorder_volume: float


def parse_member_nationality(value: Optional[object]) -> Optional[str]:
    """
    Summary
    -------
    Normalize a member-nationality filter to ``"it"``, ``"foreign"``, or ``None``.

    Parameters
    ----------
    value : Optional[object]
        Raw config or CLI value describing the desired aggressive-member
        nationality filter.

    Returns
    -------
    Optional[str]
        ``"it"`` or ``"foreign"`` when a filter is active; otherwise ``None``.

    Notes
    -----
    Empty strings and the labels ``none``, ``null``, and ``all`` disable the
    filter.

    Examples
    --------
    >>> parse_member_nationality("IT")
    'it'
    >>> parse_member_nationality("all") is None
    True
    """
    if value is None:
        return None
    nationality = str(value).strip().lower()
    if nationality in {"", "none", "null", "all"}:
        return None
    if nationality not in {"it", "foreign"}:
        raise ValueError(
            "Invalid MEMBER_NATIONALITY value. Use one of: 'it', 'foreign', or null/all for no filter."
        )
    return nationality


def with_member_nationality_tag(filename: str, member_nationality: Optional[str]) -> str:
    """
    Summary
    -------
    Append the canonical member-nationality suffix to a filename when needed.

    Parameters
    ----------
    filename : str
        Filename or stem plus extension.
    member_nationality : Optional[str]
        Active member-nationality filter, expected to be ``"it"``,
        ``"foreign"``, or ``None``.

    Returns
    -------
    str
        Input filename unchanged when no filter is active, otherwise suffixed
        with ``_member_nationality_{value}`` before the extension.

    Notes
    -----
    This mirrors the repository naming convention used by
    ``metaorder_summary_statistics.py``, ``metaorder_distributions.py``, and
    ``metaorder_computation.py``.

    Examples
    --------
    >>> with_member_nationality_tag("figure.png", None)
    'figure.png'
    >>> with_member_nationality_tag("figure.png", "it")
    'figure_member_nationality_it.png'
    """
    normalized = parse_member_nationality(member_nationality)
    if normalized is None:
        return filename
    stem, ext = Path(filename).stem, Path(filename).suffix
    return f"{stem}_member_nationality_{normalized}{ext}"


def list_metaorder_parquet_paths(data_dir: Path) -> List[Path]:
    """
    Summary
    -------
    List per-ISIN parquet trade tapes in a directory.

    Parameters
    ----------
    data_dir : Path
        Directory expected to contain one ``*.parquet`` file per ISIN.

    Returns
    -------
    List[Path]
        Sorted list of parquet files in ``data_dir``.

    Notes
    -----
    The search is non-recursive and ignores non-parquet files.

    Examples
    --------
    >>> isinstance(list_metaorder_parquet_paths(Path("data/parquet")), list)
    True
    """
    if not data_dir.exists():
        return []
    return [path for path in sorted(data_dir.iterdir()) if path.suffix.lower() == ".parquet"]


def load_trades_filtered_for_stats(
    path: Path,
    proprietary: Optional[bool],
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
    member_nationality: Optional[str] = None,
    members_nationality_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Summary
    -------
    Load one per-ISIN trade tape and apply the canonical statistics filters.

    Parameters
    ----------
    path : Path
        Parquet path for one ISIN trade tape.
    proprietary : Optional[bool]
        If True, keep only proprietary aggressive trades; if False, keep only
        client aggressive trades; if None, keep all trades.
    trading_hours : Tuple[str, str], default=("09:30:00", "17:30:00")
        Inclusive trading-hours filter applied on ``Trade Time``.
    member_nationality : Optional[str], default=None
        Optional aggressive-member nationality filter.
    members_nationality_path : Optional[Path], default=None
        Optional metadata parquet used to backfill
        ``Aggressive Member Nationality`` from ``ID Member`` when needed.

    Returns
    -------
    pd.DataFrame
        Filtered trade tape with stable row ordering and an ``ISIN`` column.

    Notes
    -----
    A stable ``__row_id__`` column is added before mergesort to keep the metaorder
    index references aligned with the saved dictionaries.

    Examples
    --------
    >>> # Example is illustrative and requires a real parquet file on disk.
    >>> isinstance(path := Path("data/parquet"), Path)
    True
    """
    trades = pd.read_parquet(path)
    return _filter_trades_for_stats(
        trades,
        proprietary=proprietary,
        trading_hours=trading_hours,
        member_nationality=member_nationality,
        isin=path.stem,
        members_nationality_path=members_nationality_path,
    )


def collect_metaorder_distribution_samples(
    metaorders_dict_path: Path,
    parquet_dir: Path,
    proprietary: bool,
    trading_hours: Tuple[str, str] = ("09:30:00", "17:30:00"),
    member_nationality: Optional[str] = None,
    members_nationality_path: Optional[Path] = None,
    include_counts_by_member: bool = True,
    show_progress: bool = True,
) -> MetaorderDistributionSamples:
    """
    Summary
    -------
    Recompute the metaorder-level samples used by the distribution statistics.

    Parameters
    ----------
    metaorders_dict_path : Path
        Pickle path to the ``metaorders_dict_all_*`` object produced by
        ``metaorder_computation.py``.
    parquet_dir : Path
        Directory containing the per-ISIN parquet trade tapes.
    proprietary : bool
        Whether the metaorder dictionary corresponds to proprietary flow
        (`True`) or client flow (`False`).
    trading_hours : Tuple[str, str], default=("09:30:00", "17:30:00")
        Inclusive trading-hours filter applied to numerator and denominator
        trade tapes.
    member_nationality : Optional[str], default=None
        Optional aggressive-member nationality filter. This must match the
        filter used when the input dictionary was generated.
    members_nationality_path : Optional[Path], default=None
        Optional metadata parquet used to infer aggressive-member nationality
        from ``ID Member`` when the tape does not already contain that column.
    include_counts_by_member : bool, default=True
        If True, aggregate the number of detected metaorders for each member.
        Set this to False when the caller only needs metaorder-level samples.
    show_progress : bool, default=True
        If True, display a ``tqdm`` progress bar across ISIN files.

    Returns
    -------
    MetaorderDistributionSamples
        Aggregated arrays and counts needed by the existing distribution plots.

    Notes
    -----
    The logic intentionally mirrors the sample construction in
    ``scripts/metaorder_summary_statistics.py`` and
    ``scripts/metaorder_distributions.py``:
    durations and inter-arrivals are returned in minutes, while ``Q``, ``Q/V``,
    and participation rate are returned in their raw units. When
    ``include_counts_by_member`` is False, the returned `counts_by_member`
    mapping is empty.

    Examples
    --------
    >>> # Example is illustrative and requires real repository outputs on disk.
    >>> isinstance(metaorders_dict_path := Path("out_files"), Path)
    True
    """
    if not metaorders_dict_path.exists():
        raise FileNotFoundError(f"Metaorder dictionary not found: {metaorders_dict_path}")

    parquet_paths = list_metaorder_parquet_paths(parquet_dir)
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    normalized_member_nationality = parse_member_nationality(member_nationality)
    with metaorders_dict_path.open("rb") as handle:
        metaorders_dict_all = pickle.load(handle)

    total_metaorders = 0
    counts_by_member: Dict[object, int] = {}
    durations_minutes: List[float] = []
    inter_arrivals_minutes: List[float] = []
    meta_volumes: List[float] = []
    q_over_v: List[float] = []
    participation_rates: List[float] = []
    n_buy_metaorders = 0
    n_sell_metaorders = 0
    n_tie_metaorders = 0
    trade_volume_by_nationality: Dict[str, float] = {"it": 0.0, "foreign": 0.0}
    unknown_trade_volume = 0.0
    nationality_counts: Dict[str, int] = {"it": 0, "foreign": 0}
    metaorder_volume_by_nationality: Dict[str, float] = {"it": 0.0, "foreign": 0.0}
    unknown_nationality_metaorders = 0
    unknown_metaorder_volume = 0.0
    mixed_nationality_metaorders = 0
    mixed_metaorder_volume = 0.0

    iterator = tqdm(parquet_paths, desc=f"ISINs Proprietary: {proprietary}", disable=not show_progress)
    for path in iterator:
        trades_full = load_trades_filtered_for_stats(
            path,
            proprietary=None,
            trading_hours=trading_hours,
            members_nationality_path=members_nationality_path,
        )
        trades = load_trades_filtered_for_stats(
            path,
            proprietary=proprietary,
            trading_hours=trading_hours,
            member_nationality=normalized_member_nationality,
            members_nationality_path=members_nationality_path,
        )

        times = trades["Trade Time"].to_numpy()
        q_buy = trades["Total Quantity Buy"].to_numpy(dtype=float)
        q_sell = trades["Total Quantity Sell"].to_numpy(dtype=float)
        nationality_arr = trades[AGGRESSIVE_MEMBER_NATIONALITY_COL].to_numpy()
        trade_volume_arr = q_buy + q_sell

        nationality_series = pd.Series(nationality_arr, dtype="object")
        for nationality in ("it", "foreign"):
            trade_volume_by_nationality[nationality] += float(trade_volume_arr[nationality_series.eq(nationality)].sum())
        unknown_trade_volume += float(trade_volume_arr[~nationality_series.isin(["it", "foreign"]).to_numpy()].sum())

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

        if include_counts_by_member:
            for member, metas in metaorders_dict.items():
                counts_by_member[member] = counts_by_member.get(member, 0) + len(metas)

        for metas in metaorders_dict.values():
            for meta in metas:
                if not meta:
                    continue
                meta_indices = np.asarray(meta, dtype=np.int64)
                if int(meta_indices.min()) < 0 or int(meta_indices.max()) >= len(trades):
                    raise IndexError(
                        f"Metaorder indices for ISIN {path.stem} fall outside the filtered trade tape. "
                        "Check that the dictionary and the selected filters match."
                    )

                start_idx = int(meta_indices[0])
                end_idx = int(meta_indices[-1])

                meta_nat, is_mixed = _infer_metaorder_nationality(nationality_arr[meta_indices])
                if meta_nat in nationality_counts:
                    nationality_counts[meta_nat] += 1
                else:
                    unknown_nationality_metaorders += 1
                if is_mixed:
                    mixed_nationality_metaorders += 1

                start_time_np = times[start_idx]
                end_time_np = times[end_idx]

                dur_minutes = (end_time_np - start_time_np) / np.timedelta64(1, "m")
                durations_minutes.append(float(dur_minutes))

                buy_qty = float(q_buy[meta_indices].sum())
                sell_qty = float(q_sell[meta_indices].sum())
                volume = buy_qty + sell_qty
                meta_volumes.append(volume)
                if meta_nat in metaorder_volume_by_nationality:
                    metaorder_volume_by_nationality[meta_nat] += volume
                else:
                    unknown_metaorder_volume += volume
                if is_mixed:
                    mixed_metaorder_volume += volume

                # Use the dominant aggressive-side quantity to infer the metaorder sign.
                if buy_qty > sell_qty:
                    n_buy_metaorders += 1
                elif sell_qty > buy_qty:
                    n_sell_metaorders += 1
                else:
                    n_tie_metaorders += 1

                start_date = pd.Timestamp(start_time_np).date()
                day_volume = daily_vols.get(start_date, 0.0)
                if day_volume != 0:
                    q_over_v.append(float(volume / day_volume))

                start_ns = np.int64(pd.Timestamp(start_time_np).value)
                end_ns = np.int64(pd.Timestamp(end_time_np).value)
                slice_volume = _volume_over_window(times_full_ns, csum_vol_full, start_ns, end_ns)
                if slice_volume != 0:
                    participation_rates.append(float(volume / slice_volume))

                if len(meta_indices) > 1:
                    meta_times = times[meta_indices]
                    diffs_minutes = (meta_times[1:] - meta_times[:-1]) / np.timedelta64(1, "m")
                    inter_arrivals_minutes.extend(diffs_minutes.tolist())

        del trades_full
        del trades
        gc.collect()

    return MetaorderDistributionSamples(
        total_metaorders=total_metaorders,
        counts_by_member=counts_by_member,
        durations_minutes=np.asarray(durations_minutes, dtype=float),
        inter_arrivals_minutes=np.asarray(inter_arrivals_minutes, dtype=float),
        meta_volumes=np.asarray(meta_volumes, dtype=float),
        q_over_v=np.asarray(q_over_v, dtype=float),
        participation_rates=np.asarray(participation_rates, dtype=float),
        n_buy_metaorders=n_buy_metaorders,
        n_sell_metaorders=n_sell_metaorders,
        n_tie_metaorders=n_tie_metaorders,
        trade_volume_by_nationality=trade_volume_by_nationality,
        unknown_trade_volume=float(unknown_trade_volume),
        nationality_counts=nationality_counts,
        metaorder_volume_by_nationality=metaorder_volume_by_nationality,
        unknown_nationality_metaorders=unknown_nationality_metaorders,
        unknown_metaorder_volume=float(unknown_metaorder_volume),
        mixed_nationality_metaorders=mixed_nationality_metaorders,
        mixed_metaorder_volume=float(mixed_metaorder_volume),
    )

def _default_members_nationality_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "members_nationality.parquet"


@lru_cache(maxsize=4)
def _load_members_nationality_map_cached(path_str: str) -> pd.Series:
    path = Path(path_str)
    if not path.exists():
        return pd.Series(dtype="string")
    df = pd.read_parquet(path, columns=["FIRM_ID_MODIF", "NAZIONALITA"]).copy()
    df = df.dropna(subset=["FIRM_ID_MODIF"])
    df["FIRM_ID_MODIF"] = pd.to_numeric(df["FIRM_ID_MODIF"], errors="coerce")
    df = df.dropna(subset=["FIRM_ID_MODIF"]).copy()
    df["FIRM_ID_MODIF"] = df["FIRM_ID_MODIF"].astype(np.int64)
    df["NAZIONALITA"] = df["NAZIONALITA"].map(_normalize_member_nationality)
    df = df.drop_duplicates(subset=["FIRM_ID_MODIF"], keep="first")
    return df.set_index("FIRM_ID_MODIF")["NAZIONALITA"]


def _normalize_member_nationality(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    label = str(value).strip().lower()
    if label in {"it", "italy", "italian"}:
        return "it"
    if label in {"foreign"}:
        return "foreign"
    return None


def _ensure_aggressive_member_nationality(
    trades: pd.DataFrame,
    members_nationality_path: Optional[Path],
) -> pd.DataFrame:
    if AGGRESSIVE_MEMBER_NATIONALITY_COL in trades.columns:
        out = trades.copy()
        out[AGGRESSIVE_MEMBER_NATIONALITY_COL] = out[AGGRESSIVE_MEMBER_NATIONALITY_COL].map(
            _normalize_member_nationality
        )
        return out

    out = trades.copy()
    if "ID Member" not in out.columns:
        out[AGGRESSIVE_MEMBER_NATIONALITY_COL] = pd.Series([None] * len(out), index=out.index, dtype="object")
        return out

    metadata_path = members_nationality_path or _default_members_nationality_path()
    nationality_map = _load_members_nationality_map_cached(str(metadata_path.resolve()))
    member_ids = pd.to_numeric(out["ID Member"], errors="coerce")
    out[AGGRESSIVE_MEMBER_NATIONALITY_COL] = member_ids.map(nationality_map)
    return out


def _infer_metaorder_nationality(labels: Sequence[object]) -> Tuple[Optional[str], bool]:
    norm = pd.Series(labels).map(_normalize_member_nationality).dropna()
    if norm.empty:
        return None, False
    counts = norm.value_counts()
    top_count = counts.max()
    tied = sorted(counts[counts == top_count].index.tolist())
    inferred = tied[0]
    is_mixed = len(counts) > 1
    return inferred, is_mixed


def _filter_trades_for_stats(
    trades: pd.DataFrame,
    proprietary: Optional[bool],
    trading_hours: Optional[Tuple[str, str]],
    member_nationality: Optional[str],
    isin: Optional[str],
    members_nationality_path: Optional[Path],
) -> pd.DataFrame:
    trades = _ensure_aggressive_member_nationality(trades, members_nationality_path)
    if proprietary is True:
        trades = trades[trades["Trade Type Aggressive"] == "Dealing_on_own_account"].copy()
    elif proprietary is False:
        trades = trades[trades["Trade Type Aggressive"] != "Dealing_on_own_account"].copy()

    if member_nationality is not None:
        nat = trades[AGGRESSIVE_MEMBER_NATIONALITY_COL].astype("string").str.strip().str.lower()
        trades = trades.loc[nat.eq(member_nationality).fillna(False)].copy()

    if trading_hours is not None:
        start, end = trading_hours
        trades = trades[
            (trades["Trade Time"].dt.time >= pd.to_datetime(start).time())
            & (trades["Trade Time"].dt.time <= pd.to_datetime(end).time())
        ].copy()

    trades = trades.reset_index(drop=True)
    trades["__row_id__"] = np.arange(len(trades), dtype=np.int64)
    trades.sort_values(["Trade Time", "__row_id__"], kind="mergesort", inplace=True)
    trades.reset_index(drop=True, inplace=True)
    if isin is not None:
        trades["ISIN"] = str(isin)
    return trades


def _volume_over_window(ts_ns: np.ndarray, csum_vol: np.ndarray, start_ns: np.int64, end_ns: np.int64) -> float:
    start_idx = np.searchsorted(ts_ns, start_ns, side="left")
    end_idx = np.searchsorted(ts_ns, end_ns, side="right") - 1
    if end_idx < start_idx or end_idx < 0 or start_idx >= ts_ns.size:
        return 0.0
    prev = csum_vol[start_idx - 1] if start_idx > 0 else 0.0
    return float(csum_vol[end_idx] - prev)
