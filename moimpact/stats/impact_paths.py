"""
Bootstrap helpers for normalized impact-path retention comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class RetentionDifferenceBootstrapResult:
    """
    Summary
    -------
    Store proprietary-vs-client retention estimates and bootstrap inference.

    Parameters
    ----------
    tau_start : float
        Reference normalized time in the denominator (for example `1.0`).
    tau_end : float
        Post-execution normalized time in the numerator (for example `3.0`).
    proprietary_retention : float
        Observed proprietary retention ratio
        `mean(I(tau_end)) / mean(I(tau_start))`.
    client_retention : float
        Observed client retention ratio
        `mean(I(tau_end)) / mean(I(tau_start))`.
    delta_retention : float
        Observed difference `proprietary_retention - client_retention`.
    proprietary_ci_low : float
        Lower percentile bootstrap bound for proprietary retention.
    proprietary_ci_high : float
        Upper percentile bootstrap bound for proprietary retention.
    client_ci_low : float
        Lower percentile bootstrap bound for client retention.
    client_ci_high : float
        Upper percentile bootstrap bound for client retention.
    delta_ci_low : float
        Lower percentile bootstrap bound for the retention difference.
    delta_ci_high : float
        Upper percentile bootstrap bound for the retention difference.
    p_value : float
        Two-sided centered-bootstrap p-value for the null
        `delta_retention = 0`.
    alpha : float
        Significance level used for the confidence intervals.
    bootstrap_runs : int
        Requested number of bootstrap replications.
    random_state : int | None
        RNG seed used for the bootstrap draws.
    n_proprietary_metaorders : int
        Number of proprietary metaorders with finite impacts at both target
        times.
    n_client_metaorders : int
        Number of client metaorders with finite impacts at both target times.
    n_proprietary_clusters : int
        Number of distinct proprietary bootstrap clusters.
    n_client_clusters : int
        Number of distinct client bootstrap clusters.
    n_clusters_resampled : int
        Number of unique clusters in the pooled resampling frame.
    n_bootstrap_valid : int
        Number of bootstrap draws that produced finite retention ratios for
        both groups.
    bootstrap_proprietary_retention : np.ndarray
        Finite proprietary bootstrap retentions.
    bootstrap_client_retention : np.ndarray
        Finite client bootstrap retentions.
    bootstrap_delta_retention : np.ndarray
        Finite bootstrap draws of `proprietary_retention - client_retention`.

    Returns
    -------
    RetentionDifferenceBootstrapResult
        Immutable result container.

    Notes
    -----
    The bootstrap resamples clusters (typically trading dates) with
    replacement. Confidence intervals are percentile intervals.
    """

    tau_start: float
    tau_end: float
    proprietary_retention: float
    client_retention: float
    delta_retention: float
    proprietary_ci_low: float
    proprietary_ci_high: float
    client_ci_low: float
    client_ci_high: float
    delta_ci_low: float
    delta_ci_high: float
    p_value: float
    alpha: float
    bootstrap_runs: int
    random_state: Optional[int]
    n_proprietary_metaorders: int
    n_client_metaorders: int
    n_proprietary_clusters: int
    n_client_clusters: int
    n_clusters_resampled: int
    n_bootstrap_valid: int
    bootstrap_proprietary_retention: np.ndarray
    bootstrap_client_retention: np.ndarray
    bootstrap_delta_retention: np.ndarray

    def summary_dict(self) -> dict[str, float | int | None]:
        """
        Summary
        -------
        Convert the scalar part of the bootstrap result to a flat mapping.

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, float | int | None]
            Dictionary suitable for CSV/JSON export.

        Notes
        -----
        The potentially large bootstrap draw arrays are excluded.
        """
        out = asdict(self)
        out.pop("bootstrap_proprietary_retention", None)
        out.pop("bootstrap_client_retention", None)
        out.pop("bootstrap_delta_retention", None)
        return out


def bootstrap_retention_difference(
    proprietary_metaorders: pd.DataFrame,
    client_metaorders: pd.DataFrame,
    *,
    tau_start: float = 1.0,
    tau_end: float = 3.0,
    duration_multiplier: float = 2.0,
    period_col: str = "Period",
    partial_col: str = "partial_impact",
    aftermath_col: str = "aftermath_impact",
    cluster_col: Optional[str] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> RetentionDifferenceBootstrapResult:
    """
    Summary
    -------
    Compare proprietary and client impact-path retention via cluster bootstrap.

    Parameters
    ----------
    proprietary_metaorders : pd.DataFrame
        Proprietary metaorder dataframe containing packed impact-path columns.
    client_metaorders : pd.DataFrame
        Client metaorder dataframe containing packed impact-path columns.
    tau_start : float, default=1.0
        Normalized time used in the denominator of the retention ratio.
    tau_end : float, default=3.0
        Normalized time used in the numerator of the retention ratio.
    duration_multiplier : float, default=2.0
        Aftermath horizon used when the stored path is interpolated. The path is
        assumed to cover the range `[0, 1 + duration_multiplier]`.
    period_col : str, default="Period"
        Column containing metaorder start/end timestamps. Used to derive the
        trading-day cluster label when `cluster_col` is not supplied.
    partial_col : str, default="partial_impact"
        Column with packed in-execution impact paths.
    aftermath_col : str, default="aftermath_impact"
        Column with packed aftermath impact paths.
    cluster_col : str | None, default=None
        Optional existing cluster column. When None, the bootstrap clusters are
        the trading dates extracted from `period_col`.
    alpha : float, default=0.05
        Significance level for two-sided percentile confidence intervals.
    n_bootstrap : int, default=1000
        Number of cluster bootstrap replications.
    random_state : int | None, default=None
        Optional seed for reproducible resampling.

    Returns
    -------
    RetentionDifferenceBootstrapResult
        Point estimates, percentile confidence intervals, and bootstrap draws
        for proprietary retention, client retention, and their difference.

    Notes
    -----
    - The proprietary and client retentions are defined as
      `R = mean(I(tau_end)) / mean(I(tau_start))` on the subset of metaorders
      with finite interpolated impacts at both target times.
    - The bootstrap resamples pooled trading-day clusters with replacement and
      recomputes the difference `R_prop - R_client` on each replicate.
    - If a bootstrap replicate yields a non-finite denominator for either
      group, that replicate is dropped from the reported intervals.

    Examples
    --------
    >>> demo = pd.DataFrame(
    ...     {
    ...         "Period": [[pd.Timestamp("2024-01-02").value, pd.Timestamp("2024-01-02 00:01:00").value]],
    ...         "partial_impact": [np.asarray([1.0, 2.0], dtype=np.float32)],
    ...         "aftermath_impact": [np.asarray([2.0, 1.0], dtype=np.float32)],
    ...     }
    ... )
    >>> result = bootstrap_retention_difference(demo, demo, n_bootstrap=10, random_state=0)
    >>> np.isfinite(result.delta_retention)
    True
    """
    if tau_end <= tau_start:
        raise ValueError("tau_end must be strictly larger than tau_start.")
    if duration_multiplier <= 0.0:
        raise ValueError("duration_multiplier must be positive.")

    proprietary_sample = _prepare_retention_sample(
        proprietary_metaorders,
        tau_start=tau_start,
        tau_end=tau_end,
        duration_multiplier=duration_multiplier,
        period_col=period_col,
        partial_col=partial_col,
        aftermath_col=aftermath_col,
        cluster_col=cluster_col,
    )
    client_sample = _prepare_retention_sample(
        client_metaorders,
        tau_start=tau_start,
        tau_end=tau_end,
        duration_multiplier=duration_multiplier,
        period_col=period_col,
        partial_col=partial_col,
        aftermath_col=aftermath_col,
        cluster_col=cluster_col,
    )

    prop_summary = _aggregate_retention_by_cluster(proprietary_sample)
    client_summary = _aggregate_retention_by_cluster(client_sample)

    proprietary_retention = _retention_from_cluster_sums(prop_summary["sum_start"], prop_summary["sum_end"])
    client_retention = _retention_from_cluster_sums(client_summary["sum_start"], client_summary["sum_end"])
    delta_retention = float(proprietary_retention - client_retention)

    pooled_clusters = pd.Index(
        sorted(set(prop_summary.index.tolist()).union(set(client_summary.index.tolist()))),
        dtype="object",
    )
    n_clusters_resampled = int(len(pooled_clusters))
    if n_clusters_resampled < 2:
        raise ValueError(
            "At least two bootstrap clusters are required to compare retentions. "
            "Check that the selected sample spans multiple trading days."
        )

    prop_start = prop_summary["sum_start"].reindex(pooled_clusters, fill_value=0.0).to_numpy(dtype=float)
    prop_end = prop_summary["sum_end"].reindex(pooled_clusters, fill_value=0.0).to_numpy(dtype=float)
    client_start = client_summary["sum_start"].reindex(pooled_clusters, fill_value=0.0).to_numpy(dtype=float)
    client_end = client_summary["sum_end"].reindex(pooled_clusters, fill_value=0.0).to_numpy(dtype=float)

    boot_prop = np.full(int(n_bootstrap), np.nan, dtype=float)
    boot_client = np.full(int(n_bootstrap), np.nan, dtype=float)
    boot_delta = np.full(int(n_bootstrap), np.nan, dtype=float)
    if int(n_bootstrap) > 0:
        rng = np.random.default_rng(random_state)
        for b in tqdm(range(int(n_bootstrap)), desc="Bootstrap replicates", unit="replicate"):
            sampled = rng.integers(0, n_clusters_resampled, size=n_clusters_resampled)
            freq = np.bincount(sampled, minlength=n_clusters_resampled).astype(float)
            prop_ratio = _retention_from_cluster_sums(freq * prop_start, freq * prop_end)
            client_ratio = _retention_from_cluster_sums(freq * client_start, freq * client_end)
            if np.isfinite(prop_ratio) and np.isfinite(client_ratio):
                boot_prop[b] = prop_ratio
                boot_client[b] = client_ratio
                boot_delta[b] = prop_ratio - client_ratio

    valid_mask = np.isfinite(boot_prop) & np.isfinite(boot_client) & np.isfinite(boot_delta)
    valid_prop = boot_prop[valid_mask]
    valid_client = boot_client[valid_mask]
    valid_delta = boot_delta[valid_mask]

    prop_lo, prop_hi = _percentile_interval(valid_prop, alpha=alpha)
    client_lo, client_hi = _percentile_interval(valid_client, alpha=alpha)
    delta_lo, delta_hi = _percentile_interval(valid_delta, alpha=alpha)
    p_value = _centered_bootstrap_p_value(valid_delta, observed=delta_retention)

    return RetentionDifferenceBootstrapResult(
        tau_start=float(tau_start),
        tau_end=float(tau_end),
        proprietary_retention=float(proprietary_retention),
        client_retention=float(client_retention),
        delta_retention=float(delta_retention),
        proprietary_ci_low=float(prop_lo),
        proprietary_ci_high=float(prop_hi),
        client_ci_low=float(client_lo),
        client_ci_high=float(client_hi),
        delta_ci_low=float(delta_lo),
        delta_ci_high=float(delta_hi),
        p_value=float(p_value),
        alpha=float(alpha),
        bootstrap_runs=int(n_bootstrap),
        random_state=random_state,
        n_proprietary_metaorders=int(len(proprietary_sample)),
        n_client_metaorders=int(len(client_sample)),
        n_proprietary_clusters=int(prop_summary.shape[0]),
        n_client_clusters=int(client_summary.shape[0]),
        n_clusters_resampled=int(n_clusters_resampled),
        n_bootstrap_valid=int(valid_delta.size),
        bootstrap_proprietary_retention=valid_prop,
        bootstrap_client_retention=valid_client,
        bootstrap_delta_retention=valid_delta,
    )


def _prepare_retention_sample(
    metaorders: pd.DataFrame,
    *,
    tau_start: float,
    tau_end: float,
    duration_multiplier: float,
    period_col: str,
    partial_col: str,
    aftermath_col: str,
    cluster_col: Optional[str],
) -> pd.DataFrame:
    required = {period_col, partial_col, aftermath_col}
    missing = required.difference(metaorders.columns)
    if cluster_col is not None and cluster_col not in metaorders.columns:
        missing = set(missing)
        missing.add(cluster_col)
    if missing:
        raise KeyError(f"Missing required columns for retention bootstrap: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    grid = np.asarray([float(tau_start), float(tau_end)], dtype=float)
    for row in metaorders.itertuples(index=False):
        partial = _unpack_path(getattr(row, partial_col))
        aftermath = _unpack_path(getattr(row, aftermath_col))
        interpolated = _interpolate_impact_path(partial, aftermath, grid, duration_multiplier)
        if interpolated is None or interpolated.size != 2:
            continue
        impact_start = float(interpolated[0])
        impact_end = float(interpolated[1])
        if not (np.isfinite(impact_start) and np.isfinite(impact_end)):
            continue

        if cluster_col is None:
            cluster_value = _extract_period_start_date(getattr(row, period_col))
        else:
            cluster_value = getattr(row, cluster_col)
        if cluster_value is None or pd.isna(cluster_value):
            continue

        rows.append(
            {
                "cluster": cluster_value,
                "impact_start": impact_start,
                "impact_end": impact_end,
            }
        )

    if not rows:
        raise ValueError(
            "No valid impact-path observations were found at the requested taus. "
            "Check that the input sample contains packed path columns and a wide enough aftermath horizon."
        )
    return pd.DataFrame(rows)


def _aggregate_retention_by_cluster(sample: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sample.groupby("cluster", sort=False, dropna=False)
        .agg(
            n_metaorders=("impact_start", "size"),
            sum_start=("impact_start", "sum"),
            sum_end=("impact_end", "sum"),
        )
        .sort_index()
    )
    if grouped.empty:
        raise ValueError("No valid clusters remain after preparing the retention sample.")
    return grouped


def _retention_from_cluster_sums(sum_start: Sequence[float], sum_end: Sequence[float]) -> float:
    total_start = float(np.sum(np.asarray(sum_start, dtype=float)))
    total_end = float(np.sum(np.asarray(sum_end, dtype=float)))
    if not np.isfinite(total_start) or abs(total_start) <= 0.0:
        return float("nan")
    if not np.isfinite(total_end):
        return float("nan")
    return float(total_end / total_start)


def _percentile_interval(draws: np.ndarray, *, alpha: float) -> tuple[float, float]:
    if draws.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    )


def _centered_bootstrap_p_value(draws: np.ndarray, *, observed: float) -> float:
    if draws.size == 0 or not np.isfinite(observed):
        return float("nan")
    centered = draws - observed
    exceedances = int(np.count_nonzero(np.abs(centered) >= abs(observed)))
    # Add-one correction avoids reporting p=0 with a finite number of bootstrap draws.
    return float((exceedances + 1) / (draws.size + 1))


def _unpack_path(blob: Optional[bytes | bytearray | memoryview | list[float] | np.ndarray]) -> Optional[np.ndarray]:
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return np.frombuffer(blob, dtype=np.float32)
    return np.asarray(blob, dtype=np.float32)


def _interpolate_impact_path(
    partial: Optional[Iterable[float]],
    aftermath: Optional[Iterable[float]],
    time_grid: np.ndarray,
    duration_multiplier: float,
) -> Optional[np.ndarray]:
    times: list[float] = []
    values: list[float] = []

    if partial is not None:
        partial_arr = np.asarray(list(partial), dtype=float).ravel()
        if partial_arr.size > 0:
            times.extend(np.linspace(0.0, 1.0, partial_arr.size))
            values.extend(partial_arr.tolist())
    if aftermath is not None:
        aftermath_arr = np.asarray(list(aftermath), dtype=float).ravel()
        if aftermath_arr.size > 0:
            times.extend(1.0 + np.linspace(0.0, duration_multiplier, aftermath_arr.size))
            values.extend(aftermath_arr.tolist())

    if not times or not values:
        return None

    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    if int(mask.sum()) < 2:
        return None

    t = t[mask]
    v = v[mask]
    order = np.argsort(t)
    t_sorted = t[order]
    v_sorted = v[order]
    dedup_idx = np.concatenate(([0], np.where(np.diff(t_sorted) > 0)[0] + 1))
    t_unique = t_sorted[dedup_idx]
    v_unique = v_sorted[dedup_idx]
    if t_unique.size < 2:
        return None
    if float(time_grid.min()) < float(t_unique.min()) or float(time_grid.max()) > float(t_unique.max()):
        return None
    return np.interp(time_grid, t_unique, v_unique)


def _extract_period_start_date(period: object) -> Optional[pd.Timestamp]:
    if period is None:
        return None
    try:
        start, _ = period
    except Exception:
        return None

    start_ts: Optional[pd.Timestamp]
    try:
        start_val = int(start)
    except Exception:
        try:
            start_ts = pd.Timestamp(start)
        except Exception:
            return None
    else:
        start_ts = pd.Timestamp(start_val)

    if start_ts is None or pd.isna(start_ts):
        return None
    return pd.Timestamp(start_ts).normalize()
