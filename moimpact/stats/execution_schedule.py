"""
Cluster-bootstrap helpers for scalar execution-schedule comparisons.

The helpers in this module compare proprietary vs client execution schedules
through scalar features derived from each metaorder's cumulative child-trade
path while respecting date-level dependence.

The implementation is designed for speed:

- child-trade paths are decoded once
- scalar features are extracted once per metaorder
- bootstrap replicates resample pooled clusters and aggregate precomputed
  cluster-level histograms to approximate the group medians
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from moimpact.execution_typology import build_cumulative_schedule_curve, unpack_float32_path


GROUP_NAMES = ("proprietary", "client")
_GROUP_TO_CODE = {name: idx for idx, name in enumerate(GROUP_NAMES)}
SCALAR_METRIC_NAMES = (
    "front25_share",
    "front50_share",
    "time_to_25",
    "time_to_50",
    "center_of_mass",
    "twap_l1_distance",
)


@dataclass(frozen=True)
class PreparedExecutionScheduleSample:
    """Prepared per-metaorder scalar features reused across inference routines."""

    scalar_values: np.ndarray
    scalar_metric_names: tuple[str, ...]
    group_codes: np.ndarray
    cluster_codes: np.ndarray
    cluster_labels: np.ndarray
    n_input_rows: Mapping[str, int]
    n_valid_metaorders: Mapping[str, int]
    skipped_reasons: Mapping[str, Mapping[str, int]]


def _period_start_date(period: object) -> Optional[pd.Timestamp]:
    if period is None:
        return None
    try:
        start, _ = period
    except Exception:
        return None
    try:
        start_value = int(start)
    except Exception:
        try:
            start_ts = pd.Timestamp(start)
        except Exception:
            return None
    else:
        start_ts = pd.Timestamp(start_value)
    if start_ts is None or pd.isna(start_ts):
        return None
    return pd.Timestamp(start_ts).normalize()


def _resolve_cluster_values(
    frame: pd.DataFrame,
    *,
    cluster_col: Optional[str],
) -> np.ndarray:
    if cluster_col is None:
        if "Period" not in frame.columns:
            raise KeyError("Execution-schedule inference requires a 'Period' column when cluster_col is None.")
        cluster_values = frame["Period"].map(_period_start_date)
    else:
        if cluster_col not in frame.columns:
            raise KeyError(f"Execution-schedule inference missing cluster column: {cluster_col}")
        cluster_values = frame[cluster_col]
    return np.asarray(cluster_values.to_numpy(), dtype=object)


def _curve_time_to_threshold(
    curve: np.ndarray,
    tau_grid: np.ndarray,
    *,
    threshold: float,
) -> float:
    curve_arr = np.asarray(curve, dtype=float)
    tau_arr = np.asarray(tau_grid, dtype=float)
    if curve_arr.ndim != 1 or tau_arr.ndim != 1 or curve_arr.size != tau_arr.size:
        return float("nan")
    if not np.all(np.isfinite(curve_arr)) or not np.all(np.isfinite(tau_arr)):
        return float("nan")
    if threshold <= float(curve_arr[0]):
        return float(tau_arr[0])
    hit = np.flatnonzero(curve_arr >= threshold)
    if hit.size == 0:
        return float("nan")
    idx = int(hit[0])
    if idx == 0:
        return float(tau_arr[0])
    left_y = float(curve_arr[idx - 1])
    right_y = float(curve_arr[idx])
    left_x = float(tau_arr[idx - 1])
    right_x = float(tau_arr[idx])
    if right_y <= left_y:
        return right_x
    frac = (float(threshold) - left_y) / (right_y - left_y)
    frac = float(np.clip(frac, 0.0, 1.0))
    return left_x + frac * (right_x - left_x)


def _extract_scalar_metrics(
    child_time_norm: np.ndarray,
    child_volume_fraction: np.ndarray,
    curve: np.ndarray,
    tau_grid: np.ndarray,
) -> np.ndarray:
    time_arr = np.asarray(child_time_norm, dtype=float).ravel()
    volume_arr = np.asarray(child_volume_fraction, dtype=float).ravel()
    volume_sum = float(volume_arr.sum())
    if volume_sum <= 0.0 or not np.isfinite(volume_sum):
        return np.full(len(SCALAR_METRIC_NAMES), np.nan, dtype=np.float32)
    time_arr = np.clip(time_arr, 0.0, 1.0)
    volume_arr = volume_arr / volume_sum
    return np.asarray(
        [
            float(volume_arr[time_arr <= 0.25].sum()),
            float(volume_arr[time_arr <= 0.50].sum()),
            _curve_time_to_threshold(curve, tau_grid, threshold=0.25),
            _curve_time_to_threshold(curve, tau_grid, threshold=0.50),
            float(np.sum(time_arr * volume_arr)),
            float(np.trapezoid(np.abs(np.asarray(curve, dtype=float) - tau_grid), tau_grid)),
        ],
        dtype=np.float32,
    )


def prepare_execution_schedule_sample(
    proprietary_metaorders: pd.DataFrame,
    client_metaorders: pd.DataFrame,
    *,
    n_time_grid: int,
    n_heatmap_bins_y: int = 0,
    cluster_col: Optional[str] = None,
) -> PreparedExecutionScheduleSample:
    """
    Decode packed child paths and prepare reusable scalar features for inference.
    """
    del n_heatmap_bins_y
    n_time_grid = int(n_time_grid)
    if n_time_grid < 2:
        raise ValueError("n_time_grid must be at least 2.")

    tau_grid = np.linspace(0.0, 1.0, n_time_grid, dtype=float)
    scalar_rows: list[np.ndarray] = []
    group_codes: list[int] = []
    cluster_labels: list[object] = []
    n_input_rows = {
        "proprietary": int(len(proprietary_metaorders)),
        "client": int(len(client_metaorders)),
    }
    n_valid_metaorders = {name: 0 for name in GROUP_NAMES}
    skipped_reasons = {name: Counter() for name in GROUP_NAMES}

    for group_name, frame in (
        ("proprietary", proprietary_metaorders),
        ("client", client_metaorders),
    ):
        missing = {"child_time_norm", "child_volume_fraction"}.difference(frame.columns)
        if missing:
            raise KeyError(
                f"Execution-schedule inference missing required columns for {group_name}: {sorted(missing)}"
            )
        cluster_values = _resolve_cluster_values(frame, cluster_col=cluster_col)
        rows = frame[["child_time_norm", "child_volume_fraction"]].itertuples(index=False, name=None)
        for (time_blob, volume_blob), cluster_value in zip(rows, cluster_values, strict=True):
            if pd.isna(cluster_value):
                skipped_reasons[group_name]["missing_cluster"] += 1
                continue
            child_time = unpack_float32_path(time_blob)
            child_volume = unpack_float32_path(volume_blob)
            curve, reason = build_cumulative_schedule_curve(
                child_time,
                child_volume,
                n_grid=n_time_grid,
            )
            if curve is None:
                skipped_reasons[group_name][str(reason or "invalid_schedule")] += 1
                continue

            scalar_values_arr = _extract_scalar_metrics(
                np.asarray(child_time, dtype=float),
                np.asarray(child_volume, dtype=float),
                np.asarray(curve, dtype=float),
                tau_grid,
            )
            if not np.all(np.isfinite(scalar_values_arr)):
                skipped_reasons[group_name]["invalid_scalar_metrics"] += 1
                continue

            scalar_rows.append(scalar_values_arr.astype(np.float32, copy=False))
            group_codes.append(_GROUP_TO_CODE[group_name])
            cluster_labels.append(cluster_value)
            n_valid_metaorders[group_name] += 1

    if not scalar_rows:
        raise ValueError("No valid execution schedules were available for inference.")

    cluster_codes, unique_clusters = pd.factorize(np.asarray(cluster_labels, dtype=object), sort=True)
    return PreparedExecutionScheduleSample(
        scalar_values=np.vstack(scalar_rows).astype(np.float32, copy=False),
        scalar_metric_names=tuple(SCALAR_METRIC_NAMES),
        group_codes=np.asarray(group_codes, dtype=np.int8),
        cluster_codes=np.asarray(cluster_codes, dtype=np.int32),
        cluster_labels=np.asarray(unique_clusters, dtype=object),
        n_input_rows={str(k): int(v) for k, v in n_input_rows.items()},
        n_valid_metaorders={str(k): int(v) for k, v in n_valid_metaorders.items()},
        skipped_reasons={
            str(group): {str(key): int(value) for key, value in sorted(counter.items())}
            for group, counter in skipped_reasons.items()
        },
    )


def _aggregate_cluster_scalar_histograms(
    prepared: PreparedExecutionScheduleSample,
    *,
    n_histogram_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_groups = len(GROUP_NAMES)
    n_clusters = int(prepared.cluster_labels.size)
    n_metrics = int(prepared.scalar_values.shape[1])
    group_cluster_codes = prepared.group_codes.astype(np.int64) * n_clusters + prepared.cluster_codes.astype(np.int64)
    bin_edges = np.linspace(0.0, 1.0, int(n_histogram_bins) + 1, dtype=np.float64)

    values = np.clip(prepared.scalar_values.astype(np.float64, copy=False), 0.0, 1.0)
    bin_idx = np.searchsorted(bin_edges, values, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, int(n_histogram_bins) - 1).astype(np.int16, copy=False)
    metric_idx = np.broadcast_to(np.arange(n_metrics, dtype=np.int64), bin_idx.shape)
    gc_idx = np.broadcast_to(group_cluster_codes[:, None], bin_idx.shape)

    hist_counts = np.zeros((n_groups * n_clusters, n_metrics, int(n_histogram_bins)), dtype=np.uint32)
    np.add.at(
        hist_counts,
        (
            gc_idx.ravel(order="C"),
            metric_idx.ravel(order="C"),
            bin_idx.ravel(order="C"),
        ),
        1,
    )
    return hist_counts.reshape(n_groups, n_clusters, n_metrics, int(n_histogram_bins)), bin_edges


def _histogram_quantiles_last_axis(
    histogram_batch: np.ndarray,
    bin_edges: np.ndarray,
    *,
    quantile: float,
) -> np.ndarray:
    counts = np.asarray(histogram_batch, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    squeeze = False
    if counts.ndim == 2:
        counts = counts[None, :, :]
        squeeze = True
    if counts.ndim != 3:
        raise ValueError("histogram_batch must have shape (batch, metric, bins) or (metric, bins).")
    if edges.ndim != 1 or edges.size != counts.shape[-1] + 1:
        raise ValueError("bin_edges must have length n_bins + 1.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")

    totals = counts.sum(axis=-1)
    cdf = np.cumsum(counts, axis=-1)
    target = quantile * totals
    hit_idx = np.argmax(cdf >= target[..., None], axis=-1)

    left = edges[hit_idx]
    right = edges[hit_idx + 1]
    prev_idx = np.clip(hit_idx - 1, 0, counts.shape[-1] - 1)
    prev_mass = np.take_along_axis(cdf, prev_idx[..., None], axis=-1)[..., 0]
    prev_mass = np.where(hit_idx > 0, prev_mass, 0.0)
    bin_mass = np.take_along_axis(counts, hit_idx[..., None], axis=-1)[..., 0]

    frac = np.divide(
        target - prev_mass,
        bin_mass,
        out=np.full_like(target, 0.5, dtype=np.float64),
        where=bin_mass > 0.0,
    )
    frac = np.clip(frac, 0.0, 1.0)
    quantiles = left + frac * (right - left)
    quantiles[totals <= 0.0] = np.nan
    return quantiles[0] if squeeze else quantiles


def _bootstrap_weight_batch(
    rng: np.random.Generator,
    *,
    n_replicates: int,
    n_clusters: int,
) -> np.ndarray:
    draws = rng.integers(0, n_clusters, size=(n_replicates, n_clusters), endpoint=False)
    weights = np.zeros((n_replicates, n_clusters), dtype=np.int16 if n_clusters < 32_000 else np.int32)
    rows = np.repeat(np.arange(n_replicates, dtype=np.int64), n_clusters)
    np.add.at(weights, (rows, draws.ravel(order="C")), 1)
    return weights.astype(np.float64, copy=False)


def infer_execution_schedule_scalar_summaries(
    prepared: PreparedExecutionScheduleSample,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
    batch_size: int = 256,
    n_histogram_bins: int = 1024,
) -> pd.DataFrame:
    """
    Compare scalar schedule features via pooled-cluster bootstrap of group medians.
    """
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")
    batch_size = max(int(batch_size), 1)
    n_histogram_bins = int(n_histogram_bins)
    if n_histogram_bins < 2:
        raise ValueError("n_histogram_bins must be >= 2.")

    n_clusters = int(prepared.cluster_labels.size)
    if n_clusters < 2:
        raise ValueError(
            "At least two bootstrap clusters are required for execution-schedule inference."
        )

    prop_idx = _GROUP_TO_CODE["proprietary"]
    client_idx = _GROUP_TO_CODE["client"]
    prop_mask = prepared.group_codes == prop_idx
    client_mask = prepared.group_codes == client_idx
    proprietary_value = np.median(prepared.scalar_values[prop_mask].astype(np.float64, copy=False), axis=0)
    client_value = np.median(prepared.scalar_values[client_mask].astype(np.float64, copy=False), axis=0)
    delta_value = proprietary_value - client_value

    hist_counts, bin_edges = _aggregate_cluster_scalar_histograms(
        prepared,
        n_histogram_bins=n_histogram_bins,
    )
    bootstrap_delta = np.full((n_bootstrap, prepared.scalar_values.shape[1]), np.nan, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    cursor = 0
    while cursor < n_bootstrap:
        stop = min(cursor + batch_size, n_bootstrap)
        weights = _bootstrap_weight_batch(
            rng,
            n_replicates=stop - cursor,
            n_clusters=n_clusters,
        )
        prop_hist = np.tensordot(weights, hist_counts[prop_idx].astype(np.float64, copy=False), axes=(1, 0))
        client_hist = np.tensordot(weights, hist_counts[client_idx].astype(np.float64, copy=False), axes=(1, 0))
        prop_medians = _histogram_quantiles_last_axis(prop_hist, bin_edges, quantile=0.5)
        client_medians = _histogram_quantiles_last_axis(client_hist, bin_edges, quantile=0.5)
        bootstrap_delta[cursor:stop, :] = prop_medians - client_medians
        cursor = stop

    rows: list[dict[str, float | int | str | None]] = []
    for metric_idx, metric_name in enumerate(prepared.scalar_metric_names):
        valid = bootstrap_delta[np.isfinite(bootstrap_delta[:, metric_idx]), metric_idx]
        if valid.size == 0:
            ci_low = float("nan")
            ci_high = float("nan")
            p_value = float("nan")
            bootstrap_valid_runs = 0
        else:
            ci_low = float(np.quantile(valid, alpha / 2.0))
            ci_high = float(np.quantile(valid, 1.0 - alpha / 2.0))
            centered = valid - float(delta_value[metric_idx])
            p_value = float(np.mean(np.abs(centered) >= abs(float(delta_value[metric_idx]))))
            bootstrap_valid_runs = int(valid.size)
        rows.append(
            {
                "metric": metric_name,
                "proprietary_value": float(proprietary_value[metric_idx]),
                "client_value": float(client_value[metric_idx]),
                "delta_value": float(delta_value[metric_idx]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "alpha": float(alpha),
                "bootstrap_runs": int(n_bootstrap),
                "bootstrap_valid_runs": int(bootstrap_valid_runs),
                "summary_stat": "median",
                "random_state": float("nan") if random_state is None else int(random_state),
                "histogram_bins": int(n_histogram_bins),
                "n_proprietary_metaorders": int(prepared.n_valid_metaorders["proprietary"]),
                "n_client_metaorders": int(prepared.n_valid_metaorders["client"]),
                "n_clusters": int(n_clusters),
            }
        )
    return pd.DataFrame(rows)
