"""
Helpers for pooled execution-typology analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


_EPS_FLOOR = 1.0e-12


@dataclass(frozen=True)
class PathFeatureSpec:
    """Feature-extraction settings for packed execution and impact paths."""

    twap_grid_size: int = 21


def unpack_float32_path(
    blob: Optional[bytes | bytearray | memoryview | Sequence[float] | np.ndarray],
) -> Optional[np.ndarray]:
    """Decode the packed float32 arrays written by `metaorder_computation.py`."""
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return np.frombuffer(blob, dtype=np.float32)
    return np.asarray(blob, dtype=np.float32)


def period_duration_seconds(period: Any) -> float:
    """Compute metaorder duration from a `[start_ns, end_ns]` period pair."""
    try:
        start_ns, end_ns = period
        return float((int(end_ns) - int(start_ns)) * 1.0e-9)
    except Exception:
        return float("nan")


def _safe_abs_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= _EPS_FLOOR:
        return float("nan")
    return float(abs(num) / abs(den))


def _curve_twap_l1_distance(curve: np.ndarray, tau_grid: np.ndarray) -> float:
    twap = tau_grid
    return float(np.trapezoid(np.abs(curve - twap), tau_grid))


def build_cumulative_schedule_curve(
    child_time_norm: Optional[np.ndarray],
    child_volume_fraction: Optional[np.ndarray],
    *,
    n_grid: int,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    if child_time_norm is None or child_volume_fraction is None:
        return None, "missing_schedule"

    time_arr = np.asarray(child_time_norm, dtype=float).ravel()
    vol_arr = np.asarray(child_volume_fraction, dtype=float).ravel()
    if time_arr.size == 0 or vol_arr.size == 0:
        return None, "empty_schedule"
    if time_arr.size != vol_arr.size:
        return None, "length_mismatch"
    if not np.all(np.isfinite(time_arr)) or not np.all(np.isfinite(vol_arr)):
        return None, "non_finite_values"
    if np.any(vol_arr < 0.0):
        return None, "negative_volumes"

    vol_sum = float(vol_arr.sum())
    if not np.isfinite(vol_sum) or vol_sum <= 0.0:
        return None, "nonpositive_volume_sum"

    time_arr = np.clip(time_arr, 0.0, 1.0)
    vol_arr = vol_arr / vol_sum
    order = np.argsort(time_arr, kind="mergesort")
    time_sorted = time_arr[order]
    vol_sorted = vol_arr[order]
    cum_sorted = np.cumsum(vol_sorted)
    if not np.all(np.isfinite(cum_sorted)):
        return None, "invalid_cumsum"

    x = np.concatenate(([0.0], time_sorted, [1.0]))
    y = np.concatenate(([0.0], cum_sorted, [1.0]))
    y = np.clip(y, 0.0, 1.0)
    keep_last = np.concatenate((x[1:] > x[:-1], [True]))
    x_unique = x[keep_last]
    y_unique = y[keep_last]
    if x_unique.size < 2:
        return None, "insufficient_unique_times"

    tau_grid = np.linspace(0.0, 1.0, int(n_grid), dtype=float)
    curve = np.interp(tau_grid, x_unique, y_unique)
    if not np.all(np.isfinite(curve)):
        return None, "invalid_interpolation"
    curve[0] = 0.0
    curve[-1] = 1.0
    return curve.astype(np.float32, copy=False), None


def extract_schedule_features(
    child_time_norm: Optional[np.ndarray],
    child_volume_fraction: Optional[np.ndarray],
    *,
    spec: Optional[PathFeatureSpec] = None,
) -> dict[str, float]:
    """Engineer schedule-shape features from normalized child paths."""
    spec = spec or PathFeatureSpec()
    nan_features = {
        "schedule_front25_share": float("nan"),
        "schedule_front50_share": float("nan"),
        "schedule_back25_share": float("nan"),
        "schedule_center_of_mass": float("nan"),
        "schedule_hhi": float("nan"),
        "schedule_twap_l1_distance": float("nan"),
    }
    if child_time_norm is None or child_volume_fraction is None:
        return nan_features

    time_arr = np.asarray(child_time_norm, dtype=float).ravel()
    vol_arr = np.asarray(child_volume_fraction, dtype=float).ravel()
    if (
        time_arr.size == 0
        or vol_arr.size == 0
        or time_arr.size != vol_arr.size
        or not np.all(np.isfinite(time_arr))
        or not np.all(np.isfinite(vol_arr))
        or np.any(vol_arr < 0.0)
    ):
        return nan_features

    vol_sum = float(vol_arr.sum())
    if not np.isfinite(vol_sum) or vol_sum <= 0.0:
        return nan_features

    time_arr = np.clip(time_arr, 0.0, 1.0)
    vol_arr = vol_arr / vol_sum
    curve, _ = build_cumulative_schedule_curve(time_arr, vol_arr, n_grid=spec.twap_grid_size)
    if curve is None:
        return nan_features

    tau_grid = np.linspace(0.0, 1.0, int(spec.twap_grid_size), dtype=float)
    return {
        "schedule_front25_share": float(vol_arr[time_arr <= 0.25].sum()),
        "schedule_front50_share": float(vol_arr[time_arr <= 0.50].sum()),
        "schedule_back25_share": float(vol_arr[time_arr >= 0.75].sum()),
        "schedule_center_of_mass": float(np.sum(time_arr * vol_arr)),
        "schedule_hhi": float(np.sum(np.square(vol_arr))),
        "schedule_twap_l1_distance": _curve_twap_l1_distance(curve.astype(float), tau_grid),
    }


def extract_impact_shape_features(
    *,
    impact_end: float,
    impact_1m: float,
    impact_10m: float,
    impact_30m: float,
    impact_60m: float,
    partial_impact: Optional[np.ndarray],
) -> dict[str, float]:
    """Engineer impact-shape features from scalar and packed impact paths."""
    partial_arr = (
        np.asarray(partial_impact, dtype=float).ravel()
        if partial_impact is not None
        else np.empty(0, dtype=float)
    )
    if partial_arr.size == 0 or not np.any(np.isfinite(partial_arr)):
        peak_abs_partial = float("nan")
        overshoot = float("nan")
    else:
        finite_partial = partial_arr[np.isfinite(partial_arr)]
        peak_abs_partial = float(np.max(np.abs(finite_partial)))
        overshoot = _safe_abs_ratio(peak_abs_partial, impact_end)

    return {
        "abs_impact_end": float(abs(impact_end)) if np.isfinite(impact_end) else float("nan"),
        "abs_impact_1m": float(abs(impact_1m)) if np.isfinite(impact_1m) else float("nan"),
        "abs_impact_10m": float(abs(impact_10m)) if np.isfinite(impact_10m) else float("nan"),
        "abs_impact_30m": float(abs(impact_30m)) if np.isfinite(impact_30m) else float("nan"),
        "abs_impact_60m": float(abs(impact_60m)) if np.isfinite(impact_60m) else float("nan"),
        "retention_30_over_end": _safe_abs_ratio(impact_30m, impact_end),
        "retention_60_over_end": _safe_abs_ratio(impact_60m, impact_end),
        "peak_abs_partial_impact": peak_abs_partial,
        "overshoot_peak_over_end": overshoot,
    }


def engineer_path_features_frame(
    frame: pd.DataFrame,
    *,
    spec: Optional[PathFeatureSpec] = None,
) -> pd.DataFrame:
    """Extract schedule and impact-shape features for one dataframe chunk."""
    spec = spec or PathFeatureSpec()
    rows: list[dict[str, float]] = []
    for row in frame.itertuples(index=False):
        child_time = unpack_float32_path(getattr(row, "child_time_norm", None))
        child_volume = unpack_float32_path(getattr(row, "child_volume_fraction", None))
        partial = unpack_float32_path(getattr(row, "partial_impact", None))

        schedule_features = extract_schedule_features(
            child_time,
            child_volume,
            spec=spec,
        )
        impact_features = extract_impact_shape_features(
            impact_end=float(getattr(row, "Impact", np.nan)),
            impact_1m=float(getattr(row, "Impact_1m", np.nan)),
            impact_10m=float(getattr(row, "Impact_10m", np.nan)),
            impact_30m=float(getattr(row, "Impact_30m", np.nan)),
            impact_60m=float(getattr(row, "Impact_60m", np.nan)),
            partial_impact=partial,
        )
        rows.append({**schedule_features, **impact_features})
    return pd.DataFrame(rows, index=frame.index)


def aggregate_schedule_profiles_chunk(
    frame: pd.DataFrame,
    *,
    tau_grid: np.ndarray,
    group_col: str = "Group",
    cluster_col: str = "Cluster",
) -> dict[tuple[str, int], dict[str, Any]]:
    """Aggregate cumulative schedule profile statistics for one dataframe chunk."""
    n_grid = int(tau_grid.size)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    cols = [group_col, cluster_col, "child_time_norm", "child_volume_fraction"]
    for row in frame[cols].itertuples(index=False, name=None):
        group_value, cluster_value, time_blob, volume_blob = row
        key = (str(group_value), int(cluster_value))
        state = out.setdefault(
            key,
            {
                "sum_curve": np.zeros(n_grid, dtype=float),
                "sumsq_curve": np.zeros(n_grid, dtype=float),
                "count_curve": np.zeros(n_grid, dtype=float),
                "n_valid_metaorders": 0,
                "skipped_reasons": {},
            },
        )
        curve, reason = build_cumulative_schedule_curve(
            unpack_float32_path(time_blob),
            unpack_float32_path(volume_blob),
            n_grid=n_grid,
        )
        if curve is None:
            reason_key = str(reason or "invalid_schedule")
            skipped = state["skipped_reasons"]
            skipped[reason_key] = int(skipped.get(reason_key, 0)) + 1
            continue
        curve_f = np.asarray(curve, dtype=float)
        state["sum_curve"] += curve_f
        state["sumsq_curve"] += np.square(curve_f)
        state["count_curve"] += 1.0
        state["n_valid_metaorders"] = int(state["n_valid_metaorders"]) + 1
    return out


def build_cluster_label_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ranking metrics used to order and label execution types."""
    out = summary_df.copy()
    urgency_source = np.log1p(pd.to_numeric(out.get("Participation Rate_median"), errors="coerce"))
    urgency_source += np.log1p(pd.to_numeric(out.get("Q/V_median"), errors="coerce"))
    urgency_source -= np.log1p(pd.to_numeric(out.get("DurationSeconds_median"), errors="coerce"))
    front_source = pd.to_numeric(out.get("schedule_front25_share_median"), errors="coerce")
    front_source -= pd.to_numeric(out.get("schedule_back25_share_median"), errors="coerce")
    persistence_source = pd.to_numeric(out.get("retention_60_over_end_median"), errors="coerce")

    out["urgency_score"] = _zscore_series(urgency_source)
    out["front_loading_score"] = _zscore_series(front_source)
    out["persistence_score"] = _zscore_series(persistence_source)
    return out


def assign_auto_type_labels(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic cluster codes and hybrid-readable labels."""
    out = build_cluster_label_metrics(summary_df)
    labels: list[str] = []
    codes: list[str] = []
    for row in out.itertuples(index=False):
        urgency_bucket = _bucket_from_score(float(getattr(row, "urgency_score")))
        front_bucket = _bucket_from_score(float(getattr(row, "front_loading_score")))
        persistence_bucket = _bucket_from_score(float(getattr(row, "persistence_score")))
        code = f"{urgency_bucket}_{front_bucket}_{persistence_bucket}"
        label = " ".join(
            [
                _render_bucket_phrase(urgency_bucket, "urgency"),
                _render_bucket_phrase(front_bucket, "schedule"),
                _render_bucket_phrase(persistence_bucket, "impact"),
            ]
        )
        codes.append(code)
        labels.append(label)
    out["type_code"] = codes
    out["auto_type_label"] = labels
    return out


def apply_type_label_overrides(
    summary_df: pd.DataFrame,
    overrides: Optional[Mapping[Any, str]],
) -> pd.DataFrame:
    """Apply optional user-provided label overrides by cluster id."""
    out = summary_df.copy()
    out["type_label"] = out["auto_type_label"]
    if not overrides:
        return out
    normalized: dict[int, str] = {}
    for key, value in overrides.items():
        try:
            normalized[int(key)] = str(value)
        except Exception:
            continue
    if not normalized:
        return out
    mask = out["Cluster"].isin(normalized)
    out.loc[mask, "type_label"] = out.loc[mask, "Cluster"].map(normalized)
    return out


def order_clusters_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic display order from computed urgency metrics."""
    out = summary_df.copy()
    order = (
        out.sort_values(
            ["urgency_score", "front_loading_score", "persistence_score", "Cluster"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )
    order["display_order"] = np.arange(len(order), dtype=int)
    return order


def _zscore_series(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if not np.any(finite):
        return out
    sub = arr[finite]
    mean = float(np.mean(sub))
    std = float(np.std(sub))
    if not np.isfinite(std) or std <= _EPS_FLOOR:
        out[finite] = 0.0
        return out
    out[finite] = (sub - mean) / std
    return out


def _bucket_from_score(score: float) -> str:
    if not np.isfinite(score):
        return "neutral"
    if score >= 0.5:
        return "high"
    if score <= -0.5:
        return "low"
    return "neutral"


def _render_bucket_phrase(bucket: str, axis: str) -> str:
    if axis == "urgency":
        mapping = {
            "high": "High urgency",
            "neutral": "Balanced urgency",
            "low": "Low urgency",
        }
        return mapping[bucket]
    if axis == "schedule":
        mapping = {
            "high": "front-loaded schedule",
            "neutral": "balanced schedule",
            "low": "back-loaded schedule",
        }
        return mapping[bucket]
    mapping = {
        "high": "persistent impact",
        "neutral": "moderate persistence",
        "low": "fast-decaying impact",
    }
    return mapping[bucket]
