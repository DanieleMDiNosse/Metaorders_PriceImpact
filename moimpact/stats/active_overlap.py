"""
Active-overlap statistics for member prop-client flow.

The helpers in this module compute interval-weighted client environments for
proprietary metaorders. They are intentionally independent from plotting and
file I/O so they can be tested on small synthetic panels.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


GROUP_PROPRIETARY = "proprietary"
GROUP_CLIENT = "client"
SCOPE_SAME_ISIN = "same_isin"
SCOPE_ALL_ISIN = "all_isin"
BUCKET_ALL_ACTIVE = "all_active"
BUCKET_PREEXISTING = "preexisting_at_prop_start"
BUCKET_STARTS_DURING = "starts_during_prop"

VALID_SCOPES = (SCOPE_SAME_ISIN, SCOPE_ALL_ISIN)
VALID_LEAD_LAG_BUCKETS = (BUCKET_ALL_ACTIVE, BUCKET_PREEXISTING, BUCKET_STARTS_DURING)


def compute_overlap_fraction_matrix(
    *,
    target_start_ns: np.ndarray,
    target_end_ns: np.ndarray,
    target_duration_ns: np.ndarray,
    other_start_ns: np.ndarray,
    other_end_ns: np.ndarray,
) -> np.ndarray:
    """
    Summary
    -------
    Compute target-normalized active-overlap fractions.

    Parameters
    ----------
    target_start_ns : np.ndarray
        One-dimensional array of target start timestamps in nanoseconds.
    target_end_ns : np.ndarray
        One-dimensional array of target end timestamps in nanoseconds.
    target_duration_ns : np.ndarray
        One-dimensional array of target durations in nanoseconds.
    other_start_ns : np.ndarray
        One-dimensional array of environment start timestamps in nanoseconds.
    other_end_ns : np.ndarray
        One-dimensional array of environment end timestamps in nanoseconds.

    Returns
    -------
    np.ndarray
        Matrix with shape `(n_targets, n_others)`. Entry `(i, j)` is the
        overlap length between target `i` and other `j`, divided by target
        duration. Zero-duration targets use the point-in-time active rule.

    Notes
    -----
    For positive-duration targets, the weight is:

    `max(0, min(end_i, end_j) - max(start_i, start_j)) / (end_i - start_i)`.

    For zero-duration targets, the weight is one when
    `start_j <= start_i <= end_j`, and zero otherwise.

    Examples
    --------
    >>> starts = np.array([0, 5], dtype=np.int64)
    >>> ends = np.array([10, 5], dtype=np.int64)
    >>> duration = np.maximum(ends - starts, 0)
    >>> frac = compute_overlap_fraction_matrix(
    ...     target_start_ns=starts,
    ...     target_end_ns=ends,
    ...     target_duration_ns=duration,
    ...     other_start_ns=np.array([5], dtype=np.int64),
    ...     other_end_ns=np.array([15], dtype=np.int64),
    ... )
    >>> float(frac[0, 0])
    0.5
    """
    start_i = np.asarray(target_start_ns, dtype=np.int64)[:, None]
    end_i = np.asarray(target_end_ns, dtype=np.int64)[:, None]
    duration_i = np.asarray(target_duration_ns, dtype=np.int64)
    start_j = np.asarray(other_start_ns, dtype=np.int64)[None, :]
    end_j = np.asarray(other_end_ns, dtype=np.int64)[None, :]

    overlap_ns = np.minimum(end_i, end_j) - np.maximum(start_i, start_j)
    overlap_ns = np.maximum(overlap_ns, 0)
    frac = np.zeros(overlap_ns.shape, dtype=float)

    positive_duration = duration_i > 0
    if np.any(positive_duration):
        frac[positive_duration, :] = (
            overlap_ns[positive_duration, :].astype(float)
            / duration_i[positive_duration, None].astype(float)
        )

    zero_duration = ~positive_duration
    if np.any(zero_duration):
        point_active = (start_j <= start_i[zero_duration, :]) & (start_i[zero_duration, :] <= end_j)
        frac[zero_duration, :] = point_active.astype(float)

    return frac


def compute_member_active_overlap_features(
    df: pd.DataFrame,
    *,
    scopes: Sequence[str] = VALID_SCOPES,
    lead_lag_buckets: Sequence[str] = VALID_LEAD_LAG_BUCKETS,
    batch_size: int = 2048,
    n_jobs: int = 1,
    group_col: str = "group",
    member_col: str = "Member",
    isin_col: str = "ISIN",
    date_col: str = "Date",
    start_col: str = "StartTimestamp",
    end_col: str = "EndTimestamp",
    direction_col: str = "Direction",
    q_col: str = "Q",
) -> pd.DataFrame:
    """
    Summary
    -------
    Compute active same-member client imbalances for proprietary targets.

    Parameters
    ----------
    df : pd.DataFrame
        Combined metaorder table containing proprietary and client rows.
    scopes : Sequence[str], default=("same_isin", "all_isin")
        Client environment scopes. `same_isin` restricts clients to the same
        member, ISIN, and date; `all_isin` restricts only to member and date.
    lead_lag_buckets : Sequence[str], default=all supported buckets
        Temporal buckets to compute. Supported values are `all_active`,
        `preexisting_at_prop_start`, and `starts_during_prop`.
    batch_size : int, default=2048
        Number of proprietary target rows processed per matrix batch.
    n_jobs : int, default=1
        Number of process workers used over independent `(Member, Date)` blocks.
        Use `1` for deterministic serial execution and easier debugging.
    group_col, member_col, isin_col, date_col, start_col, end_col, direction_col, q_col : str
        Column names in `df`.

    Returns
    -------
    pd.DataFrame
        Long target-level table. Each proprietary target appears once for every
        requested `(scope, lead_lag_bucket)` pair, with active-client imbalance
        and related diagnostics.

    Notes
    -----
    - Only client rows from the same `Member` are eligible.
    - `active_client_imbalance` is not signed by the proprietary direction.
      The companion column `active_client_alignment` equals
      `Direction_prop * active_client_imbalance`.
    - Rows with no positive active client overlap have `NaN` imbalance and are
      meant to be excluded from correlation estimates.
    - Parallelism is by `(Member, Date)` block. Results are sorted by original
      target row, scope, and bucket, so output ordering is deterministic.

    Examples
    --------
    >>> frame = pd.DataFrame({
    ...     "group": ["proprietary", "client"],
    ...     "Member": ["M1", "M1"],
    ...     "ISIN": ["A", "A"],
    ...     "Date": [pd.Timestamp("2024-01-02")] * 2,
    ...     "StartTimestamp": [pd.Timestamp("2024-01-02 10:00"), pd.Timestamp("2024-01-02 09:55")],
    ...     "EndTimestamp": [pd.Timestamp("2024-01-02 10:10"), pd.Timestamp("2024-01-02 10:05")],
    ...     "Direction": [1.0, -1.0],
    ...     "Q": [100.0, 50.0],
    ... })
    >>> out = compute_member_active_overlap_features(frame, scopes=["same_isin"], n_jobs=1)
    >>> float(out.loc[out["lead_lag_bucket"] == "all_active", "active_client_imbalance"].iat[0])
    -1.0
    """
    scopes = _validate_requested_values(scopes, VALID_SCOPES, label="scopes")
    lead_lag_buckets = _validate_requested_values(
        lead_lag_buckets,
        VALID_LEAD_LAG_BUCKETS,
        label="lead_lag_buckets",
    )
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    if int(n_jobs) <= 0:
        raise ValueError("n_jobs must be positive after option resolution.")

    prepared = _prepare_input_frame(
        df,
        group_col=group_col,
        member_col=member_col,
        isin_col=isin_col,
        date_col=date_col,
        start_col=start_col,
        end_col=end_col,
        direction_col=direction_col,
        q_col=q_col,
    )

    items = [
        (group.copy(), tuple(scopes), tuple(lead_lag_buckets), int(batch_size))
        for _, group in prepared.groupby([member_col, date_col], sort=False, dropna=False)
    ]
    if n_jobs > 1 and len(items) > 1:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as executor:
            parts = list(executor.map(_member_date_worker, items))
    else:
        parts = [_member_date_worker(item) for item in items]

    non_empty_parts = [part for part in parts if not part.empty]
    if non_empty_parts:
        out = pd.concat(non_empty_parts, ignore_index=True)
    else:
        out = _empty_output_frame()

    if out.empty:
        return out
    scope_order = {scope: idx for idx, scope in enumerate(scopes)}
    bucket_order = {bucket: idx for idx, bucket in enumerate(lead_lag_buckets)}
    out["__scope_order__"] = out["scope"].map(scope_order)
    out["__bucket_order__"] = out["lead_lag_bucket"].map(bucket_order)
    out = (
        out.sort_values(["target_row_id", "__scope_order__", "__bucket_order__"], kind="mergesort")
        .drop(columns=["__scope_order__", "__bucket_order__"])
        .reset_index(drop=True)
    )
    return out


def _validate_requested_values(values: Sequence[str], valid: Sequence[str], *, label: str) -> list[str]:
    requested = [str(value) for value in values]
    if not requested:
        raise ValueError(f"{label} must contain at least one value.")
    invalid = [value for value in requested if value not in valid]
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}. Valid values: {list(valid)}")
    return requested


def _prepare_input_frame(
    df: pd.DataFrame,
    *,
    group_col: str,
    member_col: str,
    isin_col: str,
    date_col: str,
    start_col: str,
    end_col: str,
    direction_col: str,
    q_col: str,
) -> pd.DataFrame:
    required = [group_col, member_col, isin_col, date_col, start_col, end_col, direction_col, q_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for active-overlap features: {missing}")

    rename_map = {
        group_col: "group",
        member_col: "Member",
        isin_col: "ISIN",
        date_col: "Date",
        start_col: "StartTimestamp",
        end_col: "EndTimestamp",
        direction_col: "Direction",
        q_col: "Q",
    }
    out = df[required].copy().rename(columns=rename_map).reset_index(drop=True)
    out["_row_id"] = np.arange(len(out), dtype=np.int64)
    out["_group_norm"] = out["group"].map(_normalize_group_label)
    if out["_group_norm"].isna().any():
        bad = sorted(out.loc[out["_group_norm"].isna(), "group"].astype(str).unique().tolist())
        raise ValueError(f"Unsupported group labels: {bad}")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    start_ts = pd.to_datetime(out["StartTimestamp"], errors="coerce")
    end_ts = pd.to_datetime(out["EndTimestamp"], errors="coerce")
    if out["Date"].isna().any() or start_ts.isna().any() or end_ts.isna().any():
        raise ValueError("Failed to parse Date, StartTimestamp, or EndTimestamp for some rows.")

    out["StartTimestamp"] = start_ts
    out["EndTimestamp"] = end_ts
    out["_start_ns"] = start_ts.astype("int64")
    out["_end_ns"] = end_ts.astype("int64")
    if (out["_end_ns"].to_numpy(dtype=np.int64) < out["_start_ns"].to_numpy(dtype=np.int64)).any():
        raise ValueError("Found rows with EndTimestamp earlier than StartTimestamp.")

    out["Direction"] = pd.to_numeric(out["Direction"], errors="coerce")
    out["Q"] = pd.to_numeric(out["Q"], errors="coerce")
    if out["Direction"].isna().any() or out["Q"].isna().any():
        raise ValueError("Found non-numeric Direction or Q values.")
    if (~np.isfinite(out["Direction"].to_numpy(dtype=float))).any() or (
        ~np.isfinite(out["Q"].to_numpy(dtype=float))
    ).any():
        raise ValueError("Found non-finite Direction or Q values.")
    if (out["Q"].to_numpy(dtype=float) < 0.0).any():
        raise ValueError("Found negative Q values.")
    return out


def _normalize_group_label(value: object) -> str | None:
    text = str(value).strip().lower()
    if text in {"proprietary", "prop", "own", "own_account"}:
        return GROUP_PROPRIETARY
    if text in {"client", "non_proprietary", "non-proprietary", "nonprop", "non_prop"}:
        return GROUP_CLIENT
    return None


def _member_date_worker(payload: tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], int]) -> pd.DataFrame:
    group, scopes, lead_lag_buckets, batch_size = payload
    return _compute_member_date_features(
        group,
        scopes=scopes,
        lead_lag_buckets=lead_lag_buckets,
        batch_size=int(batch_size),
    )


def _compute_member_date_features(
    group: pd.DataFrame,
    *,
    scopes: Iterable[str],
    lead_lag_buckets: Iterable[str],
    batch_size: int,
) -> pd.DataFrame:
    prop = group[group["_group_norm"] == GROUP_PROPRIETARY]
    client = group[group["_group_norm"] == GROUP_CLIENT]
    if prop.empty:
        return _empty_output_frame()

    parts: list[pd.DataFrame] = []
    for scope in scopes:
        if scope == SCOPE_ALL_ISIN:
            parts.append(
                _compute_scope_features(
                    prop,
                    client,
                    scope=scope,
                    lead_lag_buckets=lead_lag_buckets,
                    batch_size=batch_size,
                )
            )
        elif scope == SCOPE_SAME_ISIN:
            for _, prop_isin in prop.groupby("ISIN", sort=False, dropna=False):
                isin = prop_isin["ISIN"].iloc[0]
                client_isin = client[client["ISIN"].isna()] if pd.isna(isin) else client[client["ISIN"].eq(isin)]
                parts.append(
                    _compute_scope_features(
                        prop_isin,
                        client_isin,
                        scope=scope,
                        lead_lag_buckets=lead_lag_buckets,
                        batch_size=batch_size,
                    )
                )
    non_empty_parts = [part for part in parts if not part.empty]
    if not non_empty_parts:
        return _empty_output_frame()
    return pd.concat(non_empty_parts, ignore_index=True)


def _compute_scope_features(
    prop: pd.DataFrame,
    client: pd.DataFrame,
    *,
    scope: str,
    lead_lag_buckets: Iterable[str],
    batch_size: int,
) -> pd.DataFrame:
    n_prop = len(prop)
    if n_prop == 0:
        return _empty_output_frame()

    arrays = _allocate_feature_arrays(n_prop)
    prop_start = prop["_start_ns"].to_numpy(dtype=np.int64)
    prop_end = prop["_end_ns"].to_numpy(dtype=np.int64)
    prop_duration = np.maximum(prop_end - prop_start, 0)
    prop_q = prop["Q"].to_numpy(dtype=float)
    prop_dir = np.sign(prop["Direction"].to_numpy(dtype=float))

    client_start = client["_start_ns"].to_numpy(dtype=np.int64)
    client_end = client["_end_ns"].to_numpy(dtype=np.int64)
    client_q = client["Q"].to_numpy(dtype=float)
    client_dir = np.sign(client["Direction"].to_numpy(dtype=float))

    for left in range(0, n_prop, int(batch_size)):
        right = min(left + int(batch_size), n_prop)
        if len(client) == 0:
            frac = np.zeros((right - left, 0), dtype=float)
        else:
            frac = compute_overlap_fraction_matrix(
                target_start_ns=prop_start[left:right],
                target_end_ns=prop_end[left:right],
                target_duration_ns=prop_duration[left:right],
                other_start_ns=client_start,
                other_end_ns=client_end,
            )

        bucket_weights = {
            BUCKET_ALL_ACTIVE: frac,
            BUCKET_PREEXISTING: frac
            * (
                (client_start[None, :] <= prop_start[left:right, None])
                & (prop_start[left:right, None] <= client_end[None, :])
            ),
            BUCKET_STARTS_DURING: frac
            * (
                (client_start[None, :] > prop_start[left:right, None])
                & (client_start[None, :] <= prop_end[left:right, None])
            ),
        }
        for bucket, weights in bucket_weights.items():
            _fill_arrays_for_bucket(
                arrays[bucket],
                rows=slice(left, right),
                weights=weights,
                client_q=client_q,
                client_dir=client_dir,
                prop_q=prop_q[left:right],
                prop_dir=prop_dir[left:right],
            )

    base = prop[
        [
            "_row_id",
            "Member",
            "ISIN",
            "Date",
            "StartTimestamp",
            "EndTimestamp",
            "Direction",
            "Q",
        ]
    ].copy()
    base = base.rename(
        columns={
            "_row_id": "target_row_id",
            "Direction": "target_direction",
            "Q": "target_Q",
            "StartTimestamp": "target_start",
            "EndTimestamp": "target_end",
        }
    )
    base["target_duration_minutes"] = (prop_duration.astype(float) / 60_000_000_000.0)
    base["scope"] = scope
    base["n_client_candidates"] = int(len(client))

    parts = []
    for bucket in lead_lag_buckets:
        part = base.copy()
        part["lead_lag_bucket"] = bucket
        for column, values in arrays[bucket].items():
            part[column] = values
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _allocate_feature_arrays(n_rows: int) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for bucket in VALID_LEAD_LAG_BUCKETS:
        arrays[bucket] = {
            "active_client_any_count": np.zeros(n_rows, dtype=float),
            "active_client_count_tw": np.zeros(n_rows, dtype=float),
            "active_client_gross_q_tw": np.zeros(n_rows, dtype=float),
            "active_client_gross_q_tw_over_Q": np.full(n_rows, np.nan, dtype=float),
            "active_client_signed_q_tw": np.zeros(n_rows, dtype=float),
            "active_client_signed_q_tw_over_Q": np.full(n_rows, np.nan, dtype=float),
            "active_client_same_q_tw": np.zeros(n_rows, dtype=float),
            "active_client_opp_q_tw": np.zeros(n_rows, dtype=float),
            "active_client_imbalance": np.full(n_rows, np.nan, dtype=float),
            "active_client_alignment": np.full(n_rows, np.nan, dtype=float),
        }
    return arrays


def _fill_arrays_for_bucket(
    arrays: dict[str, np.ndarray],
    *,
    rows: slice,
    weights: np.ndarray,
    client_q: np.ndarray,
    client_dir: np.ndarray,
    prop_q: np.ndarray,
    prop_dir: np.ndarray,
) -> None:
    if weights.shape[1] == 0:
        gross = np.zeros(weights.shape[0], dtype=float)
        signed = np.zeros(weights.shape[0], dtype=float)
        pos_q = np.zeros(weights.shape[0], dtype=float)
        neg_q = np.zeros(weights.shape[0], dtype=float)
    else:
        gross = weights @ client_q
        signed = weights @ (client_q * client_dir)
        pos_q = weights @ (client_q * (client_dir > 0.0))
        neg_q = weights @ (client_q * (client_dir < 0.0))

    same = np.where(prop_dir > 0.0, pos_q, np.where(prop_dir < 0.0, neg_q, np.nan))
    opp = np.where(prop_dir > 0.0, neg_q, np.where(prop_dir < 0.0, pos_q, np.nan))
    imbalance = np.divide(
        signed,
        gross,
        out=np.full_like(signed, np.nan, dtype=float),
        where=gross > 0.0,
    )

    arrays["active_client_any_count"][rows] = (weights > 0.0).sum(axis=1).astype(float)
    arrays["active_client_count_tw"][rows] = weights.sum(axis=1)
    arrays["active_client_gross_q_tw"][rows] = gross
    arrays["active_client_signed_q_tw"][rows] = signed
    arrays["active_client_same_q_tw"][rows] = same
    arrays["active_client_opp_q_tw"][rows] = opp
    arrays["active_client_imbalance"][rows] = imbalance
    arrays["active_client_alignment"][rows] = prop_dir * imbalance
    arrays["active_client_gross_q_tw_over_Q"][rows] = np.divide(
        gross,
        prop_q,
        out=np.full_like(gross, np.nan, dtype=float),
        where=prop_q > 0.0,
    )
    arrays["active_client_signed_q_tw_over_Q"][rows] = np.divide(
        signed,
        prop_q,
        out=np.full_like(signed, np.nan, dtype=float),
        where=prop_q > 0.0,
    )


def _empty_output_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "target_row_id",
            "Member",
            "ISIN",
            "Date",
            "target_start",
            "target_end",
            "target_direction",
            "target_Q",
            "target_duration_minutes",
            "scope",
            "n_client_candidates",
            "lead_lag_bucket",
            "active_client_any_count",
            "active_client_count_tw",
            "active_client_gross_q_tw",
            "active_client_gross_q_tw_over_Q",
            "active_client_signed_q_tw",
            "active_client_signed_q_tw_over_Q",
            "active_client_same_q_tw",
            "active_client_opp_q_tw",
            "active_client_imbalance",
            "active_client_alignment",
        ]
    )
