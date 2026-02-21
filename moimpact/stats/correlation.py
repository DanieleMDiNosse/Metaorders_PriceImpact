"""
Correlation, bootstrap CI, and permutation helpers.

These functions are used in crowding analyses where dependence within trading
days matters. The cluster bootstrap implemented here resamples trading dates
with replacement.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def corr_with_ci(
    x: Iterable[float],
    y: Iterable[float],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, int]:
    """
    Summary
    -------
    Compute Pearson correlation and a percentile-bootstrap confidence interval.

    Parameters
    ----------
    x : Iterable[float]
        1D array-like of observations.
    y : Iterable[float]
        1D array-like of observations (same length as `x`).
    alpha : float, default=0.05
        Significance level for a two-sided `(1 - alpha)` CI.
    n_bootstrap : int, default=1000
        Number of bootstrap replications over (x, y) pairs.
    random_state : Optional[int], default=None
        Optional RNG seed for reproducibility.

    Returns
    -------
    r : float
        Pearson correlation coefficient.
    lo : float
        Lower bound of the percentile-bootstrap CI.
    hi : float
        Upper bound of the percentile-bootstrap CI.
    n : int
        Number of valid `(x, y)` pairs used (after dropping NaNs).

    Notes
    -----
    - For very small samples (n <= 3) the correlation is treated as undefined
      and NaNs are returned for `(r, lo, hi)`.

    Examples
    --------
    >>> r, lo, hi, n = corr_with_ci([1, 2, 3, 4], [1.0, 2.0, 1.0, 2.0], n_bootstrap=200, random_state=0)
    >>> n
    4
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = int(x_arr.size)

    if n <= 3:
        return float("nan"), float("nan"), float("nan"), n

    r = float(np.corrcoef(x_arr, y_arr)[0, 1])
    if n_bootstrap <= 0 or not np.isfinite(r):
        return r, float("nan"), float("nan"), n

    rng = np.random.default_rng(random_state)
    boot_rs = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        xb = x_arr[idx]
        yb = y_arr[idx]
        if np.all(xb == xb[0]) or np.all(yb == yb[0]):
            boot_rs[i] = np.nan
        else:
            boot_rs[i] = np.corrcoef(xb, yb)[0, 1]

    valid = boot_rs[~np.isnan(boot_rs)]
    if valid.size == 0:
        lo = hi = float("nan")
    else:
        lo = float(np.quantile(valid, alpha / 2.0))
        hi = float(np.quantile(valid, 1.0 - alpha / 2.0))
    return r, lo, hi, n


def corr_with_bootstrap_p(
    x: Iterable[float],
    y: Iterable[float],
    *,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float, int]:
    """
    Summary
    -------
    Compute Pearson correlation and a two-sided permutation p-value.

    Parameters
    ----------
    x : Iterable[float]
        1D array-like of observations.
    y : Iterable[float]
        1D array-like of observations (same length as `x`).
    n_bootstrap : int, default=1000
        Number of permutations used to approximate the null distribution.
    random_state : Optional[int], default=None
        Optional RNG seed for reproducibility.

    Returns
    -------
    r : float
        Pearson correlation coefficient.
    p_value : float
        Two-sided permutation p-value (NaN when `n_bootstrap <= 0` or when the
        correlation is undefined).
    n : int
        Number of valid `(x, y)` pairs used (after dropping NaNs).

    Notes
    -----
    - Despite the name, this is a permutation test (shuffle `y`), not a bootstrap.
    - The p-value is computed as `2 * min(P(r_perm >= r_obs), P(r_perm <= r_obs))`.

    Examples
    --------
    >>> r, p, n = corr_with_bootstrap_p([1, -1, 1], [0.1, -0.2, 0.0], n_bootstrap=200, random_state=0)
    >>> n
    3
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = int(x_arr.size)
    if n < 3:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(x_arr, y_arr)[0, 1])
    if n_bootstrap <= 0 or not np.isfinite(r):
        return r, float("nan"), n

    rng = np.random.default_rng(random_state)
    perm_rs = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        y_perm = rng.permutation(y_arr)
        perm_rs[i] = np.corrcoef(x_arr, y_perm)[0, 1]

    if np.isnan(perm_rs).all():
        p_val = float("nan")
    else:
        greater = float(np.mean(perm_rs >= r))
        smaller = float(np.mean(perm_rs <= r))
        p_val = 2 * min(greater, smaller)
        p_val = min(1.0, p_val)
    return r, p_val, n


def _pearsonr_from_sums(
    n: float,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
) -> float:
    """Compute Pearson correlation from sufficient statistics."""
    if n < 3:
        return float("nan")

    cov = sum_xy - (sum_x * sum_y) / n
    var_x = sum_x2 - (sum_x * sum_x) / n
    var_y = sum_y2 - (sum_y * sum_y) / n
    denom = math.sqrt(var_x * var_y) if (var_x > 0.0 and var_y > 0.0) else 0.0
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(cov / denom)


def corr_with_cluster_bootstrap_ci_and_permutation_p(
    x: Iterable[float],
    y: Iterable[float],
    cluster: Iterable[object],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    n_permutations: Optional[int] = None,
    y_const_tol: float = 0.0,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float, float, int, int]:
    """
    Summary
    -------
    Compute a cluster-robust correlation CI and p-value via resampling.

    Parameters
    ----------
    x : Iterable[float]
        1D iterable of numeric values.
    y : Iterable[float]
        1D iterable of numeric values (same length as `x`).
    cluster : Iterable[object]
        Cluster labels (e.g., trading Date). Bootstrap resamples clusters.
    alpha : float, default=0.05
        Significance level for the `(1 - alpha)` CI.
    n_bootstrap : int, default=1000
        Number of cluster bootstrap replications used to form the CI.
    n_permutations : Optional[int], default=None
        Number of cluster permutations used to approximate the two-sided p-value.
        If None, defaults to `n_bootstrap`.
    y_const_tol : float, default=0.0
        Tolerance to check that `y` is approximately constant within clusters for
        permutation validity.
    random_state : Optional[int], default=None
        Optional RNG seed for reproducible resampling.

    Returns
    -------
    r : float
        Pearson correlation on the original sample.
    lo : float
        Lower bound of the cluster bootstrap percentile CI.
    hi : float
        Upper bound of the cluster bootstrap percentile CI.
    p : float
        Two-sided permutation p-value (NaN if permutation is invalid).
    n_obs : int
        Number of observations used after dropping invalid pairs.
    n_clusters : int
        Number of unique clusters used after dropping invalid pairs.

    Notes
    -----
    - Cluster bootstrap resamples clusters with replacement and concatenates all
      observations in the sampled clusters.
    - Cluster permutation p-value permutes cluster-level `y` values across clusters.
      This requires `y` to be (approximately) constant within each cluster.

    Examples
    --------
    >>> x = [1, -1, 1, 1, -1]
    >>> y = [0.2, 0.2, -0.1, -0.1, -0.1]
    >>> d = ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"]
    >>> r, lo, hi, p, n_obs, n_clusters = corr_with_cluster_bootstrap_ci_and_permutation_p(
    ...     x, y, d, n_bootstrap=200, random_state=0
    ... )
    >>> n_clusters
    2
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    cluster_arr = np.asarray(list(cluster), dtype=object)

    if not (x_arr.shape == y_arr.shape == cluster_arr.shape):
        raise ValueError("x, y, and cluster must have the same length.")

    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (~pd.isna(cluster_arr))
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    cluster_arr = cluster_arr[mask]

    n_obs = int(x_arr.size)
    if n_obs < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), n_obs, 0

    cluster_codes, _ = pd.factorize(cluster_arr, sort=False)
    n_clusters = int(cluster_codes.max() + 1) if cluster_codes.size else 0
    if n_clusters < 1:
        return float("nan"), float("nan"), float("nan"), float("nan"), n_obs, 0

    df = pd.DataFrame(
        {
            "cluster": cluster_codes,
            "x": x_arr,
            "y": y_arr,
            "x2": x_arr * x_arr,
            "y2": y_arr * y_arr,
            "xy": x_arr * y_arr,
        }
    )
    agg = (
        df.groupby("cluster", sort=False, dropna=False)
        .agg(
            n=("x", "size"),
            sum_x=("x", "sum"),
            sum_y=("y", "sum"),
            sum_x2=("x2", "sum"),
            sum_y2=("y2", "sum"),
            sum_xy=("xy", "sum"),
            y_min=("y", "min"),
            y_max=("y", "max"),
        )
        .reset_index(drop=True)
    )
    n_k = agg["n"].to_numpy(dtype=float)
    sum_x_k = agg["sum_x"].to_numpy(dtype=float)
    sum_y_k = agg["sum_y"].to_numpy(dtype=float)
    sum_x2_k = agg["sum_x2"].to_numpy(dtype=float)
    sum_y2_k = agg["sum_y2"].to_numpy(dtype=float)
    sum_xy_k = agg["sum_xy"].to_numpy(dtype=float)
    y_min_k = agg["y_min"].to_numpy(dtype=float)
    y_max_k = agg["y_max"].to_numpy(dtype=float)

    n_total = float(n_k.sum())
    sum_x_total = float(sum_x_k.sum())
    sum_y_total = float(sum_y_k.sum())
    sum_x2_total = float(sum_x2_k.sum())
    sum_y2_total = float(sum_y2_k.sum())
    sum_xy_total = float(sum_xy_k.sum())

    r_obs = _pearsonr_from_sums(
        n_total, sum_x_total, sum_y_total, sum_x2_total, sum_y2_total, sum_xy_total
    )

    lo = hi = float("nan")
    if n_bootstrap > 0 and np.isfinite(r_obs) and n_clusters >= 2:
        rng = np.random.default_rng(random_state)
        boot_rs = np.empty(int(n_bootstrap), dtype=float)
        for b in range(int(n_bootstrap)):
            sampled = rng.integers(0, n_clusters, size=n_clusters)
            freq = np.bincount(sampled, minlength=n_clusters).astype(float)

            n_b = float(np.sum(freq * n_k))
            sx_b = float(np.sum(freq * sum_x_k))
            sy_b = float(np.sum(freq * sum_y_k))
            sx2_b = float(np.sum(freq * sum_x2_k))
            sy2_b = float(np.sum(freq * sum_y2_k))
            sxy_b = float(np.sum(freq * sum_xy_k))
            boot_rs[b] = _pearsonr_from_sums(n_b, sx_b, sy_b, sx2_b, sy2_b, sxy_b)

        valid = boot_rs[np.isfinite(boot_rs)]
        if valid.size > 0:
            lo = float(np.quantile(valid, alpha / 2.0))
            hi = float(np.quantile(valid, 1.0 - alpha / 2.0))

    p_val = float("nan")
    n_perm_eff = int(n_permutations) if n_permutations is not None else int(n_bootstrap)
    if (
        n_perm_eff > 0
        and np.isfinite(r_obs)
        and n_clusters >= 2
        and np.nanmax(np.abs(y_max_k - y_min_k)) <= float(y_const_tol)
    ):
        rng = np.random.default_rng(random_state)
        y_cluster = sum_y_k / n_k

        perm_rs = np.empty(n_perm_eff, dtype=float)
        for b in range(n_perm_eff):
            y_perm = rng.permutation(y_cluster)
            sy_p = float(np.sum(n_k * y_perm))
            sy2_p = float(np.sum(n_k * y_perm * y_perm))
            sxy_p = float(np.sum(sum_x_k * y_perm))
            perm_rs[b] = _pearsonr_from_sums(
                n_total, sum_x_total, sy_p, sum_x2_total, sy2_p, sxy_p
            )

        valid = perm_rs[np.isfinite(perm_rs)]
        if valid.size > 0:
            extreme = np.sum(np.abs(valid) >= abs(r_obs))
            p_val = float((1.0 + extreme) / (1.0 + valid.size))

    return float(r_obs), float(lo), float(hi), float(p_val), n_obs, n_clusters
