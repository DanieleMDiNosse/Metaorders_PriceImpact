"""
Shared impact-fit helpers used by the metaorder impact scripts.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

from moimpact.plot_style import THEME_COLORWAY
from moimpact.plotting import COLOR_NEUTRAL


def weights_from_sigma(sigma: np.ndarray) -> np.ndarray:
    """
    Summary
    -------
    Convert standard deviations or standard errors into WLS weights.

    Parameters
    ----------
    sigma : np.ndarray
        One-dimensional array of positive uncertainty scales.

    Returns
    -------
    np.ndarray
        Array with elementwise weights ``1 / sigma**2`` where valid and zero
        elsewhere.

    Notes
    -----
    Non-finite or non-positive entries are mapped to zero weight.

    Examples
    --------
    >>> weights_from_sigma(np.array([2.0, 0.0])).tolist()
    [0.25, 0.0]
    """
    sigma = np.asarray(sigma, dtype=float)
    w = np.zeros_like(sigma, dtype=float)
    ok = np.isfinite(sigma) & (sigma > 0)
    w[ok] = 1.0 / np.square(sigma[ok])
    return w


def weighted_r2(y: np.ndarray, yhat: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """
    Summary
    -------
    Compute a weighted coefficient of determination.

    Parameters
    ----------
    y : np.ndarray
        Observed values.
    yhat : np.ndarray
        Fitted values with the same shape as `y`.
    w : Optional[np.ndarray], default=None
        Optional positive observation weights. When omitted, the computation is
        unweighted.

    Returns
    -------
    float
        Weighted ``R^2`` value, or ``nan`` when it is not well-defined.

    Notes
    -----
    The statistic is

    ``1 - sum_i w_i (y_i - yhat_i)^2 / sum_i w_i (y_i - ybar_w)^2``.

    Examples
    --------
    >>> weighted_r2(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    1.0
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


def power_law(x: np.ndarray, prefactor: float, exponent: float) -> np.ndarray:
    """
    Summary
    -------
    Evaluate a power-law curve.

    Parameters
    ----------
    x : np.ndarray
        Positive input values, typically ``Q/V``.
    prefactor : float
        Multiplicative constant.
    exponent : float
        Power-law exponent.

    Returns
    -------
    np.ndarray
        Values ``prefactor * x**exponent``.

    Notes
    -----
    This is the univariate impact specification used by the repository's WLS
    fits.
    """
    return prefactor * np.power(x, exponent)


def logarithmic_impact(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Summary
    -------
    Evaluate the logarithmic impact benchmark.

    Parameters
    ----------
    x : np.ndarray
        Positive input values, typically ``Q/V``.
    a : float
        Multiplicative coefficient.
    b : float
        Positive slope parameter inside the logarithm.

    Returns
    -------
    np.ndarray
        Values ``a * log10(1 + b * x)``.

    Notes
    -----
    This matches the alternative specification plotted alongside the power-law
    fit in `metaorder_computation.py`.
    """
    return a * np.log10(1.0 + b * x)


def filter_metaorders_info_for_fits(df: pd.DataFrame, min_qv: float) -> pd.DataFrame:
    """
    Summary
    -------
    Apply the canonical fit-ready filter to a metaorder info table.

    Parameters
    ----------
    df : pd.DataFrame
        Metaorder-level table expected to contain `Q/V`, `Price Change`,
        `Direction`, and `Daily Vol`.
    min_qv : float
        Strict lower bound for `Q/V`.

    Returns
    -------
    pd.DataFrame
        Copy of the input with numeric columns sanitized and a derived `Impact`
        column equal to ``Price Change * Direction / Daily Vol``.

    Notes
    -----
    Rows with missing or non-finite `Q/V` or `Impact` are dropped. This helper
    deliberately does not apply the participation-rate cap; callers should do
    that explicitly so the sample size bookkeeping remains visible.

    Examples
    --------
    >>> demo = pd.DataFrame({"Q/V": [1e-4], "Price Change": [0.1], "Direction": [1], "Daily Vol": [0.2]})
    >>> out = filter_metaorders_info_for_fits(demo, min_qv=1e-5)
    >>> float(out["Impact"].iat[0])
    0.5
    """
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
        out["Price Change"] * out["Direction"] / out["Daily Vol"],
        errors="coerce",
    )

    numeric_cols = [
        c
        for c in ["Q/V", "Vt/V", "Impact", "Participation Rate", "Price Change", "Daily Vol", "Q"]
        if c in out.columns
    ]
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return out.dropna(subset=["Q/V", "Impact"]).reset_index(drop=True)


def fit_power_law_logbins_wls_new(
    subdf: pd.DataFrame,
    n_logbins: int = 30,
    min_count: int = 100,
    use_median: bool = False,
    control_cols: Optional[List[str]] = None,
) -> Tuple[
    pd.DataFrame,
    Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        Optional[pd.Series],
        Optional[pd.Series],
    ],
]:
    """
    Summary
    -------
    Fit the repository's univariate power-law impact curve on log-binned data.

    Parameters
    ----------
    subdf : pd.DataFrame
        Input table containing `Q/V` and `Impact`, and optionally bin-level
        control columns listed in `control_cols`.
    n_logbins : int, default=30
        Number of logarithmic bins used for `Q/V`.
    min_count : int, default=100
        Minimum number of metaorders required for a bin to enter the fit.
    use_median : bool, default=False
        If True, use the bin median impact instead of the mean.
    control_cols : Optional[List[str]], default=None
        Optional additional columns aggregated at the bin level and included as
        linear controls in the log-space WLS.

    Returns
    -------
    tuple[pd.DataFrame, tuple]
        The retained binned table and the fitted parameter bundle
        ``(Y_hat, Y_se, gamma_hat, gamma_se, R2_log, R2_lin,
        beta_controls, beta_controls_se)``.

    Notes
    -----
    The fit is linear in log space:

    ``log(I/sigma) = log(Y) + gamma * log(Q/V) + beta' controls``.

    Bin-level weights use the delta-method variance of ``log(mean_imp)``.
    """
    qv = pd.to_numeric(subdf["Q/V"], errors="coerce")
    impact = pd.to_numeric(subdf["Impact"], errors="coerce")
    mask = (qv > 0) & np.isfinite(impact)

    controls_numeric: dict[str, pd.Series] = {}
    if control_cols is not None:
        for col in control_cols:
            ctrl = pd.to_numeric(subdf[col], errors="coerce")
            controls_numeric[col] = ctrl
            mask &= np.isfinite(ctrl)

    sub = subdf.loc[mask].copy()
    sub["Q/V"] = qv.loc[mask].astype(float)
    sub["Impact"] = impact.loc[mask].astype(float)
    for col, ctrl in controls_numeric.items():
        sub[col] = ctrl.loc[mask].astype(float)
    if sub.empty:
        raise ValueError("No valid rows (Q/V>0 and finite Impact/controls).")

    x = sub["Q/V"].to_numpy()
    x_min = x.min()
    x_max = x.max()
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("Invalid Q/V range for log binning.")

    edges = np.logspace(np.log10(x_min), np.log10(x_max), n_logbins + 1)
    bin_idx = np.digitize(x, edges) - 1
    mask = (bin_idx >= 0) & (bin_idx < n_logbins)
    sub = sub.iloc[mask].copy()
    sub["bin"] = bin_idx[mask]

    grp = sub.groupby("bin")
    agg_imp = grp["Impact"].agg(
        mean_imp="mean",
        median_imp="median",
        std_imp=lambda s: s.std(ddof=1),
        count="size",
    ).sort_index()

    if control_cols is not None and len(control_cols) > 0:
        agg = agg_imp.join(grp[control_cols].mean().sort_index())
    else:
        agg = agg_imp

    y_stat = agg["median_imp"] if use_median else agg["mean_imp"]
    y_std = agg["std_imp"].to_numpy()
    counts = agg["count"].to_numpy()
    sem = y_std / np.sqrt(np.maximum(counts, 1))

    bins_present = agg.index.to_numpy()
    left_edges = edges[bins_present]
    right_edges = edges[bins_present + 1]
    x_center = np.sqrt(left_edges * right_edges)

    cols: dict[str, np.ndarray] = {
        "center_QV": x_center,
        "mean_imp": y_stat.to_numpy(),
        "std_imp": y_std,
        "sem_imp": sem,
        "count": counts,
    }
    if control_cols is not None and len(control_cols) > 0:
        for col in control_cols:
            cols[col] = agg[col].to_numpy()

    binned = pd.DataFrame(cols).sort_values("center_QV").reset_index(drop=True)

    cond = (
        (binned["count"] >= min_count)
        & np.isfinite(binned["mean_imp"])
        & np.isfinite(binned["sem_imp"])
        & (binned["sem_imp"] > 0)
        & (binned["mean_imp"] > 0)
    )
    if control_cols is not None:
        for col in control_cols:
            cond &= np.isfinite(binned[col])
    binned = binned[cond].reset_index(drop=True)

    n_required = 2 + (len(control_cols) if control_cols else 0)
    if len(binned) < n_required:
        raise ValueError(
            f"Not enough valid bins after filtering (got {len(binned)}; need at least {n_required})."
        )

    x_log = np.log(binned["center_QV"].to_numpy())
    z_log = np.log(binned["mean_imp"].to_numpy())
    var_logy = (binned["sem_imp"].to_numpy() / binned["mean_imp"].to_numpy()) ** 2
    w = np.where(np.isfinite(var_logy) & (var_logy > 0), 1.0 / var_logy, 0.0)

    design_cols = [np.ones_like(x_log), x_log]
    control_names: list[str] = []
    if control_cols is not None and len(control_cols) > 0:
        for col in control_cols:
            design_cols.append(binned[col].to_numpy())
            control_names.append(col)
    design = np.vstack(design_cols).T

    sqrt_w = np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(design * sqrt_w[:, None], z_log * sqrt_w, rcond=None)

    a_hat = float(coef[0])
    gamma_hat = float(coef[1])
    beta_controls = pd.Series(coef[2:], index=control_names) if control_names else None
    y_hat = float(np.exp(a_hat))

    residuals = z_log - (design @ coef)
    rss = np.sum(w * residuals**2)
    dof = max(len(z_log) - design.shape[1], 1)
    s2 = rss / dof
    xtwx = design.T @ (w[:, None] * design)
    cov = s2 * np.linalg.inv(xtwx)
    se_all = np.sqrt(np.diag(cov))

    a_se = float(se_all[0])
    gamma_se = float(se_all[1])
    y_se = float(y_hat * a_se)
    beta_controls_se = pd.Series(se_all[2:], index=control_names) if control_names else None

    z_hat = design @ coef
    r2_log = weighted_r2(z_log, z_hat, w=w)

    yhat_linear = power_law(binned["center_QV"].to_numpy(), y_hat, gamma_hat)
    w_lin = weights_from_sigma(binned["sem_imp"].to_numpy())
    r2_lin = weighted_r2(binned["mean_imp"].to_numpy(), yhat_linear, w=w_lin)

    params = (y_hat, y_se, gamma_hat, gamma_se, r2_log, r2_lin, beta_controls, beta_controls_se)
    return binned, params


def fit_logarithmic_from_binned(
    binned: pd.DataFrame,
) -> Tuple[float, float, float, float, float]:
    """
    Summary
    -------
    Fit the logarithmic impact benchmark on already binned data.

    Parameters
    ----------
    binned : pd.DataFrame
        Log-binned impact table containing `center_QV`, `mean_imp`, and
        `sem_imp`.

    Returns
    -------
    tuple[float, float, float, float, float]
        ``(a_hat, a_se, b_hat, b_se, R2_lin)`` for the logarithmic curve
        ``a * log10(1 + b * Q/V)``.

    Notes
    -----
    Bin-level `sem_imp` values are passed as `sigma` with
    `absolute_sigma=True`.
    """
    if binned.empty:
        raise ValueError("No bins available for logarithmic fit.")

    x = binned["center_QV"].to_numpy()
    y = binned["mean_imp"].to_numpy()
    sigma = binned["sem_imp"].to_numpy()

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite values encountered in binned data for logarithmic fit.")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Non-positive or non-finite SEM values encountered for logarithmic fit.")

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

    try:
        popt, pcov = curve_fit(
            logarithmic_impact,
            x,
            y,
            p0=(a0, b0),
            sigma=sigma,
            absolute_sigma=True,
            maxfev=20000,
            bounds=((0.0, 0.0), (np.inf, np.inf)),
        )
    except Exception as exc:
        raise ValueError(f"Nonlinear logarithmic fit failed: {exc}") from exc

    a_hat = float(popt[0])
    b_hat = float(popt[1])
    if pcov is None or not np.all(np.isfinite(pcov)):
        a_se = float("nan")
        b_se = float("nan")
    else:
        perr = np.sqrt(np.diag(pcov))
        a_se = float(perr[0])
        b_se = float(perr[1])

    yhat = logarithmic_impact(x, a_hat, b_hat)
    w_lin = weights_from_sigma(sigma)
    r2_lin = weighted_r2(y, yhat, w=w_lin)
    return a_hat, a_se, b_hat, b_se, r2_lin


def plot_fit(
    fig: go.Figure,
    binned: pd.DataFrame,
    params,
    label_prefix: Optional[str] = None,
    label_size: int = 16,
    legend_size: int = 14,
    log_params: Optional[Tuple[float, float, float, float, float]] = None,
    series_color: Optional[str] = None,
    log_line_color: Optional[str] = None,
) -> None:
    """
    Summary
    -------
    Add one power-law fit series, and optionally one logarithmic series, to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to update in place.
    binned : pd.DataFrame
        Binned data returned by `fit_power_law_logbins_wls_new`.
    params : tuple
        Parameter bundle returned by `fit_power_law_logbins_wls_new`.
    label_prefix : Optional[str], default=None
        Optional text prepended to legend labels when multiple series are
        overlaid.
    label_size : int, default=16
        Axis label and tick font size.
    legend_size : int, default=14
        Legend font size.
    log_params : Optional[tuple], default=None
        Optional logarithmic-fit parameter bundle from
        `fit_logarithmic_from_binned`.
    series_color : Optional[str], default=None
        Override for the marker and power-law line color.
    log_line_color : Optional[str], default=None
        Override for the dashed logarithmic line color.

    Returns
    -------
    None
        The figure is modified in place.

    Notes
    -----
    The y-axis is kept on a log scale, so only positive binned mean impacts are
    admissible.
    """
    y_hat, y_err, gamma, gamma_err, _, _, _, _ = params
    if series_color is None:
        series_idx = len(fig.data) // (3 if log_params is not None else 2)
        series_color = THEME_COLORWAY[series_idx % len(THEME_COLORWAY)]

    prefix = f"{label_prefix}: " if label_prefix is not None else ""
    fig.add_trace(
        go.Scatter(
            x=binned["center_QV"],
            y=binned["mean_imp"],
            mode="markers",
            marker=dict(size=7, color=series_color),
            error_y=dict(type="data", array=binned["sem_imp"], visible=True, color=COLOR_NEUTRAL),
            name="Bin means +/- SEM" if label_prefix is None else f"{label_prefix}: bin means +/- SEM",
        )
    )

    x_min = float(binned["center_QV"].min())
    x_max = float(binned["center_QV"].max())
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 300)

    power_law_formula_math = (
        rf"$I/\sigma = ({y_hat:.3g}\pm{y_err:.2g})(Q/V)^{{{gamma:.3f}\pm{gamma_err:.3f}}}$"
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=power_law(x_grid, y_hat, gamma),
            mode="lines",
            line=dict(color=series_color, width=2),
            name=power_law_formula_math if label_prefix is None else f"{prefix}power-law fit",
        )
    )

    if log_params is not None:
        a_hat, a_se, b_hat, b_se, _ = log_params
        log_color = THEME_COLORWAY[2] if log_line_color is None else log_line_color
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=logarithmic_impact(x_grid, a_hat, b_hat),
                mode="lines",
                line=dict(color=log_color, width=2, dash="dash"),
                name=(
                    rf"$I/\sigma = ({a_hat:.3g}\pm{a_se:.2g})\log_{{10}}(1 + ({b_hat:.3g}\pm{b_se:.2g})\,Q/V)$"
                    if label_prefix is None
                    else f"{prefix}logarithmic fit"
                ),
            )
        )

    fig.update_xaxes(type="log", title_text="Q/V", title_font=dict(size=label_size), tickfont=dict(size=label_size))
    fig.update_yaxes(type="log", title_text=r"$I/\sigma$", title_font=dict(size=label_size), tickfont=dict(size=label_size))
    fig.update_layout(legend=dict(font=dict(size=legend_size)))
