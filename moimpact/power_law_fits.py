"""
Reusable power-law fitting helpers for distribution diagnostics and Clauset-style tests.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

_CLAUSET_LR_P_THRESHOLD = 0.1


@dataclass(frozen=True)
class PowerLawFitResult:
    """
    Summary
    -------
    Container for the fitted parameters of a power-law tail.

    Parameters
    ----------
    alpha : float
        Power-law exponent.
    xmin : float
        Lower cutoff of the fitted tail.
    ks_stat : float
        Kolmogorov-Smirnov distance between the empirical and fitted tail CDFs.
    n_tail : int
        Number of observations in the fitted tail (`x >= xmin`).
    method : str, default="approx"
        Identifier of the fitting routine used to obtain the result.
    alpha_sigma : Optional[float], default=None
        Optional standard error estimate for `alpha` when the fitting backend
        exposes it.

    Returns
    -------
    PowerLawFitResult
        Dataclass instance with the fitted tail parameters.

    Notes
    -----
    The `method` field is informative only and does not change the interpretation
    of the other attributes.
    """

    alpha: float
    xmin: float
    ks_stat: float
    n_tail: int
    method: str = "approx"
    alpha_sigma: Optional[float] = None


@dataclass(frozen=True)
class PowerLawGoFResult:
    """
    Summary
    -------
    Goodness-of-fit summary from the Clauset semiparametric bootstrap.

    Parameters
    ----------
    fit_result : PowerLawFitResult
        Empirical tail fit used as the null model.
    p_value : float
        Bootstrap goodness-of-fit p-value.
    bootstrap_runs : int
        Number of successful bootstrap refits contributing to `p_value`.
    tail_fraction : float
        Empirical tail mass `n_tail / n`.
    random_state : Optional[int], default=None
        Seed used for the bootstrap RNG, when provided.
    fit_method : str, default="approx"
        Tail-fitting backend used inside the bootstrap refits.

    Returns
    -------
    PowerLawGoFResult
        Dataclass instance with the Clauset bootstrap summary.

    Notes
    -----
    The reported `bootstrap_runs` can be smaller than the requested number of
    replicates if some synthetic refits are invalid.
    """

    fit_result: PowerLawFitResult
    p_value: float
    bootstrap_runs: int
    tail_fraction: float
    random_state: Optional[int] = None
    fit_method: str = "approx"


@dataclass(frozen=True)
class PowerLawComparisonResult:
    """
    Summary
    -------
    Likelihood-ratio comparison between a power law and an alternative tail model.

    Parameters
    ----------
    alternative : str
        Name of the alternative distribution passed to `powerlaw`.
    loglikelihood_ratio : Optional[float]
        Vuong-style log-likelihood ratio `R`, positive when the power law is favored.
    p_value : Optional[float]
        Significance of `R`.
    favored_model : str
        One of `"power_law"`, the alternative name, `"undecided"`, or
        `"unavailable"`.
    nested : bool
        Whether the comparison is treated as nested.
    fit_method : str, default="approx"
        Tail-fitting backend used to determine `xmin`.

    Returns
    -------
    PowerLawComparisonResult
        Dataclass instance summarizing one model comparison.

    Notes
    -----
    The decision in `favored_model` uses the Clauset paper's likelihood-ratio
    significance convention `p < 0.1`.
    """

    alternative: str
    loglikelihood_ratio: Optional[float]
    p_value: Optional[float]
    favored_model: str
    nested: bool
    fit_method: str = "approx"


@dataclass(frozen=True)
class TailModelComparisonResult:
    """
    Summary
    -------
    Likelihood-ratio comparison between any two fitted tail models.

    Parameters
    ----------
    model_a : str
        First model passed to `powerlaw.Fit.distribution_compare`.
    model_b : str
        Second model passed to `powerlaw.Fit.distribution_compare`.
    loglikelihood_ratio : Optional[float]
        Positive values favor `model_a`; negative values favor `model_b`.
    p_value : Optional[float]
        Significance reported by the `powerlaw` package for the comparison.
    favored_model : str
        Winner after applying the Clauset/Vuong decision rule with threshold
        `p < 0.1`; otherwise `"undecided"` or `"unavailable"`.
    nested : bool
        Whether the comparison is treated as nested.
    fit_method : str, default="approx"
        Tail-fitting backend used to determine `xmin`.

    Returns
    -------
    TailModelComparisonResult
        Dataclass instance summarizing one direct model-vs-model comparison.
    """

    model_a: str
    model_b: str
    loglikelihood_ratio: Optional[float]
    p_value: Optional[float]
    favored_model: str
    nested: bool
    fit_method: str = "approx"


@dataclass(frozen=True)
class TailDistributionScore:
    """
    Summary
    -------
    One candidate tail model scored on the common fitted tail sample.

    Parameters
    ----------
    model : str
        Distribution label understood by the `powerlaw` package.
    loglikelihood : Optional[float]
        Sum of pointwise log-likelihoods on the fitted tail sample.
    parameter_count : int
        Number of free parameters used by the fitted distribution.
    aic : Optional[float]
        Akaike information criterion computed on the fitted tail sample.
    bic : Optional[float]
        Bayesian information criterion computed on the fitted tail sample.
    ks_stat : Optional[float]
        Kolmogorov-Smirnov distance for the fitted distribution on the common tail.
    noise_flag : bool
        Whether the third-party package marked the fit as noisy/unstable.
    valid : bool
        Whether the score is considered valid for model selection.

    Returns
    -------
    TailDistributionScore
        Dataclass instance summarizing one candidate model score.
    """

    model: str
    loglikelihood: Optional[float]
    parameter_count: int
    aic: Optional[float]
    bic: Optional[float]
    ks_stat: Optional[float]
    noise_flag: bool
    valid: bool


@dataclass(frozen=True)
class PowerLawFitComparisonSummary:
    """
    Summary
    -------
    Empirical power-law fit plus likelihood-ratio comparisons to alternatives.

    Parameters
    ----------
    fit_result : PowerLawFitResult
        Estimated power-law tail parameters.
    comparisons : tuple[PowerLawComparisonResult, ...]
        Likelihood-ratio comparisons against the requested alternatives.
    pairwise_comparisons : tuple[TailModelComparisonResult, ...], default=()
        Direct likelihood-ratio comparisons across the fitted candidate models.
    best_model : str, default="power_law"
        Candidate model selected for overlay/summary on the common tail.
    best_model_criterion : str, default="aic"
        Criterion used to select `best_model`.
    model_scores : tuple[TailDistributionScore, ...], default=()
        Per-model scores computed on the common fitted tail.
    raw_fit : Any, optional
        Opaque `powerlaw.Fit` object backing the scored model family. This is
        exposed so repository scripts can evaluate PDFs of the selected model
        without refitting.
    fit_method : str, default="approx"
        Tail-fitting backend used to determine `xmin`.

    Returns
    -------
    PowerLawFitComparisonSummary
        Dataclass instance with the fitted tail and alternative-model checks.

    Notes
    -----
    This helper omits the Clauset semiparametric bootstrap. Use
    `run_power_law_clauset_pipeline` when you also need the goodness-of-fit
    p-value.
    """

    fit_result: PowerLawFitResult
    comparisons: tuple[PowerLawComparisonResult, ...]
    pairwise_comparisons: tuple[TailModelComparisonResult, ...] = tuple()
    best_model: str = "power_law"
    best_model_criterion: str = "aic"
    model_scores: tuple[TailDistributionScore, ...] = tuple()
    raw_fit: Any = field(default=None, repr=False)
    fit_method: str = "approx"


@dataclass(frozen=True)
class TailModelParameterBootstrapInterval:
    """
    Summary
    -------
    Percentile-bootstrap confidence interval for one fitted model parameter.

    Parameters
    ----------
    parameter : str
        Name of the fitted parameter as exposed by the `powerlaw` distribution.
    estimate : float
        Empirical point estimate from the original, non-bootstrap fit.
    ci_low : Optional[float]
        Lower percentile-bootstrap confidence bound, or `None` when not available.
    ci_high : Optional[float]
        Upper percentile-bootstrap confidence bound, or `None` when not available.
    std : Optional[float]
        Bootstrap standard deviation of the fitted parameter, or `None` when
        fewer than two finite bootstrap draws are available.

    Returns
    -------
    TailModelParameterBootstrapInterval
        Dataclass instance summarizing one parameter estimate and interval.

    Notes
    -----
    The interval is conditional on the configured bootstrap pipeline and on the
    candidate model family used to refit each bootstrap replicate.
    """

    parameter: str
    estimate: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    std: Optional[float]


@dataclass(frozen=True)
class ScalarBootstrapInterval:
    """
    Summary
    -------
    Percentile-bootstrap confidence interval for one scalar pipeline quantity.

    Parameters
    ----------
    quantity : str
        Label identifying the scalar quantity summarized by the interval.
    estimate : float
        Empirical point estimate from the original, non-bootstrap fit.
    ci_low : Optional[float]
        Lower percentile-bootstrap confidence bound, or `None` when not available.
    ci_high : Optional[float]
        Upper percentile-bootstrap confidence bound, or `None` when not available.
    std : Optional[float]
        Bootstrap standard deviation of the scalar quantity, or `None` when
        fewer than two finite bootstrap draws are available.
    valid_runs : int
        Number of bootstrap replicates that yielded a finite draw for this quantity.

    Returns
    -------
    ScalarBootstrapInterval
        Dataclass instance summarizing one scalar bootstrap interval.

    Notes
    -----
    This helper is used for pipeline-level quantities, such as the refitted
    Clauset cutoff `xmin`, that are not specific to a single alternative model.
    """

    quantity: str
    estimate: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    std: Optional[float]
    valid_runs: int


@dataclass(frozen=True)
class TailModelBootstrapSummary:
    """
    Summary
    -------
    Bootstrap parameter-uncertainty summary for one fitted tail model.

    Parameters
    ----------
    model : str
        Distribution label understood by the `powerlaw` package.
    parameter_intervals : tuple[TailModelParameterBootstrapInterval, ...]
        Interval summary for each parameter exposed by the fitted model.
    requested_runs : int
        Number of bootstrap replicates requested by the caller.
    valid_runs : int
        Number of bootstrap replicates that yielded a valid fit for this model.

    Returns
    -------
    TailModelBootstrapSummary
        Dataclass instance summarizing one candidate model's bootstrap output.

    Notes
    -----
    `valid_runs` can differ across models because some bootstrap refits can fail
    or be flagged as unstable by the third-party optimizer for one model but not
    another.
    """

    model: str
    parameter_intervals: tuple[TailModelParameterBootstrapInterval, ...]
    requested_runs: int
    valid_runs: int


@dataclass(frozen=True)
class FullPipelineBootstrapSummary:
    """
    Summary
    -------
    Full-sample bootstrap uncertainty summary for the Clauset fitting pipeline.

    Parameters
    ----------
    empirical_fit : PowerLawFitComparisonSummary
        Empirical fit on the original sample used as the reference estimates.
    model_summaries : tuple[TailModelBootstrapSummary, ...]
        Per-model percentile-bootstrap interval summaries.
    alpha : float
        Two-sided significance level used for the percentile interval.
    bootstrap_runs : int
        Number of bootstrap replicates requested by the caller.
    successful_pipeline_runs : int
        Number of bootstrap replicates whose full Clauset refit succeeded.
    xmin_summary : Optional[ScalarBootstrapInterval], default=None
        Percentile-bootstrap summary for the refitted Clauset cutoff `xmin`.
    random_state : Optional[int], default=None
        Seed used for the bootstrap RNG, when provided.
    resampling_scheme : str, default="nonparametric_full_sample"
        Label describing the bootstrap design used to generate the replicates.

    Returns
    -------
    FullPipelineBootstrapSummary
        Dataclass instance containing the empirical fit and the bootstrap-based
        parameter intervals for all fitted candidate models.

    Notes
    -----
    This summary targets full-pipeline uncertainty: each bootstrap replicate is
    refit from the resampled raw sample and therefore re-estimates the Clauset
    tail cutoff `xmin` before refitting the candidate tail models.
    """

    empirical_fit: PowerLawFitComparisonSummary
    model_summaries: tuple[TailModelBootstrapSummary, ...]
    alpha: float
    bootstrap_runs: int
    successful_pipeline_runs: int
    xmin_summary: Optional[ScalarBootstrapInterval] = None
    random_state: Optional[int] = None
    resampling_scheme: str = "nonparametric_full_sample"


@dataclass(frozen=True)
class ClausetPipelineResult:
    """
    Summary
    -------
    Combined result of the Clauset fit, bootstrap goodness-of-fit test, and model comparisons.

    Parameters
    ----------
    fit_result : PowerLawFitResult
        Estimated power-law tail parameters.
    gof_result : PowerLawGoFResult
        Semiparametric bootstrap goodness-of-fit summary.
    comparisons : tuple[PowerLawComparisonResult, ...]
        Likelihood-ratio comparisons against the requested alternatives.
    power_law_plausible : bool
        Whether the bootstrap p-value exceeds `p_threshold`.
    p_threshold : float
        Plausibility cutoff applied to the bootstrap p-value.
    fit_method : str, default="approx"
        Tail-fitting backend used in the pipeline.

    Returns
    -------
    ClausetPipelineResult
        Dataclass instance with the full Clauset-style workflow output.

    Notes
    -----
    `power_law_plausible` refers only to the Section 4 bootstrap test. The
    likelihood-ratio comparisons are reported separately in `comparisons`.
    """

    fit_result: PowerLawFitResult
    gof_result: PowerLawGoFResult
    comparisons: tuple[PowerLawComparisonResult, ...]
    power_law_plausible: bool
    p_threshold: float
    fit_method: str = "approx"


@dataclass(frozen=True)
class _PowerlawFitBundle:
    fit_result: PowerLawFitResult
    raw_fit: Any
    fit_score: float
    sample: np.ndarray


@dataclass(frozen=True)
class _TailModelFitCacheEntry:
    model: str
    distribution: Any = field(default=None, repr=False)
    loglikelihoods: Optional[np.ndarray] = field(default=None, repr=False)
    parameter_count: int = 0
    ks_stat: Optional[float] = None
    noise_flag: bool = True
    in_range: bool = False


def fit_power_law_clauset_continuous(
    data: Iterable[float],
    min_tail: int = 50,
    num_candidates: int = 200,
    refine_window: int = 50,
) -> Optional[PowerLawFitResult]:
    """
    Summary
    -------
    Fit a continuous power law using the repository's coarse-`xmin` Clauset backend.

    Parameters
    ----------
    data : Iterable[float]
        One-dimensional sample of positive observations.
    min_tail : int, default=50
        Minimum number of observations required in the fitted tail `x >= xmin`.
    num_candidates : int, default=200
        Number of coarse candidate cutoffs built from log-spaced tail sizes.
    refine_window : int, default=50
        Number of start-index positions explored on each side of the best
        coarse candidate during local refinement.

    Returns
    -------
    Optional[PowerLawFitResult]
        Best-fit parameters and KS distance, or `None` when no valid tail exists.

    Notes
    -----
    - This compatibility wrapper delegates to `fit_power_law_clauset` with
      `fit_method="approx"` and `discrete=False`.
    - It is fit-only and does not perform the Clauset bootstrap
      goodness-of-fit test or likelihood-ratio comparisons.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = 1.0 * np.power(1.0 - rng.random(5000), -1.0 / (2.5 - 1.0))
    >>> fit = fit_power_law_clauset_continuous(x, min_tail=100)
    >>> fit is not None
    True
    """
    return fit_power_law_clauset(
        data,
        discrete=False,
        fit_method="approx",
        min_tail=min_tail,
        num_candidates=num_candidates,
        refine_window=refine_window,
    )


def fit_power_law_clauset(
    data: Iterable[float],
    *,
    discrete: bool = False,
    fit_method: str = "approx",
    xmin: Optional[float] = None,
    xmin_distance: str = "D",
    min_tail: int = 50,
    num_candidates: int = 200,
    refine_window: int = 50,
    verbose: int = 0,
) -> Optional[PowerLawFitResult]:
    """
    Summary
    -------
    Fit a power-law tail using a Clauset-style backend with selectable `xmin` search.

    Parameters
    ----------
    data : Iterable[float]
        One-dimensional sample of positive observations.
    discrete : bool, default=False
        Whether to treat the sample as a discrete integer distribution.
    fit_method : str, default="approx"
        Backend used to determine `xmin`. `"approx"` keeps the coarse candidate
        grid used in this repository, while `"powerlaw"` delegates the search to
        the external `powerlaw` package.
    xmin : Optional[float], default=None
        Optional fixed cutoff. When provided, no `xmin` search is performed.
    xmin_distance : str, default="D"
        Distance metric passed to `powerlaw` when it optimizes `xmin`.
    min_tail : int, default=50
        Minimum number of observations required in the fitted tail.
    num_candidates : int, default=200
        Number of coarse candidate cutoffs for `fit_method="approx"`.
    refine_window : int, default=50
        Local index window for `fit_method="approx"`.
    verbose : int, default=0
        Verbosity passed to `powerlaw.Fit` on the exact backend.

    Returns
    -------
    Optional[PowerLawFitResult]
        Fitted tail parameters, or `None` when no valid tail exists.

    Notes
    -----
    - The default `"approx"` backend keeps the coarse-grained `xmin`
      candidate search but evaluates each candidate with `powerlaw`.
    - This is a fit-only helper. Use `run_power_law_clauset_pipeline` for the
      full bootstrap and likelihood-ratio workflow.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = 5.0 * np.power(1.0 - rng.random(2000), -1.0 / (2.5 - 1.0))
    >>> fit = fit_power_law_clauset(x, fit_method="approx", min_tail=100)
    >>> fit is not None
    True
    """
    bundle = _fit_power_law_clauset_bundle(
        data,
        discrete=discrete,
        fit_method=fit_method,
        xmin=xmin,
        xmin_distance=xmin_distance,
        min_tail=min_tail,
        num_candidates=num_candidates,
        refine_window=refine_window,
        verbose=verbose,
    )
    return None if bundle is None else bundle.fit_result


def fit_power_law_powerlaw_package(
    data: Iterable[float],
    xmin: Optional[float] = None,
    *,
    discrete: bool = False,
    xmin_distance: str = "D",
    verbose: int = 0,
) -> Optional[PowerLawFitResult]:
    """
    Summary
    -------
    Fit a power law using the external `powerlaw` package.

    Parameters
    ----------
    data : Iterable[float]
        One-dimensional sample of positive observations.
    xmin : Optional[float], default=None
        Optional fixed tail cutoff. If omitted, the package searches for the
        Clauset-optimal `xmin`.
    discrete : bool, default=False
        Whether to fit a discrete power law.
    xmin_distance : str, default="D"
        Distance metric used by `powerlaw.Fit` when optimizing `xmin`.
    verbose : int, default=0
        Verbosity passed to `powerlaw.Fit`.

    Returns
    -------
    Optional[PowerLawFitResult]
        Fitted tail parameters, or `None` when the input sample is too small.

    Notes
    -----
    This is a fit-only wrapper around `powerlaw.Fit`. It does not run the
    Clauset semiparametric bootstrap or alternative-model comparisons.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = 1.0 * np.power(1.0 - rng.random(1000), -1.0 / (2.5 - 1.0))
    >>> fit = fit_power_law_powerlaw_package(x, verbose=0)
    >>> fit is not None
    True
    """
    arr = _positive_finite_sample(data)
    if arr.size < 2:
        return None

    bundle = _fit_powerlaw_exact_backend(
        arr,
        discrete=discrete,
        xmin=xmin,
        xmin_distance=xmin_distance,
        min_tail=2,
        verbose=verbose,
        method="powerlaw",
    )
    return None if bundle is None else bundle.fit_result


def fit_power_law_clauset_discrete_approx(
    data: Iterable[float],
    min_tail: int = 50,
    num_candidates: int = 200,
    refine_window: int = 50,
) -> Optional[PowerLawFitResult]:
    """
    Summary
    -------
    Fit a discrete power law using the repository's coarse-`xmin` Clauset backend.

    Parameters
    ----------
    data : Iterable[float]
        One-dimensional sample of positive integer observations.
    min_tail : int, default=50
        Minimum number of observations required in the fitted tail `x >= xmin`.
    num_candidates : int, default=200
        Number of coarse candidate cutoffs built from log-spaced tail sizes.
    refine_window : int, default=50
        Number of start-index positions explored on each side of the best
        coarse candidate during local refinement.

    Returns
    -------
    Optional[PowerLawFitResult]
        Best-fit parameters and KS distance, or `None` when no valid tail exists.

    Notes
    -----
    - This compatibility wrapper delegates to `fit_power_law_clauset` with
      `fit_method="approx"` and `discrete=True`.
    - It is fit-only and does not run the full Clauset pipeline.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = np.arange(1, 1001)
    >>> fit = fit_power_law_clauset_discrete_approx(x, min_tail=50)
    >>> fit is not None
    True
    """
    return fit_power_law_clauset(
        data,
        discrete=True,
        fit_method="approx",
        min_tail=min_tail,
        num_candidates=num_candidates,
        refine_window=refine_window,
    )


def clauset_power_law_gof_test(
    data: Iterable[float],
    *,
    discrete: bool = False,
    fit_method: str = "approx",
    xmin: Optional[float] = None,
    xmin_distance: str = "D",
    min_tail: int = 50,
    num_candidates: int = 200,
    refine_window: int = 50,
    bootstrap_runs: int = 1000,
    random_state: Optional[int] = 0,
    verbose: int = 0,
) -> Optional[PowerLawGoFResult]:
    """
    Summary
    -------
    Run the Clauset semiparametric bootstrap goodness-of-fit test.

    Parameters
    ----------
    data : Iterable[float]
        One-dimensional sample of positive observations.
    discrete : bool, default=False
        Whether to treat the sample as a discrete integer distribution.
    fit_method : str, default="approx"
        Backend used to determine `xmin`.
    xmin : Optional[float], default=None
        Optional fixed cutoff. When provided, the same fixed cutoff is used for
        all bootstrap refits.
    xmin_distance : str, default="D"
        Distance metric passed to `powerlaw` when it optimizes `xmin`.
    min_tail : int, default=50
        Minimum number of observations required in the fitted tail.
    num_candidates : int, default=200
        Number of coarse candidate cutoffs for `fit_method="approx"`.
    refine_window : int, default=50
        Local index window for `fit_method="approx"`.
    bootstrap_runs : int, default=1000
        Number of bootstrap replicates to attempt.
    random_state : Optional[int], default=0
        Seed used for the bootstrap RNG. Use `None` for non-deterministic draws.
    verbose : int, default=0
        Verbosity passed to `powerlaw.Fit` on the exact backend.

    Returns
    -------
    Optional[PowerLawGoFResult]
        Bootstrap goodness-of-fit summary, or `None` when the empirical fit is invalid.

    Notes
    -----
    The synthetic samples follow Clauset's semiparametric construction: the body
    below `xmin` is resampled from the empirical data and the tail above `xmin`
    is generated from the fitted power law.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = 5.0 * np.power(1.0 - rng.random(2000), -1.0 / (2.5 - 1.0))
    >>> result = clauset_power_law_gof_test(x, bootstrap_runs=25, random_state=0)
    >>> result is not None
    True
    """
    bundle = _fit_power_law_clauset_bundle(
        data,
        discrete=discrete,
        fit_method=fit_method,
        xmin=xmin,
        xmin_distance=xmin_distance,
        min_tail=min_tail,
        num_candidates=num_candidates,
        refine_window=refine_window,
        verbose=verbose,
    )
    if bundle is None:
        return None
    return _clauset_power_law_gof_from_bundle(
        bundle,
        discrete=discrete,
        fit_method=_normalize_fit_method(fit_method),
        xmin=xmin,
        xmin_distance=xmin_distance,
        min_tail=min_tail,
        num_candidates=num_candidates,
        refine_window=refine_window,
        bootstrap_runs=bootstrap_runs,
        random_state=random_state,
    )


def compare_power_law_to_alternatives(
    data: Iterable[float],
    *,
    discrete: bool = False,
    fit_method: str = "approx",
    xmin: Optional[float] = None,
    xmin_distance: str = "D",
    min_tail: int = 50,
    num_candidates: int = 200,
    refine_window: int = 50,
    alternatives: Sequence[str] = ("lognormal", "exponential", "truncated_power_law"),
    verbose: int = 0,
) -> tuple[PowerLawComparisonResult, ...]:
    """
    Summary
    -------
    Compare a fitted power law against alternative tail distributions.

    Parameters
    ----------
    data : Iterable[float]
        One-dimensional sample of positive observations.
    discrete : bool, default=False
        Whether to treat the sample as a discrete integer distribution.
    fit_method : str, default="approx"
        Backend used to determine `xmin`.
    xmin : Optional[float], default=None
        Optional fixed cutoff.
    xmin_distance : str, default="D"
        Distance metric passed to `powerlaw` when it optimizes `xmin`.
    min_tail : int, default=50
        Minimum number of observations required in the fitted tail.
    num_candidates : int, default=200
        Number of coarse candidate cutoffs for `fit_method="approx"`.
    refine_window : int, default=50
        Local index window for `fit_method="approx"`.
    alternatives : Sequence[str], default=("lognormal", "exponential", "truncated_power_law")
        Alternative distributions passed to `powerlaw.Fit.distribution_compare`.
    verbose : int, default=0
        Verbosity passed to `powerlaw.Fit` on the exact backend.

    Returns
    -------
    tuple[PowerLawComparisonResult, ...]
        One comparison result per requested alternative. Returns an empty tuple
        when the empirical power-law fit is invalid.

    Notes
    -----
    `favored_model` uses the Clauset likelihood-ratio convention: the sign of the
    ratio is trusted only when `p < 0.1`.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = 5.0 * np.power(1.0 - rng.random(2000), -1.0 / (2.5 - 1.0))
    >>> comps = compare_power_law_to_alternatives(x, alternatives=("exponential",))
    >>> len(comps)
    1
    """
    result = fit_power_law_with_alternatives(
        data,
        discrete=discrete,
        fit_method=fit_method,
        xmin=xmin,
        xmin_distance=xmin_distance,
        min_tail=min_tail,
        num_candidates=num_candidates,
        refine_window=refine_window,
        alternatives=alternatives,
        verbose=verbose,
    )
    if result is None:
        return tuple()
    return result.comparisons


def fit_power_law_with_alternatives(
    data: Iterable[float],
    *,
    discrete: bool = False,
    fit_method: str = "approx",
    xmin: Optional[float] = None,
    xmin_distance: str = "D",
    min_tail: int = 50,
    num_candidates: int = 200,
    refine_window: int = 50,
    alternatives: Sequence[str] = ("lognormal", "exponential", "truncated_power_law"),
    verbose: int = 0,
) -> Optional[PowerLawFitComparisonSummary]:
    """
    Summary
    -------
    Fit a power-law tail and compare it to alternative tail distributions.

    Parameters
    ----------
    data : Iterable[float]
        One-dimensional sample of positive observations.
    discrete : bool, default=False
        Whether to treat the sample as a discrete integer distribution.
    fit_method : str, default="approx"
        Backend used to determine `xmin`.
    xmin : Optional[float], default=None
        Optional fixed cutoff.
    xmin_distance : str, default="D"
        Distance metric passed to `powerlaw` when it optimizes `xmin`.
    min_tail : int, default=50
        Minimum number of observations required in the fitted tail.
    num_candidates : int, default=200
        Number of coarse candidate cutoffs for `fit_method="approx"`.
    refine_window : int, default=50
        Local index window for `fit_method="approx"`.
    alternatives : Sequence[str], default=("lognormal", "exponential", "truncated_power_law")
        Alternative distributions passed to `powerlaw.Fit.distribution_compare`.
    verbose : int, default=0
        Verbosity passed to `powerlaw.Fit` on the exact backend.

    Returns
    -------
    Optional[PowerLawFitComparisonSummary]
        Fitted tail plus likelihood-ratio comparisons, or `None` when the
        empirical power-law fit is invalid.

    Notes
    -----
    This helper reuses the same fitted `xmin` and power-law parameters for the
    alternative-model checks, so it avoids paying for the fit twice.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = 5.0 * np.power(1.0 - rng.random(2000), -1.0 / (2.5 - 1.0))
    >>> result = fit_power_law_with_alternatives(x, alternatives=("exponential",))
    >>> result is not None
    True
    """
    normalized_fit_method = _normalize_fit_method(fit_method)
    bundle = _fit_power_law_clauset_bundle(
        data,
        discrete=discrete,
        fit_method=normalized_fit_method,
        xmin=xmin,
        xmin_distance=xmin_distance,
        min_tail=min_tail,
        num_candidates=num_candidates,
        refine_window=refine_window,
        verbose=verbose,
    )
    if bundle is None:
        return None

    candidate_models = _candidate_tail_models(alternatives)
    model_cache = _build_tail_model_fit_cache(
        bundle,
        candidates=candidate_models,
    )
    comparisons = _compare_power_law_to_alternatives_from_bundle(
        bundle,
        alternatives=alternatives,
        fit_method=normalized_fit_method,
        model_cache=model_cache,
    )
    pairwise_comparisons = _compare_tail_model_pairs_from_bundle(
        bundle,
        candidates=candidate_models,
        fit_method=normalized_fit_method,
        model_cache=model_cache,
    )
    model_scores = _score_tail_model_candidates(
        bundle,
        candidates=candidate_models,
        model_cache=model_cache,
    )
    return PowerLawFitComparisonSummary(
        fit_result=bundle.fit_result,
        comparisons=comparisons,
        pairwise_comparisons=pairwise_comparisons,
        best_model=_select_best_tail_model(model_scores),
        best_model_criterion="aic",
        model_scores=model_scores,
        raw_fit=bundle.raw_fit,
        fit_method=normalized_fit_method,
    )


def run_power_law_clauset_pipeline(
    data: Iterable[float],
    *,
    discrete: bool = False,
    fit_method: str = "approx",
    xmin: Optional[float] = None,
    xmin_distance: str = "D",
    min_tail: int = 50,
    num_candidates: int = 200,
    refine_window: int = 50,
    bootstrap_runs: int = 1000,
    random_state: Optional[int] = 0,
    p_threshold: float = 0.1,
    alternatives: Sequence[str] = ("lognormal", "exponential", "truncated_power_law"),
    verbose: int = 0,
) -> Optional[ClausetPipelineResult]:
    """
    Summary
    -------
    Run the full Clauset workflow with a selectable `xmin` search backend.

    Parameters
    ----------
    data : Iterable[float]
        One-dimensional sample of positive observations.
    discrete : bool, default=False
        Whether to treat the sample as a discrete integer distribution.
    fit_method : str, default="approx"
        Backend used to determine `xmin`.
    xmin : Optional[float], default=None
        Optional fixed cutoff.
    xmin_distance : str, default="D"
        Distance metric passed to `powerlaw` when it optimizes `xmin`.
    min_tail : int, default=50
        Minimum number of observations required in the fitted tail.
    num_candidates : int, default=200
        Number of coarse candidate cutoffs for `fit_method="approx"`.
    refine_window : int, default=50
        Local index window for `fit_method="approx"`.
    bootstrap_runs : int, default=1000
        Number of bootstrap replicates to attempt.
    random_state : Optional[int], default=0
        Seed used for the bootstrap RNG. Use `None` for non-deterministic draws.
    p_threshold : float, default=0.1
        Plausibility cutoff applied to the bootstrap goodness-of-fit p-value.
    alternatives : Sequence[str], default=("lognormal", "exponential", "truncated_power_law")
        Alternative distributions passed to `powerlaw.Fit.distribution_compare`.
    verbose : int, default=0
        Verbosity passed to `powerlaw.Fit` on the exact backend.

    Returns
    -------
    Optional[ClausetPipelineResult]
        Combined Clauset workflow output, or `None` when the empirical fit is invalid.

    Notes
    -----
    The default `"approx"` backend preserves the repository's sparse `xmin`
    candidate search while running the downstream Clauset bootstrap and
    likelihood-ratio comparisons through `powerlaw`.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = 5.0 * np.power(1.0 - rng.random(2000), -1.0 / (2.5 - 1.0))
    >>> result = run_power_law_clauset_pipeline(x, bootstrap_runs=25, random_state=0)
    >>> result is not None
    True
    """
    normalized_fit_method = _normalize_fit_method(fit_method)
    bundle = _fit_power_law_clauset_bundle(
        data,
        discrete=discrete,
        fit_method=normalized_fit_method,
        xmin=xmin,
        xmin_distance=xmin_distance,
        min_tail=min_tail,
        num_candidates=num_candidates,
        refine_window=refine_window,
        verbose=verbose,
    )
    if bundle is None:
        return None

    gof_result = _clauset_power_law_gof_from_bundle(
        bundle,
        discrete=discrete,
        fit_method=normalized_fit_method,
        xmin=xmin,
        xmin_distance=xmin_distance,
        min_tail=min_tail,
        num_candidates=num_candidates,
        refine_window=refine_window,
        bootstrap_runs=bootstrap_runs,
        random_state=random_state,
    )
    if gof_result is None:
        return None

    comparisons = _compare_power_law_to_alternatives_from_bundle(
        bundle,
        alternatives=alternatives,
        fit_method=normalized_fit_method,
    )
    p_threshold = float(p_threshold)
    return ClausetPipelineResult(
        fit_result=bundle.fit_result,
        gof_result=gof_result,
        comparisons=comparisons,
        power_law_plausible=bool(gof_result.p_value > p_threshold),
        p_threshold=p_threshold,
        fit_method=normalized_fit_method,
    )


def bootstrap_full_clauset_pipeline_parameters(
    data: Iterable[float],
    *,
    discrete: bool = False,
    fit_method: str = "approx",
    xmin: Optional[float] = None,
    xmin_distance: str = "D",
    min_tail: int = 50,
    num_candidates: int = 200,
    refine_window: int = 50,
    bootstrap_runs: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = 0,
    alternatives: Sequence[str] = ("lognormal", "exponential", "truncated_power_law"),
    verbose: int = 0,
    empirical_fit: Optional[PowerLawFitComparisonSummary] = None,
    show_progress: bool = False,
) -> Optional[FullPipelineBootstrapSummary]:
    """
    Summary
    -------
    Bootstrap all fitted tail-model parameters by rerunning the full Clauset pipeline.

    Parameters
    ----------
    data : Iterable[float]
        One-dimensional sample of positive observations.
    discrete : bool, default=False
        Whether to treat the sample as a discrete integer distribution.
    fit_method : str, default="approx"
        Backend used to determine `xmin` in every empirical and bootstrap refit.
    xmin : Optional[float], default=None
        Optional fixed cutoff. When `None`, each bootstrap replicate re-estimates
        `xmin` from the resampled sample.
    xmin_distance : str, default="D"
        Distance metric passed to `powerlaw` when it optimizes `xmin`.
    min_tail : int, default=50
        Minimum number of observations required in the fitted tail.
    num_candidates : int, default=200
        Number of coarse candidate cutoffs for `fit_method="approx"`.
    refine_window : int, default=50
        Local index window for `fit_method="approx"`.
    bootstrap_runs : int, default=1000
        Number of bootstrap replicates to attempt.
    alpha : float, default=0.05
        Significance level for the two-sided percentile confidence intervals.
    random_state : Optional[int], default=0
        Seed used for the bootstrap RNG. Use `None` for non-deterministic draws.
    alternatives : Sequence[str], default=("lognormal", "exponential", "truncated_power_law")
        Alternative distributions refit in every bootstrap replicate.
    verbose : int, default=0
        Verbosity passed to the empirical fit when `empirical_fit` is not supplied.
    empirical_fit : Optional[PowerLawFitComparisonSummary], default=None
        Optional precomputed empirical fit on `data`. When supplied, the helper
        reuses it for the point estimates and bootstraps only the replicate refits.
    show_progress : bool, default=False
        If True, display a `tqdm` progress bar over bootstrap replicates when
        `tqdm` is available in the active environment.

    Returns
    -------
    Optional[FullPipelineBootstrapSummary]
        Empirical fit plus percentile-bootstrap intervals for each candidate
        model's fitted parameters, or `None` when the empirical fit is invalid.

    Notes
    -----
    This is a nonparametric full-sample bootstrap. Each replicate resamples the
    original positive finite sample with replacement and reruns the Clauset
    fitting pipeline, including `xmin` selection and the alternative-model fits.
    The returned intervals therefore target uncertainty for the entire fitting
    pipeline rather than conditional-on-`xmin` uncertainty.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = 5.0 * np.power(1.0 - rng.random(1000), -1.0 / (2.5 - 1.0))
    >>> result = bootstrap_full_clauset_pipeline_parameters(
    ...     x,
    ...     fit_method="approx",
    ...     min_tail=200,
    ...     num_candidates=40,
    ...     refine_window=20,
    ...     bootstrap_runs=8,
    ...     alpha=0.1,
    ...     random_state=0,
    ... )
    >>> result is not None
    True
    """
    bootstrap_runs = int(bootstrap_runs)
    if bootstrap_runs < 1:
        raise ValueError("bootstrap_runs must be >= 1.")
    alpha = float(alpha)
    if (not np.isfinite(alpha)) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be a finite float in the open interval (0, 1).")

    normalized_fit_method = _normalize_fit_method(fit_method)
    normalized_alternatives = _normalize_alternatives(alternatives)
    sample = _positive_finite_sample(data)
    if sample.size < 2:
        return None

    resolved_empirical_fit = empirical_fit
    if resolved_empirical_fit is None:
        resolved_empirical_fit = fit_power_law_with_alternatives(
            sample,
            discrete=discrete,
            fit_method=normalized_fit_method,
            xmin=xmin,
            xmin_distance=xmin_distance,
            min_tail=min_tail,
            num_candidates=num_candidates,
            refine_window=refine_window,
            alternatives=normalized_alternatives,
            verbose=verbose,
        )
    if resolved_empirical_fit is None or resolved_empirical_fit.raw_fit is None:
        return None

    candidate_models = _candidate_tail_models(normalized_alternatives)
    empirical_parameters = _extract_empirical_tail_model_parameters(
        resolved_empirical_fit.raw_fit,
        candidates=candidate_models,
    )

    draws_by_model: dict[str, dict[str, list[float]]] = {
        model: {parameter: [] for parameter in parameter_names}
        for model, parameter_names in ((model, details["parameter_names"]) for model, details in empirical_parameters.items())
    }
    valid_runs_by_model = {model: 0 for model in empirical_parameters}
    xmin_draws: list[float] = []

    rng = np.random.default_rng(random_state)
    successful_pipeline_runs = 0
    replicate_iterator = range(bootstrap_runs)
    if show_progress and _tqdm is not None:
        replicate_iterator = _tqdm(
            replicate_iterator,
            total=bootstrap_runs,
            desc="Clauset bootstrap",
            leave=False,
        )

    for _ in replicate_iterator:
        bootstrap_sample = np.asarray(rng.choice(sample, size=sample.size, replace=True), dtype=float)
        bootstrap_fit = fit_power_law_with_alternatives(
            bootstrap_sample,
            discrete=discrete,
            fit_method=normalized_fit_method,
            xmin=xmin,
            xmin_distance=xmin_distance,
            min_tail=min_tail,
            num_candidates=num_candidates,
            refine_window=refine_window,
            alternatives=normalized_alternatives,
            verbose=0,
        )
        if bootstrap_fit is None or bootstrap_fit.raw_fit is None:
            continue
        successful_pipeline_runs += 1
        if bootstrap_fit.fit_result is not None:
            bootstrap_xmin = float(bootstrap_fit.fit_result.xmin)
            if np.isfinite(bootstrap_xmin):
                xmin_draws.append(bootstrap_xmin)

        for model, details in empirical_parameters.items():
            parameter_names = details["parameter_names"]
            if not parameter_names:
                continue
            try:
                with _suppress_powerlaw_warnings():
                    distribution = getattr(bootstrap_fit.raw_fit, model)
            except Exception:
                continue

            values: list[float] = []
            valid_model_fit = True
            for parameter_name in parameter_names:
                try:
                    value = float(getattr(distribution, parameter_name))
                except Exception:
                    valid_model_fit = False
                    break
                if not np.isfinite(value):
                    valid_model_fit = False
                    break
                values.append(value)
            if not valid_model_fit:
                continue

            valid_runs_by_model[model] += 1
            for parameter_name, value in zip(parameter_names, values):
                draws_by_model[model][parameter_name].append(value)

    model_summaries: list[TailModelBootstrapSummary] = []
    for model, details in empirical_parameters.items():
        parameter_names = details["parameter_names"]
        estimates = details["estimates"]
        valid_runs = int(valid_runs_by_model.get(model, 0))
        parameter_intervals: list[TailModelParameterBootstrapInterval] = []
        for parameter_name in parameter_names:
            draws = np.asarray(draws_by_model[model][parameter_name], dtype=float)
            ci_low: Optional[float] = None
            ci_high: Optional[float] = None
            std: Optional[float] = None
            if draws.size >= 2 and np.all(np.isfinite(draws)):
                ci_low = float(np.quantile(draws, alpha / 2.0))
                ci_high = float(np.quantile(draws, 1.0 - alpha / 2.0))
                std = float(np.std(draws, ddof=1))
            parameter_intervals.append(
                TailModelParameterBootstrapInterval(
                    parameter=parameter_name,
                    estimate=float(estimates[parameter_name]),
                    ci_low=ci_low,
                    ci_high=ci_high,
                    std=std,
                )
            )
        model_summaries.append(
            TailModelBootstrapSummary(
                model=model,
                parameter_intervals=tuple(parameter_intervals),
                requested_runs=bootstrap_runs,
                valid_runs=valid_runs,
            )
        )

    xmin_summary: Optional[ScalarBootstrapInterval] = None
    empirical_xmin = float(resolved_empirical_fit.fit_result.xmin)
    if np.isfinite(empirical_xmin):
        xmin_ci_low: Optional[float] = None
        xmin_ci_high: Optional[float] = None
        xmin_std: Optional[float] = None
        xmin_valid_runs = int(len(xmin_draws))
        if xmin_valid_runs >= 2:
            xmin_draws_array = np.asarray(xmin_draws, dtype=float)
            if np.all(np.isfinite(xmin_draws_array)):
                xmin_ci_low = float(np.quantile(xmin_draws_array, alpha / 2.0))
                xmin_ci_high = float(np.quantile(xmin_draws_array, 1.0 - alpha / 2.0))
                xmin_std = float(np.std(xmin_draws_array, ddof=1))
        xmin_summary = ScalarBootstrapInterval(
            quantity="xmin",
            estimate=empirical_xmin,
            ci_low=xmin_ci_low,
            ci_high=xmin_ci_high,
            std=xmin_std,
            valid_runs=xmin_valid_runs,
        )

    return FullPipelineBootstrapSummary(
        empirical_fit=resolved_empirical_fit,
        model_summaries=tuple(model_summaries),
        alpha=alpha,
        bootstrap_runs=bootstrap_runs,
        successful_pipeline_runs=successful_pipeline_runs,
        xmin_summary=xmin_summary,
        random_state=None if random_state is None else int(random_state),
    )


def power_law_pdf(x: Iterable[float], alpha: float, xmin: float) -> np.ndarray:
    """
    Summary
    -------
    Evaluate the continuous power-law PDF on `x` for a fitted `(alpha, xmin)` pair.

    Parameters
    ----------
    x : Iterable[float]
        Evaluation points.
    alpha : float
        Power-law exponent, expected to be greater than one.
    xmin : float
        Lower cutoff of the power-law tail, expected to be positive.

    Returns
    -------
    np.ndarray
        Density values, with zeros below `xmin`.

    Notes
    -----
    Uses `p(x) = (alpha - 1) / xmin * (x / xmin)^(-alpha)` for `x >= xmin`.

    Examples
    --------
    >>> vals = power_law_pdf([1.0, 2.0, 4.0], alpha=2.5, xmin=1.0)
    >>> vals.shape
    (3,)
    """
    x_arr = _coerce_array(x)
    density = np.zeros_like(x_arr, dtype=float)
    if alpha <= 1.0 or xmin <= 0.0:
        return density
    mask = np.isfinite(x_arr) & (x_arr >= xmin)
    if np.any(mask):
        density[mask] = ((alpha - 1.0) / xmin) * np.power(x_arr[mask] / xmin, -alpha)
    return density


def power_law_survival(x: Iterable[float], alpha: float, xmin: float) -> np.ndarray:
    """
    Summary
    -------
    Evaluate the conditional survival function of a continuous power-law tail.

    Parameters
    ----------
    x : Iterable[float]
        Evaluation points.
    alpha : float
        Power-law exponent, expected to be greater than one.
    xmin : float
        Lower cutoff of the power-law tail, expected to be positive.

    Returns
    -------
    np.ndarray
        Conditional survival values `P(X >= x | X >= xmin)` for `x >= xmin`,
        with zeros below `xmin`.

    Notes
    -----
    For a continuous power law, the conditional tail survival is
    `(x / xmin) ** (1 - alpha)` for `x >= xmin`.

    Examples
    --------
    >>> vals = power_law_survival([1.0, 2.0, 4.0], alpha=2.5, xmin=1.0)
    >>> np.allclose(vals, np.array([1.0, 2.0 ** -1.5, 4.0 ** -1.5]))
    True
    """
    x_arr = _coerce_array(x)
    survival = np.zeros_like(x_arr, dtype=float)
    if alpha <= 1.0 or xmin <= 0.0:
        return survival
    mask = np.isfinite(x_arr) & (x_arr >= xmin)
    if np.any(mask):
        survival[mask] = np.power(x_arr[mask] / xmin, 1.0 - alpha)
    return survival


def power_law_survival_discrete(x: Iterable[float], alpha: float, xmin: float) -> np.ndarray:
    """
    Summary
    -------
    Evaluate the conditional survival function of a discrete power-law tail.

    Parameters
    ----------
    x : Iterable[float]
        Evaluation points, interpreted on the integer support of the fitted tail.
    alpha : float
        Power-law exponent, expected to be greater than one.
    xmin : float
        Lower cutoff of the fitted discrete tail.

    Returns
    -------
    np.ndarray
        Conditional survival values `P(X >= x | X >= xmin)` using the discrete
        power-law model, with zeros below `xmin`.

    Notes
    -----
    The discrete survival is `zeta(alpha, ceil(x)) / zeta(alpha, xmin)` for
    `x >= xmin`, where `zeta` is the Hurwitz zeta function.

    Examples
    --------
    >>> vals = power_law_survival_discrete([2, 3, 4], alpha=2.5, xmin=2)
    >>> vals.shape
    (3,)
    """
    x_arr = _coerce_array(x)
    survival = np.zeros_like(x_arr, dtype=float)
    if alpha <= 1.0 or xmin < 1.0:
        return survival
    mask = np.isfinite(x_arr) & (x_arr >= xmin)
    if np.any(mask):
        k = np.ceil(x_arr[mask]).astype(int)
        survival[mask] = _hurwitz_zeta(alpha, k) / _hurwitz_zeta(alpha, int(np.ceil(xmin)))
    return survival


def _fit_power_law_clauset_bundle(
    data: Iterable[float],
    *,
    discrete: bool,
    fit_method: str,
    xmin: Optional[float],
    xmin_distance: str,
    min_tail: int,
    num_candidates: int,
    refine_window: int,
    verbose: int,
) -> Optional[_PowerlawFitBundle]:
    arr = _prepare_powerlaw_sample(data, discrete=discrete)
    fit_method = _normalize_fit_method(fit_method)
    min_tail, num_candidates, refine_window = _coerce_search_controls(
        min_tail=min_tail,
        num_candidates=num_candidates,
        refine_window=refine_window,
    )

    if arr.size < min_tail:
        return None

    if xmin is not None:
        bundle = _fit_powerlaw_fixed_xmin(
            arr,
            xmin=float(xmin),
            discrete=discrete,
            xmin_distance=xmin_distance,
            verbose=0,
            method=fit_method,
        )
        if bundle is None or bundle.fit_result.n_tail < min_tail:
            return None
        return bundle

    if fit_method == "approx":
        return _fit_powerlaw_approx_backend(
            arr,
            discrete=discrete,
            xmin_distance=xmin_distance,
            min_tail=min_tail,
            num_candidates=num_candidates,
            refine_window=refine_window,
        )

    return _fit_powerlaw_exact_backend(
        arr,
        discrete=discrete,
        xmin=None,
        xmin_distance=xmin_distance,
        min_tail=min_tail,
        verbose=verbose,
        method="powerlaw",
    )


def _fit_powerlaw_approx_backend(
    arr: np.ndarray,
    *,
    discrete: bool,
    xmin_distance: str,
    min_tail: int,
    num_candidates: int,
    refine_window: int,
) -> Optional[_PowerlawFitBundle]:
    x_sorted = np.sort(arr)
    if x_sorted.size < min_tail:
        return None

    coarse_indices = _candidate_start_indices_for_sorted_sample(
        x_sorted,
        min_tail=min_tail,
        num_candidates=num_candidates,
    )
    if coarse_indices.size == 0:
        return None

    best_bundle: Optional[_PowerlawFitBundle] = None
    best_start_idx: Optional[int] = None
    for start_idx in coarse_indices:
        bundle = _fit_powerlaw_fixed_xmin(
            arr,
            xmin=float(x_sorted[int(start_idx)]),
            discrete=discrete,
            xmin_distance=xmin_distance,
            verbose=0,
            method="approx",
        )
        if bundle is None or bundle.fit_result.n_tail < min_tail:
            continue
        if _bundle_is_better(bundle, best_bundle):
            best_bundle = bundle
            best_start_idx = int(start_idx)

    if best_bundle is None or best_start_idx is None:
        return None

    if refine_window <= 0:
        return best_bundle

    refine_lo = max(0, best_start_idx - refine_window)
    refine_hi = min(x_sorted.size - 2, best_start_idx + refine_window)
    refine_tail_sizes = x_sorted.size - np.arange(refine_lo, refine_hi + 1, dtype=int)
    refine_indices = _candidate_start_indices_from_tail_sizes(x_sorted, refine_tail_sizes)

    for start_idx in refine_indices:
        bundle = _fit_powerlaw_fixed_xmin(
            arr,
            xmin=float(x_sorted[int(start_idx)]),
            discrete=discrete,
            xmin_distance=xmin_distance,
            verbose=0,
            method="approx",
        )
        if bundle is None or bundle.fit_result.n_tail < min_tail:
            continue
        if _bundle_is_better(bundle, best_bundle):
            best_bundle = bundle

    return best_bundle


def _fit_powerlaw_exact_backend(
    arr: np.ndarray,
    *,
    discrete: bool,
    xmin: Optional[float],
    xmin_distance: str,
    min_tail: int,
    verbose: int,
    method: str,
) -> Optional[_PowerlawFitBundle]:
    xmin_arg: Any = None
    if xmin is not None:
        xmin_arg = float(xmin)
    else:
        xmin_arg = _powerlaw_xmin_search_range(np.sort(arr), discrete=discrete, min_tail=min_tail)

    fit_obj = _call_powerlaw_fit(
        arr,
        xmin=xmin_arg,
        discrete=discrete,
        xmin_distance=xmin_distance,
        verbose=verbose,
    )
    bundle = _bundle_from_powerlaw_fit(
        arr,
        fit_obj,
        method=method,
        score_name=xmin_distance,
    )
    if bundle is None or bundle.fit_result.n_tail < min_tail:
        return None
    return bundle


def _fit_powerlaw_fixed_xmin(
    arr: np.ndarray,
    *,
    xmin: float,
    discrete: bool,
    xmin_distance: str,
    verbose: int,
    method: str,
) -> Optional[_PowerlawFitBundle]:
    if (not np.isfinite(xmin)) or xmin <= 0:
        return None
    fit_obj = _call_powerlaw_fit(
        arr,
        xmin=float(xmin),
        discrete=discrete,
        xmin_distance=xmin_distance,
        verbose=verbose,
    )
    return _bundle_from_powerlaw_fit(
        arr,
        fit_obj,
        method=method,
        score_name=xmin_distance,
    )


def _bundle_from_powerlaw_fit(
    arr: np.ndarray,
    fit_obj: Any,
    *,
    method: str,
    score_name: str,
) -> Optional[_PowerlawFitBundle]:
    with _suppress_powerlaw_warnings():
        power_law_fit = fit_obj.power_law
        alpha = float(getattr(power_law_fit, "alpha", np.nan))
        xmin_fit = float(getattr(power_law_fit, "xmin", np.nan))
        ks_stat = float(getattr(power_law_fit, "D", np.nan))
        fit_score = float(getattr(power_law_fit, score_name, np.nan))
    # The third-party package raises `noise_flag` when the optimizer fails or
    # parks the parameters on the admissible boundary. Treat those fits as
    # invalid so downstream plots skip unstable overlays.
    if bool(getattr(power_law_fit, "noise_flag", False)):
        return None
    if (not np.isfinite(alpha)) or (not np.isfinite(xmin_fit)) or (not np.isfinite(ks_stat)) or (not np.isfinite(fit_score)):
        return None

    n_tail = int(np.count_nonzero(arr >= xmin_fit))
    if n_tail < 2:
        return None

    alpha_sigma_raw = getattr(power_law_fit, "standard_err", None)
    if alpha_sigma_raw is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            alpha_sigma_raw = getattr(power_law_fit, "sigma", None)
    alpha_sigma = float(alpha_sigma_raw) if alpha_sigma_raw is not None and np.isfinite(alpha_sigma_raw) else None
    fit_result = PowerLawFitResult(
        alpha=alpha,
        xmin=xmin_fit,
        ks_stat=ks_stat,
        n_tail=n_tail,
        method=method,
        alpha_sigma=alpha_sigma,
    )
    return _PowerlawFitBundle(
        fit_result=fit_result,
        raw_fit=fit_obj,
        fit_score=fit_score,
        sample=arr,
    )


def _call_powerlaw_fit(
    arr: np.ndarray,
    *,
    xmin: Any,
    discrete: bool,
    xmin_distance: str,
    verbose: int,
) -> Any:
    powerlaw = _import_powerlaw()
    with _suppress_powerlaw_warnings():
        return powerlaw.Fit(
            arr,
            xmin=xmin,
            discrete=discrete,
            xmin_distance=xmin_distance,
            verbose=verbose,
        )


def _clauset_power_law_gof_from_bundle(
    bundle: _PowerlawFitBundle,
    *,
    discrete: bool,
    fit_method: str,
    xmin: Optional[float],
    xmin_distance: str,
    min_tail: int,
    num_candidates: int,
    refine_window: int,
    bootstrap_runs: int,
    random_state: Optional[int],
) -> Optional[PowerLawGoFResult]:
    bootstrap_runs = max(int(bootstrap_runs), 1)
    rng = np.random.default_rng(random_state)
    fit_result = bundle.fit_result
    n_total = bundle.sample.size
    n_tail = fit_result.n_tail
    n_body = n_total - n_tail
    body = bundle.sample[bundle.sample < fit_result.xmin]

    ks_stats: list[float] = []
    for _ in range(bootstrap_runs):
        synthetic = _draw_semiparametric_bootstrap_sample(
            body=body,
            n_body=n_body,
            n_tail=n_tail,
            fitted_power_law=bundle.raw_fit,
            rng=rng,
        )
        synthetic_bundle = _fit_power_law_clauset_bundle(
            synthetic,
            discrete=discrete,
            fit_method=fit_method,
            xmin=xmin,
            xmin_distance=xmin_distance,
            min_tail=min_tail,
            num_candidates=num_candidates,
            refine_window=refine_window,
            verbose=0,
        )
        if synthetic_bundle is None:
            continue
        ks_stats.append(float(synthetic_bundle.fit_result.ks_stat))

    if not ks_stats:
        return None

    ks_array = np.asarray(ks_stats, dtype=float)
    p_value = float(np.mean(ks_array >= fit_result.ks_stat))
    return PowerLawGoFResult(
        fit_result=fit_result,
        p_value=p_value,
        bootstrap_runs=int(ks_array.size),
        tail_fraction=float(fit_result.n_tail / n_total),
        random_state=None if random_state is None else int(random_state),
        fit_method=fit_method,
    )


def _compare_power_law_to_alternatives_from_bundle(
    bundle: _PowerlawFitBundle,
    *,
    alternatives: Sequence[str],
    fit_method: str,
    model_cache: Optional[Mapping[str, _TailModelFitCacheEntry]] = None,
) -> tuple[PowerLawComparisonResult, ...]:
    results: list[PowerLawComparisonResult] = []
    for alternative in _normalize_alternatives(alternatives):
        comparison = _compare_two_tail_models_from_bundle(
            bundle,
            model_a="power_law",
            model_b=alternative,
            fit_method=fit_method,
            model_cache=model_cache,
        )
        results.append(
            PowerLawComparisonResult(
                alternative=alternative,
                loglikelihood_ratio=comparison.loglikelihood_ratio,
                p_value=comparison.p_value,
                favored_model=comparison.favored_model,
                nested=comparison.nested,
                fit_method=comparison.fit_method,
            )
        )
    return tuple(results)


def _compare_tail_model_pairs_from_bundle(
    bundle: _PowerlawFitBundle,
    *,
    candidates: Sequence[str],
    fit_method: str,
    model_cache: Optional[Mapping[str, _TailModelFitCacheEntry]] = None,
) -> tuple[TailModelComparisonResult, ...]:
    normalized_candidates = tuple(dict.fromkeys(str(model).strip() for model in candidates if str(model).strip()))
    results: list[TailModelComparisonResult] = []
    for index, model_a in enumerate(normalized_candidates):
        for model_b in normalized_candidates[index + 1 :]:
            results.append(
                _compare_two_tail_models_from_bundle(
                    bundle,
                    model_a=model_a,
                    model_b=model_b,
                    fit_method=fit_method,
                    model_cache=model_cache,
                )
            )
    return tuple(results)


def _compare_two_tail_models_from_bundle(
    bundle: _PowerlawFitBundle,
    *,
    model_a: str,
    model_b: str,
    fit_method: str,
    model_cache: Optional[Mapping[str, _TailModelFitCacheEntry]] = None,
) -> TailModelComparisonResult:
    nested = _is_nested_model_comparison(model_a, model_b)
    entry_a = None if model_cache is None else model_cache.get(model_a)
    entry_b = None if model_cache is None else model_cache.get(model_b)
    if (
        entry_a is not None
        and entry_b is not None
        and entry_a.loglikelihoods is not None
        and entry_b.loglikelihoods is not None
        and entry_a.loglikelihoods.shape == entry_b.loglikelihoods.shape
    ):
        try:
            from powerlaw.statistics import loglikelihood_ratio as powerlaw_loglikelihood_ratio

            ratio, p_value = powerlaw_loglikelihood_ratio(
                entry_a.loglikelihoods.copy(),
                entry_b.loglikelihoods.copy(),
                nested=nested,
            )
            ratio = float(ratio)
            p_value = float(p_value)
        except Exception:
            ratio = np.nan
            p_value = np.nan
        else:
            if np.isfinite(ratio) and np.isfinite(p_value):
                return TailModelComparisonResult(
                    model_a=model_a,
                    model_b=model_b,
                    loglikelihood_ratio=ratio,
                    p_value=p_value,
                    favored_model=_classify_pairwise_likelihood_ratio(ratio, p_value, model_a, model_b),
                    nested=nested,
                    fit_method=fit_method,
                )

    try:
        with _suppress_powerlaw_warnings():
            ratio, p_value = bundle.raw_fit.distribution_compare(model_a, model_b, nested=nested)
        ratio = float(ratio)
        p_value = float(p_value)
    except Exception:
        return TailModelComparisonResult(
            model_a=model_a,
            model_b=model_b,
            loglikelihood_ratio=None,
            p_value=None,
            favored_model="unavailable",
            nested=nested,
            fit_method=fit_method,
        )

    if (not np.isfinite(ratio)) or (not np.isfinite(p_value)):
        return TailModelComparisonResult(
            model_a=model_a,
            model_b=model_b,
            loglikelihood_ratio=None,
            p_value=None,
            favored_model="unavailable",
            nested=nested,
            fit_method=fit_method,
        )

    return TailModelComparisonResult(
        model_a=model_a,
        model_b=model_b,
        loglikelihood_ratio=ratio,
        p_value=p_value,
        favored_model=_classify_pairwise_likelihood_ratio(ratio, p_value, model_a, model_b),
        nested=nested,
        fit_method=fit_method,
    )


def _candidate_tail_models(alternatives: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(("power_law", *_normalize_alternatives(alternatives))))


def _extract_empirical_tail_model_parameters(
    raw_fit: Any,
    *,
    candidates: Sequence[str],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for model in candidates:
        try:
            with _suppress_powerlaw_warnings():
                distribution = getattr(raw_fit, model)
                parameter_names = tuple(getattr(distribution, "parameter_names", ()))
        except Exception:
            summaries[model] = {"parameter_names": tuple(), "estimates": {}}
            continue

        estimates: dict[str, float] = {}
        for parameter_name in parameter_names:
            try:
                value = float(getattr(distribution, parameter_name))
            except Exception:
                value = np.nan
            estimates[parameter_name] = value
        summaries[model] = {
            "parameter_names": parameter_names,
            "estimates": estimates,
        }
    return summaries


def _build_tail_model_fit_cache(
    bundle: _PowerlawFitBundle,
    *,
    candidates: Sequence[str],
) -> dict[str, _TailModelFitCacheEntry]:
    tail_sample = np.asarray(getattr(bundle.raw_fit, "data", bundle.sample), dtype=float)
    cache: dict[str, _TailModelFitCacheEntry] = {}
    for model in candidates:
        try:
            with _suppress_powerlaw_warnings():
                distribution = getattr(bundle.raw_fit, model)
                loglikelihoods = np.asarray(distribution.loglikelihoods(tail_sample), dtype=float)
                ks_stat = float(getattr(distribution, "D", np.nan))
                noise_flag = bool(getattr(distribution, "noise_flag", False))
                parameter_names = tuple(getattr(distribution, "parameter_names", ()))
                in_range = bool(distribution.in_range())
        except Exception:
            cache[model] = _TailModelFitCacheEntry(model=model)
            continue
        cache[model] = _TailModelFitCacheEntry(
            model=model,
            distribution=distribution,
            loglikelihoods=loglikelihoods,
            parameter_count=max(int(len(parameter_names)), 1),
            ks_stat=None if not np.isfinite(ks_stat) else ks_stat,
            noise_flag=noise_flag,
            in_range=in_range,
        )
    return cache


def _score_tail_model_candidates(
    bundle: _PowerlawFitBundle,
    *,
    candidates: Sequence[str],
    model_cache: Optional[Mapping[str, _TailModelFitCacheEntry]] = None,
) -> tuple[TailDistributionScore, ...]:
    tail_sample = np.asarray(getattr(bundle.raw_fit, "data", bundle.sample), dtype=float)
    n_obs = max(int(tail_sample.size), 1)
    scores: list[TailDistributionScore] = []
    resolved_cache = (
        dict(model_cache)
        if model_cache is not None
        else _build_tail_model_fit_cache(bundle, candidates=candidates)
    )

    for model in candidates:
        entry = resolved_cache.get(model)
        if entry is None or entry.loglikelihoods is None:
            scores.append(
                TailDistributionScore(
                    model=model,
                    loglikelihood=None,
                    parameter_count=0,
                    aic=None,
                    bic=None,
                    ks_stat=None,
                    noise_flag=True,
                    valid=False,
                )
            )
            continue

        loglikelihoods = entry.loglikelihoods
        ks_stat = np.nan if entry.ks_stat is None else float(entry.ks_stat)
        noise_flag = bool(entry.noise_flag)
        parameter_count = max(int(entry.parameter_count), 1)
        in_range = bool(entry.in_range)
        loglikelihood = float(np.sum(loglikelihoods)) if loglikelihoods.size and np.all(np.isfinite(loglikelihoods)) else np.nan
        aic = float(2 * parameter_count - 2 * loglikelihood) if np.isfinite(loglikelihood) else np.nan
        bic = (
            float(math.log(n_obs) * parameter_count - 2 * loglikelihood)
            if np.isfinite(loglikelihood)
            else np.nan
        )
        valid = bool(np.isfinite(loglikelihood) and np.isfinite(aic) and np.isfinite(ks_stat) and in_range and (not noise_flag))
        scores.append(
            TailDistributionScore(
                model=model,
                loglikelihood=None if not np.isfinite(loglikelihood) else loglikelihood,
                parameter_count=parameter_count,
                aic=None if not np.isfinite(aic) else aic,
                bic=None if not np.isfinite(bic) else bic,
                ks_stat=None if not np.isfinite(ks_stat) else ks_stat,
                noise_flag=noise_flag,
                valid=valid,
            )
        )

    return tuple(scores)


def _select_best_tail_model(model_scores: Sequence[TailDistributionScore]) -> str:
    valid_scores = [score for score in model_scores if score.valid and score.aic is not None]
    if not valid_scores:
        return "power_law"
    best = min(
        valid_scores,
        key=lambda score: (
            float(score.aic),
            math.inf if score.ks_stat is None else float(score.ks_stat),
            score.model,
        ),
    )
    return best.model


def _draw_semiparametric_bootstrap_sample(
    *,
    body: np.ndarray,
    n_body: int,
    n_tail: int,
    fitted_power_law: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_body > 0:
        body_draw = np.asarray(rng.choice(body, size=n_body, replace=True), dtype=float)
    else:
        body_draw = np.empty(0, dtype=float)

    tail_draw = _draw_powerlaw_tail_sample(fitted_power_law, size=n_tail, rng=rng)
    if body_draw.size == 0:
        return tail_draw
    if tail_draw.size == 0:
        return body_draw
    return np.concatenate([body_draw, tail_draw]).astype(float, copy=False)


def _draw_powerlaw_tail_sample(
    fitted_power_law: Any,
    *,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=float)

    seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint64))
    with _suppress_powerlaw_warnings(), _temporary_numpy_random_seed(seed):
        draws = fitted_power_law.power_law.generate_random(size=size)
    return np.asarray(draws, dtype=float).reshape(-1)


@contextlib.contextmanager
def _suppress_powerlaw_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@contextlib.contextmanager
def _temporary_numpy_random_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _normalize_fit_method(fit_method: str) -> str:
    fit_method = str(fit_method).strip().lower()
    if fit_method not in {"approx", "powerlaw"}:
        raise ValueError("fit_method must be one of {'approx', 'powerlaw'}.")
    return fit_method


def _normalize_alternatives(alternatives: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for alternative in alternatives:
        label = str(alternative).strip()
        if label:
            normalized.append(label)
    return tuple(normalized)


def _coerce_search_controls(
    *,
    min_tail: int,
    num_candidates: int,
    refine_window: int,
) -> tuple[int, int, int]:
    return max(int(min_tail), 2), max(int(num_candidates), 5), max(int(refine_window), 0)


def _candidate_start_indices_for_sorted_sample(
    x_sorted: np.ndarray,
    *,
    min_tail: int,
    num_candidates: int,
) -> np.ndarray:
    n = x_sorted.size
    n_coarse = min(num_candidates, max(n - min_tail + 1, 1))
    tail_sizes = np.rint(np.geomspace(min_tail, n, num=n_coarse)).astype(int)
    tail_sizes = np.clip(tail_sizes, min_tail, n)
    return _candidate_start_indices_from_tail_sizes(x_sorted, np.unique(tail_sizes))


def _candidate_start_indices_from_tail_sizes(x_sorted: np.ndarray, tail_sizes: np.ndarray) -> np.ndarray:
    n = x_sorted.size
    start_indices = n - np.asarray(tail_sizes, dtype=int)
    start_indices = start_indices[(start_indices >= 0) & (start_indices < (n - 1))]
    if start_indices.size == 0:
        return np.empty(0, dtype=int)

    start_indices = np.unique(start_indices.astype(int))
    distinct: list[int] = []
    last_xmin: Optional[float] = None
    for idx in np.sort(start_indices):
        xmin = float(x_sorted[int(idx)])
        if last_xmin is None or xmin != last_xmin:
            distinct.append(int(idx))
            last_xmin = xmin
    return np.asarray(distinct, dtype=int)


def _bundle_is_better(candidate: _PowerlawFitBundle, incumbent: Optional[_PowerlawFitBundle]) -> bool:
    if incumbent is None:
        return True
    if candidate.fit_score < incumbent.fit_score:
        return True
    return bool(
        np.isclose(candidate.fit_score, incumbent.fit_score, rtol=0.0, atol=1e-12)
        and candidate.fit_result.n_tail > incumbent.fit_result.n_tail
    )


def _powerlaw_xmin_search_range(
    x_sorted: np.ndarray,
    *,
    discrete: bool,
    min_tail: int,
) -> Optional[tuple[float, float]]:
    if x_sorted.size < max(min_tail, 2):
        return None

    max_allowed = float(x_sorted[x_sorted.size - min_tail])
    greater = x_sorted[x_sorted > max_allowed]
    if greater.size == 0:
        return None

    first_disallowed = float(greater[0])
    if discrete:
        upper = float(first_disallowed + 0.5)
    else:
        upper = float(np.nextafter(first_disallowed, math.inf))
    lower = float(x_sorted[0])
    if (not np.isfinite(upper)) or upper <= lower:
        return None
    return (lower, upper)


def _classify_likelihood_ratio(ratio: float, p_value: float, alternative: str) -> str:
    return _classify_pairwise_likelihood_ratio(ratio, p_value, "power_law", alternative)

def _is_nested_powerlaw_comparison(alternative: str) -> bool:
    return _is_nested_model_comparison("power_law", alternative)


def _classify_pairwise_likelihood_ratio(
    ratio: float,
    p_value: float,
    model_a: str,
    model_b: str,
) -> str:
    if p_value >= _CLAUSET_LR_P_THRESHOLD:
        return "undecided"
    if ratio > 0:
        return model_a
    if ratio < 0:
        return model_b
    return "undecided"


def _is_nested_model_comparison(model_a: str, model_b: str) -> bool:
    return {str(model_a).strip(), str(model_b).strip()} == {"power_law", "truncated_power_law"}


def _prepare_powerlaw_sample(data: Iterable[float], *, discrete: bool) -> np.ndarray:
    arr = _positive_finite_sample(data)
    if not discrete:
        return arr
    arr = np.rint(arr)
    return arr[arr >= 1]


def _import_powerlaw():
    try:
        import powerlaw
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "The 'powerlaw' package is required for Clauset power-law fitting. "
            "Install it in the active environment or use the local approximation helpers."
        ) from exc
    return powerlaw


def _coerce_array(x: Iterable[float]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=float)
    return np.asarray(list(x), dtype=float)


def _positive_finite_sample(data: Iterable[float]) -> np.ndarray:
    arr = _coerce_array(data)
    return arr[np.isfinite(arr) & (arr > 0)]


def _hurwitz_zeta(alpha: float, q) -> np.ndarray:
    try:
        from scipy.special import zeta as scipy_zeta
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for discrete power-law fitting and survival evaluation.") from exc
    return scipy_zeta(alpha, q)
