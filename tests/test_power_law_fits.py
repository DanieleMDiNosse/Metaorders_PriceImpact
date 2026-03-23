from __future__ import annotations

import unittest
import warnings
from unittest import mock

import numpy as np

from moimpact.power_law_fits import (
    ClausetPipelineResult,
    FullPipelineBootstrapSummary,
    PowerLawFitComparisonSummary,
    PowerLawFitResult,
    _bundle_from_powerlaw_fit,
    bootstrap_full_clauset_pipeline_parameters,
    compare_power_law_to_alternatives,
    fit_power_law_clauset,
    fit_power_law_with_alternatives,
    run_power_law_clauset_pipeline,
)


def _sample_continuous_power_law(*, seed: int, n: int, alpha: float, xmin: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return xmin * np.power(1.0 - rng.random(n), -1.0 / (alpha - 1.0))


def _format_fit_result(label: str, fit: PowerLawFitResult) -> str:
    return (
        f"{label}: alpha={fit.alpha:.6f}, xmin={fit.xmin:.6f}, "
        f"ks={fit.ks_stat:.6f}, n_tail={fit.n_tail}, method={fit.method}"
    )


def _format_pipeline_result(label: str, result: ClausetPipelineResult) -> str:
    comparisons = ", ".join(
        (
            f"{comp.alternative}: favored={comp.favored_model}, "
            f"R={comp.loglikelihood_ratio if comp.loglikelihood_ratio is not None else 'NA'}, "
            f"p={comp.p_value if comp.p_value is not None else 'NA'}"
        )
        for comp in result.comparisons
    )
    return (
        f"{label}: alpha={result.fit_result.alpha:.6f}, xmin={result.fit_result.xmin:.6f}, "
        f"ks={result.fit_result.ks_stat:.6f}, gof_p={result.gof_result.p_value:.6f}, "
        f"plausible={result.power_law_plausible}, comparisons=[{comparisons}]"
    )


class TestPowerLawFits(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import powerlaw  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"powerlaw is not available: {exc}") from exc

    def test_fit_power_law_clauset_approx_is_close_to_powerlaw_reference(self) -> None:
        sample = _sample_continuous_power_law(seed=1, n=5000, alpha=2.5, xmin=5.0)

        approx = fit_power_law_clauset(
            sample,
            fit_method="approx",
            min_tail=1000,
            num_candidates=2000,
            refine_window=80,
        )
        reference = fit_power_law_clauset(
            sample,
            fit_method="powerlaw",
            min_tail=1000,
        )

        self.assertIsNotNone(approx)
        self.assertIsNotNone(reference)
        assert approx is not None
        assert reference is not None
        print(_format_fit_result("approx_fit", approx))
        print(_format_fit_result("reference_fit", reference))

        summary = (
            f"approx(alpha={approx.alpha:.6f}, xmin={approx.xmin:.6f}, ks={approx.ks_stat:.6f}, n_tail={approx.n_tail}) "
            f"reference(alpha={reference.alpha:.6f}, xmin={reference.xmin:.6f}, ks={reference.ks_stat:.6f}, n_tail={reference.n_tail})"
        )
        self.assertLess(abs(approx.alpha - reference.alpha), 0.01, msg=summary)
        self.assertLess(abs(approx.xmin - reference.xmin), 0.5, msg=summary)
        self.assertGreaterEqual(approx.ks_stat + 1e-12, reference.ks_stat, msg=summary)
        self.assertLess(approx.ks_stat - reference.ks_stat, 5e-4, msg=summary)

    def test_full_clauset_pipeline_approx_accepts_synthetic_pareto(self) -> None:
        sample = _sample_continuous_power_law(seed=0, n=3000, alpha=2.5, xmin=5.0)

        result = run_power_law_clauset_pipeline(
            sample,
            fit_method="approx",
            min_tail=1000,
            num_candidates=60,
            refine_window=30,
            bootstrap_runs=12,
            random_state=0,
        )

        self.assertIsNotNone(result)
        assert result is not None
        print(_format_pipeline_result("approx_pipeline_seed0", result))
        self.assertGreater(result.gof_result.p_value, 0.1)
        comparisons = {comp.alternative: comp for comp in result.comparisons}
        self.assertIn("exponential", comparisons)
        self.assertEqual(comparisons["exponential"].favored_model, "power_law")
        self.assertIsNotNone(comparisons["exponential"].p_value)
        assert comparisons["exponential"].p_value is not None
        self.assertLess(comparisons["exponential"].p_value, 0.1)

    def test_full_clauset_pipeline_powerlaw_reference_accepts_synthetic_pareto(self) -> None:
        sample = _sample_continuous_power_law(seed=0, n=3000, alpha=2.5, xmin=5.0)

        result = run_power_law_clauset_pipeline(
            sample,
            fit_method="powerlaw",
            min_tail=1000,
            bootstrap_runs=12,
            random_state=0,
        )

        self.assertIsNotNone(result)
        assert result is not None
        print(_format_pipeline_result("reference_pipeline_seed0", result))
        self.assertGreater(result.gof_result.p_value, 0.1)
        comparisons = {comp.alternative: comp for comp in result.comparisons}
        self.assertIn("exponential", comparisons)
        self.assertEqual(comparisons["exponential"].favored_model, "power_law")
        self.assertIsNotNone(comparisons["exponential"].p_value)
        assert comparisons["exponential"].p_value is not None
        self.assertLess(comparisons["exponential"].p_value, 0.1)

    def test_approx_and_powerlaw_pipeline_results_agree_on_decision(self) -> None:
        sample = _sample_continuous_power_law(seed=2, n=3000, alpha=2.5, xmin=5.0)

        approx = run_power_law_clauset_pipeline(
            sample,
            fit_method="approx",
            min_tail=1000,
            num_candidates=60,
            refine_window=30,
            bootstrap_runs=12,
            random_state=0,
        )
        reference = run_power_law_clauset_pipeline(
            sample,
            fit_method="powerlaw",
            min_tail=1000,
            bootstrap_runs=12,
            random_state=0,
        )

        self.assertIsNotNone(approx)
        self.assertIsNotNone(reference)
        assert approx is not None
        assert reference is not None
        print(_format_pipeline_result("approx_pipeline_seed2", approx))
        print(_format_pipeline_result("reference_pipeline_seed2", reference))

        summary = (
            f"approx(alpha={approx.fit_result.alpha:.6f}, xmin={approx.fit_result.xmin:.6f}, "
            f"ks={approx.fit_result.ks_stat:.6f}, p={approx.gof_result.p_value:.6f}) "
            f"reference(alpha={reference.fit_result.alpha:.6f}, xmin={reference.fit_result.xmin:.6f}, "
            f"ks={reference.fit_result.ks_stat:.6f}, p={reference.gof_result.p_value:.6f})"
        )
        self.assertLess(abs(approx.fit_result.alpha - reference.fit_result.alpha), 0.01, msg=summary)
        self.assertLess(abs(approx.fit_result.xmin - reference.fit_result.xmin), 0.5, msg=summary)
        self.assertLess(approx.fit_result.ks_stat - reference.fit_result.ks_stat, 5e-4, msg=summary)
        self.assertEqual(approx.power_law_plausible, reference.power_law_plausible, msg=summary)

    def test_compare_power_law_to_alternatives_returns_expected_default_set(self) -> None:
        sample = _sample_continuous_power_law(seed=0, n=3000, alpha=2.5, xmin=5.0)
        comparisons = compare_power_law_to_alternatives(sample, fit_method="approx", min_tail=1000)
        self.assertEqual(
            tuple(comp.alternative for comp in comparisons),
            ("lognormal", "exponential", "truncated_power_law"),
        )

    def test_fit_power_law_with_alternatives_reuses_fit_and_returns_default_comparisons(self) -> None:
        sample = _sample_continuous_power_law(seed=0, n=3000, alpha=2.5, xmin=5.0)
        result = fit_power_law_with_alternatives(sample, fit_method="approx", min_tail=1000)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIsInstance(result, PowerLawFitComparisonSummary)
        self.assertEqual(
            tuple(comp.alternative for comp in result.comparisons),
            ("lognormal", "exponential", "truncated_power_law"),
        )
        self.assertEqual(result.best_model, "power_law")
        self.assertEqual(result.best_model_criterion, "aic")
        self.assertTrue(any(score.model == "power_law" for score in result.model_scores))
        self.assertEqual(
            {(comp.model_a, comp.model_b) for comp in result.pairwise_comparisons},
            {
                ("power_law", "lognormal"),
                ("power_law", "exponential"),
                ("power_law", "truncated_power_law"),
                ("lognormal", "exponential"),
                ("lognormal", "truncated_power_law"),
                ("exponential", "truncated_power_law"),
            },
        )
        pairwise = {(comp.model_a, comp.model_b): comp for comp in result.pairwise_comparisons}
        self.assertTrue(pairwise[("power_law", "truncated_power_law")].nested)
        self.assertFalse(pairwise[("lognormal", "exponential")].nested)
        comparisons = {comp.alternative: comp for comp in result.comparisons}
        self.assertIn("exponential", comparisons)
        self.assertEqual(comparisons["exponential"].favored_model, "power_law")
        self.assertIsNotNone(comparisons["exponential"].p_value)

    def test_bundle_from_powerlaw_fit_suppresses_lazy_warnings(self) -> None:
        class _FakeDistribution:
            alpha = 2.4
            xmin = 2.0
            D = 0.12
            standard_err = 0.08
            noise_flag = False

        class _WarningFit:
            @property
            def power_law(self):
                warnings.warn("lazy third-party warning", UserWarning)
                return _FakeDistribution()

        sample = np.array([1.0, 2.0, 3.0, 5.0, 8.0], dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bundle = _bundle_from_powerlaw_fit(sample, _WarningFit(), method="approx", score_name="D")

        self.assertIsNotNone(bundle)
        self.assertEqual(len(caught), 0)

    def test_bundle_from_powerlaw_fit_rejects_noisy_distribution(self) -> None:
        class _NoisyDistribution:
            alpha = 2.4
            xmin = 2.0
            D = 0.12
            standard_err = 0.08
            noise_flag = True

        class _NoisyFit:
            @property
            def power_law(self):
                return _NoisyDistribution()

        sample = np.array([1.0, 2.0, 3.0, 5.0, 8.0], dtype=float)
        bundle = _bundle_from_powerlaw_fit(sample, _NoisyFit(), method="approx", score_name="D")
        self.assertIsNone(bundle)

    def test_full_pipeline_parameter_bootstrap_is_deterministic(self) -> None:
        sample = np.random.default_rng(0).lognormal(mean=1.0, sigma=0.8, size=2000)
        empirical_fit = fit_power_law_with_alternatives(
            sample,
            fit_method="approx",
            min_tail=120,
            num_candidates=40,
            refine_window=20,
        )

        self.assertIsNotNone(empirical_fit)
        assert empirical_fit is not None

        result_a = bootstrap_full_clauset_pipeline_parameters(
            sample,
            fit_method="approx",
            min_tail=120,
            num_candidates=40,
            refine_window=20,
            bootstrap_runs=8,
            alpha=0.1,
            random_state=11,
            empirical_fit=empirical_fit,
        )
        result_b = bootstrap_full_clauset_pipeline_parameters(
            sample,
            fit_method="approx",
            min_tail=120,
            num_candidates=40,
            refine_window=20,
            bootstrap_runs=8,
            alpha=0.1,
            random_state=11,
            empirical_fit=empirical_fit,
        )

        self.assertIsNotNone(result_a)
        self.assertIsNotNone(result_b)
        assert result_a is not None
        assert result_b is not None
        self.assertIsInstance(result_a, FullPipelineBootstrapSummary)
        self.assertEqual(result_a.bootstrap_runs, 8)
        self.assertEqual(result_a.successful_pipeline_runs, result_b.successful_pipeline_runs)

        for summary_a, summary_b in zip(result_a.model_summaries, result_b.model_summaries):
            self.assertEqual(summary_a.model, summary_b.model)
            self.assertEqual(summary_a.valid_runs, summary_b.valid_runs)
            self.assertEqual(summary_a.requested_runs, summary_b.requested_runs)
            self.assertEqual(
                tuple(interval.parameter for interval in summary_a.parameter_intervals),
                tuple(interval.parameter for interval in summary_b.parameter_intervals),
            )
            np.testing.assert_allclose(
                [interval.estimate for interval in summary_a.parameter_intervals],
                [interval.estimate for interval in summary_b.parameter_intervals],
            )
            np.testing.assert_allclose(
                [
                    np.nan if interval.ci_low is None else interval.ci_low
                    for interval in summary_a.parameter_intervals
                ],
                [
                    np.nan if interval.ci_low is None else interval.ci_low
                    for interval in summary_b.parameter_intervals
                ],
                equal_nan=True,
            )
            np.testing.assert_allclose(
                [
                    np.nan if interval.ci_high is None else interval.ci_high
                    for interval in summary_a.parameter_intervals
                ],
                [
                    np.nan if interval.ci_high is None else interval.ci_high
                    for interval in summary_b.parameter_intervals
                ],
                equal_nan=True,
            )
            np.testing.assert_allclose(
                [
                    np.nan if interval.std is None else interval.std
                    for interval in summary_a.parameter_intervals
                ],
                [
                    np.nan if interval.std is None else interval.std
                    for interval in summary_b.parameter_intervals
                ],
                equal_nan=True,
            )
        if result_a.xmin_summary is not None and result_b.xmin_summary is not None:
            self.assertEqual(result_a.xmin_summary.valid_runs, result_b.xmin_summary.valid_runs)
            np.testing.assert_allclose(
                [
                    np.nan if result_a.xmin_summary.ci_low is None else result_a.xmin_summary.ci_low,
                    np.nan if result_a.xmin_summary.ci_high is None else result_a.xmin_summary.ci_high,
                    np.nan if result_a.xmin_summary.std is None else result_a.xmin_summary.std,
                ],
                [
                    np.nan if result_b.xmin_summary.ci_low is None else result_b.xmin_summary.ci_low,
                    np.nan if result_b.xmin_summary.ci_high is None else result_b.xmin_summary.ci_high,
                    np.nan if result_b.xmin_summary.std is None else result_b.xmin_summary.std,
                ],
                equal_nan=True,
            )

    def test_full_pipeline_parameter_bootstrap_summarizes_all_candidate_models(self) -> None:
        sample = np.random.default_rng(1).lognormal(mean=1.0, sigma=0.8, size=2200)
        result = bootstrap_full_clauset_pipeline_parameters(
            sample,
            fit_method="approx",
            min_tail=120,
            num_candidates=40,
            refine_window=20,
            bootstrap_runs=8,
            alpha=0.1,
            random_state=3,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            tuple(summary.model for summary in result.model_summaries),
            ("power_law", "lognormal", "exponential", "truncated_power_law"),
        )
        self.assertGreater(result.successful_pipeline_runs, 0)
        for summary in result.model_summaries:
            self.assertEqual(summary.requested_runs, 8)
            self.assertGreaterEqual(summary.valid_runs, 0)
            self.assertGreater(len(summary.parameter_intervals), 0)
            for interval in summary.parameter_intervals:
                self.assertTrue(hasattr(interval, "std"))
        self.assertIsNotNone(result.xmin_summary)
        assert result.xmin_summary is not None
        self.assertTrue(hasattr(result.xmin_summary, "std"))

    def test_full_pipeline_parameter_bootstrap_reruns_xmin_selection(self) -> None:
        sample = np.random.default_rng(2).lognormal(mean=1.0, sigma=0.8, size=1800)
        empirical_fit = fit_power_law_with_alternatives(
            sample,
            fit_method="approx",
            min_tail=120,
            num_candidates=30,
            refine_window=15,
        )

        self.assertIsNotNone(empirical_fit)
        assert empirical_fit is not None

        with mock.patch(
            "moimpact.power_law_fits.fit_power_law_with_alternatives",
            autospec=True,
            return_value=empirical_fit,
        ) as mocked_fit:
            result = bootstrap_full_clauset_pipeline_parameters(
                sample,
                fit_method="approx",
                min_tail=120,
                num_candidates=30,
                refine_window=15,
                bootstrap_runs=3,
                alpha=0.1,
                random_state=5,
                empirical_fit=empirical_fit,
            )

        self.assertIsNotNone(result)
        self.assertEqual(mocked_fit.call_count, 3)
        for call in mocked_fit.call_args_list:
            self.assertIsNone(call.kwargs["xmin"])


if __name__ == "__main__":
    unittest.main()
