from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

import scripts.metaorder_distributions as metaorder_distributions
from scripts.metaorder_distributions import (
    _annotation_text_from_fit_row,
    _build_best_vs_second_review_table,
    _compact_fit_summary_table,
    _resolve_powerlaw_fit_worker_count,
    build_distribution_figure_from_saved_outputs,
)


class TestMetaorderDistributionsReview(unittest.TestCase):
    def test_annotation_text_includes_bootstrap_intervals(self) -> None:
        text = _annotation_text_from_fit_row(
            {
                "sample_size": 120,
                "fit_success": True,
                "best_fit_model": "lognormal",
                "xmin": 1.25,
                "bootstrap_full_pipeline_enabled": True,
                "bootstrap_alpha": 0.05,
                "bootstrap_xmin_ci_low": 1.0,
                "bootstrap_xmin_ci_high": 1.5,
                "alpha": 2.3456,
                "power_law_alpha_ci_low": 2.1,
                "power_law_alpha_ci_high": 2.6,
            }
        )

        self.assertIn("Best fit = lognormal", text)
        self.assertIn("x_min = 1.25", text)
        self.assertIn("95% CI [1, 1.5]", text)
        self.assertIn("alpha = 2.3456", text)
        self.assertIn("95% CI [2.1000, 2.6000]", text)

    def test_best_vs_second_review_includes_bootstrap_intervals(self) -> None:
        fit_summary = pd.DataFrame(
            [
                {
                    "group": "client",
                    "metric": "q_over_v",
                    "power_law_loglikelihood": -102.0,
                    "power_law_alpha": 2.4,
                    "power_law_alpha_ci_low": 2.2,
                    "power_law_alpha_ci_high": 2.6,
                    "power_law_alpha_std": 0.11,
                    "lognormal_loglikelihood": -99.0,
                    "lognormal_mu": 0.5,
                    "lognormal_mu_ci_low": 0.4,
                    "lognormal_mu_ci_high": 0.6,
                    "lognormal_mu_std": 0.07,
                    "lognormal_sigma": 0.8,
                    "lognormal_sigma_ci_low": 0.7,
                    "lognormal_sigma_ci_high": 0.9,
                    "lognormal_sigma_std": 0.05,
                    "truncated_power_law_loglikelihood": -100.0,
                    "truncated_power_law_alpha": 1.6,
                    "truncated_power_law_alpha_ci_low": 1.3,
                    "truncated_power_law_alpha_ci_high": 1.9,
                    "truncated_power_law_alpha_std": 0.09,
                    "truncated_power_law_lambda": 0.3,
                    "truncated_power_law_lambda_ci_low": 0.2,
                    "truncated_power_law_lambda_ci_high": 0.4,
                    "truncated_power_law_lambda_std": 0.03,
                    "loglikelihood_ratio_lognormal_vs_truncated_power_law": 1.5,
                    "p_value_lognormal_vs_truncated_power_law": 0.01,
                }
            ]
        )

        review = _build_best_vs_second_review_table(fit_summary)

        self.assertEqual(review.loc[0, "Best by AIC"], "lognormal")
        self.assertEqual(review.loc[0, "2nd"], "truncated_power_law")
        self.assertEqual(
            review.loc[0, "Best fit parameters"],
            "mu=0.5000 [0.4000, 0.6000]; sigma=0.8000 [0.7000, 0.9000]",
        )
        self.assertEqual(
            review.loc[0, "2nd fit parameters"],
            "alpha=1.6000 [1.3000, 1.9000]; lambda=0.3000 [0.2000, 0.4000]",
        )

    def test_best_vs_second_review_uses_point_estimates_without_bootstrap(self) -> None:
        fit_summary = pd.DataFrame(
            [
                {
                    "group": "proprietary",
                    "metric": "meta_volumes",
                    "power_law_loglikelihood": -105.0,
                    "power_law_alpha": 2.1,
                    "lognormal_loglikelihood": -101.0,
                    "lognormal_mu": 0.4,
                    "lognormal_sigma": 0.9,
                    "truncated_power_law_loglikelihood": -103.0,
                    "truncated_power_law_alpha": 1.8,
                    "truncated_power_law_lambda": 0.2,
                    "loglikelihood_ratio_lognormal_vs_truncated_power_law": 2.0,
                    "p_value_lognormal_vs_truncated_power_law": 0.02,
                }
            ]
        )

        review = _build_best_vs_second_review_table(fit_summary)

        self.assertEqual(review.loc[0, "Best fit parameters"], "mu=0.4000; sigma=0.9000")
        self.assertEqual(review.loc[0, "2nd fit parameters"], "alpha=1.8000; lambda=0.2000")

    def test_compact_fit_summary_keeps_bootstrap_std_fields(self) -> None:
        fit_summary = pd.DataFrame(
            [
                {
                    "metric": "q_over_v",
                    "panel_title": "Relative size",
                    "group": "client",
                    "sample_size": 250,
                    "fit_enabled": True,
                    "fit_success": True,
                    "fit_method": "approx",
                    "alpha": 2.3,
                    "xmin": 1.2,
                    "ks_stat": 0.08,
                    "n_tail": 120,
                    "best_fit_model": "lognormal",
                    "best_fit_criterion": "aic",
                    "best_fit_aic": 24.0,
                    "powerlaw_compare_summary": "lognormal=lognormal",
                    "bootstrap_full_pipeline_enabled": True,
                    "bootstrap_distribution": "power_law,lognormal",
                    "bootstrap_resampling_scheme": "nonparametric_full_sample",
                    "bootstrap_runs_requested": 20,
                    "bootstrap_alpha": 0.05,
                    "bootstrap_random_state": 3,
                    "bootstrap_pipeline_valid_runs": 18,
                    "bootstrap_xmin_valid_runs": 18,
                    "bootstrap_xmin_ci_low": 1.0,
                    "bootstrap_xmin_ci_high": 1.4,
                    "bootstrap_xmin_std": 0.08,
                    "model_power_law_loglikelihood": -12.0,
                    "model_power_law_bootstrap_valid_runs": 18,
                    "model_lognormal_loglikelihood": -10.0,
                    "model_lognormal_bootstrap_valid_runs": 17,
                    "param_power_law_alpha": 2.3,
                    "param_power_law_alpha_ci_low": 2.1,
                    "param_power_law_alpha_ci_high": 2.5,
                    "param_power_law_alpha_std": 0.12,
                    "param_lognormal_mu": 0.4,
                    "param_lognormal_mu_ci_low": 0.2,
                    "param_lognormal_mu_ci_high": 0.6,
                    "param_lognormal_mu_std": 0.09,
                }
            ]
        )

        compact = _compact_fit_summary_table(fit_summary)

        self.assertIn("bootstrap_distribution", compact.columns)
        self.assertIn("bootstrap_xmin_std", compact.columns)
        self.assertIn("sample_size", compact.columns)
        self.assertIn("fit_success", compact.columns)
        self.assertIn("best_fit_model", compact.columns)
        self.assertIn("best_fit_aic", compact.columns)
        self.assertIn("powerlaw_compare_summary", compact.columns)
        self.assertIn("power_law_alpha_std", compact.columns)
        self.assertIn("lognormal_mu_std", compact.columns)

    def test_build_distribution_figure_from_saved_outputs_reuses_saved_curves(self) -> None:
        fit_rows = []
        for metric in metaorder_distributions._distribution_metric_specs():
            for group in ("client", "proprietary"):
                fit_rows.append(
                    {
                        "metric": metric.field_name,
                        "panel_title": metric.panel_title,
                        "group": group,
                        "sample_size": 0,
                        "fit_success": False,
                        "alpha": float("nan"),
                        "xmin": float("nan"),
                        "ks_stat": float("nan"),
                        "n_tail": 0,
                        "best_fit_model": "unavailable",
                        "best_fit_aic": float("nan"),
                        "powerlaw_compare_summary": "none",
                        "bootstrap_full_pipeline_enabled": False,
                        "bootstrap_alpha": float("nan"),
                        "bootstrap_xmin_ci_low": float("nan"),
                        "bootstrap_xmin_ci_high": float("nan"),
                        "power_law_alpha_ci_low": float("nan"),
                        "power_law_alpha_ci_high": float("nan"),
                    }
                )

        fit_summary = pd.DataFrame(fit_rows)
        fit_summary.loc[
            (fit_summary["metric"] == "q_over_v") & (fit_summary["group"] == "client"),
            [
                "sample_size",
                "fit_success",
                "alpha",
                "xmin",
                "ks_stat",
                "n_tail",
                "best_fit_model",
                "best_fit_aic",
                "powerlaw_compare_summary",
                "bootstrap_full_pipeline_enabled",
                "bootstrap_alpha",
                "bootstrap_xmin_ci_low",
                "bootstrap_xmin_ci_high",
                "power_law_alpha_ci_low",
                "power_law_alpha_ci_high",
            ],
        ] = [
            200,
            True,
            2.3456,
            0.12,
            0.08,
            75,
            "lognormal",
            11.2,
            "lognormal=lognormal",
            True,
            0.05,
            0.1,
            0.15,
            2.1,
            2.6,
        ]

        plot_data = pd.DataFrame(
            [
                {
                    "metric": "q_over_v",
                    "panel_title": "Relative size",
                    "group": "client",
                    "label": "Client",
                    "row_idx": 4,
                    "col_idx": 1,
                    "trace_kind": "density",
                    "point_index": 0,
                    "x": 0.1,
                    "y": 10.0,
                },
                {
                    "metric": "q_over_v",
                    "panel_title": "Relative size",
                    "group": "client",
                    "label": "Client",
                    "row_idx": 4,
                    "col_idx": 1,
                    "trace_kind": "density",
                    "point_index": 1,
                    "x": 0.2,
                    "y": 8.0,
                },
                {
                    "metric": "q_over_v",
                    "panel_title": "Relative size",
                    "group": "client",
                    "label": "Client",
                    "row_idx": 4,
                    "col_idx": 1,
                    "trace_kind": "best_fit",
                    "point_index": 0,
                    "x": 0.12,
                    "y": 7.5,
                },
                {
                    "metric": "q_over_v",
                    "panel_title": "Relative size",
                    "group": "client",
                    "label": "Client",
                    "row_idx": 4,
                    "col_idx": 1,
                    "trace_kind": "best_fit",
                    "point_index": 1,
                    "x": 0.2,
                    "y": 6.0,
                },
                {
                    "metric": "q_over_v",
                    "panel_title": "Relative size",
                    "group": "client",
                    "label": "Client",
                    "row_idx": 4,
                    "col_idx": 1,
                    "trace_kind": "power_law_overlay",
                    "point_index": 0,
                    "x": 0.12,
                    "y": 7.0,
                },
            ]
        )

        fig = build_distribution_figure_from_saved_outputs(
            fit_summary,
            plot_data,
            show_progress=False,
        )

        annotation_texts = [annotation.text for annotation in fig.layout.annotations]
        self.assertEqual(len(fig.data), 3)
        self.assertTrue(any("Best fit = lognormal" in text for text in annotation_texts))
        self.assertTrue(any("95% CI [0.1, 0.15]" in text for text in annotation_texts))


class TestMetaorderDistributionsParallelism(unittest.TestCase):
    @mock.patch("scripts.metaorder_distributions.os.cpu_count", return_value=8)
    def test_worker_count_defaults_to_panel_count_cap(self, _cpu_count: mock.Mock) -> None:
        with mock.patch.object(metaorder_distributions, "POWERLAW_FIT_ENABLED", True):
            with mock.patch.object(metaorder_distributions, "POWERLAW_FULL_BOOTSTRAP_ENABLED", True):
                with mock.patch.object(metaorder_distributions, "POWERLAW_FIT_MAX_WORKERS", None):
                    self.assertEqual(_resolve_powerlaw_fit_worker_count(3), 3)

    @mock.patch("scripts.metaorder_distributions.os.cpu_count", return_value=16)
    def test_worker_count_auto_caps_bootstrap_heavy_runs(self, _cpu_count: mock.Mock) -> None:
        with mock.patch.object(metaorder_distributions, "POWERLAW_FIT_ENABLED", True):
            with mock.patch.object(metaorder_distributions, "POWERLAW_FULL_BOOTSTRAP_ENABLED", True):
                with mock.patch.object(metaorder_distributions, "POWERLAW_FIT_MAX_WORKERS", None):
                    self.assertEqual(_resolve_powerlaw_fit_worker_count(10), 4)

    @mock.patch("scripts.metaorder_distributions.os.cpu_count", return_value=8)
    def test_worker_count_respects_configured_cap(self, _cpu_count: mock.Mock) -> None:
        with mock.patch.object(metaorder_distributions, "POWERLAW_FIT_ENABLED", True):
            with mock.patch.object(metaorder_distributions, "POWERLAW_FIT_MAX_WORKERS", 2):
                self.assertEqual(_resolve_powerlaw_fit_worker_count(5), 2)

    def test_worker_count_falls_back_to_serial_when_fit_disabled(self) -> None:
        with mock.patch.object(metaorder_distributions, "POWERLAW_FIT_ENABLED", False):
            with mock.patch.object(metaorder_distributions, "POWERLAW_FIT_MAX_WORKERS", 8):
                self.assertEqual(_resolve_powerlaw_fit_worker_count(5), 1)


if __name__ == "__main__":
    unittest.main()
