from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.crowding_impact_analysis as crowding_impact


def _period(date_str: str, minute_offset: int) -> list[int]:
    start = pd.Timestamp(f"{date_str} 10:{minute_offset:02d}:00")
    end = start + pd.Timedelta(minutes=5)
    return [int(start.value), int(end.value)]


def _build_group_frame(group: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_scale = 0.20 if group == "client" else 0.28
    crowd_levels = [-0.65, -0.15, 0.55]
    eta_levels = [0.08, 0.14, 0.22, 0.38, 0.52, 0.68]
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]

    for date_idx, date_str in enumerate(dates):
        for crowd_idx, aligned_crowding in enumerate(crowd_levels):
            crowd_multiplier = 1.0 + 0.30 * aligned_crowding
            for q_idx, qv in enumerate(np.logspace(-4.5, -2.1, 6)):
                direction = 1 if ((q_idx + crowd_idx + (0 if group == "client" else 1)) % 2 == 0) else -1
                eta = eta_levels[(q_idx + crowd_idx + date_idx) % len(eta_levels)]
                vtv = float(qv / eta)
                daily_vol = 0.20 + 0.01 * date_idx
                impact = group_scale * crowd_multiplier * float(qv ** 0.52) * (1.0 + 0.02 * date_idx + 0.01 * q_idx)
                price_change = impact * daily_vol / direction
                rows.append(
                    {
                        "ISIN": f"ISIN_{q_idx % 2}",
                        "Date": pd.Timestamp(date_str),
                        "Period": _period(date_str, minute_offset=crowd_idx * 10 + q_idx),
                        "Direction": direction,
                        "Price Change": price_change,
                        "Daily Vol": daily_vol,
                        "Q": float(qv * 1_000_000.0),
                        "Q/V": float(qv),
                        "Vt/V": vtv,
                        "Participation Rate": eta,
                        "Impact": impact,
                        # Store the raw imbalance so that Direction * imbalance
                        # equals the intended aligned-crowding level.
                        "imbalance_local": aligned_crowding * direction,
                    }
                )
    return pd.DataFrame(rows)


def _build_working_frame() -> pd.DataFrame:
    client = _build_group_frame("client").assign(group="client", group_label="Client", metaorder_id=lambda df: [f"c_{i}" for i in range(len(df))])
    prop = _build_group_frame("proprietary").assign(
        group="proprietary",
        group_label="Proprietary",
        metaorder_id=lambda df: [f"p_{i}" for i in range(len(df))],
    )
    out = pd.concat([client, prop], ignore_index=True)
    out["aligned_crowding"] = out["Direction"] * out["imbalance_local"]
    return out[
        [
            "metaorder_id",
            "Date",
            "ISIN",
            "group",
            "group_label",
            "Direction",
            "Q",
            "Q/V",
            "Vt/V",
            "Participation Rate",
            "Impact",
            "imbalance_local",
            "aligned_crowding",
        ]
    ].copy()


def _build_joint_regression_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-02-01", periods=5, freq="D")
    eta_levels = [0.08, 0.16, 0.32]
    vtv_levels = [0.02, 0.05, 0.11]
    abs_imb_levels = [0.10, 0.35, 0.65]
    group_scales = {"client": 0.09, "proprietary": 0.12}
    group_imb_slopes = {"client": 0.35, "proprietary": 0.55}

    for group in ["client", "proprietary"]:
        for date_idx, date in enumerate(dates):
            for eta_idx, eta in enumerate(eta_levels):
                for vtv_idx, vtv in enumerate(vtv_levels):
                    for imb_idx, abs_imb in enumerate(abs_imb_levels):
                        for rep in range(4):
                            direction = 1 if (rep + date_idx + eta_idx) % 2 == 0 else -1
                            qv = eta * vtv
                            noise = 1.0 + 0.02 * rep + 0.01 * date_idx
                            impact = (
                                group_scales[group]
                                * (eta ** 0.35)
                                * (vtv ** 0.28)
                                * (1.0 + group_imb_slopes[group] * abs_imb)
                                * noise
                            )
                            rows.append(
                                {
                                    "metaorder_id": f"{group}_{date_idx}_{eta_idx}_{vtv_idx}_{imb_idx}_{rep}",
                                    "Date": date,
                                    "ISIN": f"ISIN_{(eta_idx + vtv_idx) % 2}",
                                    "group": group,
                                    "group_label": "Client" if group == "client" else "Proprietary",
                                    "Direction": direction,
                                    "Q": float(qv * 1_000_000.0),
                                    "Q/V": float(qv),
                                    "Vt/V": float(vtv),
                                    "Participation Rate": float(eta),
                                    "Impact": float(impact),
                                    "imbalance_local": float(direction * abs_imb),
                                    "aligned_crowding": float(abs_imb),
                                    "abs_imbalance": float(abs_imb),
                                    "crowding_input_col": "imbalance_local",
                                }
                            )
    return pd.DataFrame(rows)


def _build_pooled_log_regression_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-03-01", periods=6, freq="D")
    eta_levels = [0.08, 0.16, 0.32]
    vtv_levels = [0.02, 0.05, 0.11]
    abs_imb_levels = [0.10, 0.30, 0.60]

    alpha = -2.50
    beta_eta = 0.45
    beta_vtv = 0.25
    beta_abs_imb = 0.20
    beta_prop = 0.35

    for group in ["client", "proprietary"]:
        prop_dummy = 1.0 if group == "proprietary" else 0.0
        for date_idx, date in enumerate(dates):
            date_effect = 0.03 * date_idx
            for eta_idx, eta in enumerate(eta_levels):
                for vtv_idx, vtv in enumerate(vtv_levels):
                    for imb_idx, abs_imb in enumerate(abs_imb_levels):
                        for rep in range(3):
                            direction = 1 if (rep + eta_idx + date_idx) % 2 == 0 else -1
                            qv = eta * vtv
                            noise = 0.01 * rep
                            log_abs_impact = (
                                alpha
                                + beta_eta * np.log(eta)
                                + beta_vtv * np.log(vtv)
                                + beta_abs_imb * abs_imb
                                + beta_prop * prop_dummy
                                + date_effect
                                + noise
                            )
                            mean_like_impact = float(np.exp(log_abs_impact))
                            rows.append(
                                {
                                    "metaorder_id": f"{group}_{date_idx}_{eta_idx}_{vtv_idx}_{imb_idx}_{rep}",
                                    "Date": date,
                                    "ISIN": f"ISIN_{(eta_idx + vtv_idx) % 3}",
                                    "group": group,
                                    "group_label": "Client" if group == "client" else "Proprietary",
                                    "Direction": direction,
                                    "Q": float(qv * 1_000_000.0),
                                    "Q/V": float(qv),
                                    "Vt/V": float(vtv),
                                    "Participation Rate": float(eta),
                                    "Impact": mean_like_impact,
                                    "imbalance_local": float(direction * abs_imb),
                                    "aligned_crowding": float(abs_imb),
                                    "abs_imbalance": float(abs_imb),
                                    "crowding_input_col": "imbalance_local",
                                }
                            )
    return pd.DataFrame(rows)


class TestCrowdingImpactHelpers(unittest.TestCase):
    def test_period_start_ns_handles_common_formats(self) -> None:
        self.assertEqual(crowding_impact._period_start_ns([1, 2]), 1)
        self.assertEqual(crowding_impact._period_start_ns(np.array([3, 4], dtype=np.int64)), 3)
        self.assertEqual(crowding_impact._period_start_ns("[5, 6]"), 5)
        self.assertIsNone(crowding_impact._period_start_ns([]))

    def test_main_variant_builds_positive_high_minus_low_gap(self) -> None:
        working = _build_working_frame()
        out = crowding_impact._analyse_main_variant(
            working,
            n_crowding_quantiles=3,
            benchmark_phis=(1.0e-3, 1.0e-2),
            n_logbins=5,
            min_count=2,
            store_binned=True,
        )

        self.assertEqual(sorted(out.sample_sizes["group"].unique().tolist()), ["client", "proprietary"])
        self.assertEqual(out.cutpoints["crowding_bin"].nunique(), 3)
        self.assertTrue((out.fit_summary["status"] == "ok").all())

        high_low = out.contrasts[out.contrasts["contrast_name"] == "high_minus_low"]
        self.assertEqual(high_low.shape[0], 4)
        self.assertTrue((high_low["point_estimate"] > 0).all())

    def test_bootstrap_sampler_preserves_nonempty_groups(self) -> None:
        working = _build_working_frame()
        sampler = crowding_impact.DateBootstrapSampler.from_frame(working)
        sample = sampler.sample(np.random.default_rng(0))
        self.assertGreater(len(sample), 0)
        self.assertEqual(sorted(sample["group"].unique().tolist()), ["client", "proprietary"])

    def test_prepare_working_sample_all_scope_produces_finite_crowding(self) -> None:
        client = _build_group_frame("client").assign(
            metaorder_id=lambda df: [f"c_{i}" for i in range(len(df))],
            group="client",
            group_label="Client",
        )[
            [
                "metaorder_id",
                "Date",
                "ISIN",
                "group",
                "group_label",
                "Direction",
                "Q",
                "Q/V",
                "Vt/V",
                "Participation Rate",
                "Impact",
                "imbalance_local",
            ]
        ]
        prop = _build_group_frame("proprietary").assign(
            metaorder_id=lambda df: [f"p_{i}" for i in range(len(df))],
            group="proprietary",
            group_label="Proprietary",
        )[
            [
                "metaorder_id",
                "Date",
                "ISIN",
                "group",
                "group_label",
                "Direction",
                "Q",
                "Q/V",
                "Vt/V",
                "Participation Rate",
                "Impact",
                "imbalance_local",
            ]
        ]
        client_result = crowding_impact.GroupLoadResult(
            frame=client,
            raw_n=len(client),
            fit_filtered_n=len(client),
            eta_filtered_n=len(client),
            imbalance_source="reused",
        )
        prop_result = crowding_impact.GroupLoadResult(
            frame=prop,
            raw_n=len(prop),
            fit_filtered_n=len(prop),
            eta_filtered_n=len(prop),
            imbalance_source="reused",
        )

        working, counts = crowding_impact._prepare_working_sample(
            client_result,
            prop_result,
            crowding_scope="all",
        )

        self.assertEqual(int(counts["n_rows"].iloc[-1]), int((working["group"] == "proprietary").sum()))
        self.assertTrue(np.isfinite(working["aligned_crowding"]).all())
        self.assertGreater(working["aligned_crowding"].nunique(), 2)
        self.assertEqual(working["crowding_input_col"].iloc[0], crowding_impact.COL_ALL_IMBALANCE)
        self.assertTrue(np.isfinite(working["abs_imbalance"]).all())

    def test_joint_bin_regression_recovers_positive_abs_imbalance_slope(self) -> None:
        working = _build_joint_regression_frame()
        out = crowding_impact._analyse_joint_bin_regression(
            working,
            n_eta_bins=3,
            n_vtv_bins=3,
            n_abs_imb_bins=3,
            min_cell_count=3,
        )

        self.assertEqual(sorted(out.fit_summary["group"].tolist()), ["client", "proprietary"])
        self.assertTrue((out.fit_summary["status"] == "ok").all())
        self.assertTrue((out.fit_summary["beta_abs_imb_hat"] > 0).all())
        self.assertGreater(int(out.cell_data["retained_for_fit"].sum()), 20)
        self.assertTrue((out.group_differences["prop_minus_client_beta_abs_imb_hat"] > 0).all())

    def test_main_curve_figure_keeps_group_titles_and_omits_benchmark_note(self) -> None:
        working = _build_working_frame()
        outputs = crowding_impact._analyse_main_variant(
            working,
            n_crowding_quantiles=3,
            benchmark_phis=(1.0e-3, 1.0e-2),
            n_logbins=5,
            min_count=2,
            store_binned=True,
        )
        dirs = crowding_impact.make_plot_output_dirs(Path("images/test_crowding_layout"))

        with mock.patch.object(crowding_impact, "_export_plotly_figure") as export_mock:
            crowding_impact._plot_main_curves(
                outputs,
                dirs,
                benchmark_phis=(1.0e-3, 1.0e-2),
                write_html=False,
                write_png=False,
            )

        fig = export_mock.call_args.args[0]
        annotation_texts = [ann.text for ann in fig.layout.annotations]
        self.assertIn("Client", annotation_texts)
        self.assertIn("Proprietary", annotation_texts)
        self.assertNotIn("Benchmarks: 1e-03, 1e-02", annotation_texts)
        self.assertEqual(fig.layout.legend.title.text, "Crowding quantile")

    def test_multi_panel_figures_share_axes_for_comparison(self) -> None:
        working = _build_working_frame()
        main_outputs = crowding_impact._analyse_main_variant(
            working,
            n_crowding_quantiles=3,
            benchmark_phis=(1.0e-3, 1.0e-2),
            n_logbins=5,
            min_count=2,
            store_binned=True,
        )
        eta_outputs = crowding_impact._analyse_eta_variant(
            working,
            n_eta_bins=2,
            n_crowding_quantiles=3,
            benchmark_phis=(1.0e-3, 1.0e-2),
            n_logbins=5,
            min_count=2,
            store_binned=True,
        )
        joint_outputs = crowding_impact._analyse_joint_bin_regression(
            _build_joint_regression_frame(),
            n_eta_bins=3,
            n_vtv_bins=3,
            n_abs_imb_bins=3,
            min_cell_count=3,
        )
        joint_plot_ci, joint_plot_diff_ci = crowding_impact._build_joint_regression_analytic_plot_tables(
            joint_outputs.fit_summary,
            alpha=0.05,
        )

        prediction_ci = crowding_impact._bootstrap_ci_from_table(
            main_outputs.predictions,
            pd.DataFrame(),
            id_cols=[
                crowding_impact.COL_GROUP,
                crowding_impact.COL_GROUP_LABEL,
                crowding_impact.COL_CROWDING_BIN,
                crowding_impact.COL_CROWDING_LABEL,
                "benchmark_phi",
            ],
            value_cols=["predicted_impact"],
            alpha=0.05,
        )
        contrasts_ci = crowding_impact._bootstrap_ci_from_table(
            main_outputs.contrasts,
            pd.DataFrame(),
            id_cols=[crowding_impact.COL_GROUP, crowding_impact.COL_GROUP_LABEL, "benchmark_phi", "contrast_name"],
            value_cols=["point_estimate"],
            alpha=0.05,
        )
        group_diff_ci = crowding_impact._bootstrap_ci_from_table(
            main_outputs.group_differences,
            pd.DataFrame(),
            id_cols=["benchmark_phi", "contrast_name"],
            value_cols=["prop_high_minus_low", "client_high_minus_low", "prop_minus_client"],
            alpha=0.05,
        )
        dirs = crowding_impact.make_plot_output_dirs(Path("images/test_crowding_shared_axes"))

        def _capture_figure(plot_fn, *args, **kwargs):
            with mock.patch.object(crowding_impact, "_export_plotly_figure") as export_mock:
                plot_fn(*args, **kwargs)
            return export_mock.call_args.args[0]

        figures = {
            "main": _capture_figure(
                crowding_impact._plot_main_curves,
                main_outputs,
                dirs,
                benchmark_phis=(1.0e-3, 1.0e-2),
                write_html=False,
                write_png=False,
            ),
            "predicted": _capture_figure(
                crowding_impact._plot_predicted_impacts,
                main_outputs.predictions,
                prediction_ci,
                dirs,
                write_html=False,
                write_png=False,
            ),
            "difference": _capture_figure(
                crowding_impact._plot_difference_figure,
                contrasts_ci,
                group_diff_ci,
                dirs,
                write_html=False,
                write_png=False,
            ),
            "eta": _capture_figure(
                crowding_impact._plot_eta_robustness,
                eta_outputs,
                dirs,
                write_html=False,
                write_png=False,
            ),
            "joint_regression": _capture_figure(
                crowding_impact._plot_joint_regression_coefficients,
                joint_plot_ci,
                joint_plot_diff_ci,
                dirs,
                write_html=False,
                write_png=False,
            ),
        }

        for name, fig in figures.items():
            layout = fig.to_plotly_json()["layout"]
            x_matches = [
                axis.get("matches")
                for axis_name, axis in layout.items()
                if axis_name.startswith("xaxis") and isinstance(axis, dict)
            ]
            y_matches = [
                axis.get("matches")
                for axis_name, axis in layout.items()
                if axis_name.startswith("yaxis") and isinstance(axis, dict)
            ]
            self.assertTrue(any(match is not None for match in x_matches), msg=f"{name} should share x-axes.")
            self.assertTrue(any(match is not None for match in y_matches), msg=f"{name} should share y-axes.")

    def test_pooled_log_regression_plot_uses_bootstrap_intervals(self) -> None:
        coefficients = pd.DataFrame(
            [
                {
                    "term": "log_eta",
                    "term_label": "log(η)",
                    "term_order": 1,
                    "estimate": 0.23,
                    "analytic_ci_low": 0.20,
                    "analytic_ci_high": 0.26,
                },
                {
                    "term": crowding_impact.COL_PROP_DUMMY,
                    "term_label": "1{proprietary}",
                    "term_order": 4,
                    "estimate": 0.17,
                    "analytic_ci_low": 0.10,
                    "analytic_ci_high": 0.24,
                },
            ]
        )
        bootstrap_ci = pd.DataFrame(
            [
                {
                    "term": "log_eta",
                    "term_label": "log(η)",
                    "term_order": 1,
                    "metric": "estimate",
                    "point_estimate": 0.23,
                    "ci_low": 0.21,
                    "ci_high": 0.24,
                },
                {
                    "term": crowding_impact.COL_PROP_DUMMY,
                    "term_label": "1{proprietary}",
                    "term_order": 4,
                    "metric": "estimate",
                    "point_estimate": 0.17,
                    "ci_low": 0.15,
                    "ci_high": 0.19,
                },
            ]
        )
        dirs = crowding_impact.make_plot_output_dirs(Path("images/test_pooled_ci"))

        with mock.patch.object(crowding_impact, "_export_plotly_figure") as export_mock:
            crowding_impact._plot_pooled_log_regression_coefficients(
                coefficients,
                bootstrap_ci,
                dirs,
                write_html=False,
                write_png=False,
            )

        fig = export_mock.call_args.args[0]
        trace = fig.data[0]
        self.assertAlmostEqual(float(trace.error_x.array[0]), 0.02)
        self.assertAlmostEqual(float(trace.error_x.array[1]), 0.01)
        self.assertAlmostEqual(float(trace.error_x.arrayminus[0]), 0.02)
        self.assertAlmostEqual(float(trace.error_x.arrayminus[1]), 0.02)

    def test_predicted_impact_plot_uses_symmetric_standard_errors(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "group": "client",
                    "group_label": "Client",
                    "crowding_bin": 0,
                    "crowding_label": "Low",
                    "benchmark_phi": 1.0e-3,
                    "predicted_impact": 0.0020,
                },
                {
                    "group": "client",
                    "group_label": "Client",
                    "crowding_bin": 1,
                    "crowding_label": "Mid",
                    "benchmark_phi": 1.0e-3,
                    "predicted_impact": 0.0040,
                },
                {
                    "group": "proprietary",
                    "group_label": "Proprietary",
                    "crowding_bin": 0,
                    "crowding_label": "Low",
                    "benchmark_phi": 1.0e-3,
                    "predicted_impact": 0.0050,
                },
                {
                    "group": "proprietary",
                    "group_label": "Proprietary",
                    "crowding_bin": 1,
                    "crowding_label": "Mid",
                    "benchmark_phi": 1.0e-3,
                    "predicted_impact": 0.0060,
                },
            ]
        )
        prediction_summary = pd.DataFrame(
            [
                {
                    "group": "client",
                    "crowding_bin": 0,
                    "benchmark_phi": 1.0e-3,
                    "metric": "predicted_impact",
                    "se": 0.0003,
                    "ci_low": 0.0010,
                    "ci_high": 0.0028,
                },
                {
                    "group": "client",
                    "crowding_bin": 1,
                    "benchmark_phi": 1.0e-3,
                    "metric": "predicted_impact",
                    "se": 0.0004,
                    "ci_low": 0.0030,
                    "ci_high": 0.0049,
                },
                {
                    "group": "proprietary",
                    "crowding_bin": 0,
                    "benchmark_phi": 1.0e-3,
                    "metric": "predicted_impact",
                    "se": 0.0002,
                    "ci_low": 0.0041,
                    "ci_high": 0.0057,
                },
                {
                    "group": "proprietary",
                    "crowding_bin": 1,
                    "benchmark_phi": 1.0e-3,
                    "metric": "predicted_impact",
                    "se": 0.00025,
                    "ci_low": 0.0054,
                    "ci_high": 0.0068,
                },
            ]
        )
        dirs = crowding_impact.make_plot_output_dirs(Path("images/test_predicted_impact_se"))

        with mock.patch.object(crowding_impact, "_export_plotly_figure") as export_mock:
            crowding_impact._plot_predicted_impacts(
                predictions,
                prediction_summary,
                dirs,
                write_html=False,
                write_png=False,
            )

        fig = export_mock.call_args.args[0]
        client_trace = fig.data[0]
        proprietary_trace = fig.data[1]
        self.assertAlmostEqual(float(client_trace.error_y.array[0]), 0.0003)
        self.assertAlmostEqual(float(client_trace.error_y.arrayminus[0]), 0.0003)
        self.assertAlmostEqual(float(client_trace.error_y.array[1]), 0.0004)
        self.assertAlmostEqual(float(proprietary_trace.error_y.array[0]), 0.0002)
        self.assertAlmostEqual(float(proprietary_trace.error_y.arrayminus[1]), 0.00025)

    def test_prepare_pooled_log_regression_sample_retains_rows_but_excludes_nonpositive_cells(self) -> None:
        working = _build_pooled_log_regression_frame()
        target_mask = (
            (working["group"] == "client")
            & np.isclose(working["Participation Rate"], 0.08)
            & np.isclose(working["abs_imbalance"], 0.10)
            & (working["Vt/V"] <= 0.05)
        )
        working.loc[target_mask, "Impact"] = -10.0 * np.abs(working.loc[target_mask, "Impact"].to_numpy(dtype=float))

        sample, counts, _ = crowding_impact._prepare_pooled_log_regression_sample(
            working,
            impact_mode="signed_mean",
            n_eta_bins=3,
            n_vtv_bins=3,
            n_abs_imb_bins=3,
            min_cell_count=3,
        )

        finite_impact_n = int(
            counts.loc[
                (counts["stage"] == "after_finite_impact") & (counts["group"] == "all"),
                "n_rows",
            ].iat[0]
        )
        self.assertEqual(finite_impact_n, len(working))
        self.assertTrue((sample["count"] > 0).all())
        bad_cell = sample.loc[
            (sample["group"] == "client")
            & np.isclose(sample["mean_eta_raw"], 0.08)
            & np.isclose(sample["mean_abs_imb_raw"], 0.10)
            & (sample["mean_imp"] <= 0.0)
        ]
        self.assertGreaterEqual(len(bad_cell), 1)
        self.assertTrue((bad_cell["retained_for_fit"] == False).all())

    def test_pooled_log_regression_recovers_positive_slopes_and_prop_dummy(self) -> None:
        working = _build_pooled_log_regression_frame()
        out = crowding_impact._analyse_pooled_log_regression(
            working,
            alpha=0.05,
            impact_mode="signed_mean",
            n_eta_bins=3,
            n_vtv_bins=3,
            n_abs_imb_bins=3,
            min_cell_count=3,
            dummy_group="proprietary",
        )
        out_client = crowding_impact._analyse_pooled_log_regression(
            working,
            alpha=0.05,
            impact_mode="signed_mean",
            n_eta_bins=3,
            n_vtv_bins=3,
            n_abs_imb_bins=3,
            min_cell_count=3,
            dummy_group="client",
        )

        self.assertEqual(out.fit_summary["status"].iat[0], "ok")
        self.assertEqual(out_client.fit_summary["status"].iat[0], "ok")
        self.assertEqual(int(out.fit_summary["n_groups"].iat[0]), 2)
        self.assertGreater(int(out.fit_summary["n_obs"].iat[0]), 10)
        self.assertEqual(out.fit_summary["dummy_group"].iat[0], "proprietary")
        self.assertEqual(out_client.fit_summary["dummy_group"].iat[0], "client")

        coef = out.coefficients.set_index("term")
        coef_client = out_client.coefficients.set_index("term")
        self.assertGreater(float(coef.loc["log_eta", "estimate"]), 0.0)
        self.assertGreater(float(coef.loc["log_vtv", "estimate"]), 0.0)
        self.assertGreater(float(coef.loc["abs_imbalance", "estimate"]), 0.0)
        self.assertGreater(float(coef.loc["is_proprietary", "estimate"]), 0.0)
        self.assertLess(float(coef_client.loc["is_client", "estimate"]), 0.0)
        self.assertAlmostEqual(
            float(coef.loc["log_eta", "estimate"]),
            float(coef_client.loc["log_eta", "estimate"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(coef.loc["log_vtv", "estimate"]),
            float(coef_client.loc["log_vtv", "estimate"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(coef.loc["abs_imbalance", "estimate"]),
            float(coef_client.loc["abs_imbalance", "estimate"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(coef.loc["is_proprietary", "estimate"]),
            -float(coef_client.loc["is_client", "estimate"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(coef_client.loc["const", "estimate"]),
            float(coef.loc["const", "estimate"]) + float(coef.loc["is_proprietary", "estimate"]),
            places=10,
        )


class TestCrowdingImpactScriptSmoke(unittest.TestCase):
    def test_script_writes_tables_figures_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_root = tmp / "out"
            img_root = tmp / "images"
            prop_path = tmp / "prop.parquet"
            client_path = tmp / "client.parquet"

            _build_group_frame("proprietary").to_parquet(prop_path, index=False)
            _build_group_frame("client").to_parquet(client_path, index=False)

            rc = crowding_impact.main(
                [
                    "--prop-path",
                    str(prop_path),
                    "--client-path",
                    str(client_path),
                    "--output-file-path",
                    str(out_root),
                    "--img-output-path",
                    str(img_root),
                    "--analysis-tag",
                    "crowding_impact_smoke",
                    "--crowding-scope",
                    "within_group",
                    "--bootstrap-runs",
                    "6",
                    "--n-logbins",
                    "5",
                    "--min-count",
                    "2",
                    "--joint-regression-eta-bins",
                    "3",
                    "--joint-regression-vtv-bins",
                    "3",
                    "--joint-regression-abs-imb-bins",
                    "3",
                    "--joint-regression-min-count",
                    "2",
                    "--seed",
                    "0",
                    "--no-write-png",
                    "--no-progress",
                ]
            )

            self.assertEqual(rc, 0)

            out_dir = out_root / "crowding_impact_smoke"
            html_dir = img_root / "crowding_impact_smoke" / "html"

            self.assertTrue((out_dir / "fit_summary_main.csv").exists())
            self.assertTrue((out_dir / "bootstrap_predicted_impacts_main.csv").exists())
            self.assertTrue((out_dir / "monotonic_contrasts_main.csv").exists())
            self.assertTrue((out_dir / "group_difference_main.csv").exists())
            self.assertTrue((out_dir / "acceptance_summary.csv").exists())
            self.assertTrue((out_dir / "joint_regression_fit_summary.csv").exists())
            self.assertTrue((out_dir / "joint_regression_bootstrap_coefficients.csv").exists())
            self.assertTrue((out_dir / "pooled_log_regression_cell_data.csv").exists())
            self.assertTrue((out_dir / "pooled_log_regression_fit_summary.csv").exists())
            self.assertTrue((out_dir / "pooled_log_regression_bootstrap_coefficients.csv").exists())
            self.assertTrue((out_dir / "pooled_log_regression_client_dummy_cell_data.csv").exists())
            self.assertTrue((out_dir / "pooled_log_regression_client_dummy_fit_summary.csv").exists())
            self.assertTrue((out_dir / "pooled_log_regression_client_dummy_bootstrap_coefficients.csv").exists())
            self.assertTrue((out_dir / "run_manifest.json").exists())

            self.assertTrue((html_dir / "main_crowding_impact_curves.html").exists())
            self.assertTrue((html_dir / "predicted_impact_by_crowding_quantile.html").exists())
            self.assertTrue((html_dir / "crowding_gap_differences.html").exists())
            self.assertTrue((html_dir / "eta_robustness_crowding_impact_curves.html").exists())
            self.assertTrue((html_dir / "joint_bin_regression_coefficients.html").exists())
            self.assertTrue((html_dir / "pooled_log_regression_coefficients.html").exists())
            self.assertTrue((html_dir / "pooled_log_regression_client_dummy_coefficients.html").exists())

            manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["bootstrap_runs"], 6)
            self.assertEqual(manifest["fit_defaults"]["min_count"], 2)
            self.assertTrue(manifest["run_joint_bin_regression"])
            self.assertTrue(manifest["run_pooled_log_regression"])
            self.assertEqual(manifest["pooled_log_regression"]["impact_mode"], "signed_mean")
            self.assertEqual(manifest["pooled_log_regression"]["dummy_groups"], ["proprietary", "client"])


if __name__ == "__main__":
    unittest.main()
