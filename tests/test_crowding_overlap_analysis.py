from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import moimpact.workflows.crowding.overlap as overlap


def _period(start_ts: str, end_ts: str | None = None) -> list[int]:
    start = pd.Timestamp(start_ts)
    end = pd.Timestamp(end_ts) if end_ts is not None else start
    return [int(start.value), int(end.value)]


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ISIN": ["A", "A", "A", "B"],
            "Q": [100.0, 50.0, 30.0, 1000.0],
            "Direction": [1.0, 1.0, -1.0, 1.0],
            "group": ["proprietary", "proprietary", "client", "client"],
            "proprietary": [1, 1, 0, 0],
            "Price Change": [0.01, 0.02, 0.03, 0.04],
            "Daily Vol": [0.10, 0.10, 0.10, 0.10],
            "Q/V": [0.01, 0.02, 0.03, 0.04],
            "Participation Rate": [0.10, 0.20, 0.30, 0.40],
            "Vt/V": [0.01, 0.02, 0.03, 0.04],
            "Period": [
                _period("2024-01-02 09:30:00", "2024-01-02 09:40:00"),
                _period("2024-01-02 09:35:00", "2024-01-02 09:45:00"),
                _period("2024-01-02 09:32:00", "2024-01-02 09:37:00"),
                _period("2024-01-02 09:35:00", "2024-01-02 09:45:00"),
            ],
        }
    )


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    bin_frame = overlap._build_bin_frame(
        trading_hours=(pd.Timestamp("09:30:00").time(), pd.Timestamp("10:00:00").time()),
        bin_minutes=10,
    )
    return overlap._prepare_metaorders(
        df,
        bin_frame=bin_frame,
        trading_hours=(pd.Timestamp("09:30:00").time(), pd.Timestamp("10:00:00").time()),
    )


class TestCrowdingOverlapAnalysis(unittest.TestCase):
    def test_period_parsing_handles_common_interval_formats(self) -> None:
        self.assertEqual(overlap._period_to_start_end_ns([1, 2]), (1, 2))
        self.assertEqual(overlap._period_to_start_end_ns(np.array([3, 4], dtype=np.int64)), (3, 4))
        self.assertEqual(overlap._period_to_start_end_ns("[5, 6]"), (5, 6))
        self.assertEqual(overlap._period_to_start_end_ns("[7 8]"), (7, 8))
        self.assertEqual(overlap._period_to_start_end_ns(9), (9, 9))
        self.assertEqual(overlap._period_to_start_end_ns([]), (None, None))

    def test_overlap_fraction_matrix_matches_hand_computed_cases(self) -> None:
        starts = np.array([0, 5, 20], dtype=np.int64)
        ends = np.array([10, 15, 20], dtype=np.int64)
        duration = np.maximum(ends - starts, 0)
        frac = overlap._overlap_fraction_matrix(
            target_start_ns=starts,
            target_end_ns=ends,
            target_duration_ns=duration,
            other_start_ns=starts,
            other_end_ns=ends,
        )

        self.assertAlmostEqual(frac[0, 1], 0.5)
        self.assertAlmostEqual(frac[1, 0], 0.5)
        self.assertAlmostEqual(frac[0, 2], 0.0)
        self.assertAlmostEqual(frac[2, 2], 1.0)
        self.assertAlmostEqual(frac[2, 0], 0.0)

    def test_leave_one_out_same_opp_and_environment_features(self) -> None:
        prepared = _prepare(_base_frame())
        features = overlap.compute_overlap_features(prepared, batch_size=2, n_jobs=1)
        target = features[(features["ISIN"] == "A") & (features["Q"] == 100.0)].iloc[0]

        self.assertAlmostEqual(target["overlap_any_count_all"], 2.0)
        self.assertAlmostEqual(target["overlap_count_tw_all"], 1.0)
        self.assertAlmostEqual(target["overlap_gross_q_tw_all"], 40.0)
        self.assertAlmostEqual(target["overlap_gross_q_tw_over_Q_all"], 0.4)
        self.assertAlmostEqual(target["overlap_same_q_tw_all"], 25.0)
        self.assertAlmostEqual(target["overlap_opp_q_tw_all"], 15.0)
        self.assertAlmostEqual(target["overlap_net_signed_q_tw_all"], 10.0)
        self.assertAlmostEqual(target["overlap_net_signed_q_tw_over_Q_all"], 0.1)
        self.assertAlmostEqual(target["overlap_active_imbalance_tw_all"], 0.25)

        self.assertAlmostEqual(target["overlap_any_count_prop_env"], 1.0)
        self.assertAlmostEqual(target["overlap_gross_q_tw_prop_env"], 25.0)
        self.assertAlmostEqual(target["overlap_any_count_client_env"], 1.0)
        self.assertAlmostEqual(target["overlap_gross_q_tw_client_env"], 15.0)

        isolated = features[(features["ISIN"] == "B") & (features["Q"] == 1000.0)].iloc[0]
        self.assertAlmostEqual(isolated["overlap_any_count_all"], 0.0)
        self.assertAlmostEqual(isolated["overlap_gross_q_tw_all"], 0.0)
        self.assertTrue(np.isnan(isolated["overlap_active_imbalance_tw_all"]))

    def test_zero_duration_target_uses_point_in_time_rule(self) -> None:
        df = pd.DataFrame(
            {
                "ISIN": ["A", "A", "A"],
                "Q": [10.0, 100.0, 50.0],
                "Direction": [1.0, 1.0, -1.0],
                "group": ["proprietary", "client", "client"],
                "proprietary": [1, 0, 0],
                "Price Change": [0.01, 0.02, 0.03],
                "Daily Vol": [0.10, 0.10, 0.10],
                "Period": [
                    _period("2024-01-02 09:35:00", "2024-01-02 09:35:00"),
                    _period("2024-01-02 09:30:00", "2024-01-02 09:40:00"),
                    _period("2024-01-02 09:36:00", "2024-01-02 09:37:00"),
                ],
            }
        )
        features = overlap.compute_overlap_features(_prepare(df), batch_size=4, n_jobs=1)
        target = features[features["Q"] == 10.0].iloc[0]

        self.assertAlmostEqual(target["duration_minutes"], 0.0)
        self.assertAlmostEqual(target["overlap_any_count_all"], 1.0)
        self.assertAlmostEqual(target["overlap_count_tw_all"], 1.0)
        self.assertAlmostEqual(target["overlap_gross_q_tw_all"], 100.0)
        self.assertAlmostEqual(target["overlap_same_q_tw_all"], 100.0)
        self.assertAlmostEqual(target["overlap_opp_q_tw_all"], 0.0)

    def test_impact_fallback_uses_direction_price_change_over_daily_vol(self) -> None:
        df = pd.DataFrame(
            {
                "ISIN": ["A"],
                "Q": [10.0],
                "Direction": [-1.0],
                "group": ["client"],
                "proprietary": [0],
                "Price Change": [0.20],
                "Daily Vol": [0.10],
                "Period": [_period("2024-01-02 09:30:00", "2024-01-02 09:35:00")],
            }
        )
        prepared = _prepare(df)
        self.assertAlmostEqual(float(prepared["Impact"].iat[0]), -2.0)

    def test_binned_wls_regression_uses_qv_curve_cells_and_control_means(self) -> None:
        rows = []
        qv_centers = np.geomspace(0.001, 0.08, 6)
        for group, is_prop, multiplier in [("client", 0, 1.0), ("proprietary", 1, 1.35)]:
            for bin_idx, qv in enumerate(qv_centers):
                for obs_idx in range(5):
                    impact = multiplier * 0.03 * (qv / qv_centers[0]) ** 0.45
                    impact *= 1.0 + 0.01 * (obs_idx - 2)
                    rows.append(
                        {
                            "group": group,
                            "proprietary": is_prop,
                            "Impact": impact,
                            "Q": 1000.0,
                            "Q/V": qv * (1.0 + 0.002 * obs_idx),
                            "Participation Rate": (0.08 + 0.01 * bin_idx) * (1.0 + 0.01 * obs_idx),
                            "duration_minutes": (5.0 + bin_idx) * (1.0 + 0.01 * obs_idx),
                            "overlap_count_tw_all": 1.0 + 0.2 * bin_idx,
                            "overlap_gross_q_tw_over_Q_all": 2.0 + 0.3 * bin_idx,
                            "overlap_same_q_tw_all": 1300.0 + 80.0 * bin_idx,
                            "overlap_opp_q_tw_all": 700.0 + 50.0 * bin_idx,
                            "overlap_net_signed_q_tw_over_Q_all": 0.1 * bin_idx,
                            "overlap_active_imbalance_tw_all": 0.02 * bin_idx,
                        }
                    )
        features = pd.DataFrame(rows)

        cells, coefs = overlap._run_binned_wls_impact_regressions(
            features,
            run_regressions=True,
            n_logbins=6,
            min_cell_n=3,
        )

        prop = coefs[(coefs["model"] == "W1") & (coefs["term"] == "proprietary_flag")].iloc[0]
        retained = cells[(cells["model"] == "W1") & (cells["group"] == "client")].iloc[0]
        self.assertEqual(prop["status"], "ok")
        self.assertGreater(float(prop["estimate"]), 0.0)
        self.assertAlmostEqual(
            float(retained["log_mean_impact"]),
            float(np.log(retained["mean_impact"])),
        )
        self.assertAlmostEqual(
            float(retained["log_qv"]),
            float(np.log(retained["center_qv"])),
        )
        self.assertAlmostEqual(
            float(retained["log_mean_participation_rate"]),
            float(np.log(retained["mean_participation_rate"])),
        )
        self.assertGreater(float(retained["weight"]), 0.0)

    def test_main_smoke_writes_csvs_without_plots_or_parquet(self) -> None:
        prop = _base_frame().iloc[:2].drop(columns=["group", "proprietary"])
        client = _base_frame().iloc[2:].drop(columns=["group", "proprietary"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prop_path = tmp_path / "prop.parquet"
            client_path = tmp_path / "client.parquet"
            prop.to_parquet(prop_path, index=False)
            client.to_parquet(client_path, index=False)

            rc = overlap.main(
                [
                    "--dataset-name",
                    "demo",
                    "--prop-path",
                    str(prop_path),
                    "--client-path",
                    str(client_path),
                    "--output-file-path",
                    str(tmp_path / "out"),
                    "--img-output-path",
                    str(tmp_path / "images"),
                    "--analysis-tag",
                    "crowding_overlap_test",
                    "--start-bin-minutes",
                    "10",
                    "--n-jobs",
                    "1",
                    "--no-plots",
                    "--no-write-parquet",
                ]
            )

            out_dir = tmp_path / "out" / "crowding_overlap_test"
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "overlap_feature_summary.csv").exists())
            self.assertTrue((out_dir / "overlap_intraday_summary.csv").exists())
            self.assertTrue((out_dir / "overlap_impact_regressions.csv").exists())
            self.assertTrue((out_dir / "overlap_impact_wls_cells.csv").exists())
            self.assertTrue((out_dir / "overlap_impact_wls_regressions.csv").exists())
            self.assertTrue((out_dir / "run_manifest.json").exists())
            self.assertFalse((out_dir / "overlap_features.parquet").exists())


if __name__ == "__main__":
    unittest.main()
