from __future__ import annotations

import builtins
import importlib
import unittest

import numpy as np
import pandas as pd

import scripts.metaorder_start_event_study as event_study


def _period(start_ts: str, end_ts: str) -> list[int]:
    start = pd.Timestamp(start_ts)
    end = pd.Timestamp(end_ts)
    return [int(start.value), int(end.value)]


class TestMetaorderStartEventStudy(unittest.TestCase):
    def test_import_does_not_patch_print(self) -> None:
        original_print = builtins.print
        importlib.reload(event_study)
        self.assertIs(builtins.print, original_print)

    def test_period_start_ns_handles_common_encodings(self) -> None:
        self.assertEqual(event_study._period_start_ns([1, 2]), 1)
        self.assertEqual(event_study._period_start_ns(np.array([3, 4], dtype=np.int64)), 3)
        self.assertEqual(event_study._period_start_ns("[5, 6]"), 5)
        self.assertIsNone(event_study._period_start_ns([]))

    def test_compute_exposures_minutes_truncates_near_open(self) -> None:
        bin_spec = event_study._compute_bin_spec(window_minutes=30, bin_minutes=5)
        session_open_ns = np.int64(pd.Timestamp("2024-01-02 09:30:00").value)
        session_close_ns = np.int64(pd.Timestamp("2024-01-02 17:30:00").value)
        start_ns = np.asarray([pd.Timestamp("2024-01-02 09:32:00").value], dtype=np.int64)

        exposures = event_study._compute_exposures_minutes(
            start_ns,
            session_open_ns=session_open_ns,
            session_close_ns=session_close_ns,
            bin_spec=bin_spec,
        )

        self.assertEqual(exposures.shape, (1, 12))
        self.assertAlmostEqual(float(np.sum(exposures[0, :6])), 2.0)
        self.assertAlmostEqual(float(np.sum(exposures[0, 6:])), 30.0)

    def test_prepare_group_metrics_counts_same_and_opposite_sign_neighbors(self) -> None:
        df = pd.DataFrame(
            {
                "ISIN": ["AAA", "AAA", "AAA", "AAA"],
                "Date": pd.to_datetime(["2024-01-02"] * 4),
                "Period": [
                    _period("2024-01-02 09:58:00", "2024-01-02 09:59:00"),
                    _period("2024-01-02 10:00:00", "2024-01-02 10:01:00"),
                    _period("2024-01-02 10:03:00", "2024-01-02 10:04:00"),
                    _period("2024-01-02 10:04:00", "2024-01-02 10:05:00"),
                ],
                "Direction": [1, 1, 1, -1],
                "Participation Rate": [0.2, 0.95, 0.3, 0.4],
                "Member": ["M1", "M2", "M3", "M4"],
            }
        )
        bin_spec = event_study._compute_bin_spec(window_minutes=5, bin_minutes=5)
        out = event_study._prepare_group_metrics(
            df,
            label="demo",
            bin_spec=bin_spec,
            session_start_time=pd.Timestamp("09:30:00").time(),
            session_end_time=pd.Timestamp("17:30:00").time(),
            same_actor_col="Member",
            exclude_same_actor=False,
        )

        anchor = out.loc[out["Participation Rate"].eq(0.95)].iloc[0]
        self.assertAlmostEqual(float(anchor["same_pre_mean_rate"]), 1.0 / 5.0)
        self.assertAlmostEqual(float(anchor["same_post_mean_rate"]), 1.0 / 5.0)
        self.assertAlmostEqual(float(anchor["opp_post_mean_rate"]), 1.0 / 5.0)
        self.assertAlmostEqual(float(anchor["opp_pre_mean_rate"]), 0.0)

    def test_weighted_stat_from_strata_uses_treated_anchor_weighting(self) -> None:
        strata = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "stratum_key": ["A", "B"],
                "n_treated": [2, 1],
                "n_control": [2, 1],
                "sum_treated__same_pre_mean_rate": [4.0, 1.0],
                "sum_control__same_pre_mean_rate": [2.0, 0.0],
            }
        )
        treated, control, excess, total_treated = event_study._weighted_stat_from_strata(
            strata,
            metric_cols=["same_pre_mean_rate"],
        )

        self.assertEqual(total_treated, 3.0)
        self.assertAlmostEqual(float(treated[0]), 5.0 / 3.0)
        self.assertAlmostEqual(float(control[0]), 4.0 / 6.0)
        self.assertAlmostEqual(float(excess[0]), 1.0)

    def test_permutation_summary_stats_returns_finite_draws(self) -> None:
        df = pd.DataFrame(
            {
                "stratum_key": ["A", "A", "A", "A"],
                "high_eta": [1, 1, 0, 0],
                "same_pre_mean_rate": [2.0, 1.8, 0.2, 0.1],
                "same_post_mean_rate": [2.2, 2.0, 0.3, 0.2],
                "opp_pre_mean_rate": [0.5, 0.4, 0.6, 0.5],
                "opp_post_mean_rate": [0.4, 0.3, 0.5, 0.4],
            }
        )

        observed, draws = event_study._permutation_summary_stats(
            df,
            summary_metric_cols=[name for name, _, _, _ in event_study.SUMMARY_METRIC_SPECS],
            n_runs=20,
            seed=0,
        )

        self.assertEqual(observed.shape, (4,))
        self.assertEqual(draws.shape, (20, 4))
        self.assertTrue(np.all(np.isfinite(draws)))
