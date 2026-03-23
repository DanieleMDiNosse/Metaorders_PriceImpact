from __future__ import annotations

import builtins
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import scripts.metaorder_intraday_analysis as intraday


def _period(start_ts: str, end_ts: str) -> list[int]:
    start = pd.Timestamp(start_ts)
    end = pd.Timestamp(end_ts)
    return [int(start.value), int(end.value)]


class TestMetaorderIntradayAnalysis(unittest.TestCase):
    def test_import_does_not_patch_print(self) -> None:
        original_print = builtins.print
        importlib.reload(intraday)
        self.assertIs(builtins.print, original_print)

    def test_period_endpoint_ns_handles_common_period_encodings(self) -> None:
        self.assertEqual(intraday._period_endpoint_ns([1, 2], 0), 1)
        self.assertEqual(intraday._period_endpoint_ns(np.array([3, 4], dtype=np.int64), 1), 4)
        self.assertEqual(intraday._period_endpoint_ns("[5, 6]", 0), 5)
        self.assertIsNone(intraday._period_endpoint_ns([], 0))

    def test_classify_intraday_session_assigns_boundary_to_evening(self) -> None:
        morning_ts = pd.Timestamp("2024-01-02 10:15:00")
        evening_ts = pd.Timestamp("2024-01-02 15:15:00")
        boundary_ts = pd.Timestamp("2024-01-02 13:30:00")

        self.assertEqual(
            intraday._classify_intraday_session(morning_ts, intraday.SESSION_WINDOWS),
            "morning",
        )
        self.assertEqual(
            intraday._classify_intraday_session(evening_ts, intraday.SESSION_WINDOWS),
            "evening",
        )
        self.assertEqual(
            intraday._classify_intraday_session(boundary_ts, intraday.SESSION_WINDOWS),
            "evening",
        )

    def test_attach_intraday_session_columns_drops_cross_session_metaorders(self) -> None:
        df = pd.DataFrame(
            {
                "Period": [
                    _period("2024-01-02 09:45:00", "2024-01-02 10:00:00"),
                    _period("2024-01-02 14:00:00", "2024-01-02 14:15:00"),
                    _period("2024-01-02 13:20:00", "2024-01-02 13:40:00"),
                    _period("2024-01-02 13:30:00", "2024-01-02 13:35:00"),
                ]
            }
        )

        out = intraday._attach_intraday_session_columns(df, intraday.SESSION_WINDOWS)
        self.assertEqual(out["Session"].tolist(), ["morning", "evening", None, "evening"])

    def test_run_group_session_analysis_builds_expected_counts(self) -> None:
        df = pd.DataFrame(
            {
                "Period": [
                    _period("2024-01-02 09:40:00", "2024-01-02 09:45:00"),
                    _period("2024-01-02 09:50:00", "2024-01-02 09:55:00"),
                    _period("2024-01-02 10:10:00", "2024-01-02 10:15:00"),
                    _period("2024-01-02 10:20:00", "2024-01-02 10:25:00"),
                    _period("2024-01-02 13:40:00", "2024-01-02 13:45:00"),
                    _period("2024-01-02 13:50:00", "2024-01-02 13:55:00"),
                    _period("2024-01-02 14:10:00", "2024-01-02 14:15:00"),
                    _period("2024-01-02 14:20:00", "2024-01-02 14:25:00"),
                    _period("2024-01-02 13:20:00", "2024-01-02 13:40:00"),
                ],
                "Q/V": [0.01, 0.011, 0.20, 0.21, 0.01, 0.011, 0.20, 0.21, 0.05],
                "Price Change": [0.010, 0.011, 0.030, 0.032, 0.012, 0.013, 0.034, 0.036, 0.02],
                "Direction": [1] * 9,
                "Daily Vol": [1.0] * 9,
                "Participation Rate": [0.3] * 9,
                "Vt/V": [0.2] * 9,
                "Q": [100.0] * 9,
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            session_colors = intraday._session_color_map(intraday.SESSION_WINDOWS.keys())
            with mock.patch.object(intraday, "ANALYSIS_OUTPUT_DIR", tmp_path), mock.patch.object(
                intraday, "MIN_COUNT", 2
            ), mock.patch.object(intraday, "N_LOGBIN", 2):
                summary_rows, session_fits = intraday._run_group_session_analysis(
                    df,
                    group_tag="proprietary",
                    group_label="Proprietary",
                    session_colors=session_colors,
                )

        summary = {row["session"]: row for row in summary_rows}
        self.assertEqual(summary["morning"]["n_detected"], 4)
        self.assertEqual(summary["evening"]["n_detected"], 4)
        self.assertEqual(summary["morning"]["n_after_qv_filter"], 4)
        self.assertEqual(summary["evening"]["n_after_qv_filter"], 4)
        self.assertEqual(summary["morning"]["n_fit_sample_after_pr_cap"], 4)
        self.assertEqual(summary["evening"]["n_fit_sample_after_pr_cap"], 4)
        self.assertIn("morning", session_fits)
        self.assertIn("evening", session_fits)
        self.assertFalse(np.isnan(summary["morning"]["power_gamma"]))
        self.assertFalse(np.isnan(summary["evening"]["power_gamma"]))

