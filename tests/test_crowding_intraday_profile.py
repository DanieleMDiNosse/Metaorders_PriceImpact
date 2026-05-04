from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import moimpact.workflows.crowding.intraday as intraday_crowding


def _period(start_ts: str, end_ts: str | None = None) -> list[int]:
    start = pd.Timestamp(start_ts)
    end = pd.Timestamp(end_ts) if end_ts is not None else start + pd.Timedelta(minutes=5)
    return [int(start.value), int(end.value)]


def _build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    prop = pd.DataFrame(
        {
            "ISIN": ["A", "A", "A", "A"],
            "Q": [10.0, 10.0, 8.0, 6.0],
            "Direction": [1.0, -1.0, 1.0, -1.0],
            "Period": [
                _period("2024-01-02 09:30:00"),
                _period("2024-01-02 09:35:00"),
                _period("2024-01-03 09:30:00"),
                _period("2024-01-03 09:35:00"),
            ],
        }
    )
    client = pd.DataFrame(
        {
            "ISIN": ["A", "A", "A", "A"],
            "Q": [10.0, 5.0, 8.0, 12.0],
            "Direction": [1.0, 1.0, -1.0, 1.0],
            "Period": [
                _period("2024-01-02 09:32:00"),
                _period("2024-01-02 09:44:00"),
                _period("2024-01-03 09:31:00"),
                _period("2024-01-03 09:45:00"),
            ],
        }
    )
    return prop, client


class TestCrowdingIntradayProfile(unittest.TestCase):
    def test_period_endpoint_ns_handles_common_formats(self) -> None:
        self.assertEqual(intraday_crowding._period_endpoint_ns([1, 2], 0), 1)
        self.assertEqual(intraday_crowding._period_endpoint_ns(np.array([3, 4], dtype=np.int64), 1), 4)
        self.assertEqual(intraday_crowding._period_endpoint_ns("[5, 6]", 0), 5)
        self.assertIsNone(intraday_crowding._period_endpoint_ns([], 0))

    def test_summary_matches_expected_day_equal_averages(self) -> None:
        prop, client = _build_frames()
        bin_frame = intraday_crowding._build_bin_frame(
            trading_hours=(pd.Timestamp("09:30:00").time(), pd.Timestamp("10:00:00").time()),
            bin_minutes=10,
        )
        prepared = pd.concat(
            [
                intraday_crowding._attach_start_bin_columns(
                    prop.assign(group="proprietary", group_label="Proprietary"),
                    bin_frame=bin_frame,
                    trading_hours=(pd.Timestamp("09:30:00").time(), pd.Timestamp("10:00:00").time()),
                ),
                intraday_crowding._attach_start_bin_columns(
                    client.assign(group="client", group_label="Client"),
                    bin_frame=bin_frame,
                    trading_hours=(pd.Timestamp("09:30:00").time(), pd.Timestamp("10:00:00").time()),
                ),
            ],
            ignore_index=True,
        )
        prepared = prepared[prepared["inside_trading_hours"] & prepared["start_bin_id"].notna()].reset_index(drop=True)
        prepared = intraday_crowding._compute_intraday_all_others_crowding(prepared)
        day_panel = intraday_crowding._build_day_bin_panel(prepared)
        summary = intraday_crowding._build_summary_table(
            day_panel,
            bin_frame=bin_frame,
            min_n_day_bin=1,
            alpha=0.05,
            bootstrap_runs=0,
            seed=0,
        )

        prop_bin0 = summary[(summary["group"] == "proprietary") & (summary["bin_id"] == 0)].iloc[0]
        client_bin0 = summary[(summary["group"] == "client") & (summary["bin_id"] == 0)].iloc[0]
        prop_bin1 = summary[(summary["group"] == "proprietary") & (summary["bin_id"] == 1)].iloc[0]
        client_bin1 = summary[(summary["group"] == "client") & (summary["bin_id"] == 1)].iloc[0]

        self.assertAlmostEqual(prop_bin0["avg_daily_mean_aligned_crowding"], -0.5, places=6)
        self.assertAlmostEqual(prop_bin0["avg_daily_median_aligned_crowding"], -0.5, places=6)
        self.assertAlmostEqual(client_bin0["avg_daily_mean_aligned_crowding"], -0.07142857142857142, places=6)
        self.assertAlmostEqual(client_bin0["avg_daily_median_aligned_crowding"], -0.07142857142857142, places=6)
        self.assertEqual(int(prop_bin1["n_days"]), 0)
        self.assertEqual(int(client_bin1["n_days"]), 0)
        self.assertTrue(np.isnan(prop_bin1["avg_daily_mean_aligned_crowding"]))
        self.assertTrue(np.isnan(client_bin1["avg_daily_mean_aligned_crowding"]))

    def test_main_writes_outputs_without_plots(self) -> None:
        prop, client = _build_frames()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prop_path = tmp_path / "prop.parquet"
            client_path = tmp_path / "client.parquet"
            prop.to_parquet(prop_path, index=False)
            client.to_parquet(client_path, index=False)

            rc = intraday_crowding.main(
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
                    "crowding_intraday_test",
                    "--bin-minutes",
                    "10",
                    "--min-n-day-bin",
                    "1",
                    "--bootstrap-runs",
                    "0",
                    "--no-plots",
                    "--no-write-parquet",
                ]
            )

            self.assertEqual(rc, 0)
            day_panel_path = tmp_path / "out" / "crowding_intraday_test" / "day_bin_panel_10min.csv"
            summary_path = tmp_path / "out" / "crowding_intraday_test" / "intraday_profile_summary_10min.csv"
            self.assertTrue(day_panel_path.exists())
            self.assertTrue(summary_path.exists())

    def test_heatmap_figure_builds_two_group_panels(self) -> None:
        prop, client = _build_frames()
        bin_frame = intraday_crowding._build_bin_frame(
            trading_hours=(pd.Timestamp("09:30:00").time(), pd.Timestamp("10:00:00").time()),
            bin_minutes=10,
        )
        prepared = pd.concat(
            [
                intraday_crowding._attach_start_bin_columns(
                    prop.assign(group="proprietary", group_label="Proprietary"),
                    bin_frame=bin_frame,
                    trading_hours=(pd.Timestamp("09:30:00").time(), pd.Timestamp("10:00:00").time()),
                ),
                intraday_crowding._attach_start_bin_columns(
                    client.assign(group="client", group_label="Client"),
                    bin_frame=bin_frame,
                    trading_hours=(pd.Timestamp("09:30:00").time(), pd.Timestamp("10:00:00").time()),
                ),
            ],
            ignore_index=True,
        )
        prepared = prepared[prepared["inside_trading_hours"] & prepared["start_bin_id"].notna()].reset_index(drop=True)
        prepared = intraday_crowding._compute_intraday_all_others_crowding(prepared)
        day_panel = intraday_crowding._build_day_bin_panel(prepared)

        fig = intraday_crowding._build_day_bin_heatmap_figure(
            day_panel,
            metric="mean_aligned_crowding",
            metric_label="Daily mean aligned crowding",
            trading_hours=(pd.Timestamp("09:30:00").time(), pd.Timestamp("10:00:00").time()),
        )

        self.assertEqual(len(fig.data), 2)
        self.assertEqual(fig.data[0].type, "heatmap")
        self.assertEqual(fig.data[1].type, "heatmap")


if __name__ == "__main__":
    unittest.main()
