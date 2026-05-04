from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import moimpact.workflows.metaorders.start_time_distribution as start_dist


def _period(start_ts: str, end_ts: str | None = None) -> list[int]:
    start = pd.Timestamp(start_ts)
    end = pd.Timestamp(end_ts) if end_ts is not None else start + pd.Timedelta(minutes=5)
    return [int(start.value), int(end.value)]


class TestMetaorderStartTimeDistribution(unittest.TestCase):
    def test_period_endpoint_ns_handles_common_encodings(self) -> None:
        self.assertEqual(start_dist._period_endpoint_ns([1, 2], 0), 1)
        self.assertEqual(start_dist._period_endpoint_ns(np.array([3, 4], dtype=np.int64), 1), 4)
        self.assertEqual(start_dist._period_endpoint_ns("[5, 6]", 0), 5)
        self.assertIsNone(start_dist._period_endpoint_ns([], 0))

    def test_build_group_distribution_counts_bins_and_close_edge(self) -> None:
        trading_hours = (dt.time(9, 30), dt.time(10, 0))
        bin_frame = start_dist._build_bin_frame(trading_hours=trading_hours, bin_minutes=10)
        df = pd.DataFrame(
            {
                "Period": [
                    _period("2024-01-02 09:30:00"),
                    _period("2024-01-02 09:39:59"),
                    _period("2024-01-02 09:40:00"),
                    _period("2024-01-02 09:50:00"),
                    _period("2024-01-02 10:00:00"),
                    _period("2024-01-02 10:05:00"),
                    _period("2024-01-02 09:20:00"),
                ]
            }
        )

        out = start_dist._build_group_distribution_table(
            df,
            group=start_dist.GROUP_PROPRIETARY,
            bin_frame=bin_frame,
            trading_hours=trading_hours,
        )

        self.assertEqual(out["bin_label"].tolist(), ["09:30-09:40", "09:40-09:50", "09:50-10:00"])
        self.assertEqual(out["n_metaorders"].tolist(), [2, 1, 2])
        self.assertEqual(int(out["n_inside_trading_hours"].iloc[0]), 5)
        self.assertEqual(int(out["n_outside_trading_hours"].iloc[0]), 2)
        np.testing.assert_allclose(out["share_metaorders"].to_numpy(), np.array([0.4, 0.2, 0.4]))

    def test_combined_distribution_preserves_both_groups(self) -> None:
        trading_hours = (dt.time(9, 30), dt.time(10, 0))
        prop = pd.DataFrame({"Period": [_period("2024-01-02 09:30:00")]})
        client = pd.DataFrame(
            {"Period": [_period("2024-01-02 09:40:00"), _period("2024-01-02 09:50:00")]}
        )

        out = start_dist._build_combined_distribution_table(
            prop,
            client,
            trading_hours=trading_hours,
            bin_minutes=10,
        )

        self.assertEqual(set(out["group"]), {"proprietary", "client"})
        prop_counts = out.loc[out["group"] == "proprietary", "n_metaorders"].tolist()
        client_counts = out.loc[out["group"] == "client", "n_metaorders"].tolist()
        self.assertEqual(prop_counts, [1, 0, 0])
        self.assertEqual(client_counts, [0, 1, 1])

    def test_main_writes_csv_and_manifest_without_plots(self) -> None:
        prop = pd.DataFrame({"Period": [_period("2024-01-02 09:30:00"), _period("2024-01-02 09:40:00")]})
        client = pd.DataFrame({"Period": [_period("2024-01-02 09:50:00")]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prop_path = tmp_path / "prop.parquet"
            client_path = tmp_path / "client.parquet"
            prop.to_parquet(prop_path, index=False)
            client.to_parquet(client_path, index=False)

            rc = start_dist.main(
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
                    "start_dist_test",
                    "--bin-minutes",
                    "10",
                    "--no-plots",
                    "--no-write-parquet",
                ]
            )

            self.assertEqual(rc, 0)
            csv_path = tmp_path / "out" / "start_dist_test" / "start_time_distribution_10min.csv"
            manifest_path = tmp_path / "out" / "start_dist_test" / "run_manifest.json"
            self.assertTrue(csv_path.exists())
            self.assertTrue(manifest_path.exists())

            out = pd.read_csv(csv_path)
            self.assertEqual(set(out["group"]), {"proprietary", "client"})
            self.assertEqual(
                out.loc[out["group"] == "proprietary", "n_metaorders"].tolist()[:2],
                [1, 1],
            )
            self.assertEqual(int(out.loc[out["group"] == "client", "n_metaorders"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
