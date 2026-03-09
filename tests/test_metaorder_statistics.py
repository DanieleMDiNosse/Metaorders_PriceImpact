from __future__ import annotations

import datetime as dt
import unittest

import numpy as np
import pandas as pd

from moimpact.plotting import COLOR_CLIENT, COLOR_PROPRIETARY
from scripts.metaorder_statistics import (
    _daily_metaorder_volume,
    build_daily_metaorder_share_table,
    build_mean_daily_metaorder_share_figure,
)


class TestMetaorderStatisticsHelpers(unittest.TestCase):
    def test_daily_metaorder_volume_aggregates_by_start_day(self) -> None:
        trades = pd.DataFrame(
            {
                "Trade Time": pd.to_datetime(
                    [
                        "2024-01-02 10:00:00",
                        "2024-01-02 10:05:00",
                        "2024-01-03 11:00:00",
                    ]
                ),
                "Total Quantity Buy": [10.0, 0.0, 5.0],
                "Total Quantity Sell": [0.0, 20.0, 0.0],
            }
        )
        metaorders = {"member_a": [[0, 1], [2]]}

        out = _daily_metaorder_volume(metaorders, trades)

        self.assertEqual(float(out.loc[dt.date(2024, 1, 2)]), 30.0)
        self.assertEqual(float(out.loc[dt.date(2024, 1, 3)]), 5.0)

    def test_build_daily_metaorder_share_table_zero_fills_missing_groups(self) -> None:
        total_market_volume = pd.Series(
            {
                dt.date(2024, 1, 2): 100.0,
                dt.date(2024, 1, 3): 200.0,
            }
        )
        proprietary_metaorder_volume = pd.Series({dt.date(2024, 1, 2): 20.0})
        client_metaorder_volume = pd.Series(
            {
                dt.date(2024, 1, 2): 5.0,
                dt.date(2024, 1, 3): 40.0,
            }
        )

        out = build_daily_metaorder_share_table(
            total_market_volume,
            proprietary_metaorder_volume,
            client_metaorder_volume,
        )

        self.assertEqual(list(out["Date"]), [dt.date(2024, 1, 2), dt.date(2024, 1, 3)])
        self.assertEqual(float(out.loc[1, "proprietary_metaorder_volume"]), 0.0)
        np.testing.assert_allclose(out["proprietary_ratio"].to_numpy(dtype=float), [0.2, 0.0])
        np.testing.assert_allclose(out["client_ratio"].to_numpy(dtype=float), [0.05, 0.2])
        np.testing.assert_allclose(out["total_ratio"].to_numpy(dtype=float), [0.25, 0.2])
        self.assertAlmostEqual(
            float(out["total_ratio"].mean()),
            float(out["proprietary_ratio"].mean() + out["client_ratio"].mean()),
        )

    def test_build_mean_daily_metaorder_share_figure_uses_group_colors(self) -> None:
        daily_share_table = pd.DataFrame(
            {
                "ISIN": ["AAA", "AAA", "BBB"],
                "Date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 2)],
                "proprietary_ratio": [0.2, 0.0, 0.05],
                "client_ratio": [0.05, 0.2, 0.1],
            }
        )

        fig = build_mean_daily_metaorder_share_figure(daily_share_table)

        self.assertEqual(len(fig.data), 2)
        traces = {trace.name: trace for trace in fig.data}
        self.assertEqual(traces["Proprietary"].marker.color, COLOR_PROPRIETARY)
        self.assertEqual(traces["Client"].marker.color, COLOR_CLIENT)
        self.assertEqual(list(traces["Proprietary"].x), ["AAA", "BBB"])
        np.testing.assert_allclose(traces["Proprietary"].y, [10.0, 5.0])
        np.testing.assert_allclose(traces["Client"].y, [12.5, 10.0])
        self.assertEqual(fig.layout.barmode, "stack")
        self.assertFalse(bool(fig.layout.showlegend))
        np.testing.assert_allclose(traces["Proprietary"].customdata[:, 0], [22.5, 15.0])
        np.testing.assert_array_equal(traces["Proprietary"].customdata[:, 1], [2, 1])


if __name__ == "__main__":
    unittest.main()
