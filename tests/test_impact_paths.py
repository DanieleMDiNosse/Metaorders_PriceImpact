from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from moimpact.stats.impact_paths import bootstrap_retention_difference


class TestImpactPathRetentionBootstrap(unittest.TestCase):
    def _build_demo_frame(
        self,
        *,
        start_values: list[float],
        end_values: list[float],
        dates: list[str],
    ) -> pd.DataFrame:
        records = []
        for start_value, end_value, date_str in zip(start_values, end_values, dates, strict=True):
            start_ts = pd.Timestamp(f"{date_str} 10:00:00")
            end_ts = pd.Timestamp(f"{date_str} 10:05:00")
            records.append(
                {
                    "Period": [int(start_ts.value), int(end_ts.value)],
                    "partial_impact": np.asarray([0.0, start_value], dtype=np.float32),
                    "aftermath_impact": np.asarray([start_value, end_value], dtype=np.float32),
                }
            )
        return pd.DataFrame(records)

    def test_bootstrap_retention_difference_is_reproducible(self) -> None:
        proprietary = self._build_demo_frame(
            start_values=[2.0, 4.0],
            end_values=[1.6, 3.0],
            dates=["2024-01-02", "2024-01-03"],
        )
        client = self._build_demo_frame(
            start_values=[2.0, 4.0],
            end_values=[1.0, 2.0],
            dates=["2024-01-02", "2024-01-03"],
        )

        out1 = bootstrap_retention_difference(
            proprietary,
            client,
            tau_start=1.0,
            tau_end=3.0,
            duration_multiplier=2.0,
            n_bootstrap=200,
            random_state=0,
        )
        out2 = bootstrap_retention_difference(
            proprietary,
            client,
            tau_start=1.0,
            tau_end=3.0,
            duration_multiplier=2.0,
            n_bootstrap=200,
            random_state=0,
        )

        self.assertEqual(out1.summary_dict(), out2.summary_dict())
        np.testing.assert_allclose(out1.bootstrap_delta_retention, out2.bootstrap_delta_retention)
        self.assertAlmostEqual(out1.proprietary_retention, 4.6 / 6.0)
        self.assertAlmostEqual(out1.client_retention, 3.0 / 6.0)
        self.assertAlmostEqual(out1.delta_retention, (4.6 / 6.0) - (3.0 / 6.0))
        self.assertEqual(out1.n_proprietary_metaorders, 2)
        self.assertEqual(out1.n_client_metaorders, 2)
        self.assertEqual(out1.n_proprietary_clusters, 2)
        self.assertEqual(out1.n_client_clusters, 2)
        self.assertGreater(out1.n_bootstrap_valid, 0)
        self.assertGreater(out1.delta_ci_low, 0.0)
        self.assertLessEqual(out1.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
