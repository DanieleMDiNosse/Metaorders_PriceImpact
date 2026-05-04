from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from moimpact.stats.active_overlap import compute_member_active_overlap_features
import moimpact.workflows.crowding.member_overlap as member_overlap


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def _base_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group": "proprietary",
                "Member": "M1",
                "ISIN": "A",
                "Date": _ts("2024-01-02"),
                "StartTimestamp": _ts("2024-01-02 10:00:00"),
                "EndTimestamp": _ts("2024-01-02 10:10:00"),
                "Direction": 1.0,
                "Q": 100.0,
            },
            {
                "group": "client",
                "Member": "M1",
                "ISIN": "A",
                "Date": _ts("2024-01-02"),
                "StartTimestamp": _ts("2024-01-02 09:55:00"),
                "EndTimestamp": _ts("2024-01-02 10:05:00"),
                "Direction": 1.0,
                "Q": 60.0,
            },
            {
                "group": "client",
                "Member": "M1",
                "ISIN": "A",
                "Date": _ts("2024-01-02"),
                "StartTimestamp": _ts("2024-01-02 10:02:00"),
                "EndTimestamp": _ts("2024-01-02 10:08:00"),
                "Direction": -1.0,
                "Q": 40.0,
            },
            {
                "group": "client",
                "Member": "M1",
                "ISIN": "B",
                "Date": _ts("2024-01-02"),
                "StartTimestamp": _ts("2024-01-02 10:00:00"),
                "EndTimestamp": _ts("2024-01-02 10:10:00"),
                "Direction": -1.0,
                "Q": 100.0,
            },
            {
                "group": "client",
                "Member": "M2",
                "ISIN": "A",
                "Date": _ts("2024-01-02"),
                "StartTimestamp": _ts("2024-01-02 10:00:00"),
                "EndTimestamp": _ts("2024-01-02 10:10:00"),
                "Direction": -1.0,
                "Q": 1000.0,
            },
        ]
    )


class TestMemberActiveOverlapFeatures(unittest.TestCase):
    def test_same_isin_scope_and_lead_lag_buckets_match_hand_calculation(self) -> None:
        features = compute_member_active_overlap_features(
            _base_rows(),
            scopes=["same_isin"],
            lead_lag_buckets=["all_active", "preexisting_at_prop_start", "starts_during_prop"],
            batch_size=2,
            n_jobs=1,
        )

        all_active = features.loc[features["lead_lag_bucket"] == "all_active"].iloc[0]
        preexisting = features.loc[features["lead_lag_bucket"] == "preexisting_at_prop_start"].iloc[0]
        starts_during = features.loc[features["lead_lag_bucket"] == "starts_during_prop"].iloc[0]

        # Same-ISIN clients contribute 5/10 * 60 = 30 signed buy volume and
        # 6/10 * 40 = 24 signed sell volume.
        self.assertAlmostEqual(float(all_active["active_client_gross_q_tw"]), 54.0)
        self.assertAlmostEqual(float(all_active["active_client_signed_q_tw"]), 6.0)
        self.assertAlmostEqual(float(all_active["active_client_imbalance"]), 6.0 / 54.0)
        self.assertAlmostEqual(float(all_active["active_client_any_count"]), 2.0)

        self.assertAlmostEqual(float(preexisting["active_client_gross_q_tw"]), 30.0)
        self.assertAlmostEqual(float(preexisting["active_client_imbalance"]), 1.0)

        self.assertAlmostEqual(float(starts_during["active_client_gross_q_tw"]), 24.0)
        self.assertAlmostEqual(float(starts_during["active_client_imbalance"]), -1.0)

    def test_all_isin_scope_includes_cross_instrument_same_member_clients(self) -> None:
        features = compute_member_active_overlap_features(
            _base_rows(),
            scopes=["all_isin"],
            lead_lag_buckets=["all_active"],
            batch_size=8,
            n_jobs=1,
        )
        row = features.iloc[0]

        # The same-member cross-ISIN sell client adds 100 units with full weight.
        self.assertAlmostEqual(float(row["active_client_gross_q_tw"]), 154.0)
        self.assertAlmostEqual(float(row["active_client_signed_q_tw"]), -94.0)
        self.assertAlmostEqual(float(row["active_client_imbalance"]), -94.0 / 154.0)
        self.assertAlmostEqual(float(row["active_client_any_count"]), 3.0)

    def test_zero_duration_target_uses_point_active_rule(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "group": "proprietary",
                    "Member": "M1",
                    "ISIN": "A",
                    "Date": _ts("2024-01-02"),
                    "StartTimestamp": _ts("2024-01-02 10:00:00"),
                    "EndTimestamp": _ts("2024-01-02 10:00:00"),
                    "Direction": -1.0,
                    "Q": 10.0,
                },
                {
                    "group": "client",
                    "Member": "M1",
                    "ISIN": "A",
                    "Date": _ts("2024-01-02"),
                    "StartTimestamp": _ts("2024-01-02 09:59:00"),
                    "EndTimestamp": _ts("2024-01-02 10:01:00"),
                    "Direction": -1.0,
                    "Q": 50.0,
                },
            ]
        )
        features = compute_member_active_overlap_features(frame, scopes=["same_isin"], n_jobs=1)
        all_active = features.loc[features["lead_lag_bucket"] == "all_active"].iloc[0]

        self.assertAlmostEqual(float(all_active["target_duration_minutes"]), 0.0)
        self.assertAlmostEqual(float(all_active["active_client_count_tw"]), 1.0)
        self.assertAlmostEqual(float(all_active["active_client_imbalance"]), -1.0)
        self.assertAlmostEqual(float(all_active["active_client_alignment"]), 1.0)

    def test_no_overlap_rows_have_nan_imbalance(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "group": "proprietary",
                    "Member": "M1",
                    "ISIN": "A",
                    "Date": _ts("2024-01-02"),
                    "StartTimestamp": _ts("2024-01-02 10:00:00"),
                    "EndTimestamp": _ts("2024-01-02 10:10:00"),
                    "Direction": 1.0,
                    "Q": 10.0,
                },
                {
                    "group": "client",
                    "Member": "M1",
                    "ISIN": "A",
                    "Date": _ts("2024-01-02"),
                    "StartTimestamp": _ts("2024-01-02 10:20:00"),
                    "EndTimestamp": _ts("2024-01-02 10:30:00"),
                    "Direction": 1.0,
                    "Q": 50.0,
                },
            ]
        )
        features = compute_member_active_overlap_features(frame, scopes=["same_isin"], n_jobs=1)
        row = features.loc[features["lead_lag_bucket"] == "all_active"].iloc[0]

        self.assertAlmostEqual(float(row["active_client_gross_q_tw"]), 0.0)
        self.assertTrue(np.isnan(row["active_client_imbalance"]))


class TestMemberActiveOverlapWorkflow(unittest.TestCase):
    def test_main_smoke_writes_tables_and_manifest_without_plots(self) -> None:
        rows: list[dict[str, object]] = []
        directions = [1.0, -1.0, 1.0, -1.0]
        for idx, direction in enumerate(directions):
            start = _ts(f"2024-01-0{2 + idx // 2} 10:{10 * (idx % 2):02d}:00")
            end = start + pd.Timedelta(minutes=5)
            rows.append(
                {
                    "group": "proprietary",
                    "Member": "M1",
                    "ISIN": "A",
                    "Date": start.normalize(),
                    "StartTimestamp": start,
                    "EndTimestamp": end,
                    "Direction": direction,
                    "Q": 100.0,
                }
            )
            rows.append(
                {
                    "group": "client",
                    "Member": "M1",
                    "ISIN": "A",
                    "Date": start.normalize(),
                    "StartTimestamp": start - pd.Timedelta(minutes=1),
                    "EndTimestamp": end + pd.Timedelta(minutes=1),
                    "Direction": direction,
                    "Q": 50.0,
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "overlap_features.parquet"
            pd.DataFrame(rows).to_parquet(input_path, index=False)

            rc = member_overlap.main(
                [
                    "--dataset-name",
                    "demo",
                    "--input-path",
                    str(input_path),
                    "--output-file-path",
                    str(tmp_path / "out"),
                    "--img-output-path",
                    str(tmp_path / "images"),
                    "--analysis-tag",
                    "member_active_overlap_test",
                    "--bootstrap-runs",
                    "0",
                    "--n-jobs",
                    "1",
                    "--min-obs-per-member",
                    "3",
                    "--no-plots",
                    "--no-write-target-parquet",
                ]
            )

            out_dir = tmp_path / "out" / "member_active_overlap_test"
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "global_correlations.csv").exists())
            self.assertTrue((out_dir / "per_member_correlations.csv").exists())
            self.assertTrue((out_dir / "per_member_correlations.parquet").exists())
            self.assertTrue((out_dir / "member_window_correlations.csv").exists())
            self.assertTrue((out_dir / "member_comovement_series.csv").exists())
            self.assertTrue((out_dir / "member_comovement_series.parquet").exists())
            self.assertTrue((out_dir / "sample_counts.csv").exists())
            self.assertTrue((out_dir / "run_manifest.json").exists())
            self.assertFalse((out_dir / "active_member_overlap_targets.parquet").exists())

            global_corr = pd.read_csv(out_dir / "global_correlations.csv")
            same_isin_all = global_corr[
                (global_corr["scope"] == "same_isin")
                & (global_corr["lead_lag_bucket"] == "all_active")
            ].iloc[0]
            self.assertAlmostEqual(float(same_isin_all["r"]), 1.0)

            comovement = pd.read_csv(out_dir / "member_comovement_series.csv")
            self.assertFalse(comovement.empty)
            self.assertEqual(set(comovement["Member"].astype(str)), {"M1"})
            self.assertIn("prop_target_imbalance", comovement.columns)
            self.assertIn("active_client_imbalance", comovement.columns)


if __name__ == "__main__":
    unittest.main()
