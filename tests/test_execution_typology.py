from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from moimpact.execution_typology import (
    assign_auto_type_labels,
    extract_impact_shape_features,
    extract_schedule_features,
    unpack_float32_path,
)
from scripts.metaorder_execution_typology import main as execution_typology_main


def _pack(values: list[float]) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _period(start: str, end: str) -> list[int]:
    return [int(pd.Timestamp(start).value), int(pd.Timestamp(end).value)]


class TestExecutionTypologyHelpers(unittest.TestCase):
    def test_unpack_float32_path_decodes_bytes(self) -> None:
        blob = np.asarray([0.0, 0.25, 1.0], dtype=np.float32).tobytes()
        out = unpack_float32_path(blob)
        self.assertIsNotNone(out)
        np.testing.assert_allclose(out, np.asarray([0.0, 0.25, 1.0], dtype=np.float32))

    def test_extract_schedule_features_returns_expected_moments(self) -> None:
        out = extract_schedule_features(
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            np.asarray([0.25, 0.5, 0.25], dtype=np.float32),
        )
        self.assertAlmostEqual(out["schedule_front25_share"], 0.25)
        self.assertAlmostEqual(out["schedule_front50_share"], 0.75)
        self.assertAlmostEqual(out["schedule_back25_share"], 0.25)
        self.assertAlmostEqual(out["schedule_center_of_mass"], 0.5)
        self.assertAlmostEqual(out["schedule_hhi"], 0.375)
        self.assertTrue(np.isfinite(out["schedule_twap_l1_distance"]))

    def test_extract_impact_shape_features_returns_expected_ratios(self) -> None:
        out = extract_impact_shape_features(
            impact_end=2.0,
            impact_1m=3.0,
            impact_10m=4.0,
            impact_30m=1.0,
            impact_60m=0.5,
            partial_impact=np.asarray([0.0, 1.0, 5.0], dtype=np.float32),
        )
        self.assertAlmostEqual(out["abs_impact_end"], 2.0)
        self.assertAlmostEqual(out["retention_30_over_end"], 0.5)
        self.assertAlmostEqual(out["retention_60_over_end"], 0.25)
        self.assertAlmostEqual(out["peak_abs_partial_impact"], 5.0)
        self.assertAlmostEqual(out["overshoot_peak_over_end"], 2.5)

    def test_assign_auto_type_labels_is_deterministic(self) -> None:
        summary = pd.DataFrame(
            {
                "Cluster": [0, 1],
                "Participation Rate_median": [0.20, 0.05],
                "Q/V_median": [0.010, 0.001],
                "DurationSeconds_median": [100.0, 900.0],
                "schedule_front25_share_median": [0.70, 0.20],
                "schedule_back25_share_median": [0.10, 0.50],
                "retention_60_over_end_median": [0.80, 0.20],
            }
        )
        out1 = assign_auto_type_labels(summary)
        out2 = assign_auto_type_labels(summary)
        self.assertListEqual(out1["type_code"].tolist(), out2["type_code"].tolist())
        self.assertListEqual(out1["auto_type_label"].tolist(), out2["auto_type_label"].tolist())


class TestExecutionTypologySmoke(unittest.TestCase):
    def _build_group_frame(self, *, group: str) -> pd.DataFrame:
        base = []
        is_client = group == "client"
        offset = 1000 if is_client else 0
        for idx in range(4):
            front_loaded = idx < 2
            start = pd.Timestamp(f"2024-01-0{1 + idx} 10:00:00")
            end = start + pd.Timedelta(minutes=5 + idx if front_loaded else 20 + idx)
            qv = 0.008 + 0.001 * idx if front_loaded else 0.001 + 0.0002 * idx
            eta = 0.18 + 0.01 * idx if front_loaded else 0.04 + 0.005 * idx
            duration = (end - start).total_seconds()
            if front_loaded:
                time_grid = [0.0, 0.1, 0.25, 1.0]
                volume = [0.45, 0.30, 0.15, 0.10]
                partial = [0.0, 0.03, 0.05, 0.06]
                impact_end = 0.04 + 0.005 * idx
                impact_30m = 0.03 + 0.004 * idx
                impact_60m = 0.025 + 0.003 * idx
            else:
                time_grid = [0.0, 0.55, 0.85, 1.0]
                volume = [0.10, 0.20, 0.30, 0.40]
                partial = [0.0, 0.005, 0.008, 0.010]
                impact_end = 0.01 + 0.002 * idx
                impact_30m = 0.004 + 0.001 * idx
                impact_60m = 0.002 + 0.001 * idx
            base.append(
                {
                    "ISIN": "AAA",
                    "Member": 10 + offset + idx,
                    "Client": 100 + offset + idx,
                    "Direction": 1 if idx % 2 == 0 else -1,
                    "Price Change": impact_end,
                    "Daily Vol": 0.08 + 0.01 * idx,
                    "Q": 10000.0 + 500.0 * idx + offset,
                    "Q/V": qv,
                    "Participation Rate": eta,
                    "Vt/V": min(0.20, qv * max(duration, 1.0) / 300.0),
                    "N Child": 4 + idx,
                    "Period": _period(str(start), str(end)),
                    "child_time_norm": _pack(time_grid),
                    "child_volume_fraction": _pack(volume),
                    "partial_impact": _pack(partial),
                    "Impact_1m": impact_end * 1.05,
                    "Impact_10m": impact_end * 1.20,
                    "Impact_30m": impact_30m,
                    "Impact_60m": impact_60m,
                    "Impact": impact_end,
                }
            )
        return pd.DataFrame(base)

    def test_execution_typology_script_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_root = tmp / "out"
            img_root = tmp / "images"
            prop_path = tmp / "prop.parquet"
            client_path = tmp / "client.parquet"
            self._build_group_frame(group="prop").to_parquet(prop_path, index=False)
            self._build_group_frame(group="client").to_parquet(client_path, index=False)

            rc = execution_typology_main(
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
                    "execution_typology_smoke",
                    "--k-min",
                    "2",
                    "--k-max",
                    "2",
                    "--n-jobs",
                    "1",
                    "--chunk-size",
                    "2",
                    "--silhouette-sample-size",
                    "8",
                    "--pca-scatter-sample-size",
                    "8",
                    "--twap-grid-size",
                    "11",
                    "--no-progress",
                ]
            )
            self.assertEqual(rc, 0)

            out_dir = out_root / "execution_typology_smoke"
            img_dir = img_root / "execution_typology_smoke" / "html"
            self.assertTrue((out_dir / "clustered_metaorders.parquet").exists())
            self.assertTrue((out_dir / "cluster_summary.csv").exists())
            self.assertTrue((out_dir / "group_type_shares.csv").exists())
            self.assertTrue((out_dir / "impact_profiles_by_type.csv").exists())
            self.assertTrue((out_dir / "schedule_profiles_by_type.csv").exists())
            self.assertTrue((out_dir / "run_manifest.json").exists())
            self.assertTrue((img_dir / "execution_typology_feature_heatmap.html").exists())
            self.assertTrue((img_dir / "execution_typology_group_shares.html").exists())

            clustered = pd.read_parquet(out_dir / "clustered_metaorders.parquet")
            for col in ["Cluster", "type_code", "auto_type_label", "type_label", "Group"]:
                self.assertIn(col, clustered.columns)


if __name__ == "__main__":
    unittest.main()
