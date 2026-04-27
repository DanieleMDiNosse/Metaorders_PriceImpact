from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from moimpact.stats.execution_schedule import (
    infer_execution_schedule_scalar_summaries,
    prepare_execution_schedule_sample,
)


def _pack(values: list[float]) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _period(date_str: str, *, minute_offset: int = 0) -> list[int]:
    start = pd.Timestamp(f"{date_str} 10:{minute_offset:02d}:00")
    end = start + pd.Timedelta(minutes=5)
    return [int(start.value), int(end.value)]


def _build_schedule_frame(
    schedules: list[tuple[str, list[float], list[float]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, (date_str, times, volumes) in enumerate(schedules):
        rows.append(
            {
                "Period": _period(date_str, minute_offset=idx),
                "child_time_norm": _pack(times),
                "child_volume_fraction": _pack(volumes),
            }
        )
    return pd.DataFrame(rows)


class TestExecutionScheduleInferenceHelpers(unittest.TestCase):
    def _front_loaded_frame(self) -> pd.DataFrame:
        return _build_schedule_frame(
            [
                ("2024-01-02", [0.0, 0.10, 0.30, 1.0], [0.40, 0.30, 0.20, 0.10]),
                ("2024-01-02", [0.0, 0.08, 0.25, 1.0], [0.50, 0.25, 0.15, 0.10]),
                ("2024-01-03", [0.0, 0.12, 0.35, 1.0], [0.35, 0.30, 0.20, 0.15]),
                ("2024-01-03", [0.0, 0.05, 0.20, 1.0], [0.55, 0.20, 0.15, 0.10]),
            ]
        )

    def _back_loaded_frame(self) -> pd.DataFrame:
        return _build_schedule_frame(
            [
                ("2024-01-02", [0.0, 0.60, 0.85, 1.0], [0.10, 0.20, 0.30, 0.40]),
                ("2024-01-02", [0.0, 0.55, 0.80, 1.0], [0.10, 0.25, 0.25, 0.40]),
                ("2024-01-03", [0.0, 0.65, 0.88, 1.0], [0.10, 0.15, 0.30, 0.45]),
                ("2024-01-03", [0.0, 0.58, 0.82, 1.0], [0.10, 0.20, 0.25, 0.45]),
            ]
        )

    def test_prepare_sample_skips_invalid_rows(self) -> None:
        proprietary = pd.concat(
            [
                self._front_loaded_frame().iloc[[0]],
                pd.DataFrame(
                    [
                        {
                            "Period": _period("2024-01-04"),
                            "child_time_norm": _pack([0.0, 0.5]),
                            "child_volume_fraction": _pack([0.2]),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        client = self._back_loaded_frame().iloc[[0]].copy()

        prepared = prepare_execution_schedule_sample(
            proprietary,
            client,
            n_time_grid=21,
            n_heatmap_bins_y=31,
        )

        self.assertEqual(prepared.n_input_rows["proprietary"], 2)
        self.assertEqual(prepared.n_valid_metaorders["proprietary"], 1)
        self.assertIn("length_mismatch", prepared.skipped_reasons["proprietary"])
        self.assertEqual(prepared.n_valid_metaorders["client"], 1)
        self.assertEqual(prepared.scalar_values.shape, (2, 6))

    def test_scalar_inference_median_is_reproducible(self) -> None:
        proprietary = self._front_loaded_frame()
        client = self._back_loaded_frame()
        prepared = prepare_execution_schedule_sample(
            proprietary,
            client,
            n_time_grid=31,
            n_heatmap_bins_y=41,
        )

        out1 = infer_execution_schedule_scalar_summaries(
            prepared,
            alpha=0.10,
            n_bootstrap=200,
            random_state=0,
            batch_size=64,
            n_histogram_bins=512,
        ).set_index("metric")
        out2 = infer_execution_schedule_scalar_summaries(
            prepared,
            alpha=0.10,
            n_bootstrap=200,
            random_state=0,
            batch_size=64,
            n_histogram_bins=512,
        ).set_index("metric")

        np.testing.assert_allclose(out1["delta_value"].to_numpy(), out2["delta_value"].to_numpy())
        np.testing.assert_allclose(out1["ci_low"].to_numpy(), out2["ci_low"].to_numpy())
        np.testing.assert_allclose(out1["ci_high"].to_numpy(), out2["ci_high"].to_numpy())
        self.assertEqual(out1.loc["front25_share", "summary_stat"], "median")
        self.assertGreater(out1.loc["front25_share", "delta_value"], 0.15)
        self.assertEqual(int(out1.loc["front25_share", "bootstrap_valid_runs"]), 200)

    def test_scalar_inference_median_is_zero_for_identical_groups(self) -> None:
        proprietary = self._front_loaded_frame()
        client = proprietary.copy(deep=True)
        prepared = prepare_execution_schedule_sample(
            proprietary,
            client,
            n_time_grid=31,
            n_heatmap_bins_y=41,
        )

        out = infer_execution_schedule_scalar_summaries(
            prepared,
            alpha=0.10,
            n_bootstrap=120,
            random_state=0,
            batch_size=64,
            n_histogram_bins=512,
        ).set_index("metric")

        np.testing.assert_allclose(out["delta_value"].to_numpy(), 0.0, atol=1.0e-8)
        self.assertGreaterEqual(float(out.loc["front25_share", "p_value"]), 0.50)
        self.assertEqual(out.loc["front25_share", "summary_stat"], "median")

    def test_scalar_summary_matches_hand_computed_values(self) -> None:
        proprietary = _build_schedule_frame(
            [
                ("2024-01-02", [0.0, 0.20, 1.0], [0.60, 0.20, 0.20]),
                ("2024-01-03", [0.0, 0.20, 1.0], [0.60, 0.20, 0.20]),
            ]
        )
        client = _build_schedule_frame(
            [
                ("2024-01-02", [0.0, 0.80, 1.0], [0.20, 0.30, 0.50]),
                ("2024-01-03", [0.0, 0.80, 1.0], [0.20, 0.30, 0.50]),
            ]
        )
        prepared = prepare_execution_schedule_sample(
            proprietary,
            client,
            n_time_grid=21,
            n_heatmap_bins_y=21,
        )

        out = infer_execution_schedule_scalar_summaries(
            prepared,
            alpha=0.10,
            n_bootstrap=120,
            random_state=0,
            batch_size=64,
        ).set_index("metric")

        self.assertAlmostEqual(out.loc["front25_share", "proprietary_value"], 0.80, places=6)
        self.assertAlmostEqual(out.loc["front25_share", "client_value"], 0.20, places=6)
        self.assertAlmostEqual(out.loc["front25_share", "delta_value"], 0.60, places=6)
        self.assertAlmostEqual(out.loc["front50_share", "proprietary_value"], 0.80, places=6)
        self.assertAlmostEqual(out.loc["front50_share", "client_value"], 0.20, places=6)
        self.assertLess(out.loc["time_to_25", "proprietary_value"], out.loc["time_to_25", "client_value"])
        self.assertLess(out.loc["center_of_mass", "proprietary_value"], out.loc["center_of_mass", "client_value"])
        self.assertEqual(out.loc["front25_share", "summary_stat"], "median")
        self.assertLessEqual(out.loc["front25_share", "p_value"], 1.0)

    def test_requires_at_least_two_clusters(self) -> None:
        proprietary = _build_schedule_frame(
            [("2024-01-02", [0.0, 0.2, 1.0], [0.5, 0.3, 0.2])]
        )
        client = _build_schedule_frame(
            [("2024-01-02", [0.0, 0.8, 1.0], [0.2, 0.3, 0.5])]
        )
        prepared = prepare_execution_schedule_sample(
            proprietary,
            client,
            n_time_grid=21,
            n_heatmap_bins_y=21,
        )

        with self.assertRaisesRegex(ValueError, "At least two bootstrap clusters"):
            infer_execution_schedule_scalar_summaries(prepared, n_bootstrap=20)


class TestExecutionScheduleScriptSmoke(unittest.TestCase):
    def _group_frame(self, *, front_loaded: bool) -> pd.DataFrame:
        schedules: list[tuple[str, list[float], list[float]]] = []
        for idx, date_str in enumerate(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]):
            if front_loaded:
                schedules.append(
                    (
                        date_str,
                        [0.0, 0.10 + 0.01 * idx, 0.30 + 0.01 * idx, 1.0],
                        [0.40, 0.25, 0.20, 0.15],
                    )
                )
            else:
                schedules.append(
                    (
                        date_str,
                        [0.0, 0.55 + 0.01 * idx, 0.80 + 0.01 * idx, 1.0],
                        [0.10, 0.20, 0.25, 0.45],
                    )
                )
        return _build_schedule_frame(schedules)

    def test_script_writes_inference_tables_and_manifest(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_root = tmp / "out"
            img_root = tmp / "images"
            prop_path = tmp / "prop.parquet"
            client_path = tmp / "client.parquet"
            cfg_path = tmp / "metaorder_execution_schedule.yml"

            self._group_frame(front_loaded=True).to_parquet(prop_path, index=False)
            self._group_frame(front_loaded=False).to_parquet(client_path, index=False)
            legacy_out_dir = out_root / "member_metaorder_execution_schedule"
            legacy_out_dir.mkdir(parents=True, exist_ok=True)
            legacy_curve = legacy_out_dir / "execution_schedule_curve_inference_prop_vs_client_median.parquet"
            legacy_curve.write_text("stale", encoding="utf-8")
            cfg_path.write_text(
                textwrap.dedent(
                    f"""
                    TICK_FONT_SIZE: 12
                    LABEL_FONT_SIZE: 12
                    TITLE_FONT_SIZE: 12
                    LEGEND_FONT_SIZE: 10
                    DATASET_NAME: testdataset
                    LEVEL: member
                    MEMBER_NATIONALITY: null
                    OUTPUT_FILE_PATH: {out_root.as_posix()}
                    IMG_OUTPUT_PATH: {img_root.as_posix()}
                    PROPRIETARY_PATH: {prop_path.as_posix()}
                    CLIENT_PATH: {client_path.as_posix()}
                    RUN_EXECUTION_SCHEDULE: true
                    N_TIME_GRID: 21
                    N_HEATMAP_BINS_Y: 31
                    HEATMAP_COLORSCALE: Turbo
                    CURVE_OVERLAY_STAT: mean
                    RUN_EXECUTION_SCHEDULE_INFERENCE: true
                    BOOTSTRAP_RUNS: 30
                    BOOTSTRAP_BATCH_SIZE: 16
                    SCALAR_HISTOGRAM_BINS: 256
                    ALPHA: 0.10
                    RANDOM_STATE: 0
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["METAORDER_EXECUTION_SCHEDULE_CONFIG"] = str(cfg_path)
            env["MPLBACKEND"] = "Agg"
            proc = subprocess.run(
                [sys.executable, "scripts/metaorder_execution_schedule.py"],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                self.fail(f"metaorder_execution_schedule.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            analysis_dir = out_root / "member_metaorder_execution_schedule"
            scalar_inference = analysis_dir / "execution_schedule_scalar_inference_prop_vs_client.parquet"
            manifest_path = analysis_dir / "run_manifest.json"

            self.assertFalse(legacy_curve.exists())
            self.assertTrue(scalar_inference.exists())
            self.assertTrue(manifest_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["run_execution_schedule_inference"])
            self.assertEqual(manifest["scalar_summary_stat"], "median")
            self.assertEqual(
                Path(manifest["output_paths"]["scalar_inference_table"]).resolve(),
                scalar_inference.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
