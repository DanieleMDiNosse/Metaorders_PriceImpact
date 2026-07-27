from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest import mock
import io
import os
import unittest

import moimpact.workflows.paper.figures as paper_figures
from moimpact.workflows.paper.figures import _parse_args


class TestPaperFiguresRunnerStyleArgs(unittest.TestCase):
    def test_style_mode_and_paper_style_config_are_cli_options(self) -> None:
        args = _parse_args(
            [
                "--style-mode",
                "global",
                "--paper-style-config",
                "config_ymls/paper_figure_styles_large.yml",
                "--dry-run",
            ]
        )

        self.assertEqual(args.style_mode, "global")
        self.assertEqual(args.paper_style_config, "config_ymls/paper_figure_styles_large.yml")


class TestPaperFigurePathDiscovery(unittest.TestCase):
    def test_ignores_commented_and_iffalse_figures(self) -> None:
        tex = """\
\\includegraphics{images/active_before}
% \\includegraphics{images/commented}
\\iffalse
\\includegraphics{images/disabled}
\\iffalse
\\includegraphics{images/nested_disabled}
\\fi
\\fi
\\includegraphics{images/active_after} % trailing comment
"""

        with mock.patch.object(Path, "read_text", return_value=tex):
            figures = paper_figures._paper_figure_paths(Path("paper/main.tex"))

        self.assertEqual(figures, ("images/active_before", "images/active_after"))


class TestPaperFiguresCrowdingConfig(unittest.TestCase):
    def test_crowding_analysis_disables_nested_eta(self) -> None:
        captured: dict[str, object] = {}

        @contextmanager
        def fake_temporary_yaml_copy(cfg_path: Path, updates):
            captured["cfg_path"] = cfg_path
            captured["updates"] = dict(updates)
            yield Path("/tmp/fake-crowding-analysis.yml")

        with mock.patch.object(paper_figures, "_temporary_yaml_copy", fake_temporary_yaml_copy), mock.patch.object(
            paper_figures, "_run_logged_command"
        ) as run_logged_command:
            paper_figures._run_crowding_analysis(
                dataset_name="ftsemib",
                img_output_root=Path("/tmp/paper-images"),
                log_dir=Path("/tmp/paper-logs"),
                style_updates={},
                dry_run=True,
            )

        self.assertIn("updates", captured)
        self.assertIs(captured["cfg_path"], paper_figures.CROWDING_CFG)
        self.assertFalse(captured["updates"]["RUN_CROWDING_VS_PART_RATE"])
        run_logged_command.assert_called_once()


class TestPaperFiguresExecutionScheduleConfig(unittest.TestCase):
    def test_execution_schedule_forces_median_overlay_to_match_paper_stem(self) -> None:
        captured: dict[str, object] = {}

        @contextmanager
        def fake_temporary_yaml_copy(cfg_path: Path, updates):
            captured["cfg_path"] = cfg_path
            captured["updates"] = dict(updates)
            yield Path("/tmp/fake-execution-schedule.yml")

        with mock.patch.object(paper_figures, "_temporary_yaml_copy", fake_temporary_yaml_copy), mock.patch.object(
            paper_figures, "_run_logged_command"
        ) as run_logged_command:
            paper_figures._run_metaorder_execution_schedule(
                dataset_name="ftsemib",
                img_output_root=Path("/tmp/paper-images"),
                log_dir=Path("/tmp/paper-logs"),
                style_updates={},
                dry_run=True,
            )

        self.assertIn("updates", captured)
        self.assertIs(captured["cfg_path"], paper_figures.METAORDER_EXECUTION_SCHEDULE_CFG)
        self.assertEqual(captured["updates"]["CURVE_OVERLAY_STAT"], "median")
        run_logged_command.assert_called_once()


class TestPaperFiguresSummaryCompatibilityCopy(unittest.TestCase):
    def test_summary_compatibility_copy_includes_pdf_sidecar(self) -> None:
        with mock.patch.object(paper_figures, "_copy_if_exists") as copy_if_exists:
            paper_figures._summary_compatibility_copy(Path("/tmp/paper-images"), dry_run=False)

        expected = [
            mock.call(
                Path("/tmp/paper-images/member_metaorder_summary_statistics/png/mean_daily_metaorder_volume_share.png"),
                Path("/tmp/paper-images/prop_vs_nonprop/png/mean_daily_metaorder_volume_share.png"),
                dry_run=False,
            ),
            mock.call(
                Path("/tmp/paper-images/member_metaorder_summary_statistics/html/mean_daily_metaorder_volume_share.html"),
                Path("/tmp/paper-images/prop_vs_nonprop/html/mean_daily_metaorder_volume_share.html"),
                dry_run=False,
            ),
            mock.call(
                Path("/tmp/paper-images/member_metaorder_summary_statistics/png/mean_daily_metaorder_volume_share.pdf"),
                Path("/tmp/paper-images/prop_vs_nonprop/png/mean_daily_metaorder_volume_share.pdf"),
                dry_run=False,
            ),
        ]
        copy_if_exists.assert_has_calls(expected)


class TestPaperFiguresConsoleOutput(unittest.TestCase):
    def test_run_overview_is_sectioned_and_not_raw_json_only(self) -> None:
        manifest = {
            "dataset_name": "ftsemib",
            "img_output_root": "/tmp/paper/images",
            "targets": ["execution_schedule", "crowding"],
            "selected_figures": [
                "images/member_metaorder_execution_schedule/png/execution_schedule_heatmap_prop_vs_client_median",
                "images/crowding_impact/png/main_crowding_impact_curves.png",
            ],
            "tasks": ["crowding_impact", "execution_schedule"],
            "stage_all": "FILTERED",
            "stage_it": "NONE",
            "max_workers": 1,
            "write_pdf": True,
            "style_mode": "per-figure",
            "paper_style_config": "/tmp/paper_figure_styles.yml",
            "dry_run": True,
        }
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            paper_figures._print_run_overview(manifest, Path("/tmp/logs/run_manifest.json"))

        output = buffer.getvalue()
        self.assertIn("Paper figure run", output)
        self.assertIn("Mode", output)
        self.assertIn("dry-run", output)
        self.assertIn("Dataset", output)
        self.assertIn("ftsemib", output)
        self.assertIn("Selected figures", output)
        self.assertNotEqual(output.lstrip()[:1], "{")

    def test_selected_figures_are_grouped_by_task(self) -> None:
        figures = [
            "images/member_metaorder_execution_schedule/png/execution_schedule_heatmap_prop_vs_client_median",
            "images/crowding_impact/png/main_crowding_impact_curves.png",
        ]
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            paper_figures._print_selected_figures_grouped(figures)

        output = buffer.getvalue()
        self.assertIn("execution_schedule", output)
        self.assertIn("crowding_impact", output)
        self.assertIn("execution_schedule_heatmap_prop_vs_client_median", output)
        self.assertIn("main_crowding_impact_curves.png", output)

    def test_print_titles_use_color_when_forced(self) -> None:
        with mock.patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=True):
            title = paper_figures._format_print_title("Paper figure run")

        self.assertEqual(title, "\033[1;36mPaper figure run\033[0m")

    def test_print_titles_remain_plain_without_tty(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            title = paper_figures._format_print_title("Paper figure run")

        self.assertEqual(title, "Paper figure run")


class TestPaperFiguresOutputVerification(unittest.TestCase):
    def test_missing_selected_figure_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Missing generated paper figure"):
            paper_figures._assert_selected_figures_exist(
                ["images/prop_vs_nonprop/png/mean_daily_metaorder_volume_share"],
                paper_dir=Path("/tmp/definitely-missing-paper-dir"),
            )


if __name__ == "__main__":
    unittest.main()
