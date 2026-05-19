from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest import mock
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


if __name__ == "__main__":
    unittest.main()
