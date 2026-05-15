from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
