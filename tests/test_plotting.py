from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import plotly.graph_objects as go

from moimpact.plotting import ensure_plot_dirs, make_plot_output_dirs, save_plotly_figure


class TestPlottingHelpers(unittest.TestCase):
    def test_make_plot_output_dirs_default_subdirs(self) -> None:
        dirs = make_plot_output_dirs(Path("images/demo"), use_subdirs=True)
        self.assertEqual(dirs.html_dir, Path("images/demo/html"))
        self.assertEqual(dirs.png_dir, Path("images/demo/png"))

    def test_make_plot_output_dirs_flat(self) -> None:
        dirs = make_plot_output_dirs(Path("images/demo"), use_subdirs=False)
        self.assertEqual(dirs.html_dir, Path("images/demo"))
        self.assertEqual(dirs.png_dir, Path("images/demo"))

    def test_save_plotly_figure_html_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = make_plot_output_dirs(Path(tmpdir) / "plots", use_subdirs=True)
            ensure_plot_dirs(dirs)

            fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 4], mode="lines")])
            html_path, png_path = save_plotly_figure(
                fig,
                stem="demo_plot",
                dirs=dirs,
                write_html=True,
                write_png=False,
            )

            self.assertIsNotNone(html_path)
            self.assertTrue((dirs.html_dir / "demo_plot.html").exists())
            self.assertIsNone(png_path)
