from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_save_plotly_figure_disables_legend_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = make_plot_output_dirs(Path(tmpdir) / "plots", use_subdirs=True)
            ensure_plot_dirs(dirs)

            fig = go.Figure(
                data=[
                    go.Scatter(x=[1, 2], y=[1, 4], mode="lines", name="A"),
                    go.Scatter(x=[1, 2], y=[2, 3], mode="lines", name="B"),
                ]
            )
            fig.update_layout(showlegend=True)

            with mock.patch.dict(os.environ, {"DISABLE_PLOT_LEGENDS": "true"}, clear=False):
                html_path, png_path = save_plotly_figure(
                    fig,
                    stem="legend_off",
                    dirs=dirs,
                    write_html=True,
                    write_png=False,
                )

            self.assertFalse(bool(fig.layout.showlegend))
            self.assertIsNotNone(html_path)
            self.assertIsNone(png_path)
