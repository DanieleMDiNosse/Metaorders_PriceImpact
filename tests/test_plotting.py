from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import plotly.graph_objects as go

from moimpact.paper_figure_styles import (
    PAPER_FIGURE_STYLES_ENV,
    PAPER_FIGURE_STYLE_MODE_ENV,
    PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
)
from moimpact.plotting import (
    ensure_plot_dirs,
    make_plot_output_dirs,
    plotly_export_size_kwargs,
    plotly_figure_size_from_config,
    plotly_layout_size_kwargs,
    save_plotly_figure,
)


class TestPlottingHelpers(unittest.TestCase):
    def test_make_plot_output_dirs_default_subdirs(self) -> None:
        dirs = make_plot_output_dirs(Path("images/demo"), use_subdirs=True)
        self.assertEqual(dirs.html_dir, Path("images/demo/html"))
        self.assertEqual(dirs.png_dir, Path("images/demo/png"))

    def test_make_plot_output_dirs_flat(self) -> None:
        dirs = make_plot_output_dirs(Path("images/demo"), use_subdirs=False)
        self.assertEqual(dirs.html_dir, Path("images/demo"))
        self.assertEqual(dirs.png_dir, Path("images/demo"))

    def test_plotly_figure_size_from_config_parses_nullable_dimensions(self) -> None:
        size = plotly_figure_size_from_config(
            {"IMPACT_FIT_FIGURE_WIDTH": "1200", "IMPACT_FIT_FIGURE_HEIGHT": "null"}
        )
        self.assertEqual(size, {"width": 1200})
        self.assertEqual(plotly_figure_size_from_config({}), {})
        self.assertEqual(plotly_layout_size_kwargs(None, default_width=800, default_height=500), {"width": 800, "height": 500})
        self.assertEqual(plotly_layout_size_kwargs(size, default_width=800, default_height=500), {"width": 1200, "height": 500})
        self.assertEqual(plotly_export_size_kwargs(size), {"width": 1200})

    def test_plotly_figure_size_from_config_rejects_invalid_dimensions(self) -> None:
        for bad_value in (0, -1, True, 1200.9, "1200.9", "wide"):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(ValueError):
                    plotly_figure_size_from_config({"IMPACT_FIT_FIGURE_WIDTH": bad_value})

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

    def test_save_plotly_figure_applies_paper_style_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg_path = base / "paper_figure_styles.yml"
            cfg_path.write_text(
                """
figures:
  styled_output:
    width: 777
    height: 555
    tick_font_size: 33
    label_font_size: 44
    line_width: 6
""",
                encoding="utf-8",
            )
            dirs = make_plot_output_dirs(base / "plots", use_subdirs=True)
            ensure_plot_dirs(dirs)
            fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4], mode="lines")])

            with mock.patch.dict(
                os.environ,
                {
                    PAPER_FIGURE_STYLE_MODE_ENV: PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
                    PAPER_FIGURE_STYLES_ENV: str(cfg_path),
                },
                clear=True,
            ):
                html_path, png_path = save_plotly_figure(
                    fig,
                    stem="styled_output",
                    dirs=dirs,
                    write_html=True,
                    write_png=False,
                )

        self.assertIsNotNone(html_path)
        self.assertIsNone(png_path)
        self.assertEqual(fig.layout.width, 777)
        self.assertEqual(fig.layout.height, 555)
        self.assertEqual(fig.layout.xaxis.tickfont.size, 33)
        self.assertEqual(fig.layout.xaxis.title.font.size, 44)
        self.assertEqual(fig.data[0].line.width, 6)

    def test_save_plotly_figure_enables_mathjax_in_html_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = make_plot_output_dirs(Path(tmpdir) / "plots", use_subdirs=True)
            ensure_plot_dirs(dirs)

            fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 4], mode="lines")])
            fig.update_layout(
                title="$x$",
                annotations=[dict(text="$x_{\\min}$", x=1, y=4, showarrow=False)],
            )
            html_path, png_path = save_plotly_figure(
                fig,
                stem="mathjax_on",
                dirs=dirs,
                write_html=True,
                write_png=False,
            )

            self.assertIsNotNone(html_path)
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("MathJax.js", html_text)
            self.assertIn("$x_{\\\\min}$", html_text)
            self.assertEqual(html_path, dirs.html_dir / "mathjax_on.html")
            self.assertIsNone(png_path)
