from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import plotly.graph_objects as go

from moimpact.paper_figure_styles import (
    PAPER_FIGURE_STYLES_ENV,
    PAPER_FIGURE_STYLE_MODE_ENV,
    PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
    apply_plotly_paper_figure_style,
    paper_figure_style,
    plotly_size_from_paper_style,
)


class TestPaperFigureStyles(unittest.TestCase):
    def test_lookup_merges_defaults_and_matches_stem_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "paper_figure_styles.yml"
            cfg_path.write_text(
                """
defaults:
  showlegend: false
  tick_font_size: 24
figures:
  demo_figure:
    width: "1200"
    height: 1000
    label_font_size: 66
    line_width: 5
    margin:
      l: 130
      r: 40
""",
                encoding="utf-8",
            )

            style = paper_figure_style(
                "images/demo/png/demo_figure.png",
                path=cfg_path,
                mode=PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
            )

        self.assertEqual(style["width"], 1200)
        self.assertEqual(style["height"], 1000)
        self.assertEqual(style["tick_font_size"], 24)
        self.assertEqual(style["label_font_size"], 66)
        self.assertEqual(style["line_width"], 5.0)
        self.assertFalse(style["showlegend"])
        self.assertEqual(style["margin"], {"l": 130, "r": 40})
        self.assertEqual(
            plotly_size_from_paper_style(style, default_width=800, default_height=600),
            {"width": 1200, "height": 1000},
        )

    def test_plotly_size_uses_fallbacks_when_style_omits_dimensions(self) -> None:
        self.assertEqual(
            plotly_size_from_paper_style({}, default_width=800, default_height=600),
            {"width": 800, "height": 600},
        )

    def test_invalid_style_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "paper_figure_styles.yml"
            cfg_path.write_text(
                """
figures:
  demo_figure:
    widht: 1200
""",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                paper_figure_style(
                    "demo_figure",
                    path=cfg_path,
                    mode=PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
                )

    def test_global_style_mode_ignores_per_figure_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "paper_figure_styles.yml"
            cfg_path.write_text(
                """
defaults:
  tick_font_size: 24
figures:
  demo_figure:
    label_font_size: 66
""",
                encoding="utf-8",
            )

            with mock.patch.dict(
                "os.environ",
                {PAPER_FIGURE_STYLE_MODE_ENV: "global"},
                clear=False,
            ):
                style = paper_figure_style("demo_figure", path=cfg_path)

        self.assertEqual(style, {})

    def test_invalid_style_mode_is_rejected(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {PAPER_FIGURE_STYLE_MODE_ENV: "figure-nine"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                paper_figure_style("demo_figure")

    def test_unset_style_mode_uses_global_style_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "paper_figure_styles.yml"
            cfg_path.write_text(
                """
figures:
  demo_figure:
    label_font_size: 66
""",
                encoding="utf-8",
            )

            with mock.patch.dict("os.environ", {}, clear=True):
                style = paper_figure_style("demo_figure", path=cfg_path)

        self.assertEqual(style, {})

    def test_line_width_override_preserves_invisible_band_boundary_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "paper_figure_styles.yml"
            cfg_path.write_text(
                """
figures:
  demo_figure:
    line_width: 5
""",
                encoding="utf-8",
            )
            fig = go.Figure(
                data=[
                    go.Scatter(x=[1, 2], y=[3, 4], mode="lines", line=dict(width=0)),
                    go.Scatter(x=[1, 2], y=[4, 5], mode="lines", line=dict(width=1)),
                ]
            )

            with mock.patch.dict(
                "os.environ",
                {
                    PAPER_FIGURE_STYLE_MODE_ENV: PAPER_FIGURE_STYLE_MODE_PER_FIGURE,
                    PAPER_FIGURE_STYLES_ENV: str(cfg_path),
                },
                clear=True,
            ):
                apply_plotly_paper_figure_style(fig, "demo_figure")

        self.assertEqual(fig.data[0].line.width, 0)
        self.assertEqual(fig.data[1].line.width, 5)


if __name__ == "__main__":
    unittest.main()
