from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from moimpact.workflows.paper import style_preview


class TestPaperStylePreview(unittest.TestCase):
    def test_no_data_preview_generates_placeholder_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_paper = root / "source_paper"
            source_paper.mkdir()
            (source_paper / "images" / "old" / "png").mkdir(parents=True)
            (source_paper / "images" / "old" / "png" / "stale.png").write_bytes(b"not copied")
            tex_path = source_paper / "main.tex"
            tex_path.write_text(
                r"""
\documentclass{article}
\usepackage{graphicx}
\begin{document}
\includegraphics[width=0.8\linewidth]{images/demo/png/demo_figure}
\end{document}
""".strip(),
                encoding="utf-8",
            )
            style_config = root / "paper_figure_styles.yml"
            style_config.write_text(
                """
figures:
  demo_figure:
    width: 420
    height: 300
    tick_font_size: 20
    label_font_size: 24
    showlegend: false
""".strip(),
                encoding="utf-8",
            )
            output_root = root / "preview"

            exit_code = style_preview.main(
                [
                    "--paper-tex",
                    str(tex_path),
                    "--style-config",
                    str(style_config),
                    "--output-root",
                    str(output_root),
                    "--no-compile",
                ]
            )

            self.assertEqual(exit_code, 0)
            placeholder = output_root / "paper" / "images" / "demo" / "png" / "demo_figure.png"
            self.assertTrue(placeholder.exists())
            with Image.open(placeholder) as image:
                self.assertEqual(image.size, (420, 300))
            self.assertTrue((output_root / "paper" / "main.tex").exists())
            self.assertFalse((output_root / "paper" / "images" / "old" / "png" / "stale.png").exists())
            manifest = (output_root / "preview_manifest.json").read_text(encoding="utf-8")
            self.assertIn("images/demo/png/demo_figure", manifest)
            self.assertIn('"success": null', manifest)

    def test_requested_figure_can_be_selected_by_basename(self) -> None:
        figures = (
            "images/a/png/first_figure",
            "images/b/png/second_figure",
        )

        selected = style_preview._select_figures(figures, ["second_figure"])

        self.assertEqual(selected, ("images/b/png/second_figure",))

    def test_placeholder_path_rejects_traversal_outside_scratch_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scratch_paper = Path(tmpdir) / "preview" / "paper"
            scratch_paper.mkdir(parents=True)

            with self.assertRaises(ValueError):
                style_preview._placeholder_path(scratch_paper, "../outside", default_format="png")

    def test_overwrite_refuses_non_preview_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_paper = root / "source_paper"
            source_paper.mkdir()
            tex_path = source_paper / "main.tex"
            tex_path.write_text(
                r"\documentclass{article}\usepackage{graphicx}\begin{document}\includegraphics{images/demo}\end{document}",
                encoding="utf-8",
            )
            style_config = root / "paper_figure_styles.yml"
            style_config.write_text("figures: {}\n", encoding="utf-8")
            important_dir = root / "important"
            important_dir.mkdir()
            sentinel = important_dir / "sentinel.txt"
            sentinel.write_text("keep me", encoding="utf-8")

            with self.assertRaises(ValueError):
                style_preview.main(
                    [
                        "--paper-tex",
                        str(tex_path),
                        "--style-config",
                        str(style_config),
                        "--output-root",
                        str(important_dir),
                        "--overwrite",
                        "--no-compile",
                    ]
                )

            self.assertTrue(sentinel.exists())

    def test_output_root_inside_paper_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_paper = Path(tmpdir) / "source_paper"
            source_paper.mkdir()
            output_root = source_paper / "tmp_paper_style_preview_inside"

            with self.assertRaises(ValueError):
                style_preview._validate_output_root(output_root, paper_source_dir=source_paper)

    def test_detected_figures_ignore_inline_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "main.tex"
            tex_path.write_text(
                r"""
\includegraphics{images/active}
Some text % \includegraphics{images/commented}
Escaped percent \% \includegraphics{images/still_active}
""".strip(),
                encoding="utf-8",
            )

            figures = style_preview._paper_figure_paths(tex_path)

        self.assertEqual(figures, ("images/active", "images/still_active"))

    def test_list_figures_does_not_require_style_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "main.tex"
            tex_path.write_text(r"\includegraphics{images/demo}", encoding="utf-8")
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                exit_code = style_preview.main(
                    [
                        "--paper-tex",
                        str(tex_path),
                        "--style-config",
                        str(Path(tmpdir) / "missing.yml"),
                        "--list-figures",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("images/demo", output.getvalue())


if __name__ == "__main__":
    unittest.main()
