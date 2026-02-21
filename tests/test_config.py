from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from moimpact.config import cfg_require, format_path_template, load_yaml_mapping, resolve_repo_path


class TestConfigHelpers(unittest.TestCase):
    def test_format_path_template_rejects_unknown_placeholder(self) -> None:
        with self.assertRaises(KeyError):
            format_path_template("out/{DATASET_NAME}/{MISSING}", {"DATASET_NAME": "ftsemib"})

    def test_resolve_repo_path_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_dir = Path(tmpdir)
            resolved = resolve_repo_path(script_dir, "a/b/c.txt")
            self.assertTrue(resolved.is_absolute())
            self.assertEqual(resolved, (script_dir / "a/b/c.txt").resolve())

    def test_cfg_require_missing_key(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            cfg_require({}, "MISSING", Path("config.yml"))
        self.assertIn("Missing required key", str(ctx.exception))

    def test_load_yaml_mapping_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cfg.yml"
            path.write_text("A: 1\nB: test\n", encoding="utf-8")
            cfg = load_yaml_mapping(path)
            self.assertEqual(cfg["A"], 1)
            self.assertEqual(cfg["B"], "test")

    def test_load_yaml_mapping_requires_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cfg.yml"
            path.write_text("- 1\n- 2\n", encoding="utf-8")
            with self.assertRaises(TypeError):
                load_yaml_mapping(path)

