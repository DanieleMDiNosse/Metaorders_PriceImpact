from __future__ import annotations

import builtins
import importlib
import os
import unittest


class TestImportSafety(unittest.TestCase):
    def test_imports_do_not_patch_print(self) -> None:
        # Ensure Matplotlib uses a headless-friendly backend before importing scripts.
        os.environ.setdefault("MPLBACKEND", "Agg")

        original_print = builtins.print

        importlib.import_module("scripts.metaorder_summary_statistics")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("scripts.metaorder_distributions")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("scripts.crowding_analysis")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("scripts.metaorder_start_event_study")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("scripts.metaorder_computation")
        self.assertIs(builtins.print, original_print)
