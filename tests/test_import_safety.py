from __future__ import annotations

import builtins
import importlib
import os
import unittest


class TestImportSafety(unittest.TestCase):
    def test_imports_do_not_patch_print(self) -> None:
        # Ensure Matplotlib uses a headless-friendly backend before importing workflows.
        os.environ.setdefault("MPLBACKEND", "Agg")

        original_print = builtins.print

        importlib.import_module("moimpact.workflows.metaorders.summary")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("moimpact.workflows.metaorders.distributions")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("moimpact.workflows.crowding.daily")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("moimpact.workflows.metaorders.start_event_study")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("moimpact.workflows.metaorders.start_time_distribution")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("moimpact.workflows.crowding.intraday")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("moimpact.workflows.crowding.overlap")
        self.assertIs(builtins.print, original_print)

        importlib.import_module("moimpact.workflows.metaorders.compute")
        self.assertIs(builtins.print, original_print)
