from __future__ import annotations

import builtins
import io
import tempfile
import unittest
from pathlib import Path

from moimpact.logging_utils import PrintTee, setup_file_logger


class TestLoggingUtils(unittest.TestCase):
    def test_print_tee_restores_print_and_writes_log(self) -> None:
        original_print = builtins.print
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "demo.log"
            logger = setup_file_logger("demo", log_path, mode="w", reset_handlers=True)

            with PrintTee(logger):
                buffer = io.StringIO()
                print("hello", 1, 2, sep="|", file=buffer)

            self.assertIs(builtins.print, original_print)
            contents = log_path.read_text(encoding="utf-8")
            self.assertIn("hello|1|2", contents)
