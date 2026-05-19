from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock
import unittest

import moimpact.workflows.crowding.daily as crowding_daily


class TestCrowdingEtaDelegation(unittest.TestCase):
    @mock.patch("moimpact.workflows.crowding.eta.main", autospec=True, return_value=0)
    def test_run_crowding_vs_part_rate_analysis_uses_eta_module(self, eta_main: mock.Mock) -> None:
        with mock.patch.object(crowding_daily, "RUN_CROWDING_VS_PART_RATE", True), redirect_stdout(
            io.StringIO()
        ):
            crowding_daily.run_crowding_vs_part_rate_analysis(
                Path("prop.parquet"),
                Path("client.parquet"),
                Path("out_files/test"),
                Path("images/test"),
                Path("config_ymls/crowding_analysis.yml"),
            )

        eta_main.assert_called_once()
        argv = eta_main.call_args.args[0]
        self.assertGreater(len(argv), 0)
        self.assertEqual(argv[0], "--config-path")
        self.assertIn("--analysis-tag", argv)
        self.assertIn("crowding_vs_part_rate", argv)


if __name__ == "__main__":
    unittest.main()
