from __future__ import annotations

import math
import unittest

from moimpact.stats.correlation import (
    corr_with_ci,
    corr_with_cluster_bootstrap_ci_and_permutation_p,
)


class TestCorrelationHelpers(unittest.TestCase):
    def test_corr_with_ci_reproducible_with_seed(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 2.0, 1.0, 2.0]
        out1 = corr_with_ci(x, y, alpha=0.05, n_bootstrap=200, random_state=0)
        out2 = corr_with_ci(x, y, alpha=0.05, n_bootstrap=200, random_state=0)
        self.assertEqual(out1, out2)

    def test_cluster_corr_reproducible_with_seed(self) -> None:
        x = [1, -1, 1, 1, -1]
        y = [0.2, 0.2, -0.1, -0.1, -0.1]
        d = ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"]
        out1 = corr_with_cluster_bootstrap_ci_and_permutation_p(x, y, d, n_bootstrap=200, random_state=0)
        out2 = corr_with_cluster_bootstrap_ci_and_permutation_p(x, y, d, n_bootstrap=200, random_state=0)
        self.assertEqual(out1, out2)

    def test_cluster_permutation_invalid_when_y_varies_within_cluster(self) -> None:
        x = [1, -1, 1, -1]
        y = [0.0, 1.0, 0.2, 0.2]  # first cluster varies (0,1)
        cluster = ["d1", "d1", "d2", "d2"]
        r, lo, hi, p, n_obs, n_clusters = corr_with_cluster_bootstrap_ci_and_permutation_p(
            x,
            y,
            cluster,
            n_bootstrap=50,
            n_permutations=50,
            y_const_tol=0.0,
            random_state=0,
        )
        self.assertTrue(math.isnan(p))
        self.assertEqual(n_obs, 4)
        self.assertEqual(n_clusters, 2)

