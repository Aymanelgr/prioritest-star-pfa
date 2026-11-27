import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import stats as S  # noqa: E402


class TestStats(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
        self.loc = np.array([10, 500, 20, 300, 50, 40, 80, 100], dtype=float)

    def test_popt_oracle_is_one(self):
        oracle = self.y / self.loc          # defective first, smallest first
        self.assertAlmostEqual(S.popt(oracle, self.y, self.loc), 1.0, places=6)

    def test_popt_worst_is_zero(self):
        worst = -(self.y + 1e-3 * (1 / self.loc))   # clean first, largest first
        self.assertAlmostEqual(S.popt(worst, self.y, self.loc), 0.0, places=6)

    def test_ifa(self):
        self.assertEqual(S.ifa(np.array([.9, .8, .1, .7, 0, 0, 0, 0]), self.y), 0)
        self.assertEqual(S.ifa(np.array([.1, .9, .2, .8, 0, 0, 0, 0]), self.y), 2)

    def test_recall_at_loc_prefix(self):
        s = -np.arange(8.)                      # inspect in index order; cumulative LOC 10, 510, ...
        self.assertAlmostEqual(S.recall_at_loc(s, self.y, self.loc, .2), 1 / 3)   # 510/1100 > 20 %

    def test_bootstrap_contains_point(self):
        rng = np.random.default_rng(0); y = rng.integers(0, 2, 200); s = y + rng.normal(0, 1, 200)
        e, lo, hi = S.bootstrap_ci(s, y, np.ones(200), 'AP', 300)
        self.assertTrue(lo <= e <= hi)

    def test_paired_bootstrap_identical_models(self):
        rng = np.random.default_rng(1); y = rng.integers(0, 2, 100); s = rng.random(100)
        d, lo, hi, p = S.paired_bootstrap(s, s, y, np.ones(100), 'AP', 200)
        self.assertEqual((d, lo, hi), (0.0, 0.0, 0.0)); self.assertEqual(p, 1.0)

    def test_wilcoxon_and_cliffs(self):
        self.assertEqual(S.wilcoxon_seeds([1, 2, 3], [1, 2, 3]), 1.0)
        d, mag = S.cliffs_delta([5, 6, 7], [1, 2, 3]); self.assertEqual((d, mag), (1.0, 'large'))


if __name__ == '__main__':
    unittest.main()


class TestScottKnott(unittest.TestCase):
    def test_groups(self):
        rng = np.random.default_rng(0)
        s = {'a': rng.normal(.9, .01, 200), 'b': rng.normal(.9, .01, 200), 'c': rng.normal(.5, .01, 200)}
        r = S.scott_knott_esd(s)
        self.assertEqual(r['a'], r['b']); self.assertEqual(r['a'], 1); self.assertEqual(r['c'], 2)

    def test_negligible_difference_merged(self):
        rng = np.random.default_rng(1)
        s = {'a': rng.normal(.80, .05, 500), 'b': rng.normal(.795, .05, 500)}
        r = S.scott_knott_esd(s); self.assertEqual(r['a'], r['b'])
