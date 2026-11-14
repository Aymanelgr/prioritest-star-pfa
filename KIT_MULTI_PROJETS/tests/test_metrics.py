"""Tests des métriques d'évaluation (AP, Recall@20 % classes, Recall@20 % LOC)."""
import sys, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from train import stats  # noqa: E402


class TestStats(unittest.TestCase):
    def test_perfect_ranking(self):
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        s = np.linspace(1, 0, 10)
        r = stats(s, y, np.ones(10))
        self.assertEqual(r['AP'], 1.0)
        self.assertEqual(r['ROC_AUC'], 1.0)
        self.assertEqual(r['Recall@20pct_classes'], 1.0)  # 20 % de 10 = 2 classes = les 2 défauts
        self.assertEqual(r['Recall@20pct_LOC'], 1.0)

    def test_loc_budget_is_strict(self):
        y = np.array([1, 0, 1, 0, 0])
        s = np.array([.9, .8, .7, .1, .0])
        loc = np.array([1000, 10, 10, 10, 10])  # la 1re classe dépasse le budget (20 % de 1040 = 208)
        r = stats(s, y, loc)
        self.assertEqual(r['classes_in_budget'], 4)
        self.assertEqual(r['Recall@20pct_LOC'], 0.5)

    def test_zero_loc_is_clamped(self):
        r = stats(np.array([.5, .4]), np.array([1, 0]), np.array([0, 0]))
        self.assertTrue(np.isfinite(r['Recall@20pct_LOC']))

    def test_single_class_auc_none(self):
        self.assertIsNone(stats(np.array([.2, .1, .3]), np.array([0, 0, 1]) * 0 + 1, np.ones(3))['ROC_AUC'])


if __name__ == '__main__':
    unittest.main()
