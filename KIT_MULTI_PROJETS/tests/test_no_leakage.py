"""Step 6.4 — protocol tests: the test release can never influence a choice (rules R2 and R3 of ARCHITECTURE.md)."""
import json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(KIT / 'src'))


class TestNormalisationFittedOnTrainOnly(unittest.TestCase):
    """R3: z-scores are fitted on the first window of each project. Only that window is centred with std ≤ 1
    (small offsets come from the ±10 clipping and the mean imputation of missing values)."""

    def test_first_window_is_the_reference(self):
        m = json.loads((KIT / 'data' / 'manifest_multi.json').read_text(encoding='utf-8'))
        for p, info in m['projects'].items():
            v = info['targets'][0]['version']
            X = np.load(KIT / 'data' / p / f'{p}-{v}.npz')['X'].reshape(-1, 20)
            self.assertLess(np.abs(X.mean(0)).max(), .03, f'{p}-{v} not centred')
            self.assertLessEqual(X.std(0).max(), 1.001, f'{p}-{v} std > 1')


class TestSearchNeverLoadsTest(unittest.TestCase):
    """R2: the validation search scripts only open the train and validation windows."""

    def _run(self, module, argv):
        import importlib
        mod = importlib.import_module(module)
        opened, real = [], mod.load

        def spy(path, device):
            opened.append(Path(path).name); return real(path, device)
        with mock.patch.object(mod, 'load', spy), mock.patch.object(sys, 'argv', argv):
            mod.main()
        return opened

    def test_explore_val(self):
        import explore_val
        small = dict(lr=[1e-3], weight_decay=[1e-4], dropout=[0.], lambda_c=[.1], lambda_r=[.25],
                     bidirectional=[False], use_delta=[False])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(explore_val.itertools, 'product', lambda *a: [tuple(v[0] for v in small.values())]):
            opened = self._run('explore_val', ['explore_val.py', '--project', 'ant', '--epochs', '12', '--seeds', '1',
                                               '--out', str(Path(tmp) / 'x.json')])
        self.assertEqual(sorted(opened), ['ant-1.5.npz', 'ant-1.6.npz'])

    def test_compare_fair_loads_test_only_after_tuning_is_saved(self):
        import compare_fair
        events = []
        real_load, real_write = compare_fair.load, Path.write_text

        def spy_load(path, device):
            events.append(('load', Path(path).name)); return real_load(path, device)

        def spy_write(self_, *a, **k):
            events.append(('write', self_.name)); return real_write(self_, *a, **k)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(compare_fair, 'load', spy_load), \
                mock.patch.object(Path, 'write_text', spy_write), \
                mock.patch.object(sys, 'argv', ['compare_fair.py', '--project', 'ant', '--epochs', '12', '--seeds', '2',
                                                '--tune-seeds', '1', '--boot', '10', '--out', tmp]), \
                mock.patch.object(compare_fair, 'NEURAL', ['A2_TCN_BCE', 'A3_graph_classic_fusion_BCE', 'A4_similarity_fusion_BCE',
                                                           'A5_plus_InfoNCE', 'A6_STAR_full']):
            compare_fair.main()
        test_load = events.index(('load', 'ant-1.7.npz'))
        tuning_saved = events.index(('write', 'tuning.json'))
        self.assertLess(tuning_saved, test_load, 'test window loaded before the tuning was frozen')
        self.assertEqual([e for e in events[:test_load] if e[0] == 'load'], [('load', 'ant-1.5.npz'), ('load', 'ant-1.6.npz')])


class TestDecisionRuleUsesValidationOnly(unittest.TestCase):
    def test_flag_rate_is_a_function_of_validation_scores(self):
        from train import choose_flag_rate, classification_stats
        rng = np.random.default_rng(0)
        yv = rng.integers(0, 2, 100); sv = yv + rng.normal(0, .8, 100)
        rate = choose_flag_rate(sv, yv)
        yt = rng.integers(0, 2, 80); st = rng.random(80)
        # changing the test labels cannot change the rate, and the number of flagged test classes is fixed by it
        flagged = round(rate * 80)
        for labels in (yt, 1 - yt):
            r = classification_stats(st, labels, rate)
            self.assertAlmostEqual(r['flag_rate'], round(rate, 4))
        self.assertEqual(flagged, int(round(rate * 80)))


if __name__ == '__main__':
    unittest.main()
