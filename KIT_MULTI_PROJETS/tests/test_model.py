"""Tests unitaires du modèle STAR. Lancer depuis KIT_MULTI_PROJETS : python -m unittest discover -s tests -v"""
import sys, unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from model import (CrossViewSTAR, MeanSAGE, cross_view_infonce, label_aware_infonce,  # noqa: E402
                   pairwise_rank_loss, effort_aware_rank_loss)

N, T, FEAT, D = 12, 3, 20, 32


def toy(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, T, FEAT, generator=g)
    y = torch.tensor([1., 0.] * (N // 2))
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [6, 7], [8, 9]])
    loc = torch.arange(1, N + 1, dtype=torch.float32) * 50
    return x, y, edges, loc


CONFIGS = [
    dict(use_graph=False, similarity_gate=False),
    dict(use_graph=True, similarity_gate=False),
    dict(use_graph=True, similarity_gate=True),
    dict(use_graph=True, similarity_gate=True, use_delta=True, bidirectional=True, dropout=.1),
]


class TestShapes(unittest.TestCase):
    def test_outputs_shapes_and_finite(self):
        x, _, e, _ = toy()
        for cfg in CONFIGS:
            with self.subTest(cfg=cfg):
                out = CrossViewSTAR(FEAT, d=D, **cfg).eval()(x, e)
                self.assertEqual(out['logits'].shape, (N,))
                self.assertEqual(out['rank'].shape, (N,))
                self.assertEqual(out['temporal'].shape, (N, D))
                self.assertEqual(out['structural'].shape, (N, D))
                self.assertEqual(out['agreement'].shape, (N,))
                self.assertEqual(out['gate'].shape, (N, 2))
                for k, v in out.items():
                    self.assertTrue(torch.isfinite(v).all(), k)
                self.assertTrue(torch.allclose(out['gate'].sum(1), torch.ones(N), atol=1e-5))

    def test_real_data_shapes(self):
        f = np.load(ROOT / 'data' / 'ant' / 'ant-1.5.npz')
        self.assertEqual(f['X'].ndim, 3)
        self.assertEqual(f['X'].shape[1], 3)
        self.assertEqual(len(f['y']), f['X'].shape[0])
        self.assertTrue(f['edges'].max() < f['X'].shape[0])
        out = CrossViewSTAR(f['X'].shape[2]).eval()(torch.as_tensor(f['X']), torch.as_tensor(f['edges']))
        self.assertTrue(torch.isfinite(out['logits']).all())

    def test_no_edges(self):
        x, _, _, _ = toy()
        out = CrossViewSTAR(FEAT, bidirectional=True)(x, torch.zeros(0, 2, dtype=torch.long))
        self.assertTrue(torch.isfinite(out['logits']).all())


class TestCausality(unittest.TestCase):
    def test_tcn_is_causal(self):
        """Modifier la dernière version ne doit pas changer la sortie TCN aux pas précédents."""
        torch.manual_seed(0)
        for use_delta in (False, True):
            net = CrossViewSTAR(FEAT, use_graph=False, use_delta=use_delta).eval()
            x, _, _, _ = toy()
            x2 = x.clone(); x2[:, -1] += 10.

            def hidden(inp):
                t = net.temporal_input(inp).transpose(1, 2)
                t = torch.relu(net.tcn1(t))[:, :, :T]
                return torch.relu(net.tcn2(t))[:, :, :T]
            h1, h2 = hidden(x), hidden(x2)
            self.assertTrue(torch.allclose(h1[:, :, :-1], h2[:, :, :-1]), f'fuite future (use_delta={use_delta})')
            self.assertFalse(torch.allclose(h1[:, :, -1], h2[:, :, -1]))


class TestGraph(unittest.TestCase):
    def test_isolated_node_unaffected_by_others(self):
        torch.manual_seed(0)
        sage = MeanSAGE(D, bidirectional=True).eval()
        x = torch.randn(4, D); e = torch.tensor([[0, 1], [1, 2]])
        x2 = x.clone(); x2[0] += 5
        self.assertTrue(torch.allclose(sage(x, e)[3], sage(x2, e)[3]))  # nœud 3 isolé


class TestLossesAndGradients(unittest.TestCase):
    def test_losses_finite_and_gradients(self):
        x, y, e, loc = toy()
        net = CrossViewSTAR(FEAT, use_delta=True, bidirectional=True, dropout=.1)
        out = net(x, e)
        losses = {
            'infonce': cross_view_infonce(out['temporal'], out['structural']),
            'label_aware': label_aware_infonce(out['temporal'], out['structural'], y),
            'label_aware_all': label_aware_infonce(out['temporal'], out['structural'], y, defect_only=False),
            'rank': pairwise_rank_loss(out['rank'], y),
            'effort_rank': effort_aware_rank_loss(out['rank'], y, loc),
        }
        for k, v in losses.items():
            self.assertTrue(torch.isfinite(v), k)
        total = torch.nn.functional.binary_cross_entropy_with_logits(out['logits'], y) + sum(losses.values())
        total.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_rank_loss_degenerate(self):
        s = torch.randn(5)
        self.assertEqual(float(pairwise_rank_loss(s, torch.zeros(5))), 0.)
        self.assertEqual(float(effort_aware_rank_loss(s, torch.ones(5), torch.ones(5))), 0.)

    def test_rank_loss_prefers_correct_order(self):
        y = torch.tensor([1., 0.])
        self.assertLess(pairwise_rank_loss(torch.tensor([2., -2.]), y), pairwise_rank_loss(torch.tensor([-2., 2.]), y))

    def test_label_aware_masks_defect_pairs(self):
        """Avec tous les exemples défectueux, chaque ligne n'a plus que son positif -> perte nulle."""
        z = torch.randn(4, D)
        self.assertAlmostEqual(float(label_aware_infonce(z, torch.randn(4, D), torch.ones(4))), 0., places=5)

    def test_infonce_aligned_lower(self):
        z = torch.randn(8, D)
        self.assertLess(cross_view_infonce(z, z), cross_view_infonce(z, torch.randn(8, D)))


if __name__ == '__main__':
    unittest.main()
