import unittest
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

from stbench.simulate import simulate, perturb, hex_grid
from stbench import methods as M, ensemble as E, svg_aggregate as A


class SimulatorTests(unittest.TestCase):
    def test_shapes_and_truth(self):
        s = simulate(n_x=20, n_y=18, n_genes=300, n_domain_genes=40, n_gradient_genes=20, n_state_genes=30, seed=0)
        self.assertEqual(s.counts.shape, (360, 300))
        self.assertEqual(int(s.is_svg.sum()), 60)
        self.assertTrue(((s.gene_domain >= 0) == (s.gene_class == "domain")).all())

    def test_perturb_thinning_and_subsampling(self):
        s = simulate(n_x=20, n_y=18, n_genes=200, n_domain_genes=20, n_gradient_genes=10, n_state_genes=20, seed=0)
        p, idx = perturb(s, np.random.default_rng(0), depth=0.5, spot_frac=0.5)
        self.assertEqual(len(idx), 180)
        self.assertLess(p.counts.sum(), s.counts[idx].sum())


class TestCalibration(unittest.TestCase):
    def test_null_tests_are_calibrated(self):
        rng = np.random.default_rng(1)
        coords = hex_grid(30, 30)
        x = np.log1p(rng.poisson(2.0, (len(coords), 400)))  # no spatial structure
        W = M.knn_weights(coords)
        _, p1 = M.morans_i(x, W); _, p2 = M.gearys_c(x, W); p3 = M.sparkx_like(x, coords)
        for p in (p1, p2, p3):
            self.assertLess(abs((p < 0.05).mean() - 0.05), 0.03)

    def test_acat_bounds(self):
        p = np.array([[0.01, 0.5], [0.02, 0.6], [0.5, 0.9]])
        out = M.acat(p, axis=0)
        self.assertTrue(out[0] < 0.05 and out[1] > 0.3)


class EnsembleTests(unittest.TestCase):
    def test_align_and_vote_recover_permuted_labels(self):
        rng = np.random.default_rng(0)
        truth = rng.integers(0, 4, 200)
        perms = [rng.permutation(4)[truth] for _ in range(5)]
        lab, conf = E.majority_vote(perms, 4)
        self.assertEqual(ari(truth, lab), 1.0); self.assertTrue((conf == 1).all())

    def test_coassociation_consensus(self):
        rng = np.random.default_rng(0)
        truth = np.repeat(np.arange(3), 50)
        noisy = [np.where(rng.random(150) < 0.15, rng.integers(0, 3, 150), rng.permutation(3)[truth]) for _ in range(9)]
        C = E.coassociation(noisy)
        self.assertTrue(np.allclose(C, C.T)); self.assertTrue(np.allclose(np.diag(C), 1))
        lab, soft, conf = E.consensus_from_coassociation(C, 3)
        self.assertGreater(ari(truth, lab), 0.9)
        self.assertTrue(np.allclose(soft.sum(1), 1))

    def test_pruning_drops_noise_method(self):
        coords = hex_grid(15, 14)
        truth = (coords[:, 1] > np.median(coords[:, 1])).astype(int)
        rng = np.random.default_rng(0)
        members = {"good_a": [truth.copy() for _ in range(3)], "good_b": [truth.copy() for _ in range(3)],
                   "good_c": [truth.copy() for _ in range(3)], "noise": [rng.integers(0, 2, len(truth)) for _ in range(3)]}
        sc = E.method_scores(members, coords)
        self.assertNotIn("noise", E.prune(sc, mad_k=1.0))


class UQTests(unittest.TestCase):
    def test_stability_selection_bound(self):
        calls = np.zeros((10, 100), bool); calls[:, :5] = True; calls[::2, 50] = True
        pi, sel, bound = A.stability_selection(calls, 0.6)
        self.assertEqual(sel.sum(), 5); self.assertAlmostEqual(pi[50], 0.5)
        self.assertAlmostEqual(bound, (5.5 ** 2) / (0.2 * 100))

    def test_agreement_score(self):
        self.assertEqual(A.agreement(np.array([[1, 1, 0], [1, 1, 0]], bool)), 1.0)
        self.assertEqual(A.agreement(np.array([[1, 0], [0, 1]], bool)), 0.0)


if __name__ == "__main__":
    unittest.main()
