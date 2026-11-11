"""Tests for the v2 additions: count splitting, depth-offset NB test, spatial gate,
negative-control calibration, EnSDD-style weights, gap pruning, PCE."""
import unittest
import numpy as np

from stbench.simulate import hex_grid, simulate
from stbench.splitting import count_split, estimate_nb_size
from stbench.domain_svg import test_domains, spatial_gate, add_negative_controls, nc_calibrated_threshold
from stbench import ensemble as E


class SplittingTests(unittest.TestCase):
    def test_poisson_split_independent_and_additive(self):
        rng = np.random.default_rng(0)
        X = rng.poisson(5.0, (5000, 3))
        X1, X2 = count_split(X, 0.5, rng, model="poisson")
        self.assertTrue((X1 + X2 == X).all())
        self.assertLess(abs(np.corrcoef(X1[:, 0], X2[:, 0])[0, 1]), 0.05)

    def test_nb_split_independent(self):
        rng = np.random.default_rng(1)
        b, mu = 2.0, 6.0
        X = rng.negative_binomial(b, b / (b + mu), (20000, 2))
        X1, X2 = count_split(X, 0.5, rng, model="nb", size=np.array([b, b]))
        self.assertLess(abs(np.corrcoef(X1[:, 0], X2[:, 0])[0, 1]), 0.03)
        # naive binomial thinning of NB data leaves positive correlation
        P1 = rng.binomial(X, 0.5); P2 = X - P1
        self.assertGreater(np.corrcoef(P1[:, 0], P2[:, 0])[0, 1], 0.1)

    def test_nb_size_estimate(self):
        rng = np.random.default_rng(2)
        X = rng.negative_binomial(3.0, 3.0 / (3.0 + 8.0), (20000, 40))
        self.assertAlmostEqual(float(np.median(estimate_nb_size(X))), 3.0, delta=0.4)


class TestNullCalibration(unittest.TestCase):
    def test_nb_offset_test_null_with_depth_confounded_labels(self):
        rng = np.random.default_rng(3)
        n, G = 2000, 300
        sf = np.exp(rng.normal(0, 0.5, n))
        labels = (sf > np.median(sf)).astype(int)        # "domains" = depth strata (worst case)
        mu = np.exp(rng.uniform(np.log(0.5), np.log(5), G)); b = 2.0
        m = sf[:, None] * mu[None]
        X = rng.negative_binomial(b, b / (b + m))
        hit, Q, _ = test_domains(X, labels, method="nb_offset", fold=1.0)
        self.assertLess(hit.any(1).mean(), 0.05)
        hitw, _, _ = test_domains(X, labels, method="wilcoxon", fold=1.0, det_diff=0.0)
        self.assertGreater(hitw.any(1).mean(), 0.3)       # the log-normalised Wilcoxon rule is fooled by depth

    def test_spatial_gate_rejects_nonspatial(self):
        rng = np.random.default_rng(4)
        coords = hex_grid(30, 30)
        X = rng.poisson(3.0, (len(coords), 200))
        self.assertLess(spatial_gate(X, coords)[0].mean(), 0.03)

    def test_negative_controls_keep_depth(self):
        rng = np.random.default_rng(5)
        sf = np.exp(rng.normal(0, 0.6, 3000))
        X = rng.poisson(sf[:, None] * np.full(50, 4.0))
        aug, n_nc = add_negative_controls(X, 0.2, rng, kind="nb")
        self.assertEqual(aug.shape[1], 60)
        self.assertGreater(np.corrcoef(aug[:, 50:].sum(1), sf)[0, 1], 0.5)

    def test_nc_threshold(self):
        pi_real = np.r_[np.ones(50), np.full(500, 0.55), np.zeros(450)]
        pi_nc = np.r_[np.full(10, 0.55), np.zeros(90)]
        thr, table = nc_calibrated_threshold(pi_real, pi_nc, 0.1)
        self.assertEqual(thr, 0.6)


class EnsembleV2Tests(unittest.TestCase):
    def test_ensdd_weights_favour_consensus_members(self):
        rng = np.random.default_rng(0)
        truth = np.repeat(np.arange(3), 60)
        good = [rng.permutation(3)[truth] for _ in range(4)]
        bad = [rng.integers(0, 3, len(truth)) for _ in range(2)]
        w = E.ensdd_weights(good + bad)
        self.assertAlmostEqual(w.sum(), 1.0, places=6)
        self.assertGreater(w[:4].mean(), w[4:].mean())

    def test_prune_gap(self):
        sc = {"a": {"score": 2.0}, "b": {"score": 1.5}, "c": {"score": -1.0}, "d": {"score": -2.0}}
        self.assertEqual(E.prune_gap(sc), ["a", "b"])
        sc2 = {"a": {"score": 3.0}, "b": {"score": 0.0}, "c": {"score": -1.0}}
        self.assertEqual(E.prune_gap(sc2), ["a", "b"])   # min_keep = 2

    def test_pce_runs_and_returns_confidence(self):
        s = simulate(n_x=20, n_y=18, n_genes=300, n_domain_genes=80, n_gradient_genes=10, n_state_genes=20, seed=1)
        rng = np.random.default_rng(0)
        members = {"a": [s.domains.copy() for _ in range(2)], "b": [s.domains.copy() for _ in range(2)],
                   "noise": [rng.integers(0, 6, len(s.domains)) for _ in range(2)]}
        res = E.pce(members, s.coords, 6)
        self.assertNotIn("noise", res["kept"])
        self.assertEqual(len(res["confidence"]), len(s.domains))


if __name__ == "__main__":
    unittest.main()
