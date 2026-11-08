"""Checks for the NumPy engine used by the v2 study.

1. Hand-written gradients match central finite differences (float64).
2. Parameter counts equal arch.parameter_count for every model family.
3. If PyTorch is installed (as in CI), forward values and gradients equal the
   reference implementation in models.py after copying the same weights.
"""
import unittest
import numpy as np
from kan_mlp.arch import MODELS, architecture, parameter_count
from kan_mlp import engine_np as E

try:
    import torch
    from kan_mlp.models import build
except Exception:  # pragma: no cover - torch optional for this file
    torch = None


def _loss(model, x, y, task):
    return E.loss_only(model(x), y, task)


class EngineTests(unittest.TestCase):
    def _fd_check(self, kind, dims, task):
        rng = np.random.default_rng(3)
        m = E.Model(kind, dims, 8, rng, dtype=np.float64)
        for p in m.params:  # move away from scale=1 / bias=0 symmetry
            p += rng.normal(0, 0.05, p.shape)
        x = rng.uniform(-1.2, 1.2, (7, dims[0]))
        y = rng.integers(0, dims[-1], 7) if task == "classification" else rng.normal(size=(7, dims[-1]))
        pred, cache = m.forward(x, keep=True)
        _, g = E.loss_and_grad(pred, y, task)
        grads = m.backward(cache, g)
        eps = 1e-6
        for p, gp in zip(m.params, grads):
            flat, gflat = p.reshape(-1), gp.reshape(-1)
            for k in rng.choice(flat.size, min(12, flat.size), replace=False):
                old = flat[k]
                flat[k] = old + eps; up = _loss(m, x, y, task)
                flat[k] = old - eps; dn = _loss(m, x, y, task)
                flat[k] = old
                self.assertAlmostEqual((up - dn) / (2 * eps), gflat[k], delta=1e-5 + 1e-4 * abs(gflat[k]))

    def test_gradients_finite_difference(self):
        for kind in MODELS:
            self._fd_check(kind, [3, 4, 2], "regression")
            self._fd_check(kind, [3, 5, 4, 3], "classification")

    def test_parameter_counts(self):
        for kind in MODELS:
            for inputs, outputs, depth, budget in [(2, 1, 1, 1024), (8, 10, 2, 4096), (36, 1, 2, 4096), (2, 10, 1, 1024)]:
                dims, p = architecture(kind, inputs, outputs, depth, budget)
                m = E.Model(kind, dims)
                self.assertEqual(m.n_parameters(), p)
                self.assertEqual(p, parameter_count(kind, dims))

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_matches_torch_reference(self):
        rng = np.random.default_rng(5)
        for kind in MODELS:
            dims = [3, 6, 2]
            m = E.Model(kind, dims, 8, rng, dtype=np.float64)
            ref = build(kind, dims).double()
            if kind.startswith("kan-"):
                # models.py creates its grid centres and k*pi frequencies as float32
                # buffers (.double() only widens them); use the identical constants.
                m.basis.centers = ref[0].centers.numpy().copy()
                m.basis.freq = ref[0].frequencies.numpy().copy()
            with torch.no_grad():
                for t, p in zip(ref.parameters(), m.params):
                    t.copy_(torch.from_numpy(p))
            x = rng.uniform(-1.3, 1.3, (9, 3)); y = rng.normal(size=(9, 2))
            pred, cache = m.forward(x, keep=True)
            _, g = E.loss_and_grad(pred, y, "regression")
            grads = m.backward(cache, g)
            tp = ref(torch.from_numpy(x))
            np.testing.assert_allclose(tp.detach().numpy(), pred, rtol=1e-10, atol=1e-10)
            ((tp - torch.from_numpy(y)) ** 2).mean().backward()
            for t, gp in zip(ref.parameters(), grads):
                np.testing.assert_allclose(t.grad.numpy(), gp, rtol=1e-8, atol=1e-10)

    def test_training_reduces_loss_and_restores_best(self):
        from kan_mlp.data import synthetic
        d = synthetic("waves", 1, n=200)
        cfg = dict(terms=8, weight_decay=1e-4, max_epochs=300, check_every=10, lr_patience=3, min_delta=1e-3, patience=12)
        for kind in ["mlp-silu", "kan-rbf"]:
            dims, _ = architecture(kind, 2, 1, 1, 512)
            model, rec = E.fit(kind, dims, d, cfg, 0, 0.01)
            first = rec["history"][0]["val_loss"]
            self.assertLess(rec["val_loss"], first)
            vx, vy, _ = d["val"]
            self.assertAlmostEqual(E.loss_only(model(vx), vy, "regression"), rec["val_loss"], places=6)


if __name__ == "__main__":
    unittest.main()
