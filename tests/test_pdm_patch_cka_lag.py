"""Numerical definitions, gradients, degeneracy and temporal-order semantics."""
import unittest
import numpy as np
import torch
from utils.patch_cka_lag_loss import parts, center, METRIC_KEYS
from scripts.audit_pdm_patch_cka_lag import numpy_relation_parts


class PatchCKALagTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(198)

    def check_formula(self, device):
        p = torch.randn(2, 24, 3, device=device, dtype=torch.float64, requires_grad=True)
        y = torch.randn_like(p, requires_grad=True)
        rng = (torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state()).clone()
        expected = numpy_relation_parts(p.detach().cpu().numpy(), y.detach().cpu().numpy())
        for mode in ('cka', 'patch_lag'):
            value, diag = parts(p, y, objective=mode)
            self.assertAlmostEqual(float(value), expected[mode], places=12)
            for key in METRIC_KEYS:
                self.assertAlmostEqual(diag[key], expected[key], places=12, msg=key)
            gp, gy = torch.autograd.grad(value, (p, y), allow_unused=True)
            self.assertIsNone(gy)
            self.assertTrue(torch.isfinite(gp).all())
            self.assertGreater(float(gp.norm()), 0.)
            # First and last patches only enter one directed endpoint, yet both get gradients.
            self.assertGreater(float(gp[:, :4].norm()), 0.)
            self.assertGreater(float(gp[:, -4:].norm()), 0.)
        after = torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state()
        self.assertTrue(torch.equal(rng, after))
        # A sample/channel cannot influence another sample/channel.
        changed = p.detach().clone(); changed[0, :, 0] += torch.arange(24, device=device)
        _, one = parts(p[1:, :, 1:], y[1:, :, 1:])
        _, two = parts(changed[1:, :, 1:], y[1:, :, 1:])
        self.assertEqual(one, two)

    def test_formula_cpu(self):
        self.check_formula('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_formula_cuda(self):
        self.check_formula('cuda')

    def test_explicit_shared_rkhs_feature_operator(self):
        # RBF Gram eigendecomposition supplies an exact finite feature map shared
        # by both sequences; directly build non-symmetric left-to-right operators.
        p, y = torch.randn(1, 24, 1, dtype=torch.float64), torch.randn(1, 24, 1, dtype=torch.float64)
        _, diag = parts(p, y, objective='patch_lag')
        points = np.concatenate([p.numpy().reshape(6, 4), y.numpy().reshape(6, 4)])
        k = np.exp(-.5 * ((points[:, None]-points[None])**2).sum(-1))
        eigenvalues, vectors = np.linalg.eigh(k)
        phi = vectors * np.sqrt(eigenvalues.clip(0))[None, :]
        a, b = phi[:6], phi[6:]
        for lag in (1, 2):
            n = 6-lag; j = np.eye(n)-np.ones((n, n))/n
            cp, cy = a[:-lag].T @ j @ a[lag:] / n, b[:-lag].T @ j @ b[lag:] / n
            self.assertGreater(np.linalg.norm(cp-cp.T), 1e-3)
            self.assertAlmostEqual(diag[f'lag{lag}_distance'], float(((cp-cy)**2).sum()), places=12)
            self.assertAlmostEqual(diag[f'lag{lag}_prediction_norm2'], float((cp**2).sum()), places=12)
            self.assertAlmostEqual(diag[f'lag{lag}_cross_inner_product'], float((cp*cy).sum()), places=12)

    def test_gradcheck(self):
        p, y = torch.randn(1, 16, 1, dtype=torch.float64, requires_grad=True), torch.randn(1, 16, 1, dtype=torch.float64)
        for mode in ('cka', 'patch_lag'):
            self.assertTrue(torch.autograd.gradcheck(lambda x: parts(x, y, objective=mode)[0], (p,), eps=1e-6, atol=2e-5))

    def test_equal_and_degenerate_inputs(self):
        random = torch.randn(1, 24, 2, dtype=torch.float64)
        for value in (random, torch.ones_like(random)):
            for mode in ('cka', 'patch_lag'):
                p = value.clone().requires_grad_(); loss, diag = parts(p, value, objective=mode)
                self.assertLess(float(loss), 1e-12)
                self.assertTrue(torch.isfinite(torch.autograd.grad(loss, p)[0]).all())
        p = torch.ones_like(random).requires_grad_()
        loss, diag = parts(p, random)
        self.assertAlmostEqual(float(loss), 1., places=12)
        self.assertEqual(diag['cka_prediction_degenerate_fraction'], 1.)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, p)[0]).all())
        loss, diag = parts(random, torch.zeros_like(random))
        self.assertEqual(float(loss), 0.)
        self.assertEqual(diag['cka_target_degenerate_fraction'], 1.)

    def test_matrix_centering_and_cka_scale(self):
        a = torch.randn(2, 6, 4, dtype=torch.float64)
        b = torch.randn_like(a)
        ka, kb = center(a@a.transpose(-1,-2)), center(b@b.transpose(-1,-2))
        self.assertLess(float(ka.mean(-1).abs().max()), 1e-14)
        def align(x, y):
            return (x*y).sum((-1,-2))/(x.norm(dim=(-1,-2))*y.norm(dim=(-1,-2)))
        torch.testing.assert_close(align(ka,kb), align(ka*3,kb*7))

    def test_time_order_and_invariances(self):
        p, y = torch.randn(1, 32, 1, dtype=torch.float64), torch.randn(1, 32, 1, dtype=torch.float64)
        def reorder(z, indices):
            return z.reshape(1,8,4,1)[:,indices].reshape_as(z)
        perm = torch.tensor([4,0,6,2,7,1,5,3]); reverse = torch.arange(7,-1,-1)
        _, original = parts(p,y)
        _, joint = parts(reorder(p,perm),reorder(y,perm))
        _, independent = parts(reorder(p,perm),y)
        self.assertAlmostEqual(original['cka'],joint['cka'],places=12)
        self.assertGreater(abs(original['cka']-independent['cka']),1e-6)
        self.assertGreater(abs(original['patch_lag']-joint['patch_lag']),1e-6)
        _, reversed_both = parts(reorder(p,reverse),reorder(y,reverse))
        self.assertAlmostEqual(original['patch_lag'],reversed_both['patch_lag'],places=12)
        _, reversed_only = parts(reorder(p,reverse),y)
        self.assertGreater(abs(original['patch_lag']-reversed_only['patch_lag']),1e-6)
        # CKA depends on each self-relation: independent translations preserve it.
        _, translated = parts(p+3,y-5)
        self.assertAlmostEqual(original['cka'],translated['cka'],places=12)
        # Shared translation preserves cross kernels and therefore operator distance.
        _, translated = parts(p+3,y+3)
        self.assertAlmostEqual(original['patch_lag'],translated['patch_lag'],places=12)

    def test_invalid_inputs(self):
        p = torch.randn(1,24,1)
        for kwargs in ({'bandwidth':0},{'bandwidth':float('nan')},{'length':5},{'objective':'kl'}):
            with self.assertRaises(ValueError):parts(p,p,**kwargs)
        for bad in (p[:,:3],p.long(),p*float('nan')):
            with self.assertRaises(ValueError):parts(bad,bad)


if __name__ == '__main__':
    unittest.main()
