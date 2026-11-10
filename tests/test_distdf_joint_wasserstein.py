"""Independent value/gradient checks for the vendored official DistDF branch."""
import ast
from pathlib import Path
import unittest

import numpy as np
import torch

from utils.distdf_joint_wasserstein import distdf_loss


ROOT = Path(__file__).resolve().parents[1]


def numpy_distance(p, y, h):
    x = np.concatenate((h, p), axis=1).transpose(2, 0, 1)
    z = np.concatenate((h, y), axis=1).transpose(2, 0, 1)
    xm, zm = x.mean(1, keepdims=True), z.mean(1, keepdims=True)
    x, z = x-xm, z-zm
    identity = np.eye(x.shape[-1])[None]
    a = x.transpose(0, 2, 1) @ x / x.shape[1] + .005*identity
    b = z.transpose(0, 2, 1) @ z / z.shape[1] + .005*identity
    mean = ((xm-zm)**2).sum((1, 2))
    cov = np.trace(a, axis1=1, axis2=2) + np.trace(b, axis1=1, axis2=2)
    cov -= 2*np.sqrt(np.maximum((a*b.transpose(0, 2, 1)).sum((1, 2)), 1e-8))
    return np.sqrt(np.maximum(mean+.002*cov, 1e-8)).mean()


def official_function():
    """Execute exact official functions without unrelated optional packages."""
    source = (ROOT/'third_party/DistDF/utils/fft_ot.py').read_text()
    names = {'cal_wasserstein', 'batch_empirical_bures_wasserstein_distance',
             'batch_upper_bound_distance'}
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<official DistDF>', 'exec'), namespace)
    return namespace['cal_wasserstein']


class DistDFTests(unittest.TestCase):
    def test_official_value_gradient_cpu_cuda(self):
        official = official_function()
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for dtype in (torch.float32, torch.float64):
                generator = torch.Generator().manual_seed(192)
                tensors = [torch.randn(5, t, 3, generator=generator, dtype=dtype).to(device)
                           for t in (6, 6, 6)]
                p, y, h = [v.requires_grad_() for v in tensors]
                actual = distdf_loss(p, y, h)
                expected = official(torch.cat((h.detach(), p), 1),
                                    torch.cat((h.detach(), y.detach()), 1),
                                    'wasserstein_empirical_per_dim',
                                    ot_type='upper_bound', normalize=1,
                                    reg_sk=.005, var_weight=.002, mean_weight=1.)
                self.assertTrue(torch.equal(actual, expected))
                ga = torch.autograd.grad(actual, p, retain_graph=True)[0]
                ge = torch.autograd.grad(expected, p)[0]
                self.assertTrue(torch.equal(ga, ge))
                self.assertTrue(torch.isfinite(ga).all())
                actual.backward()
                self.assertIsNone(y.grad)
                self.assertIsNone(h.grad)
                self.assertAlmostEqual(actual.item(), numpy_distance(
                    *[v.detach().cpu().double().numpy() for v in (p, y, h)]),
                    places=5 if dtype == torch.float32 else 10)

    def test_finite_difference_and_singleton(self):
        generator = torch.Generator().manual_seed(193)
        p, y, h = [torch.randn(3, 4, 2, generator=generator, dtype=torch.float64)
                   for _ in range(3)]
        p.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x: distdf_loss(x, y, h),
                                                (p,), eps=1e-6, atol=1e-5))
        singleton = distdf_loss(p[:1], y[:1], h[:1])
        self.assertTrue(torch.isfinite(singleton))
        self.assertTrue(torch.isfinite(torch.autograd.grad(singleton, p)[0]).all())

    def test_batch_permutation_and_rng(self):
        p, y, h = [torch.randn(4, 5, 2) for _ in range(3)]
        rng = torch.get_rng_state().clone()
        value = distdf_loss(p, y, h)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        order = torch.tensor([2, 0, 3, 1])
        self.assertTrue(torch.allclose(value, distdf_loss(p[order], y[order], h[order])))
        # This official covariance upper bound need not vanish for identical inputs.
        same = distdf_loss(y, y, h)
        self.assertGreater(same.item(), 0.)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
