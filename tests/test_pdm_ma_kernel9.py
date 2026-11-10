"""Independent endpoint-MA invariants for the two fixed kernel policies."""
import unittest
import numpy as np
import torch
from models.OscillatorPDMStatePatchMAKernel9 import KernelPolicyDecomposition


class Kernel9Tests(unittest.TestCase):
    def test_kernel_policies(self):
        self.assertEqual(list(KernelPolicyDecomposition("physical9").kernels.values()), [9, 5, 3, 1])
        self.assertEqual(list(KernelPolicyDecomposition("point9").kernels.values()), [9] * 4)

    def test_numpy_endpoint_convolution(self):
        rng = np.random.RandomState(813)
        for policy in ("physical9", "point9"):
            module = KernelPolicyDecomposition(policy).double()
            for length, k in module.kernels.items():
                a = rng.randn(3, length, 1)
                expected = np.stack([
                    np.convolve(np.pad(z[:, 0], (k // 2, k // 2), mode="edge"),
                                np.ones(k) / k, mode="valid")
                    for z in a
                ])[..., None]
                residual, trend = module(torch.tensor(a))
                np.testing.assert_allclose(trend.numpy(), expected, atol=2e-15, rtol=2e-15)
                np.testing.assert_allclose((trend + residual).numpy(), a, atol=5e-16, rtol=5e-16)

    def test_constant_and_kernel1(self):
        for policy in ("physical9", "point9"):
            module = KernelPolicyDecomposition(policy)
            for length in module.kernels:
                residual, trend = module(torch.full((2, length, 1), 3.))
                self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
                self.assertTrue(torch.equal(trend, torch.full_like(trend, 3.)))
        x = torch.linspace(-1., 1., 12).reshape(1, 12, 1)
        residual, trend = KernelPolicyDecomposition("physical9")(x)
        self.assertTrue(torch.equal(trend, x))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_no_parameters_buffers_or_rng_change(self):
        before = torch.get_rng_state().clone()
        for policy in ("physical9", "point9"):
            m = KernelPolicyDecomposition(policy)
            self.assertEqual(len(m.state_dict()), 0)
            self.assertEqual(len(list(m.parameters())), 0)
            self.assertEqual(len(list(m.buffers())), 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_batch_independence(self):
        x = torch.arange(96., dtype=torch.float64).reshape(1, 96, 1)
        for policy in ("physical9", "point9"):
            module = KernelPolicyDecomposition(policy)
            first = module(x)[1]
            combined = module(torch.cat([x, -x]))[1][:1]
            self.assertTrue(torch.equal(first, combined))

    def test_backward_finite(self):
        for policy in ("physical9", "point9"):
            module = KernelPolicyDecomposition(policy)
            for length in module.kernels:
                x = torch.linspace(-2, 3, length).reshape(1, length, 1).requires_grad_()
                residual, trend = module(x)
                (trend.square().mean() + residual.square().mean()).backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_symmetric_endpoint_impulse(self):
        for policy in ("physical9", "point9"):
            module = KernelPolicyDecomposition(policy)
            x = torch.zeros(1, 12, 1)
            x[:, 0] = 1
            left = module(x)[1]
            right = module(x.flip(1))[1].flip(1)
            self.assertTrue(torch.equal(left, right))

    def test_point9_within_all_observed_lengths(self):
        module = KernelPolicyDecomposition("point9")
        self.assertTrue(all(k <= length for length, k in module.kernels.items()))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            KernelPolicyDecomposition("unknown")
        with self.assertRaises(ValueError):
            KernelPolicyDecomposition("point9")(torch.zeros(2, 13, 1))


if __name__ == "__main__":
    unittest.main()
