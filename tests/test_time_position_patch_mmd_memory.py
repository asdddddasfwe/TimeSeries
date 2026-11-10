"""Numerical and backward regression for memory bounded full MMD."""
import unittest
import torch
from utils.time_position_patch_mmd_memory import parts


def dense_reference(prediction, target, sigma_value=2.0, sigma_time=24.0):
    p = prediction.transpose(1, 2).unfold(-1, 4, 4).double()
    y = target.detach().transpose(1, 2).unfold(-1, 4, 4).double()
    n = p.shape[-2]
    index = torch.arange(n, device=p.device, dtype=torch.float64)
    time = torch.exp(-0.5 * ((index[:, None] - index[None, :]) / sigma_time).square())

    def term(a, b):
        return (time * torch.exp(-0.5 * ((a.unsqueeze(-2) - b.unsqueeze(-3))
                                       / sigma_value).square().sum(-1))).mean((-1, -2))

    return (term(p, p) + term(y, y) - 2 * term(p, y)).clamp_min(0).mean().to(prediction.dtype)


class MemoryMMDTests(unittest.TestCase):
    def test_values_gradients_rng_and_target_detachment(self):
        torch.manual_seed(2021)
        for dtype in (torch.float32, torch.float64):
            source = torch.randn(7, 3, 40, dtype=dtype)
            y = torch.randn(7, 40, 3, dtype=dtype, requires_grad=True)
            x = source.transpose(1, 2).detach().requires_grad_()
            old = dense_reference(x, y)
            old_grad = torch.autograd.grad(old, x)[0]
            for tile in (1, 3, 4, 20):
                p = source.transpose(1, 2).detach().requires_grad_()
                before = torch.get_rng_state().clone()
                new, _ = parts(p, y, batch_tile_size=tile)
                new_grad = torch.autograd.grad(new, p)[0]
                self.assertTrue(torch.equal(old, new))
                torch.testing.assert_close(old_grad, new_grad, rtol=0, atol=0)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertIsNone(y.grad)

    def test_identity_no_grad_and_gradcheck(self):
        p = torch.randn(2, 12, 2, dtype=torch.float64, requires_grad=True)
        value, _ = parts(p, p)
        self.assertEqual(float(value.detach()), 0.0)
        self.assertTrue(torch.equal(torch.autograd.grad(value, p)[0], torch.zeros_like(p)))
        y = torch.randn_like(p)
        self.assertTrue(torch.autograd.gradcheck(lambda x: parts(x, y)[0], (p,)))
        with torch.no_grad():
            self.assertTrue(torch.equal(parts(p, y)[0], dense_reference(p, y)))

    def test_invalid_arguments(self):
        p = torch.zeros(2, 16, 2)
        for tile in (0, True, 1.5):
            with self.assertRaises(ValueError):
                parts(p, p, batch_tile_size=tile)
        with self.assertRaises(ValueError):
            parts(p, p, sigma_time=0)
        with self.assertRaises(ValueError):
            parts(p[:, :-1], p[:, :-1])


if __name__ == '__main__':
    unittest.main()
