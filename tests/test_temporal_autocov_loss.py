import unittest
import torch
from utils.temporal_autocov_loss import target_scaled_autocov_loss as loss


class TemporalAutocovLossTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(2021)
        self.y = torch.randn(2, 96, 3, generator=g, dtype=torch.float64)

    def test_identical_has_zero_loss_and_gradient(self):
        p = self.y.clone().requires_grad_()
        value = loss(p, self.y)
        value.backward()
        self.assertEqual(value.item(), 0)
        self.assertEqual(p.grad.abs().max().item(), 0)

    def test_detects_amplitude_shrinkage_ordinary_acf_misses(self):
        y = self.y - self.y.mean(1, keepdim=True)
        p = 0.5 * y
        acf = lambda z: (z[:, 1:] * z[:, :-1]).sum(1) / z.square().sum(1)
        self.assertTrue(torch.allclose(acf(p), acf(y)))
        self.assertGreater(loss(p, y).item(), 0)

    def test_mse_needed_for_offset_and_sign(self):
        for p in [self.y + 3, -self.y]:
            self.assertLess(loss(p, self.y).item(), 1e-20)
            self.assertGreater((p - self.y).square().mean().item(), 0)

    def test_flat_and_nearly_flat_are_finite(self):
        for truth in [torch.ones_like(self.y), 1 + self.y * 1e-6, self.y]:
            p = torch.ones_like(truth, requires_grad=True)
            value = (p - truth).square().mean() + 0.01 * loss(p, truth)
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_prediction_gradient_matches_finite_differences(self):
        p = (self.y[:, :8, :1] * 0.9 + 0.01).clone().requires_grad_()
        truth = self.y[:, :8, :1]
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: loss(z, truth, lags=(1, 2, 4)), (p,)))

    def test_target_is_detached_and_batch_channel_independent(self):
        p = (self.y * 0.7).clone().requires_grad_()
        truth = self.y.clone().requires_grad_()
        value = loss(p, truth)
        expected = torch.stack([loss(p[b:b+1, :, c:c+1], truth[b:b+1, :, c:c+1])
                                for b in range(2) for c in range(3)]).mean()
        self.assertTrue(torch.allclose(value, expected))
        value.backward()
        self.assertIsNone(truth.grad)

    def test_half_precision_accumulates_in_float32(self):
        p = (self.y * 0.9).half().requires_grad_()
        value = loss(p, self.y.half()); value.backward()
        self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_invalid_lags_and_floor_are_rejected(self):
        for kwargs in [dict(lags=()), dict(lags=(96,)), dict(lags=(1, 1)),
                       dict(variance_floor=0), dict(variance_floor=float('nan'))]:
            with self.assertRaises(ValueError):
                loss(self.y, self.y, **kwargs)


if __name__ == '__main__':
    unittest.main()
