import unittest

import torch
from torch.distributions import MultivariateNormal, kl_divergence

from utils.gaussian_autocov_loss import (
    DEFAULT_OFFSETS, _autocovariance_matrix, gaussian_autocov_kl_loss as loss,
)


class GaussianAutocovLossTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(2021)
        self.target = torch.randn(2, 96, 3, generator=generator, dtype=torch.float64)
        self.pred = 0.7 * self.target + torch.randn(
            2, 96, 3, generator=generator, dtype=torch.float64) * 0.2

    def covariance_by_lag(self, data, scale):
        centered = data - data.mean(1, keepdim=True)
        h = data.shape[1]
        def covariance(k):
            return (centered[:, k:] * centered[:, :h-k]).sum(1) / h / scale
        return torch.stack([torch.stack([
            covariance(abs(i-j)) for j in DEFAULT_OFFSETS
        ], dim=-1) for i in DEFAULT_OFFSETS], dim=-2) + 0.01 * torch.eye(8)

    def test_gram_equals_biased_lag_matrix_and_is_psd(self):
        centered = self.target - self.target.mean(1, keepdim=True)
        actual = _autocovariance_matrix(centered, DEFAULT_OFFSETS)
        expected = self.covariance_by_lag(self.target, torch.ones(2, 3)) - 0.01 * torch.eye(8)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-10))
        self.assertGreater(torch.linalg.eigvalsh(actual).min().item(), -1e-10)

    def test_matches_torch_gaussian_kl_in_correct_direction(self):
        scale = (self.target-self.target.mean(1, keepdim=True)).square().mean(1) + 0.01
        truth_cov = self.covariance_by_lag(self.target, scale)
        pred_cov = self.covariance_by_lag(self.pred, scale)
        zero = torch.zeros(2, 3, 8, dtype=torch.float64)
        expected = kl_divergence(MultivariateNormal(zero, truth_cov),
                                 MultivariateNormal(zero, pred_cov)) / 8
        self.assertTrue(torch.allclose(loss(self.pred, self.target, reduction='none'), expected,
                                       atol=1e-9, rtol=1e-7))

    def test_identical_is_zero_with_zero_gradient(self):
        pred = self.target.clone().requires_grad_()
        value = loss(pred, self.target); value.backward()
        self.assertLess(value.item(), 1e-12)
        self.assertLess(pred.grad.abs().max().item(), 1e-12)

    def test_amplitude_shrinkage_and_time_permutation_are_detected(self):
        self.assertGreater(loss(0.5*self.target, self.target).item(), 0.1)
        t = torch.arange(96, dtype=torch.float64)
        y = torch.sin(2*torch.pi*t/24)[None, :, None]
        perm = torch.randperm(96, generator=torch.Generator().manual_seed(1))
        self.assertGreater(loss(y[:, perm], y).item(), 0.01)

    def test_offset_and_sign_blindness_requires_mse(self):
        for pred in [self.target+3, -self.target]:
            self.assertLess(loss(pred, self.target).item(), 1e-12)
            self.assertGreater((pred-self.target).square().mean().item(), 0.1)

    def test_flat_and_nearly_flat_inputs_have_finite_gradients(self):
        for pred, target in [(torch.ones_like(self.target), self.target),
                             (self.pred, torch.ones_like(self.target)),
                             (torch.zeros_like(self.target), torch.zeros_like(self.target)),
                             (1+self.pred*1e-6, 1+self.target*1e-6)]:
            pred = pred.clone().requires_grad_()
            value = (pred-target).square().mean() + 0.01*loss(pred, target)
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_gradient_matches_finite_differences(self):
        pred = self.pred[:1, :8, :1].clone().requires_grad_()
        target = self.target[:1, :8, :1]
        self.assertTrue(torch.autograd.gradcheck(
            lambda p: loss(p, target, offsets=(0, 1, 3, 6)), (pred,)))

    def test_targets_detached_and_channels_batches_independent(self):
        pred = self.pred.clone().requires_grad_()
        target = self.target.clone().requires_grad_()
        value = loss(pred, target, reduction='none')
        for b in range(2):
            for c in range(3):
                self.assertTrue(torch.allclose(value[b, c], loss(
                    pred[b:b+1, :, c:c+1], target[b:b+1, :, c:c+1])))
        value.mean().backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_mixed_precision_uses_float32(self):
        for dtype in (torch.float16, torch.bfloat16):
            pred = self.pred.to(dtype).requires_grad_()
            with torch.autocast('cpu', dtype=torch.bfloat16):
                value = loss(pred, self.target.to(dtype))
            value.backward()
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_invalid_inputs_fail_clearly(self):
        for kwargs in [dict(offsets=()), dict(offsets=(0,)), dict(offsets=(0, 96)),
                       dict(offsets=(0, 1, 1)), dict(offsets=(1, 2)),
                       dict(offsets=(0, -1)), dict(jitter=0),
                       dict(variance_floor=float('nan')), dict(reduction='sum')]:
            with self.assertRaises(ValueError):
                loss(self.pred, self.target, **kwargs)
        with self.assertRaises(ValueError):
            loss(self.pred[:, :48], self.target[:, :48])
        with self.assertRaises(ValueError):
            loss(self.pred[..., 0], self.target[..., 0])
        with self.assertRaises(ValueError):
            loss(self.pred.long(), self.target.long())
        with self.assertRaises(ValueError):
            loss(self.pred*float('nan'), self.target)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_full_shape_forward_backward(self):
        generator = torch.Generator().manual_seed(2021)
        target = torch.randn(32, 96, 7, generator=generator).cuda()
        pred = (0.8*target).detach().requires_grad_()
        with torch.autocast('cuda', dtype=torch.float16):
            value = (pred-target).square().mean() + 0.01*loss(pred, target)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(pred.grad).all())


if __name__ == '__main__':
    unittest.main()
