"""Independent fixed-increment loss and main-model integration checks."""
import unittest
import numpy as np
import torch
from utils.fixed_increment_bregman import increments, increment_bregman, objective


class FixedIncrementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2021)
        self.p = torch.randn(2, 8, 2, dtype=torch.double, requires_grad=True)
        self.y = torch.randn_like(self.p, requires_grad=True)
        self.lags = (1, 3)

    def test_numpy_value_analytic_gradient_and_detach(self):
        value, _ = increment_bregman(self.p, self.y, lags=self.lags)
        p, y = self.p.detach().numpy(), self.y.detach().numpy()
        rp = np.concatenate([p[:, k:] - p[:, :-k] for k in self.lags], axis=1)
        ry = np.concatenate([y[:, k:] - y[:, :-k] for k in self.lags], axis=1)
        expected = (.5*(ry-rp)**2 + np.logaddexp(ry, -ry)
                    - np.logaddexp(rp, -rp) - np.tanh(rp)*(ry-rp)).mean()
        self.assertAlmostEqual(float(value.detach()), expected, places=12)
        gr = (2-np.tanh(rp)**2)*(rp-ry)/rp.size
        gp = np.zeros_like(p)
        offset = 0
        for k in self.lags:
            n = p.shape[1]-k
            gp[:, k:] += gr[:, offset:offset+n]
            gp[:, :-k] -= gr[:, offset:offset+n]
            offset += n
        actual, gy = torch.autograd.grad(value, (self.p, self.y), allow_unused=True)
        np.testing.assert_allclose(actual, gp, atol=1e-12)
        self.assertIsNone(gy)
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: increment_bregman(z, self.y, lags=self.lags)[0], (self.p,)))

    def test_quadratic_nullspace_and_extremes(self):
        v, _ = increment_bregman(self.p, self.y, beta=0., lags=self.lags)
        torch.testing.assert_close(v, .5*increments(self.p-self.y.detach(), self.lags).square().mean())
        # Relations alone cannot identify a constant offset; total MSE must stay.
        y = self.p.detach()+3
        v, _ = increment_bregman(self.p, y, lags=self.lags)
        self.assertLess(abs(float(v.detach())), 1e-12)
        cfg = dict(weight=.1, beta=1., lags=self.lags)
        self.assertAlmostEqual(float(objective(self.p, y, None, 'bregman', cfg)['loss'].detach()), 9.)
        for scale in (0., 1e-4, 1., 1e3):
            p = (self.p.detach()*scale).requires_grad_()
            v, _ = increment_bregman(p, self.y.detach()*scale, lags=self.lags)
            self.assertGreaterEqual(float(v.detach()), -1e-10)
            self.assertTrue(torch.isfinite(torch.autograd.grad(v, p)[0]).all())

    def test_mean_optimum_and_no_history_dependency(self):
        y0, y1 = self.y.detach(), torch.randn_like(self.y)
        mean = (y0+y1)/2
        p = mean.clone().requires_grad_()
        cfg = dict(weight=.1, beta=1., lags=self.lags)
        x = torch.randn_like(p, requires_grad=True)
        def loss(z, y):
            return objective(z, y, x, 'bregman', cfg)['loss']
        risk = (loss(p, y0)+loss(p, y1))/2
        gp, gx = torch.autograd.grad(risk, (p, x), allow_unused=True)
        self.assertLess(float(gp.abs().max()), 1e-12)
        self.assertIsNone(gx)
        trial = mean+.2*torch.randn_like(mean)
        gap = (loss(trial, y0)+loss(trial, y1))/2-risk.detach()
        torch.testing.assert_close(gap, loss(trial, mean), atol=1e-12, rtol=1e-10)

    def test_main_model_gradients_and_no_auxiliary_checkpoint(self):
        from scripts import train_timemix_pdm_fixed_increment_bregman as study
        from test_oscillator_mixer import config
        from types import SimpleNamespace
        cfg = config(seq_len=96, pred_len=96, moving_avg=25, e_layers=1, dropout=.2,
                     down_sampling_layers=3, down_sampling_window=2)
        cfg.seasonal_normalize = False
        models = study.initialize_pair(cfg)
        lc = study.loss_configuration(SimpleNamespace(relation_weight=.1))
        self.assertNotIn('operator_checkpoint', lc)
        self.assertEqual(lc['auxiliary_parameters'], 0)
        x, y = torch.randn(2, 96, 2), torch.randn(2, 96, 2)
        for arm, model in models.items():
            parts = objective(model(x), y, x, arm, lc)
            parts['loss'].backward()
            self.assertEqual(parts['relation_diagnostics']['pair_count'], 343)
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == '__main__':
    unittest.main()
