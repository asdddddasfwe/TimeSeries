"""Independent score, second derivatives, mean consistency, and model wiring."""
import unittest

import numpy as np
import torch

from scripts.audit_nonlinear_bayesian_relation import numpy_score
from utils import nonlinear_bayesian_relation as loss_module


class NonlinearRelationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(73)
        self.x = torch.randn(2, 12, 2, dtype=torch.float64, requires_grad=True)
        self.p = torch.randn(2, 8, 2, dtype=torch.float64, requires_grad=True)
        self.y = torch.randn_like(self.p, requires_grad=True)
        self.config = dict(weight=.1, order=4, nodes=8, chunk_size=3)

    def test_independent_numpy_posterior_score_and_finite_difference(self):
        nodes = loss_module.normal_nodes(12, 8).numpy()
        p, y, x = [v.detach().numpy() for v in (self.p, self.y, self.x)]
        ref = numpy_score(p, y, x, nodes)
        post = loss_module.posterior(self.x)
        np.testing.assert_allclose(post[0], ref['mean'], atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(post[1], ref['covariance'], atol=2e-12, rtol=2e-12)
        score, parts = loss_module.relation(self.p, self.y, post, nodes=8, chunk_size=3)
        self.assertAlmostEqual(float(score.detach()), ref['divergence'], places=11)
        for key in ('curvature_compensation', 'scalar_bregman', 'tangent_correction'):
            self.assertAlmostEqual(float(parts[key].detach()), ref[key], places=11)
        grad = torch.autograd.grad(score, self.p)[0].numpy()
        for index in ((0, 0, 0), (0, 4, 1), (1, 7, 1)):
            plus, minus = p.copy(), p.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            derivative = (numpy_score(plus, y, x, nodes)['divergence']
                          - numpy_score(minus, y, x, nodes)['divergence']) / 2e-5
            self.assertAlmostEqual(derivative, grad[index], places=8)

    def test_gradient_and_second_gradient_keep_prediction_tangent(self):
        p = self.p[:1, :6, :1].detach().requires_grad_()
        y = self.y[:1, :6, :1]
        post = loss_module.posterior(self.x[:1, :, :1])
        function = lambda z: loss_module.relation(z, y, post, nodes=2)[0]
        self.assertTrue(torch.autograd.gradcheck(function, (p,)))
        self.assertTrue(torch.autograd.gradgradcheck(function, (p,)))

    def test_population_mean_and_risk_identity_with_overlap(self):
        y0, y1 = self.y.detach(), torch.randn_like(self.y)
        mean = (y0 + 2*y1) / 3
        for arm in loss_module.ARMS:
            p = mean.clone().requires_grad_()
            def objective(z, target):
                return loss_module.objective(z, target, self.x, arm, self.config)['loss']
            risk = (objective(p, y0) + 2*objective(p, y1)) / 3
            self.assertLess(float(torch.autograd.grad(risk, p)[0].abs().max()), 2e-12)
            trial = mean + .3 * torch.randn_like(mean)
            gap = (objective(trial, y0) + 2*objective(trial, y1)) / 3 - risk.detach()
            torch.testing.assert_close(gap, objective(trial, mean), atol=2e-12, rtol=2e-12)

    def test_rng_detach_controls_and_zero_weight(self):
        before = torch.get_rng_state().clone()
        nodes = loss_module.normal_nodes(12, 16, seed=751)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(nodes[:8], -nodes[8:]))
        saved = nodes.clone()
        nodes.zero_()
        self.assertTrue(torch.equal(saved, loss_module.normal_nodes(12, 16, seed=751)))
        result = loss_module.objective(self.p, self.y, self.x, config=self.config)
        g, gy, gx = torch.autograd.grad(result['loss'], (self.p, self.y, self.x), allow_unused=True)
        self.assertIsNone(gx)
        self.assertIsNone(gy)
        self.assertTrue(torch.isfinite(g).all())
        control = loss_module.objective(self.p, self.y, self.x, 'compensation', self.config)
        self.assertAlmostEqual(float((control['loss']-control['mse']).detach()),
                               .1*result['relation_diagnostics']['curvature_compensation'], places=12)
        for arm, config in [('mse', self.config), ('nonlinear', dict(self.config, weight=0.))]:
            out = loss_module.objective(self.p, self.y, self.x, arm, config)
            self.assertTrue(torch.equal(out['loss'], (self.p-self.y.detach()).square().mean()))

    def test_local_convexity_direct_potential_and_linear_limit(self):
        basis = loss_module.fixed_basis(4)
        coef = torch.randn(12, dtype=torch.float64)
        curvature = 4/(3*np.sqrt(3))*(coef[5:].abs()*basis.square().sum(-1)).sum()
        def potential(z):
            v = z[:4]
            residual = z[4]-coef[0]-coef[1:5]@v-coef[5:]@torch.tanh(basis@v)
            return torch.hypot(residual, residual.new_ones(()))-1+.5*curvature*v.square().sum()
        for scale in (.01, 1., 100.):
            p = (scale*torch.randn(5, dtype=torch.float64)).requires_grad_()
            y = scale*torch.randn(5, dtype=torch.float64)
            hessian = torch.autograd.functional.hessian(potential, p)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(hessian).min()), -1e-10)
            expected = potential(y)-potential(p)-torch.autograd.grad(potential(p), p, create_graph=True)[0]@(y-p)
            # Single time relation has parents in reverse sequence order.
            p_seq = torch.cat((p[:4].flip(0), p[4:])).reshape(1, 5, 1)
            y_seq = torch.cat((y[:4].flip(0), y[4:])).reshape(1, 5, 1)
            actual, _ = loss_module.score_from_coefficients(p_seq, y_seq, coef.reshape(1, 1, 1, 12))
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        linear = coef.clone()
        linear[5:] = 0
        value, parts = loss_module.score_from_coefficients(p_seq, y_seq, linear.reshape(1, 1, 1, 12))
        self.assertEqual(float(parts['curvature_compensation'].detach()), 0.)
        self.assertEqual(float(parts['tangent_correction'].detach()), 0.)
        torch.testing.assert_close(value, parts['scalar_bregman'])

    def test_chunking_extremes_and_zero_error(self):
        post = loss_module.posterior(self.x)
        baseline, _ = loss_module.relation(self.p, self.y, post, nodes=8, chunk_size=8)
        baseline_grad = torch.autograd.grad(baseline, self.p)[0]
        for chunk in (1, 3):
            value, _ = loss_module.relation(self.p, self.y, post, nodes=8, chunk_size=chunk)
            torch.testing.assert_close(value, baseline, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(torch.autograd.grad(value, self.p)[0], baseline_grad, atol=1e-12, rtol=1e-12)
        for scale in (0., 1e-5, 1., 1e3):
            x = self.x.detach()*scale
            p = (self.p.detach()*scale).requires_grad_()
            y = self.y.detach()*scale
            post = loss_module.posterior(x)
            value, _ = loss_module.relation(p, y, post, nodes=8)
            self.assertGreaterEqual(float(value.detach()), -1e-10)
            self.assertTrue(torch.isfinite(torch.autograd.grad(value, p)[0]).all())
            zero, _ = loss_module.relation(p, p.detach(), post, nodes=8)
            self.assertEqual(float(zero.detach()), 0.)
            # Analytic zero; reverse-mode accumulation has FP64 roundoff.
            self.assertLess(float(torch.autograd.grad(zero, p)[0].abs().max()), 1e-12)

    def test_validation_and_nondefault_order_prior(self):
        for q in (1, 2, 4):
            post = loss_module.posterior(self.x, q, noise_variance=.7, prior_variance=2.)
            ref = numpy_score(self.p.detach().numpy(), self.y.detach().numpy(),
                              self.x.detach().numpy(), loss_module.normal_nodes(3*q, 8).numpy(),
                              q, noise_variance=.7, prior_variance=2.)
            value, _ = loss_module.relation(self.p, self.y, post, q, nodes=8)
            self.assertAlmostEqual(float(value.detach()), ref['divergence'], places=10)
        for cfg in (dict(weight=-1), dict(nodes=3), dict(order=0), dict(prior_variance=0), dict(chunk_size=0)):
            with self.assertRaises(ValueError):
                loss_module.objective(self.p, self.y, self.x, config=cfg)
        bad = self.p.detach().clone()
        bad[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            loss_module.objective(bad, self.y, self.x)

    def test_full_ma1_backward_and_eval_no_grad(self):
        from test_oscillator_mixer import config
        from models.OscillatorPDMTrendCondition import Model
        cfg = config(seq_len=96, pred_len=96, moving_avg=25, e_layers=1,
                     dropout=.2, down_sampling_layers=3, down_sampling_window=2,
                     enc_in=7, c_out=7, d_model=32, d_ff=64, top_k=5, ssm_modes=8)
        cfg.seasonal_normalize = False
        model = Model(cfg)
        x, y = torch.randn(2, 96, 7), torch.randn(2, 96, 7)
        parameters_before = sum(v.numel() for v in model.parameters())
        self.assertEqual(parameters_before, 136150)
        output = model(x)
        result = loss_module.objective(output, y, x, config=dict(weight=.1))
        result['loss'].backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertEqual(parameters_before, sum(v.numel() for v in model.parameters()))
        model.eval()
        with torch.no_grad():
            validation = loss_module.objective(model(x), y, x, config=dict(weight=.1))
        self.assertTrue(torch.isfinite(validation['loss']))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cpu_cuda_fp32_score_and_gradient(self):
        cpu_p = self.p.detach().float().requires_grad_()
        gpu_p = cpu_p.detach().cuda().requires_grad_()
        for arm in loss_module.ARMS:
            cpu = loss_module.objective(cpu_p, self.y.float(), self.x.float(), arm, self.config)['loss']
            gpu = loss_module.objective(gpu_p, self.y.float().cuda(), self.x.float().cuda(), arm, self.config)['loss']
            torch.testing.assert_close(cpu, gpu.cpu(), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(torch.autograd.grad(cpu, cpu_p)[0],
                                       torch.autograd.grad(gpu, gpu_p)[0].cpu(), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
