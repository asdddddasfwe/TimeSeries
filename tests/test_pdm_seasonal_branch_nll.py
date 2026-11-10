"""Independent branch labels/NLL references, gradient and same-forward checks."""
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from test_oscillator_mixer import config
from scripts import train_timemix_pdm_seasonal_branch_nll as study
from utils.pdm_residual_ma_diagnostics import SelectedValidationCapture
from utils.prediction_autocov_nll import prediction_autocovariance, prediction_gaussian_nll
from utils.seasonal_branch_nll import branch_losses, decompose_target, forward_branches


class SeasonalBranchTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(2021)
        self.cfg = config(seq_len=96, pred_len=96, moving_avg=25, e_layers=1,
                          dropout=.2, down_sampling_layers=3, down_sampling_window=2)
        self.cfg.seasonal_normalize = False

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_target_ma_numpy_reference_and_detach(self):
        y = torch.randn(2, 96, 3, dtype=torch.double, requires_grad=True)
        s, t = decompose_target(y)
        expected = np.empty(y.shape)
        for b in range(2):
            for c in range(3):
                expected[b, :, c] = np.convolve(np.pad(y[b, :, c].detach().numpy(), 12, mode='edge'),
                                               np.ones(25)/25, mode='valid')
        np.testing.assert_allclose(t.numpy(), expected, atol=1e-12)
        torch.testing.assert_close(s+t, y.detach(), atol=1e-12, rtol=1e-12)
        self.assertFalse(s.requires_grad or t.requires_grad)
        constant = torch.full((1, 96, 2), 4.)
        ss, tt = decompose_target(constant)
        self.assertEqual(ss.abs().max().item(), 0.)
        torch.testing.assert_close(tt, constant)

    def test_season_nll_independent_numpy_not_total_nll(self):
        s = torch.randn(2, 9, 2, dtype=torch.double)
        t = torch.randn_like(s) + 3
        y = torch.randn_like(s)
        outputs = dict(prediction=s+t, seasonal=s, trend=t)
        actual = branch_losses(outputs, y, 'nll', kernel=3)
        target_s, target_t = decompose_target(y, 3)
        values = []
        for b in range(2):
            for c in range(2):
                u = s[b, :, c].numpy(); centered = u-u.mean(); h=len(u)
                g = np.array([np.dot(centered[:h-k], centered[k:])/h for k in range(h)])
                sigma = g[np.abs(np.arange(h)[:, None]-np.arange(h)[None, :])] + .01*np.eye(h)
                self.assertGreater(np.linalg.eigvalsh(sigma).min(), .01-1e-12)
                error = target_s[b, :, c].numpy()-u
                values.append((error@np.linalg.solve(sigma, error)+np.linalg.slogdet(sigma)[1]+h*np.log(2*np.pi))/(2*h))
        expected_nll = np.mean(values)
        self.assertAlmostEqual(actual['nll'].item(), expected_nll, places=10)
        self.assertAlmostEqual(actual['loss'].item(), expected_nll+(t-target_t).square().mean().item(), places=10)
        self.assertGreater(abs(actual['nll'].item()-prediction_gaussian_nll(s+t, y).item()), .01)

    def test_full_loss_gradcheck_and_target_detach(self):
        s = torch.randn(1, 7, 2, dtype=torch.double, requires_grad=True)
        t = torch.randn_like(s, requires_grad=True)
        y = torch.randn_like(s, requires_grad=True)
        def objective(a, b):
            return branch_losses(dict(prediction=a+b, seasonal=a, trend=b), y, 'nll', kernel=3)['loss']
        self.assertTrue(torch.autograd.gradcheck(objective, (s, t), atol=1e-5, rtol=1e-4))
        loss = objective(s, t)
        gs, gt, gy = torch.autograd.grad(loss, (s, t, y), allow_unused=True)
        self.assertIsNone(gy)
        self.assertTrue(torch.isfinite(gs).all() and torch.isfinite(gt).all())
        _, target_t = decompose_target(y, 3)
        torch.testing.assert_close(gt, 2*(t-target_t)/t.numel())

    def test_covariance_gradient_retained(self):
        s = torch.randn(1, 8, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn_like(s); target_s, _ = decompose_target(y, 3)
        full = torch.autograd.grad(prediction_gaussian_nll(s, target_s), s)[0]
        cov, _ = prediction_autocovariance(s)
        e=(target_s-s).transpose(1, 2).unsqueeze(-1)
        fixed=.5*torch.linalg.solve_triangular(torch.linalg.cholesky(cov.detach()),e,upper=False).square().mean()
        detached = torch.autograd.grad(fixed, s)[0]
        self.assertGreater((full-detached).abs().max().item(), 1e-5)

    def test_offset_is_penalized_not_centered_away(self):
        y=torch.full((1, 96, 2),3.,dtype=torch.double)
        s=torch.full_like(y, .5, requires_grad=True); t=y.clone()
        parts=branch_losses(dict(prediction=s+t,seasonal=s,trend=t),y,'nll')
        expected=.5*(.5**2/.01+math.log(.01)+math.log(2*math.pi))
        self.assertAlmostEqual(parts['loss'].item(),expected,places=10)
        parts['loss'].backward();self.assertGreater(s.grad.min().item(),0.)
        original=branch_losses(dict(prediction=y,seasonal=torch.zeros_like(y),trend=y),y,'mse')['loss']
        shifted=branch_losses(dict(prediction=y,seasonal=s,trend=y-s),y,'mse')['loss']
        self.assertEqual(original.item(),0.)
        self.assertAlmostEqual(shifted.item(),.5)

    def test_forward_values_rng_normalization_and_capture(self):
        model=study.initialize_pair(self.cfg)['mse'];x=torch.randn(2,96,2)
        initial={k:v.clone() for k,v in model.state_dict().items()}
        for training in (True,False):
            model.train(training);study.seed_all(71);plain=model(x);rng=torch.get_rng_state()
            study.seed_all(71);parts=forward_branches(model,x)
            torch.testing.assert_close(plain,parts['prediction'],rtol=0,atol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            torch.testing.assert_close(parts['seasonal']+parts['trend'],plain,atol=2e-6,rtol=1e-6)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as d,torch.no_grad():
            model.eval()
            with SelectedValidationCapture(model,Path(d),'mse',2) as cap:
                out=forward_branches(model,x)
            report=cap.report(out['prediction'].numpy(),x.numpy())
            self.assertLess(report['reconstruction_max_abs_error'],2e-5)
            raw=np.load(Path(d)/'mse_validation_components.npy')
            mean=np.load(Path(d)/'mse_validation_history_mean.npy')
            np.testing.assert_allclose(out['seasonal'].numpy(),raw[...,:4].sum(-1),atol=2e-6)
            np.testing.assert_allclose(out['trend'].numpy(),raw[...,4:].sum(-1)+mean,atol=2e-6)

    def test_paired_initialization_optimizer_and_all_parameter_gradients(self):
        models=study.initialize_pair(self.cfg);a,b=models.values()
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        policies=[study.build_optimizer(study.parameter_groups(m,k),k).state_dict()['param_groups'] for k,m in models.items()]
        self.assertEqual(*policies)
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        lc=study.loss_configuration(SimpleNamespace(nll_diagonal_floor=.01))
        for arm,m in models.items():
            out=forward_branches(m,x);losses=study.training_losses(out,y,arm,lc)
            expected=losses['trend_mse']+(losses['season_mse'] if arm=='mse' else losses['nll'])
            torch.testing.assert_close(losses['loss'],expected,rtol=0,atol=0)
            losses['loss'].backward()
            for n,p in m.named_parameters():
                self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)

    def test_full_size_parameter_count_unchanged(self):
        cfg=SimpleNamespace(**dict(vars(self.cfg),d_model=32,d_ff=64,ssm_modes=8,enc_in=7,c_out=7,top_k=5))
        self.assertEqual({sum(p.numel() for p in m.parameters()) for m in study.initialize_pair(cfg).values()},{136150})

    def test_invalid_inputs(self):
        y=torch.randn(1,8,1)
        for k in (0,2,-3):
            with self.assertRaises(ValueError):decompose_target(y,k)
        out=dict(prediction=y,seasonal=y,trend=y)
        for weight in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):branch_losses(out,y,'nll',seasonal_weight=weight)
        with self.assertRaises(ValueError):branch_losses(out,y,'unknown')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_loss_and_output_gradients(self):
        s=torch.randn(2,96,7,requires_grad=True);t=torch.randn_like(s,requires_grad=True);y=torch.randn_like(s)
        cpu=branch_losses(dict(prediction=s+t,seasonal=s,trend=t),y,'nll');gc=torch.autograd.grad(cpu['loss'],(s,t))
        a=s.detach().cuda().requires_grad_();b=t.detach().cuda().requires_grad_()
        gpu=branch_losses(dict(prediction=a+b,seasonal=a,trend=b),y.cuda(),'nll');gg=torch.autograd.grad(gpu['loss'],(a,b))
        torch.testing.assert_close(cpu['loss'],gpu['loss'].cpu(),atol=2e-5,rtol=2e-5)
        for x,z in zip(gc,gg):torch.testing.assert_close(x,z.cpu(),atol=2e-5,rtol=2e-4)


if __name__=='__main__':
    unittest.main()
