"""Independent references for a target-window covariance score on total output."""
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from test_oscillator_mixer import config
from utils.target_autocov_nll import target_autocovariance, target_gaussian_nll
from scripts import train_timemix_pdm_target_nll as study
from utils.pdm_residual_ma_diagnostics import SelectedValidationCapture


class TargetNLLTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(2021)
        self.cfg = config(seq_len=96, pred_len=96, moving_avg=25, e_layers=1,
                          dropout=.2, down_sampling_layers=3, down_sampling_window=2)
        self.cfg.seasonal_normalize = False

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_numpy_covariance_joint_and_conditional(self):
        p = torch.randn(2, 9, 3, dtype=torch.double)
        y = torch.randn_like(p)
        cov, gamma = target_autocovariance(y)
        values, conditional = [], []
        for b in range(2):
            for c in range(3):
                u = y[b, :, c].numpy()
                u = u-u.mean()
                h = len(u)
                g = np.array([u[:h-k] @ u[k:]/h for k in range(h)])
                sigma = g[abs(np.arange(h)[:, None]-np.arange(h)[None, :])]+.01*np.eye(h)
                np.testing.assert_allclose(gamma[b,c].numpy(), g, atol=1e-12)
                np.testing.assert_allclose(cov[b,c].numpy(), sigma, atol=1e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(sigma).min(), .01-1e-12)
                e = (y-p)[b, :, c].numpy()
                values.append((e @ np.linalg.solve(sigma,e)+np.linalg.slogdet(sigma)[1]+h*np.log(2*np.pi))/(2*h))
                terms = []
                for i in range(h):
                    beta = np.linalg.solve(sigma[:i,:i],sigma[:i,i]) if i else np.zeros(0)
                    d = sigma[i,i]-sigma[i,:i] @ beta
                    innovation = e[i]-beta @ e[:i]
                    terms.append(.5*(innovation**2/d+np.log(d)+np.log(2*np.pi)))
                conditional.append(sum(terms)/h)
        loss = float(target_gaussian_nll(p,y))
        self.assertAlmostEqual(loss, float(np.mean(values)), places=11)
        self.assertAlmostEqual(loss, float(np.mean(conditional)), places=11)

    def test_gradcheck_detach_and_exact_mean_gradient(self):
        p = torch.randn(1,6,2,dtype=torch.double,requires_grad=True)
        y = torch.randn_like(p,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:target_gaussian_nll(z,y),(p,)))
        loss = target_gaussian_nll(p,y)
        grad, target_grad = torch.autograd.grad(loss,(p,y),allow_unused=True)
        cov, _ = target_autocovariance(y)
        self.assertFalse(cov.requires_grad)
        self.assertIsNone(target_grad)
        expected = torch.linalg.solve(cov,(p.detach()-y.detach()).transpose(1,2).unsqueeze(-1)).squeeze(-1).transpose(1,2)/p.numel()
        torch.testing.assert_close(grad,expected,atol=1e-12,rtol=1e-12)
        _, d1 = target_gaussian_nll(p,y,return_diagnostics=True)
        _, d2 = target_gaussian_nll(p*10,y,return_diagnostics=True)
        self.assertEqual(d1['logdet_per_scalar'],d2['logdet_per_scalar'])
        self.assertEqual(d1['target_variance_mean'],d2['target_variance_mean'])

    def test_constant_offset_large_targets_and_temporal_order(self):
        p = torch.randn(2,96,2,requires_grad=True)
        y = torch.full_like(p,3.)
        loss = target_gaussian_nll(p,y)
        expected = .5*((p-y).square().mean()/.01+math.log(.01)+math.log(2*math.pi))
        torch.testing.assert_close(loss,expected,rtol=2e-6,atol=1e-5)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        y = torch.randn(2,96,2,dtype=torch.double)*100
        p = torch.randn_like(y,requires_grad=True)
        a = target_gaussian_nll(p,y)
        torch.testing.assert_close(a,target_gaussian_nll(p+4,y+4),rtol=1e-10,atol=1e-10)
        self.assertTrue(torch.isfinite(torch.autograd.grad(a,p)[0]).all())
        covariance,_ = target_autocovariance(y)
        other,_ = target_autocovariance(y.flip(1).roll(7,1))
        self.assertGreater(float((covariance-other).abs().max()),1.)
        for floor in (0,-1,float('nan')):
            with self.assertRaises(ValueError):target_autocovariance(y,floor)

    def test_initial_state_optimizer_functions_and_rng(self):
        models = study.initialize_pair(self.cfg)
        a,b = models.values()
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        groups = [study.build_optimizer(study.parameter_groups(m,arm),arm).state_dict()['param_groups'] for arm,m in models.items()]
        self.assertEqual(groups[0],groups[1])
        x = torch.randn(2,96,2)
        for training in (True,False):
            a.train(training);b.train(training)
            study.seed_all(42);pa=a(x);ra=torch.get_rng_state()
            study.seed_all(42);pb=b(x);rb=torch.get_rng_state()
            torch.testing.assert_close(pa,pb,rtol=0,atol=0)
            self.assertTrue(torch.equal(ra,rb))

    def test_pure_total_objectives_gradients_capture(self):
        models = study.initialize_pair(self.cfg)
        x,y = torch.randn(2,96,2),torch.randn(2,96,2)
        lc = study.loss_configuration(SimpleNamespace(nll_diagonal_floor=.01))
        for arm,m in models.items():
            p=m(x);parts=study.training_losses(p,y,arm,lc,x)
            expected=(p-y).square().mean() if arm=='mse' else target_gaussian_nll(p,y)
            torch.testing.assert_close(parts['loss'],expected,rtol=0,atol=0)
            self.assertEqual(parts['auxiliary_nll'].item(),0)
            parts['loss'].backward()
            for n,w in m.named_parameters():
                self.assertIsNotNone(w.grad,n)
                self.assertTrue(torch.isfinite(w.grad).all(),n)
            with tempfile.TemporaryDirectory() as d,torch.no_grad():
                m.eval();before=m(x).numpy()
                with SelectedValidationCapture(m,Path(d),arm,2) as cap:after=m(x).numpy()
                np.testing.assert_array_equal(before,after)
                report=cap.report(after,y.numpy())
                self.assertLessEqual(report['reconstruction_max_abs_error'],2e-5)

    def test_parameter_count_schedule_and_no_test(self):
        cfg=SimpleNamespace(**dict(vars(self.cfg),d_model=32,d_ff=64,ssm_modes=8,enc_in=7,c_out=7,top_k=5))
        models=study.initialize_pair(cfg)
        self.assertEqual({sum(p.numel() for p in m.parameters()) for m in models.values()},{136150})
        for e in range(31):self.assertEqual(study.learning_rate('mse',e,30),study.learning_rate('nll',e,30))
        with self.assertRaises(ValueError):study.dataset(cfg,'test',0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_loss_gradient_consistency(self):
        p=torch.randn(2,96,7,requires_grad=True);y=torch.randn_like(p)
        cpu=target_gaussian_nll(p,y);g=torch.autograd.grad(cpu,p)[0]
        pc=p.detach().cuda().requires_grad_();yc=y.cuda()
        gpu=target_gaussian_nll(pc,yc);gc=torch.autograd.grad(gpu,pc)[0].cpu()
        torch.testing.assert_close(cpu.detach(),gpu.detach().cpu(),atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(g,gc,atol=2e-6,rtol=2e-4)


if __name__=='__main__':unittest.main()
