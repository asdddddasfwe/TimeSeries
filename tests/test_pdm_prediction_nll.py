"""Independent numerical references and MA1 pure-NLL training contracts."""
import math,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from test_oscillator_mixer import config
from utils.prediction_autocov_nll import prediction_autocovariance,prediction_gaussian_nll
from scripts import train_timemix_pdm_prediction_nll as study
from utils.pdm_residual_ma_diagnostics import SelectedValidationCapture

class PredictionNLLTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(2021)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=1,dropout=.2,down_sampling_layers=3,down_sampling_window=2)
        self.cfg.seasonal_normalize=False
    def tearDown(self):torch.set_num_threads(self.threads)
    def test_numpy_reference_and_psd(self):
        p=torch.randn(2,9,3,dtype=torch.double);y=torch.randn_like(p)
        cov,g=prediction_autocovariance(p,.01);values=[]
        for b in range(2):
            for c in range(3):
                u=p[b,:,c].numpy();u=u-u.mean();h=len(u)
                gamma=np.array([np.dot(u[:h-k],u[k:])/h for k in range(h)])
                sigma=gamma[np.abs(np.arange(h)[:,None]-np.arange(h)[None,:])]+.01*np.eye(h)
                np.testing.assert_allclose(g[b,c].numpy(),gamma,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(cov[b,c].numpy(),sigma,atol=1e-12,rtol=1e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(sigma).min(),.01-1e-12)
                e=y[b,:,c].numpy()-p[b,:,c].numpy()
                values.append(.5*(e@np.linalg.solve(sigma,e)+np.linalg.slogdet(sigma)[1]+h*np.log(2*np.pi))/h)
        self.assertAlmostEqual(float(prediction_gaussian_nll(p,y)),float(np.mean(values)),places=11)
    def test_gradcheck_target_detach_and_covariance_gradient(self):
        p=torch.randn(1,6,2,dtype=torch.double,requires_grad=True);y=torch.randn_like(p,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:prediction_gaussian_nll(z,y),(p,),atol=1e-5,rtol=1e-4))
        loss=prediction_gaussian_nll(p,y);a=torch.autograd.grad(loss,p,retain_graph=True)[0]
        self.assertIsNone(torch.autograd.grad(loss,y,allow_unused=True)[0])
        cov,_=prediction_autocovariance(p);chol=torch.linalg.cholesky(cov.detach());e=(y.detach()-p).transpose(1,2).unsqueeze(-1)
        fixed=.5*torch.linalg.solve_triangular(chol,e,upper=False).square().mean()
        b=torch.autograd.grad(fixed,p)[0];self.assertGreater(float((a-b).abs().max()),1e-5)
    def test_flat_sequence_floor_and_shift(self):
        p=torch.full((2,96,2),3.,requires_grad=True);y=torch.randn_like(p)
        cov,g=prediction_autocovariance(p);torch.testing.assert_close(g,torch.zeros_like(g),rtol=0,atol=0)
        expected=.5*((p-y).square().mean()/.01+math.log(.01)+math.log(2*math.pi))
        loss=prediction_gaussian_nll(p,y);torch.testing.assert_close(loss,expected,rtol=2e-6,atol=1e-5)
        loss.backward();self.assertTrue(torch.isfinite(p.grad).all())
        a=torch.randn(1,96,2,dtype=torch.double);b=torch.randn_like(a)
        torch.testing.assert_close(prediction_gaussian_nll(a,b),prediction_gaussian_nll(a+4,b+4),atol=1e-10,rtol=1e-10)
        for floor in [0.,-1.,float('nan')]:
            with self.assertRaises(ValueError):prediction_autocovariance(a,floor)
    def test_large_finite_input_and_temporal_order(self):
        p=(100*torch.randn(2,96,2)).requires_grad_();y=torch.randn_like(p)
        loss=prediction_gaussian_nll(p,y);loss.backward();self.assertTrue(torch.isfinite(p.grad).all())
        p=torch.arange(8,dtype=torch.double).reshape(1,8,1);y=torch.randn_like(p)
        cov,_=prediction_autocovariance(p);other,_=prediction_autocovariance(p[:,[0,4,1,5,2,6,3,7]])
        self.assertGreater(float((cov-other).abs().max()),1.)
    def test_initial_state_optimizer_functions_and_restore(self):
        models=study.initialize_pair(self.cfg);a,b=models.values()
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        opt=[study.build_optimizer(study.parameter_groups(m,arm),arm).state_dict()['param_groups'] for arm,m in models.items()];self.assertEqual(opt[0],opt[1])
        x=torch.randn(2,96,2)
        for training in [True,False]:
            a.train(training);b.train(training);study.seed_all(42);pa=a(x);study.seed_all(42);pb=b(x);torch.testing.assert_close(pa,pb,rtol=0,atol=0)
        self.assertEqual(a.normalize.forward.__func__,b.normalize.forward.__func__)
    def test_pure_objectives_model_gradients_and_capture(self):
        models=study.initialize_pair(self.cfg);x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        lc=study.loss_configuration(SimpleNamespace(nll_diagonal_floor=.01))
        for arm,m in models.items():
            p=m(x);parts=study.training_losses(p,y,arm,lc,x)
            expected=(p-y).square().mean() if arm=='mse' else prediction_gaussian_nll(p,y)
            torch.testing.assert_close(parts['loss'],expected,rtol=0,atol=0);self.assertEqual(parts['auxiliary_nll'].item(),0)
            parts['loss'].backward()
            for n,w in m.named_parameters():self.assertIsNotNone(w.grad,n);self.assertTrue(torch.isfinite(w.grad).all(),n)
            with tempfile.TemporaryDirectory() as d,torch.no_grad():
                m.eval();before=m(x).numpy()
                with SelectedValidationCapture(m,Path(d),arm,2) as cap:after=m(x).numpy()
                np.testing.assert_array_equal(before,after);report=cap.report(after,y.numpy());self.assertLessEqual(report['reconstruction_max_abs_error'],2e-5);self.assertEqual(report['residual_decomposition_passes'],1)
    def test_parameter_count_and_schedule(self):
        cfg=SimpleNamespace(**dict(vars(self.cfg),d_model=32,d_ff=64,ssm_modes=8,enc_in=7,c_out=7,top_k=5))
        models=study.initialize_pair(cfg);self.assertEqual({sum(p.numel() for p in m.parameters()) for m in models.values()},{136150})
        for e in range(31):self.assertEqual(study.learning_rate('mse',e,30),study.learning_rate('nll',e,30))

if __name__=='__main__':unittest.main()
