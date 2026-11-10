"""Repeated residual decomposition math and paired forecasting contracts."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from test_oscillator_mixer import config
from models.OscillatorPDMResidualMA import ResidualDecomposition, Model
from models.OscillatorPDMScaledMA import ScaleAwareDecomposition
from scripts import train_timemix_pdm_residual_ma as study
from utils.pdm_residual_ma_diagnostics import SelectedValidationCapture

class ResidualMATests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=1,dropout=.2,
                        down_sampling_layers=3,down_sampling_window=2)
        self.cfg.seasonal_normalize=False
        torch.manual_seed(2021)
    def tearDown(self):torch.set_num_threads(self.threads)
    def test_numpy_reference_reconstruction_and_one_pass_identity(self):
        module=ResidualDecomposition(25,96,[96,48,24,12],3).double()
        original=ScaleAwareDecomposition(25,96,[96,48,24,12]).double()
        single=ResidualDecomposition(25,96,[96,48,24,12],1).double()
        for length,k in [(96,25),(48,13),(24,7),(12,3)]:
            x=torch.randn(2,length,3,dtype=torch.double)
            s,t=module(x);ref=x.numpy().copy();trend=np.zeros_like(ref)
            for _ in range(3):
                smooth=np.apply_along_axis(lambda a:np.convolve(np.pad(a,k//2,mode='edge'),np.ones(k)/k,'valid'),1,ref)
                ref-=smooth;trend+=smooth
            np.testing.assert_allclose(s.numpy(),ref,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(t.numpy(),trend,atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(s+t,x,atol=1e-12,rtol=1e-12)
            for a,b in zip(single(x),original(x)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_constant_affine_equivariance_and_gradcheck(self):
        m=ResidualDecomposition(25,96,[12],3).double()
        x=torch.randn(1,12,2,dtype=torch.double,requires_grad=True)
        s,t=m(x);ss,tt=m(-1.7*x+.3)
        torch.testing.assert_close(ss,-1.7*s,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(tt,-1.7*t+.3,atol=1e-12,rtol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(m,(x,),eps=1e-6,atol=1e-5))
        flat=torch.ones_like(x)*2;s,t=m(flat)
        torch.testing.assert_close(s,torch.zeros_like(s),rtol=0,atol=0)
        for invalid in (0,True,1.5):
            with self.assertRaises(ValueError):ResidualDecomposition(25,96,[12],invalid)
    def test_initial_state_optimizer_and_config_restore(self):
        models=study.initialize_pair(self.cfg);a,b=models.values()
        self.assertEqual(a.state_dict().keys(),b.state_dict().keys())
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        opts=[study.build_optimizer(study.parameter_groups(m,arm),arm) for arm,m in models.items()]
        self.assertEqual(opts[0].state_dict()['param_groups'],opts[1].state_dict()['param_groups'])
        x=torch.randn(2,96,2)
        for arm,m in models.items():
            restored=study.VARIANTS[arm](study.arm_config(self.cfg,arm)).eval()
            restored.load_state_dict(m.state_dict(),strict=True);m.eval()
            torch.testing.assert_close(m(x),restored(x),rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        self.assertGreater(float((a(x)-b(x)).abs().max()),1e-6)
        self.assertEqual(a.normalize.forward.__func__,b.normalize.forward.__func__)
    def test_pure_mse_backward_and_schedule(self):
        models=study.initialize_pair(self.cfg);x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        cfg=study.loss_configuration(SimpleNamespace(variance_floor=.01,kl_jitter=.01))
        for arm,m in models.items():
            p=m(x);parts=study.training_losses(p,y,arm,cfg,x)
            torch.testing.assert_close(parts['loss'],(p-y).square().mean(),rtol=0,atol=0)
            self.assertEqual(parts['auxiliary_kl'].item(),0.)
            parts['loss'].backward()
            for name,w in m.named_parameters():
                self.assertIsNotNone(w.grad,name);self.assertTrue(torch.isfinite(w.grad).all(),name)
        for e in range(31):self.assertEqual(study.learning_rate('ma1',e,30),study.learning_rate('ma3',e,30))
        self.assertEqual(study.joint_stale_count(4,False),5)
    def test_same_forward_capture_and_normalization(self):
        models=study.initialize_pair(self.cfg);x=torch.randn(2,96,2);truth=np.zeros((2,96,2),dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            for arm,m in models.items():
                m.eval()
                with torch.no_grad():
                    before=m(x).numpy()
                    with SelectedValidationCapture(m,Path(d),arm,2) as cap:actual=m(x).numpy()
                np.testing.assert_array_equal(before,actual)
                report=cap.report(actual,truth)
                self.assertLessEqual(report['reconstruction_max_abs_error'],2e-5)
                self.assertEqual(report['initial_split_calls'],4)
                self.assertEqual(report['residual_decomposition_passes'],study.PASSES[arm])
                self.assertEqual(report['mixing_depth'],1)
    def test_formal_size_equal_parameters(self):
        cfg=SimpleNamespace(**dict(vars(self.cfg),d_model=32,d_ff=64,ssm_modes=8,enc_in=7,c_out=7,top_k=5))
        models=study.initialize_pair(cfg)
        self.assertEqual({sum(p.numel() for p in m.parameters()) for m in models.values()},{136150})

if __name__=='__main__':unittest.main()
