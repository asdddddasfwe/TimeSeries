"""Independent mathematics, mean consistency, and full-network gradient checks."""
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from utils.conditional_transition_bregman import TransitionOperator, bregman_transition, objective
from test_oscillator_mixer import config


class BregmanTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2021)
        self.op=TransitionOperator(history=12,horizon=12,channels=2,q=3,lags=(1,4)).double()
        self.x=torch.randn(2,12,2,dtype=torch.double)
        self.coeff=tuple(v.detach() for v in self.op(self.x))

    def test_numpy_divergence_orientation_and_gradient(self):
        p=torch.randn_like(self.x,requires_grad=True);y=torch.randn_like(p,requires_grad=True)
        v,_=bregman_transition(p,y,self.op,self.coeff,1.)
        rp=self.op.residual(p,self.coeff).detach().numpy()
        ry=self.op.residual(y,self.coeff).detach().numpy()
        ref=.5*(ry-rp)**2+np.logaddexp(ry,-ry)-np.logaddexp(rp,-rp)-np.tanh(rp)*(ry-rp)
        self.assertAlmostEqual(float(v),float(ref.mean()),places=12)
        g,gy=torch.autograd.grad(v,(p,y),allow_unused=True);self.assertIsNone(gy)
        a,_=self.coeff
        gr=(1+1-np.tanh(rp)**2)*(rp-ry)/rp.size
        expected=np.zeros((2,2,12))
        for r,j in enumerate(self.op.destinations.tolist()):
            expected[:,:,j]+=gr[:,:,r]
            for q,i in enumerate(self.op.parents[r].tolist()):expected[:,:,i]-=a.numpy()[:,:,r,q]*gr[:,:,r]
        np.testing.assert_allclose(g.numpy(),expected.transpose(0,2,1),atol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(lambda z:bregman_transition(z,y,self.op,self.coeff,1.)[0],(p,)))
        # Reversed divergence is deliberately different.
        reverse,_=bregman_transition(y.detach(),p.detach(),self.op,self.coeff,1.)
        self.assertGreater(abs(float(v-reverse)),1e-6)

    def test_beta_zero_nonnegativity_and_no_detach_prediction_tanh(self):
        p=torch.randn_like(self.x,requires_grad=True);y=torch.randn_like(p)
        rp=self.op.residual(p,self.coeff);ry=self.op.residual(y,self.coeff)
        loss,_=bregman_transition(p,y,self.op,self.coeff,0.)
        torch.testing.assert_close(loss,.5*(rp-ry).square().mean(),rtol=1e-12,atol=1e-12)
        for beta in (0.,1.):
            equal,_=bregman_transition(p,p.detach(),self.op,self.coeff,beta)
            self.assertEqual(float(equal),0.)
            v,_=bregman_transition(p,y,self.op,self.coeff,beta);self.assertGreaterEqual(float(v),0.)

    def test_mean_consistency_and_original_nonlinear_counterexample(self):
        y0=torch.randn_like(self.x);y1=torch.randn_like(self.x);m=(y0+y1)/2
        p=m.clone().requires_grad_()
        def loss(z,y):return (z-y).square().mean()+.1*bregman_transition(z,y,self.op,self.coeff,1.)[0]
        risk=(loss(p,y0)+loss(p,y1))/2
        self.assertLess(float(torch.autograd.grad(risk,p)[0].abs().max()),1e-12)
        trial=m+torch.randn_like(m)*.2
        gap=(loss(trial,y0)+loss(trial,y1))/2-risk.detach()
        torch.testing.assert_close(gap,loss(trial,m),rtol=1e-10,atol=1e-12)
        z=torch.tensor([0.,1.],requires_grad=True)
        old=(z[0].square()+1+(z[1]-1).square())+.1*(z[1]-z[0].square()).square()
        self.assertAlmostEqual(float(torch.autograd.grad(old,z)[0][1]),.2,places=6)

    def test_shapes_causality_history_only_coefficients_and_extremes(self):
        self.assertTrue((self.op.parents<self.op.destinations[:,None]).all())
        a,b=self.coeff
        self.assertLessEqual(float(a.abs().sum(-1).max()),2.)
        for scale in (0.,1e-4,1.,1e3):
            p=(torch.randn_like(self.x)*scale).requires_grad_();y=torch.randn_like(p)*scale
            v,_=bregman_transition(p,y,self.op,self.coeff,1.)
            self.assertTrue(torch.isfinite(v))
            self.assertTrue(torch.isfinite(torch.autograd.grad(v,p)[0]).all())

    def test_network_initialization_gradients_and_capture(self):
        from scripts import train_timemix_pdm_transition_bregman as study
        from utils.pdm_residual_ma_diagnostics import SelectedValidationCapture
        cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=1,dropout=.2,
                   down_sampling_layers=3,down_sampling_window=2)
        cfg.seasonal_normalize=False
        models=study.initialize_pair(cfg)
        states=[m.state_dict() for m in models.values()]
        for state in states[1:]:
            for key in states[0]:self.assertTrue(torch.equal(states[0][key],state[key]))
        groups=[study.build_optimizer(study.parameter_groups(m,n),n).state_dict()['param_groups'] for n,m in models.items()]
        self.assertTrue(all(g==groups[0] for g in groups))
        with tempfile.TemporaryDirectory() as directory:
            op=TransitionOperator(channels=2);path=Path(directory)/'operator.pth'
            torch.save(dict(options=op.options,state_dict=op.state_dict()),path)
            lc=dict(operator_checkpoint=str(path),weight=.1,beta=1.)
            x,y=torch.randn(2,96,2),torch.randn(2,96,2)
            for arm,m in models.items():
                parts=objective(m(x),y,x,arm,lc);parts['loss'].backward()
                for name,w in m.named_parameters():
                    self.assertIsNotNone(w.grad,name);self.assertTrue(torch.isfinite(w.grad).all(),name)
                m.eval()
                with torch.no_grad():
                    before=m(x).numpy()
                    with SelectedValidationCapture(m,Path(directory),arm,2) as cap:after=m(x).numpy()
                np.testing.assert_array_equal(before,after)
                self.assertLess(cap.report(after,y.numpy())['reconstruction_max_abs_error'],2e-5)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_cpu_gradient(self):
        p=torch.randn(2,12,2,requires_grad=True);y=torch.randn_like(p)
        coefficients=tuple(v.float() for v in self.coeff)
        v,_=bregman_transition(p,y,self.op,coefficients,1.)
        g=torch.autograd.grad(v,p)[0]
        pc=p.detach().cuda().requires_grad_();op=self.op.cuda()
        vc,_=bregman_transition(pc,y.cuda(),op,tuple(v.cuda() for v in coefficients),1.)
        gc=torch.autograd.grad(vc,pc)[0].cpu()
        torch.testing.assert_close(v,vc.cpu(),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(g,gc,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
