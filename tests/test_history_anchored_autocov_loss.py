"""Reference arithmetic, degeneracy, gradients and matched-model regression."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch.autograd import gradcheck
from test_oscillator_mixer import config
from utils.history_anchored_autocov_loss import history_anchored_autocov_loss as loss_fn, COV_LAGS, BOUNDARY_LAGS
from scripts.train_timemix_pdm_autocov_loss import initialize_pair, parameter_groups, training_losses, loss_configuration, output_gradient_audit
from scripts.train_timemix_pdm_decoder_study import initialize_pair as old_pair


class AutocovLossTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(2021)
        self.x=torch.randn(2,96,2,dtype=torch.float64)
        self.y=torch.randn(2,96,2,dtype=torch.float64)
        self.p=torch.randn(2,96,2,dtype=torch.float64,requires_grad=True)
        self.eps=[.01,.02]
    def tearDown(self):torch.set_num_threads(self.threads)
    def test_numpy_reference_and_per_window_reduction(self):
        x,y,p=[t.detach().numpy() for t in (self.x,self.y,self.p)];cv=[];an=[]
        def huber(e):return .5*e*e if abs(e)<1 else abs(e)-.5
        for b in range(2):
            for c in range(2):
                xx=x[b,:,c]-x[b,:,c].mean();yy=y[b,:,c]-y[b,:,c].mean();pp=p[b,:,c]-p[b,:,c].mean()
                vy=np.mean(yy**2);vx=np.mean(xx**2);e=self.eps[c];gy=vy/(vy+e);gx=vx/(vx+e)
                v=0
                for k in COV_LAGS:
                    cp=sum(pp[t]*pp[t+k] for t in range(96-k))/96
                    cy=sum(yy[t]*yy[t+k] for t in range(96-k))/96
                    v+=(96-k)*huber((cp-cy)/(vy+e))
                cv.append(gy*v/sum(96-k for k in COV_LAGS))
                v=0
                for k in BOUNDARY_LAGS:
                    bp=sum(xx[96+h-k]*pp[h] for h in range(k))/k
                    by=sum(xx[96+h-k]*yy[h] for h in range(k))/k
                    v+=k*huber((bp-by)/np.sqrt((vx+e)*(vy+e)))
                an.append(gx*gy*v/sum(BOUNDARY_LAGS))
        got=loss_fn(self.x,self.p,self.y,self.eps)
        self.assertAlmostEqual(float(got['covariance'].detach()),np.mean(cv),places=12)
        self.assertAlmostEqual(float(got['anchor'].detach()),np.mean(an),places=12)
    def test_exact_zero_and_offset_invariance(self):
        exact=loss_fn(self.x,self.y,self.y,self.eps)
        self.assertTrue(all(v==0 for v in exact.values()))
        a=loss_fn(self.x,self.p,self.y,self.eps);b=loss_fn(self.x+12,self.p-7,self.y+3,self.eps)
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=1e-10,atol=1e-12)
    def test_sign_amplitude_phase(self):
        t=torch.arange(96,dtype=torch.float64);y=torch.sin(2*torch.pi*t/24)[None,:,None];x=y.clone()
        neg=loss_fn(x,-y,y,.01);half=loss_fn(x,.5*y,y,.01);shift=loss_fn(x,torch.roll(y,6,1),y,.01)
        self.assertEqual(float(neg['covariance']),0);self.assertGreater(float(neg['anchor']),1)
        self.assertGreater(float(half['covariance']),.1)
        self.assertGreater(float(shift['anchor']),100*float(shift['covariance']))
    def test_flat_and_tiny_variance_finite_and_detached(self):
        for size in [0.,1e-12,1e-5,1.]:
            x=(self.x*size).requires_grad_();y=(self.y*size).requires_grad_();p=self.p.detach().clone().requires_grad_()
            parts=loss_fn(x,p,y,self.eps);total=(p-y.detach()).square().mean()+.025*sum(parts.values());total.backward()
            self.assertTrue(torch.isfinite(p.grad).all());self.assertIsNone(x.grad);self.assertIsNone(y.grad)
            if size==0:self.assertTrue(all(v==0 for v in parts.values()))
        p=torch.zeros_like(self.p,requires_grad=True);parts=loss_fn(self.x,p,self.y,self.eps)
        g=torch.autograd.grad(parts['covariance'],p,retain_graph=True)[0];self.assertEqual(int(torch.count_nonzero(g)),0)
        g=torch.autograd.grad(parts['anchor'],p)[0];self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)
    def test_double_gradcheck(self):
        x=self.x[:1,:12,:1];y=self.y[:1,:12,:1];p=self.p[:1,:12,:1].detach().requires_grad_()
        def fn(v):return sum(loss_fn(x,v,y,.01,(0,1,3,6),(2,6,12)).values())
        self.assertTrue(gradcheck(fn,(p,),eps=1e-6,atol=1e-5,rtol=1e-3))
    def test_bad_shapes_lags_floors(self):
        for eps in [0.,-1.,float('nan'),[.01,.02,.03]]:
            with self.assertRaises(ValueError):loss_fn(self.x,self.p,self.y,eps)
        with self.assertRaises(ValueError):loss_fn(self.x,self.p,self.y,self.eps,(0,96))
        with self.assertRaises(ValueError):loss_fn(self.x,self.p,self.y,self.eps,boundary_lags=(97,))
    def test_training_equations_and_diagnostic_gradient_preservation(self):
        cfg=loss_configuration(SimpleNamespace(variance_floor=.01,kl_jitter=.01,covariance_floor_by_channel=self.eps,train_channel_variance=[1.,2.]))
        for arm in ('mse','cov','cov_anchor'):
            p=self.p.detach().float().requires_grad_();y=self.y.float();x=self.x.float()
            parts=training_losses(p,y,arm,cfg,x)
            expected=(p-y).square().mean()
            if arm!='mse':expected=expected+.025*parts['covariance']
            if arm=='cov_anchor':expected=expected+.025*parts['anchor']
            torch.testing.assert_close(parts['loss'],expected,rtol=0,atol=0)
            reference=torch.autograd.grad(parts['loss'],p,retain_graph=True)[0]
            rng=torch.get_rng_state();diagnostic=output_gradient_audit(parts,p,arm,cfg)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));parts['loss'].backward()
            torch.testing.assert_close(p.grad,reference,rtol=0,atol=0)
            if arm=='mse':self.assertEqual(diagnostic['weighted_structure']['norm'],0)
    def test_models_initial_functions_optimizer_and_legacy(self):
        cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=1,dropout=.2,down_sampling_layers=3,down_sampling_window=2);cfg.seasonal_normalize=False
        torch.manual_seed(2021);models=initialize_pair(cfg)
        torch.manual_seed(2021);old=old_pair(cfg)['oscillator']
        for m in models.values():
            for k,v in old.state_dict().items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
        policies=[]
        for arm,m in models.items():
            gs=parameter_groups(m,arm);d={id(p):g['weight_decay'] for g in gs for p in g['params']};self.assertEqual(len(d),len(list(m.parameters())))
            policies.append({n:d[id(p)] for n,p in m.named_parameters()})
        self.assertTrue(all(p==policies[0] for p in policies))
        x=self.x.float();ys=[];states=[]
        for m in models.values():
            m.train();torch.manual_seed(17);ys.append(m(x));states.append(torch.get_rng_state())
        for y in ys:torch.testing.assert_close(y,ys[0],rtol=0,atol=0)
        self.assertTrue(all(torch.equal(states[0],s) for s in states))

if __name__=='__main__':unittest.main()
