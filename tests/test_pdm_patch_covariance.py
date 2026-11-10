"""Numerical/semantic checks for disjoint patch RKHS covariance matching."""
import unittest
import numpy as np
import torch
from utils.patch_covariance_loss import parts,patch_covariance_parts
from scripts.audit_pdm_patch_covariance import numpy_covariance_parts
from scripts import train_timemix_pdm_patch_covariance as study

class PatchCovarianceTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(2021)

    def check_formula(self,device):
        p=torch.randn(2,16,3,dtype=torch.float64,device=device,requires_grad=True)
        y=torch.randn_like(p,requires_grad=True);h=torch.randn_like(p,requires_grad=True)
        rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        d,diag=parts(p,y)
        ref=numpy_covariance_parts(p.detach().cpu().numpy(),y.detach().cpu().numpy())
        self.assertEqual(diag['patch_count'],4)
        for k,v in ref.items():self.assertAlmostEqual(diag[k],v,places=12)
        for arm,w in study.WEIGHTS.items():
            r=study.objective(p,y,h,arm,study.loss_configuration(None));m=(p-y.detach()).square().mean()
            self.assertTrue(torch.equal(r['loss'],m if w==0 else m+w*d))
            gp,gy,gh=torch.autograd.grad(r['loss'],(p,y,h),retain_graph=True,allow_unused=True)
            self.assertIsNone(gy);self.assertIsNone(gh);self.assertTrue(torch.isfinite(gp).all())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        if crng is not None:self.assertTrue(torch.equal(crng,torch.cuda.get_rng_state()))

    def test_cpu_formula(self):self.check_formula('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_formula(self):self.check_formula('cuda')

    def test_gradient(self):
        p=torch.randn(1,12,1,dtype=torch.float64,requires_grad=True);y=torch.randn_like(p)
        self.assertTrue(torch.autograd.gradcheck(lambda z:parts(z,y,.7)[0],(p,)))
        v=torch.randn_like(p);eps=1e-6;g=torch.autograd.grad(parts(p,y,.7)[0],p)[0]
        fd=(parts(p+eps*v,y,.7)[0]-parts(p-eps*v,y,.7)[0])/(2*eps)
        torch.testing.assert_close(fd,(g*v).sum(),atol=1e-9,rtol=1e-5)

    def test_zero_constant_and_cross_term(self):
        y=torch.randn(2,96,7,dtype=torch.float64)
        d,diag=parts(y,y);self.assertEqual(float(d),0.);self.assertEqual(diag['patch_count'],24)
        self.assertGreater(diag['prediction_covariance_norm2'],0.)
        # A common shift leaves each self-Gram unchanged. The cross term still detects differing operators.
        shifted,ds=parts(y+2,y);self.assertGreater(float(shifted),0.)
        self.assertAlmostEqual(ds['prediction_covariance_norm2'],diag['prediction_covariance_norm2'],places=12)
        torch.testing.assert_close(parts(y+5,y+3)[0],shifted)
        self.assertAlmostEqual(float(parts(torch.ones_like(y)*5,torch.ones_like(y)*2)[0]),0.,places=12)

    def test_disjoint_boundaries_and_permutation(self):
        y=torch.arange(96,dtype=torch.float64).reshape(1,96,1)/40
        patches=y.transpose(1,2).unfold(-1,4,4)
        self.assertEqual(patches.shape,(1,1,24,4));self.assertEqual(float(patches[0,0,1,0]),.1)
        # Independent patch-set permutation leaves lag0 covariance identical.
        p=y.reshape(1,24,4,1).flip(1).reshape_as(y)
        self.assertLess(float(parts(p,y)[0]),1e-12)
        self.assertGreater(float((p-y).square().mean()),0.)
        # Coordinate changes within only prediction patches generally alter the operator.
        p=y.reshape(1,24,4,1).flip(2).reshape_as(y)
        self.assertGreater(float(parts(p,y)[0]),1e-8)

    def test_isolation(self):
        y=torch.randn(2,16,3,dtype=torch.float64);p=y.clone();p[0,:,0]+=.7
        full=parts(p,y)[0];single=parts(p[:1,:,:1],y[:1,:,:1])[0]
        torch.testing.assert_close(full,single/6)

    def test_extremes_invalid(self):
        y=torch.zeros(1,8,1);p=torch.full_like(y,1e6,requires_grad=True)
        d,_=parts(p,y);self.assertTrue(torch.isfinite(d));self.assertTrue(torch.isfinite(torch.autograd.grad(d,p)[0]).all())
        for length in [0,3,8,True]:
            with self.assertRaises(ValueError):parts(p,y,1.,length)
        for sigma in [0.,float('nan')]:
            with self.assertRaises(ValueError):parts(p,y,sigma)
        with self.assertRaises(ValueError):parts(p*float('nan'),y)

    def test_probe_and_pure_mse(self):
        model=torch.nn.Linear(3,3);x=torch.randn(2,12,3);y=torch.randn_like(x)
        for arm,w in study.WEIGHTS.items():
            model.zero_grad(set_to_none=True);p=model(x)
            r=study.objective(p,y,x,arm,study.loss_configuration(None))
            expected=torch.autograd.grad(r['loss'],tuple(model.parameters()),retain_graph=True)
            if arm=='mse':
                exact=torch.autograd.grad((p-y).square().mean(),tuple(model.parameters()),retain_graph=True)
                for a,b in zip(expected,exact):self.assertTrue(torch.equal(a,b))
            rng=torch.get_rng_state().clone();g=study.gradient_comparison(model,r,w)
            self.assertTrue(all(v.grad is None for v in model.parameters()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            r['loss'].backward()
            for v,e in zip(model.parameters(),expected):self.assertTrue(torch.equal(v.grad,e))
        self.assertEqual(study.WEIGHTS,{'mse':0.,'patch_covariance':1.})
        stale=0
        for improves,expected in [(True,0),(False,1),(False,2),(True,0),(False,1)]:
            stale=study.joint_stale_count(stale,improves);self.assertEqual(stale,expected)
        self.assertEqual(study.selection_score({'neural_mse':.4,'relation':-1}),.4)

if __name__=='__main__':unittest.main()
