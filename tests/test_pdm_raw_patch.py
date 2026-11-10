"""Numerical and pairing checks for aligned raw-value patch kernel."""
import unittest
import numpy as np
import torch
from utils.raw_patch_loss import parts
from scripts.audit_pdm_raw_patch import numpy_raw_parts
from scripts import train_timemix_pdm_raw_patch as study


class RawPatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def check_formula(self, device):
        p=torch.randn(2,12,3,dtype=torch.float64,device=device,requires_grad=True)
        y=torch.randn_like(p,requires_grad=True);h=torch.randn_like(p,requires_grad=True)
        cfg=study.loss_configuration(None)
        rng=torch.get_rng_state().clone()
        crng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        q,k,patch=parts(p,y,1.,4)
        self.assertEqual(patch.shape,(2,9,3))
        ref=numpy_raw_parts(p.detach().cpu().numpy(),y.detach().cpu().numpy(),1.)
        self.assertAlmostEqual(float(q),ref['raw_quadratic'],places=12)
        self.assertAlmostEqual(float(k),ref['raw_kernel'],places=12)
        for arm,w in study.WEIGHTS.items():
            r=study.objective(p,y,h,arm,cfg)
            expected=(p-y.detach()).square().mean()
            if w:expected=expected+w*k
            self.assertTrue(torch.equal(r['loss'],expected))
            gp,gy,gh=torch.autograd.grad(r['loss'],(p,y,h),retain_graph=True,allow_unused=True)
            self.assertIsNone(gy);self.assertIsNone(gh);self.assertTrue(torch.isfinite(gp).all())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        if crng is not None:self.assertTrue(torch.equal(crng,torch.cuda.get_rng_state()))

    def test_cpu_formula(self):self.check_formula('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_formula(self):self.check_formula('cuda')

    def test_gradcheck_finite_difference(self):
        p=torch.randn(1,7,2,dtype=torch.float64,requires_grad=True)
        y=torch.randn_like(p)
        self.assertTrue(torch.autograd.gradcheck(lambda z:parts(z,y,.7)[1],(p,)))
        direction=torch.randn_like(p);eps=1e-6
        grad=torch.autograd.grad(parts(p,y,.7)[1],p)[0]
        fd=(parts(p+eps*direction,y,.7)[1]-parts(p-eps*direction,y,.7)[1])/(2*eps)
        torch.testing.assert_close(fd,(grad*direction).sum(),atol=1e-9,rtol=1e-6)

    def test_offset_sensitive_and_common_translation_invariant(self):
        y=torch.randn(2,96,7,dtype=torch.float64)
        self.assertEqual(float(parts(y,y)[1]),0.)
        q,k,patch=parts(y+.5,y)
        self.assertEqual(patch.shape,(2,93,7))
        self.assertAlmostEqual(float(q),.125,places=13)
        self.assertAlmostEqual(float(k),1-np.exp(-.125),places=13)
        torch.testing.assert_close(parts(y+3.5,y+3)[1],k)

    def test_isolation_and_original_units(self):
        y=torch.randn(2,9,3,dtype=torch.float64)
        p=y.clone();p[0,:,0]+=.5
        _,k,q=parts(p,y)
        self.assertEqual(float(q[1].sum()),0.)
        self.assertEqual(float(q[0,:,1:].sum()),0.)
        self.assertGreater(float(parts(2*p,2*y)[1]),float(k))

    def test_extremes_and_validation(self):
        y=torch.zeros(1,4,1);p=torch.full_like(y,1e6,requires_grad=True)
        q,k,_=parts(p,y)
        self.assertEqual(float(k),1.);self.assertTrue(torch.isfinite(q))
        self.assertTrue(torch.isfinite(torch.autograd.grad(k,p)[0]).all())
        for length in [0,5]:
            with self.assertRaises(ValueError):parts(p,y,1.,length)
        for s in [0.,float('nan')]:
            with self.assertRaises(ValueError):parts(p,y,s)
        with self.assertRaises(ValueError):parts(p*float('nan'),y)

    def test_gradient_probe_is_observational(self):
        model=torch.nn.Linear(3,3);x=torch.randn(2,12,3);y=torch.randn_like(x)
        for arm,w in study.WEIGHTS.items():
            model.zero_grad(set_to_none=True)
            r=study.objective(model(x),y,x,arm,study.loss_configuration(None))
            expected=torch.autograd.grad(r['loss'],tuple(model.parameters()),retain_graph=True)
            rng=torch.get_rng_state().clone();g=study.gradient_comparison(model,r,w)
            self.assertTrue(all(v.grad is None for v in model.parameters()))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(g['all']['weighted_auxiliary_norm']==0,arm=='mse')
            r['loss'].backward()
            for v,e in zip(model.parameters(),expected):self.assertTrue(torch.equal(v.grad,e))

    def test_joint_policy_and_pure_mse(self):
        stale=0
        for improves,expected in [(True,0),(False,1),(False,2),(True,0),(False,1)]:
            stale=study.joint_stale_count(stale,improves);self.assertEqual(stale,expected)
        self.assertEqual(study.selection_score({'neural_mse':.4,'relation':-1}),.4)
        for arm in study.ARMS:
            self.assertEqual(study.learning_rate(arm,0,30),.001)
            self.assertEqual(study.learning_rate(arm,30,30),0)
        p=torch.randn(2,96,7,requires_grad=True);y=torch.randn_like(p)
        r=study.objective(p,y,y,'mse',study.loss_configuration(None))
        a=torch.autograd.grad(r['loss'],p,retain_graph=True)[0]
        b=torch.autograd.grad((p-y).square().mean(),p)[0]
        self.assertTrue(torch.equal(a,b))


if __name__=='__main__':unittest.main()
