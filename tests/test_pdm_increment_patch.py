"""Formula, isolation and optimizer instrumentation for full increment study."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from utils.increment_patch_loss import parts
from scripts.audit_pdm_increment_patch import numpy_increment_parts
from scripts import train_timemix_pdm_increment_patch as study


class IncrementPatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def check_formula(self, device):
        p=torch.randn(2,12,3,dtype=torch.float64,device=device,requires_grad=True)
        y=torch.randn_like(p,requires_grad=True);h=torch.randn_like(p,requires_grad=True)
        cfg=study.loss_configuration(None)
        rng=torch.get_rng_state().clone()
        crng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        q,k,_=parts(p,y,study.BANDWIDTH)
        ref=numpy_increment_parts(p.detach().cpu().numpy(),y.detach().cpu().numpy(),study.BANDWIDTH)
        self.assertAlmostEqual(float(q),ref['increment_quadratic'],places=11)
        self.assertAlmostEqual(float(k),ref['increment_kernel'],places=11)
        for arm,w in study.WEIGHTS.items():
            r=study.objective(p,y,h,arm,cfg)
            expected=(p-y.detach()).square().mean()
            if w:expected=expected+w*(k if arm=='increment_kernel' else q)
            self.assertTrue(torch.equal(r['loss'],expected))
            gp,gy,gh=torch.autograd.grad(r['loss'],(p,y,h),retain_graph=True,allow_unused=True)
            self.assertIsNone(gy);self.assertIsNone(gh);self.assertTrue(torch.isfinite(gp).all())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        if crng is not None:self.assertTrue(torch.equal(crng,torch.cuda.get_rng_state()))

    def test_cpu_formula(self):self.check_formula('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_formula(self):self.check_formula('cuda')

    def test_gradcheck_and_level_invariance(self):
        p=torch.randn(1,7,2,dtype=torch.float64,requires_grad=True)
        y=torch.randn_like(p);offset=torch.randn(1,1,2,dtype=torch.float64)
        for j in (0,1):
            self.assertTrue(torch.autograd.gradcheck(lambda z:parts(z,y,.7)[j],(p,)))
            torch.testing.assert_close(parts(p,y,.7)[j],parts(p+offset,y,.7)[j],atol=1e-14,rtol=1e-14)
            self.assertEqual(float(parts(y,y,.7)[j]),0)
            self.assertLess(float(parts((p+y)/2,y,.7)[j]),float(parts(p,y,.7)[j]))

    def test_matches_frozen_trial(self):
        path=Path(__file__).resolve().parents[1]/'tmp/pdm_increment_patch_trial_20261010/loss.py'
        spec=importlib.util.spec_from_file_location('old_patch',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        p=torch.randn(2,96,7,requires_grad=True);y=torch.randn_like(p)
        for a,b in zip(parts(p,y,study.BANDWIDTH),mod.parts(p,y,study.BANDWIDTH)):
            self.assertTrue(torch.equal(a,b))

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

    def test_joint_policy(self):
        stale=0
        for improves,expected in [(True,0),(False,1),(False,2),(True,0),(False,1)]:
            stale=study.joint_stale_count(stale,improves);self.assertEqual(stale,expected)
        self.assertEqual(study.selection_score({'neural_mse':.4,'relation':-1}),.4)
        for arm in study.ARMS:
            self.assertEqual(study.learning_rate(arm,0,30),.001)
            self.assertEqual(study.learning_rate(arm,30,30),0)


if __name__=='__main__':unittest.main()
