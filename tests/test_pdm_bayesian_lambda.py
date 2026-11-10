import unittest
import numpy as np
import torch
from scripts.train_timemix_pdm_bayesian_lambda import objective,loss_configuration,WEIGHTS
from utils.bayesian_relation_bregman import objective as previous_objective
from scripts.audit_pdm_bayesian_lambda import numpy_quadratic

class LambdaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.x=torch.randn(2,96,3,dtype=torch.double,requires_grad=True)
        self.y=torch.randn(2,96,3,dtype=torch.double,requires_grad=True)
        self.p=torch.randn(2,96,3,dtype=torch.double,requires_grad=True)
        self.cfg=loss_configuration(None)

    def test_numpy_reference_and_parts(self):
        d=objective(self.p,self.y,self.x,'lambda10',self.cfg)
        ref=numpy_quadratic(self.p.detach().numpy(),self.y.detach().numpy(),self.x.detach().numpy())
        self.assertAlmostEqual(float(d['relation']),ref['quadratic'],places=11)
        self.assertAlmostEqual(float(d['loss'].detach()),float(d['mse'].detach())+10*ref['quadratic'],places=10)
        self.assertGreaterEqual(ref['uncertainty_quadratic'],0)

    def test_exact_old_quadratic_and_gradient(self):
        a=objective(self.p,self.y,self.x,'lambda01',self.cfg)
        b=previous_objective(self.p,self.y,self.x,'quadratic',dict(weight=.1,order=4,prior_variance=1.,noise_variance=1.))
        self.assertTrue(torch.equal(a['loss'],b['loss']))
        self.assertTrue(torch.equal(torch.autograd.grad(a['loss'],self.p,retain_graph=True)[0],torch.autograd.grad(b['loss'],self.p)[0]))

    def test_gradient_linear_in_lambda_and_detach(self):
        grads={a:torch.autograd.grad(objective(self.p,self.y,self.x,a,self.cfg)['loss'],self.p)[0] for a in WEIGHTS}
        torch.testing.assert_close(grads['lambda10']-grads['mse'],10*(grads['lambda1']-grads['mse']),rtol=1e-10,atol=1e-10)
        objective(self.p,self.y,self.x,'lambda10',self.cfg)['loss'].backward()
        self.assertIsNone(self.x.grad);self.assertIsNone(self.y.grad);self.assertTrue(torch.isfinite(self.p.grad).all())

    def test_zero_error_and_pure_mse(self):
        d=objective(self.y,self.y,self.x,'lambda10',self.cfg)
        self.assertEqual(float(d['loss'].detach()),0.)
        a=objective(self.p,self.y,self.x,'mse',self.cfg)
        self.assertTrue(torch.equal(a['loss'],(self.p-self.y.detach()).square().mean()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_fp32_same_old_quadratic(self):
        x,y,p=[v.detach().float().cuda() for v in (self.x,self.y,self.p)];p.requires_grad_()
        for arm,weight in WEIGHTS.items():
            a=objective(p,y,x,arm,self.cfg)
            b=previous_objective(p,y,x,'quadratic',dict(weight=weight,order=4))
            self.assertTrue(torch.equal(a['loss'],b['loss']))
            self.assertTrue(torch.isfinite(torch.autograd.grad(a['loss'],p)[0]).all())

if __name__=='__main__':unittest.main()
