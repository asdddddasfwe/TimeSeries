import unittest,math
import numpy as np
import torch
from utils.time_position_patch_mmd import patch_parts,parts,numpy_parts
from utils.patch_covariance_loss import parts as old_parts

class TimeMMDTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.manual_seed(11)
 def formula(self,device):
  p=torch.randn(2,3,7,4,dtype=torch.float64,device=device,requires_grad=True);y=torch.randn_like(p,requires_grad=True)
  d,diag=patch_parts(p,y);a=p.detach().cpu().numpy();b=y.detach().cpu().numpy();n=7
  time=np.exp(-((np.arange(n)[:,None]-np.arange(n)[None,:])**2)/8)
  pp=np.exp(-((a[..., :,None,:]-a[...,None,:,:])**2).sum(-1)/2);yy=np.exp(-((b[..., :,None,:]-b[...,None,:,:])**2).sum(-1)/2);py=np.exp(-((a[..., :,None,:]-b[...,None,:,:])**2).sum(-1)/2)
  expected=(time*(pp+yy-2*py)).mean();self.assertAlmostEqual(float(d.detach()),expected,places=12)
  grad=(-2*(time*pp)[...,None]*(a[..., :,None,:]-a[...,None,:,:])+2*(time*py)[...,None]*(a[..., :,None,:]-b[...,None,:,:])).sum(-2)/(n*n*6)
  gp,gy=torch.autograd.grad(d,[p,y],allow_unused=True);self.assertIsNone(gy)
  np.testing.assert_allclose(gp.cpu().numpy(),grad,rtol=1e-10,atol=1e-12)
  x=torch.randn(2,96,7,device=device,requires_grad=True);target=torch.randn_like(x,requires_grad=True);rng=torch.get_rng_state().clone();v,dd=parts(x,target);v.backward();self.assertIsNone(target.grad);self.assertEqual(v.dtype,torch.float32);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  independent=numpy_parts(x.detach().cpu().numpy(),target.detach().cpu().numpy())
  for key,val in independent.items():self.assertAlmostEqual(dd[key],val,places=7)
  # Identity has zero MMD and zero data gradient.
  z=x.detach().clone().requires_grad_(True);v,_=parts(z,z.detach());self.assertLess(abs(float(v.detach())),1e-12);self.assertLess(float(torch.autograd.grad(v,z)[0].abs().max()),1e-12)
 def test_explicit_value_analytic_gradient_cpu(self):self.formula('cpu')
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
 def test_explicit_value_analytic_gradient_cuda(self):self.formula('cuda')
 def test_order_level_time_limits_and_invalid_inputs(self):
  y=torch.linspace(-1,1,24).repeat_interleave(4)[None,:,None];p=y.reshape(1,24,4,1).flip(1).reshape(1,96,1)
  old,_=old_parts(p,y);new,_=parts(p,y);self.assertLess(float(old),1e-10);self.assertGreater(float(new),.01)
  bag,_=parts(p,y,sigma_time=1e9);self.assertLess(float(bag),1e-10)
  aligned,_=parts(p,y,sigma_time=1e-6);expected=2/24*(1-torch.exp(-.5*(p.reshape(24,4)-y.reshape(24,4)).square().sum(-1))).mean();torch.testing.assert_close(aligned,expected)
  p=torch.zeros_like(y,requires_grad=True);target=torch.ones_like(y);d,_=parts(p,target);self.assertGreater(float(d.detach()),.01);self.assertGreater(float(torch.autograd.grad(d,p)[0].abs().max()),0)
  for sigma in (0,-1,float('nan'),float('inf')):
   with self.assertRaises(ValueError):parts(p,target,sigma_time=sigma)
  with self.assertRaises(ValueError):parts(p,target,length=5)
  with self.assertRaises(ValueError):parts(p,torch.full_like(p,float('nan')))
if __name__=='__main__':unittest.main()
