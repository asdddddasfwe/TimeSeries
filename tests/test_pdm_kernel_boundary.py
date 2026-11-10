import unittest
import numpy as np
import torch
from utils.boundary_kernel_autocovariance_loss import boundary_kernel_autocovariance_loss as loss
from scripts.audit_pdm_kernel_boundary import numpy_boundary_kernel

class BoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(37)
        self.x=torch.randn(2,10,2,dtype=torch.double,requires_grad=True)
        self.p=torch.randn(2,8,2,dtype=torch.double,requires_grad=True)
        self.y=torch.randn(2,8,2,dtype=torch.double,requires_grad=True)
        self.lags=(1,3,4)
    def test_numpy_and_gradient(self):
        z,_=loss(self.p,self.y,self.x,self.lags)
        ref=numpy_boundary_kernel(self.p.detach().numpy(),self.y.detach().numpy(),self.x.detach().numpy(),self.lags)
        self.assertAlmostEqual(float(z.detach()),ref['kernel_distance'],places=12)
        g=torch.autograd.grad(z,self.p)[0];a=self.p.detach().numpy().copy();ix=(0,0,0);eps=1e-5
        a[ix]+=eps;hi=numpy_boundary_kernel(a,self.y.detach().numpy(),self.x.detach().numpy(),self.lags)['kernel_distance']
        a[ix]-=2*eps;lo=numpy_boundary_kernel(a,self.y.detach().numpy(),self.x.detach().numpy(),self.lags)['kernel_distance']
        self.assertAlmostEqual(float(g[ix]),(hi-lo)/(2*eps),places=8)
    def test_gradcheck_and_detach(self):
        p=self.p[:1,:5,:1].detach().requires_grad_();f=lambda p:loss(p,self.y[:1,:5,:1],self.x[:1,:,:1],(1,3))[0]
        self.assertTrue(torch.autograd.gradcheck(f,(p,),atol=1e-6));f(p).backward()
        self.assertIsNone(self.y.grad);self.assertIsNone(self.x.grad)
    def test_history_tail_and_boundary_lag1(self):
        z,d=loss(self.p,self.y,self.x,(1,))
        self.assertGreater(d['distance_by_lag']['1'],0)
        x=self.x.detach().clone();x[:,:-1]+=100
        self.assertTrue(torch.equal(z,loss(self.p,self.y,x,(1,))[0]))
        x[:,-1]+=2
        self.assertGreater(abs(float(z.detach()-loss(self.p,self.y,x,(1,))[0].detach())),1e-6)
        self.assertEqual(d['pairs_per_lag'],8);self.assertEqual(d['boundary_pairs_per_lag'],[1])
    def test_identity_isolation_rng(self):
        p=self.y.detach().clone().requires_grad_();state=torch.get_rng_state().clone()
        z,_=loss(p,self.y,self.x,self.lags);self.assertLess(float(z.detach()),1e-12)
        self.assertLess(float(torch.autograd.grad(z,p)[0].abs().max()),1e-12)
        z,_=loss(self.p,self.y,self.x,self.lags)
        parts=[loss(self.p[i:i+1,:,j:j+1],self.y[i:i+1,:,j:j+1],self.x[i:i+1,:,j:j+1],self.lags)[0] for i in range(2) for j in range(2)]
        torch.testing.assert_close(z,torch.stack(parts).mean(),atol=1e-13,rtol=1e-13)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
    def test_validation_and_extreme(self):
        with self.assertRaises(ValueError):loss(self.p,self.y,self.x,(10,))
        with self.assertRaises(ValueError):loss(self.p,self.y,self.x[:,:2],(4,))
        p=(self.p.detach()*1e6).requires_grad_();z,_=loss(p,self.y,self.x,self.lags)
        self.assertTrue(torch.isfinite(torch.autograd.grad(z,p)[0]).all())
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_full_shape(self):
        x=torch.randn(2,96,7,device='cuda');y=torch.randn_like(x);p=torch.randn_like(x,requires_grad=True)
        z,_=loss(p,y,x);ref=numpy_boundary_kernel(p.detach().cpu().numpy(),y.cpu().numpy(),x.cpu().numpy())
        self.assertAlmostEqual(float(z.detach()),ref['kernel_distance'],places=8)
        z.backward();self.assertTrue(torch.isfinite(p.grad).all())
if __name__=='__main__':unittest.main()
