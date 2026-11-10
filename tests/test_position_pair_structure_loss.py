import unittest
import numpy as np
import torch
from utils.position_pair_structure_loss import position_pair_structure_loss as loss

def reference(x,p,y,v,q,fl=(0,4,8,12,24,48),bl=(4,8,12,24,48,96)):
    def center(z):
        z=z.reshape(z.shape[0],z.shape[1]//q,q,z.shape[2]).mean(2)
        return z-z.mean(1,keepdims=True)
    x,p,y=map(center,(x,p,y));ff=[];hf=[]
    def huber(e):return np.where(abs(e)<1,.5*e**2,abs(e)-.5)
    for b in range(len(x)):
        for c in range(x.shape[-1]):
            for lag in fl:
                k=lag//q
                for h in range(p.shape[1]-k):ff.append(huber((p[b,h,c]*p[b,h+k,c]-y[b,h,c]*y[b,h+k,c])/v[c]))
            for lag in bl:
                k=lag//q
                for h in range(k):hf.append(huber(x[b,x.shape[1]-k+h,c]*(p[b,h,c]-y[b,h,c])/v[c]))
    return dict(covariance=float(np.mean(ff)),anchor=float(np.mean(hf)))

class TestPositionPairs(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77);self.x=torch.randn(2,96,2,dtype=torch.double);self.p=torch.randn_like(self.x);self.y=torch.randn_like(self.x);self.v=torch.tensor([.7,2.],dtype=torch.double)
    def test_numpy_reference(self):
        for q in [1,4]:
            actual=loss(self.x,self.p,self.y,self.v,q);expected=reference(self.x.numpy(),self.p.numpy(),self.y.numpy(),self.v.numpy(),q)
            for k in actual:self.assertAlmostEqual(float(actual[k]),expected[k],places=12)
    def test_gradcheck_detach(self):
        for q in [1,4]:
            x=torch.randn(1,8,1,dtype=torch.double,requires_grad=True);p=torch.randn_like(x,requires_grad=True);y=torch.randn_like(x,requires_grad=True);v=torch.ones(1,dtype=torch.double,requires_grad=True)
            fn=lambda z:sum(loss(x,z,y,v,q,(0,4),(4,8)).values())
            self.assertTrue(torch.autograd.gradcheck(fn,(p,)))
            fn(p).backward();self.assertIsNone(x.grad);self.assertIsNone(y.grad);self.assertIsNone(v.grad);self.assertTrue(torch.isfinite(p.grad).all())
    def test_exact_and_offset(self):
        for q in [1,4]:
            z=loss(self.x,self.y,self.y,self.v,q);self.assertEqual(float(sum(z.values())),0.)
            other=loss(self.x+2,self.p+3,self.y-1,self.v,q);base=loss(self.x,self.p,self.y,self.v,q)
            for k in base:torch.testing.assert_close(base[k],other[k],rtol=1e-12,atol=1e-12)
    def test_boundary_resolves_sign(self):
        for q in [1,4]:
            z=loss(self.x,-self.y,self.y,self.v,q)
            self.assertEqual(float(z['covariance']),0.);self.assertGreater(float(z['anchor']),0.)
    def test_pool_removes_block_zero_mean_noise(self):
        noise=torch.tensor([1.,-1.,1.,-1.],dtype=torch.double).repeat(24)[None,:,None]
        raw=loss(self.x,self.y+noise,self.y,self.v,1);pooled=loss(self.x,self.y+noise,self.y,self.v,4)
        self.assertGreater(float(sum(raw.values())),0.);self.assertLess(float(sum(pooled.values())),1e-28)
    def test_flat_and_validation(self):
        p=torch.zeros_like(self.p,requires_grad=True);z=loss(torch.zeros_like(self.x),p,self.y,self.v)
        self.assertEqual(float(z['anchor'].detach()),0.);sum(z.values()).backward();self.assertTrue(torch.isfinite(p.grad).all())
        # Exactly flat prediction can have zero FF gradient; no anti-collapse guarantee.
        self.assertEqual(float(p.grad.abs().max()),0.)
        for q in [0,5]:
            with self.assertRaises(ValueError):loss(self.x,self.p,self.y,self.v,q)
        with self.assertRaises(ValueError):loss(self.x,self.p,self.y,[-1.,1.])
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self):
        for q in [1,4]:
            p=self.p.float().requires_grad_();cpu=loss(self.x.float(),p,self.y.float(),self.v.float(),q);sum(cpu.values()).backward()
            g=self.p.float().cuda().requires_grad_();gpu=loss(self.x.float().cuda(),g,self.y.float().cuda(),self.v.float().cuda(),q);sum(gpu.values()).backward()
            for k in cpu:torch.testing.assert_close(cpu[k],gpu[k].cpu(),rtol=2e-5,atol=1e-6)
            torch.testing.assert_close(p.grad,g.grad.cpu(),rtol=2e-5,atol=1e-6)
if __name__=='__main__':unittest.main()
