import unittest
import numpy as np
import torch
from utils.kernel_autocovariance_loss import kernel_autocovariance_loss
from scripts.audit_pdm_kernel_autocovariance import numpy_kernel_autocovariance
from scripts.train_timemix_pdm_kernel_autocovariance import objective,loss_configuration


class KernelAutocovarianceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(31)
        self.p=torch.randn(2,12,2,dtype=torch.double,requires_grad=True)
        self.y=torch.randn(2,12,2,dtype=torch.double,requires_grad=True)
        self.lags=(1,3,6)

    def test_numpy_trace_and_gradient(self):
        value,_=kernel_autocovariance_loss(self.p,self.y,self.lags,.8)
        ref=numpy_kernel_autocovariance(self.p.detach().numpy(),self.y.detach().numpy(),self.lags,.8)
        self.assertAlmostEqual(float(value.detach()),ref['kernel_distance'],places=12)
        grad=torch.autograd.grad(value,self.p)[0]
        arr=self.p.detach().numpy().copy();ix=(0,5,1);eps=1e-5
        arr[ix]+=eps;a=numpy_kernel_autocovariance(arr,self.y.detach().numpy(),self.lags,.8)['kernel_distance']
        arr[ix]-=2*eps;b=numpy_kernel_autocovariance(arr,self.y.detach().numpy(),self.lags,.8)['kernel_distance']
        self.assertAlmostEqual(float(grad[ix]),(a-b)/(2*eps),places=8)

    def test_gradcheck_and_detach(self):
        p=self.p[:1,:6,:1].detach().requires_grad_();y=self.y[:1,:6,:1]
        f=lambda z:kernel_autocovariance_loss(z,y,(1,2),1.)[0]
        self.assertTrue(torch.autograd.gradcheck(f,(p,),eps=1e-6,atol=1e-6))
        f(p).backward();self.assertIsNone(self.y.grad);self.assertTrue(torch.isfinite(p.grad).all())

    def test_equal_constant_and_not_scalar_hsic_matching(self):
        p=self.y.detach().clone().requires_grad_();z,_=kernel_autocovariance_loss(p,self.y,self.lags)
        self.assertAlmostEqual(float(z.detach()),0,places=13)
        self.assertLess(float(torch.autograd.grad(z,p)[0].abs().max()),1e-12)
        value,d=kernel_autocovariance_loss(self.y+2,self.y,self.lags)
        np.testing.assert_allclose(d['prediction_operator_norm2'],d['target_operator_norm2'],atol=1e-14)
        self.assertGreater(float(value.detach()),1e-4) # Scalar HSIC matching would be zero.
        a=torch.ones_like(self.y);b=4*a
        self.assertAlmostEqual(float(kernel_autocovariance_loss(a,b,self.lags)[0]),0,places=13)
        # Centered covariance ignores constant level: MSE is necessary.

    def test_window_channel_isolation_and_rng(self):
        before=torch.get_rng_state().clone();value,_=kernel_autocovariance_loss(self.p,self.y,self.lags)
        pieces=[kernel_autocovariance_loss(self.p[i:i+1,:,j:j+1],self.y[i:i+1,:,j:j+1],self.lags)[0] for i in range(2) for j in range(2)]
        torch.testing.assert_close(value,torch.stack(pieces).mean(),atol=1e-13,rtol=1e-13)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for lags,bandwidth in [((0,),1),((12,),1),((1,1),1),((1,),0)]:
            with self.assertRaises(ValueError):kernel_autocovariance_loss(self.p,self.y,lags,bandwidth)

    def test_objective_mse_and_extremes(self):
        p=torch.randn(2,96,7,requires_grad=True);y=torch.randn_like(p,requires_grad=True);x=torch.randn_like(p,requires_grad=True);cfg=loss_configuration(None)
        a=objective(p,y,x,'mse',cfg);self.assertTrue(torch.equal(a['loss'],(p-y.detach()).square().mean()))
        b=objective(p,y,x,'kernel',cfg);torch.testing.assert_close(b['loss'],b['mse']+b['relation'])
        b['loss'].backward();self.assertIsNone(x.grad);self.assertIsNone(y.grad);self.assertTrue(torch.isfinite(p.grad).all())
        z=(self.p.detach()*1e6).requires_grad_();value,_=kernel_autocovariance_loss(z,self.y,self.lags)
        self.assertTrue(torch.isfinite(torch.autograd.grad(value,z)[0]).all())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_fp32_full_window(self):
        p=torch.randn(2,96,7,device='cuda',requires_grad=True);y=torch.randn_like(p)
        value,_=kernel_autocovariance_loss(p,y)
        ref=numpy_kernel_autocovariance(p.detach().cpu().numpy(),y.cpu().numpy())
        self.assertAlmostEqual(float(value.detach()),ref['kernel_distance'],places=8)
        value.backward();self.assertTrue(torch.isfinite(p.grad).all())

if __name__=='__main__':unittest.main()
