import unittest
import numpy as np
import torch
from utils.predictable_autocov_loss import sample_autocovariance, expected_centered_autocovariance, predictable_autocov_loss


class PredictableAutocovTests(unittest.TestCase):
    def test_numpy_non_circular_reference(self):
        rng=np.random.default_rng(2);z=rng.normal(size=(3,12,2));lags=(0,1,4,11)
        c=z-z.mean(1,keepdims=True)
        expected=np.stack([(c[:,:12-k]*c[:,k:]).sum(1)/12 for k in lags],-1)
        np.testing.assert_allclose(sample_autocovariance(torch.tensor(z),lags),expected,atol=1e-14)

    def test_full_covariance_centering(self):
        rng=np.random.default_rng(3);a=rng.normal(size=(2,12,12));s=a@a.transpose(0,2,1);p=np.eye(12)-np.ones((12,12))/12
        centered=p@s@p;lags=(0,1,6,11)
        expected=np.stack([np.diagonal(centered,offset=k,axis1=-2,axis2=-1).sum(-1)/12 for k in lags],-1)
        np.testing.assert_allclose(expected_centered_autocovariance(torch.tensor(s),lags),expected,atol=1e-13)
        # Centering even white noise induces negative expected sample nonzero-lag covariance.
        white=expected_centered_autocovariance(torch.eye(12,dtype=torch.float64),(0,1))
        np.testing.assert_allclose(white,[11/12,-11/144],atol=1e-14)

    def test_gradient_and_detach(self):
        torch.manual_seed(1);p=torch.randn(2,8,2,dtype=torch.float64,requires_grad=True)
        y=torch.randn_like(p,requires_grad=True);n=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
        scale=torch.ones(2,dtype=torch.float64,requires_grad=True);w=torch.ones(3,dtype=torch.float64,requires_grad=True)
        fn=lambda p:predictable_autocov_loss(p,y,n,scale,w,lags=(0,1,4))['structure']
        self.assertTrue(torch.autograd.gradcheck(fn,(p,),atol=1e-5))
        fn(p).backward();self.assertIsNotNone(p.grad)
        for t in (y,n,scale,w):self.assertIsNone(t.grad)

    def test_flat_prediction_and_signed_target(self):
        p=torch.zeros(2,8,1,requires_grad=True);y=torch.arange(8.).reshape(1,8,1).expand(2,-1,-1)
        out=predictable_autocov_loss(p,y,torch.tensor([[100.,-2.]]),torch.ones(1),lags=(0,2))
        out['structure'].backward(retain_graph=True);self.assertEqual(p.grad.abs().sum().item(),0)
        p.grad.zero_();out['mse'].backward();self.assertGreater(p.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(out['structure']))
        zero=predictable_autocov_loss(p,y,torch.zeros(1,2),torch.ones(1),torch.zeros(2),lags=(0,2))
        self.assertEqual(zero['structure'].item(),0)

    def test_zero_correction_formula(self):
        torch.manual_seed(8);p=torch.randn(3,8,2,dtype=torch.float64);y=torch.randn_like(p);v=torch.tensor([.5,2.])
        cp=sample_autocovariance(p,(0,1,4));cy=sample_autocovariance(y,(0,1,4))
        expected=((cp-cy)/v[None,:,None]).square().mean()
        actual=predictable_autocov_loss(p,y,torch.zeros(2,3),v,lags=(0,1,4))['structure']
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_invalid_inputs_and_cuda_parity(self):
        p=torch.zeros(2,8,1)
        with self.assertRaises(ValueError):sample_autocovariance(p,(8,))
        with self.assertRaises(ValueError):predictable_autocov_loss(p,p,torch.zeros(1,2),torch.zeros(1),lags=(0,1))
        if torch.cuda.is_available():
            torch.manual_seed(12);cpu=torch.randn(4,96,2,requires_grad=True);y=torch.randn_like(cpu);n=torch.randn(2,9);v=torch.tensor([.5,2.])
            a=predictable_autocov_loss(cpu,y,n,v)['structure'];a.backward()
            gpu=cpu.detach().cuda().requires_grad_();b=predictable_autocov_loss(gpu,y.cuda(),n.cuda(),v.cuda())['structure'];b.backward()
            torch.testing.assert_close(a,b.cpu(),rtol=3e-6,atol=3e-6)
            torch.testing.assert_close(cpu.grad,gpu.grad.cpu(),rtol=3e-5,atol=2e-6)

if __name__=='__main__':unittest.main()
