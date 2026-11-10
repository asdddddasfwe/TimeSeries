import unittest
import torch
from torch.distributions import MultivariateNormal, kl_divergence
from utils.joint_gaussian_autocov_loss import DEFAULT_OFFSETS, joint_gaussian_autocov_kl_loss as loss
from utils.gaussian_autocov_loss import gaussian_autocov_kl_loss as future_loss


class JointGaussianCovarianceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(2021)
        self.x=torch.randn(2,96,3,dtype=torch.float64)
        self.y=torch.randn_like(self.x)
        self.p=.7*self.y+.2*torch.randn_like(self.y)

    def test_matches_independent_lag_covariance_gaussian_kl(self):
        truth=torch.cat([self.x,self.y],1); pred=torch.cat([self.x,self.p],1)
        truth=truth-truth.mean(1,keepdim=True); pred=pred-pred.mean(1,keepdim=True)
        scale=truth.square().mean(1)+.01
        def cov(z):
            def lag(k): return (z[:,k:]*z[:,:192-k]).sum(1)/192/scale
            return torch.stack([torch.stack([lag(abs(a-b)) for b in DEFAULT_OFFSETS],-1) for a in DEFAULT_OFFSETS],-2)+.01*torch.eye(9)
        zero=torch.zeros(2,3,9,dtype=torch.float64)
        expected=kl_divergence(MultivariateNormal(zero,cov(truth)),MultivariateNormal(zero,cov(pred)))/9
        torch.testing.assert_close(loss(self.x,self.p,self.y,reduction='none'),expected,atol=1e-10,rtol=1e-8)

    def test_lag96_contains_only_cross_boundary_pairs(self):
        z=torch.cat([self.x,self.y],1); z=z-z.mean(1,keepdim=True)
        expected=(z[:,:96]*z[:,96:]).sum(1)/192
        from utils.gaussian_autocov_loss import _autocovariance_matrix
        actual=_autocovariance_matrix(z,DEFAULT_OFFSETS)
        torch.testing.assert_close(actual[:,:,0,-1],expected)
        self.assertGreater(torch.linalg.eigvalsh(actual).min().item(),-1e-10)

    def test_shared_history_breaks_future_sign_and_offset_ambiguity(self):
        t=torch.arange(192,dtype=torch.float64)
        z=torch.sin(2*torch.pi*t/24)[None,:,None]; x,y=z[:,:96],z[:,96:]
        self.assertLess(future_loss(-y,y).item(),1e-12)
        self.assertGreater(loss(x,-y,y).item(),.01)
        self.assertGreater(loss(x,y+2,y).item(),.01)

    def test_identical_forecast_zero_loss_gradient(self):
        p=self.y.clone().requires_grad_(); v=loss(self.x,p,self.y); v.backward()
        self.assertLess(abs(v.item()),1e-12); self.assertLess(p.grad.abs().max().item(),1e-12)

    def test_only_future_prediction_receives_gradient(self):
        x=self.x.clone().requires_grad_(); y=self.y.clone().requires_grad_(); p=self.p.clone().requires_grad_()
        loss(x,p,y).backward(); self.assertIsNone(x.grad); self.assertIsNone(y.grad)
        self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(p.grad.norm().item(),0)

    def test_gradcheck(self):
        x=self.x[:1,:5,:1]; y=self.y[:1,:5,:1]; p=self.p[:1,:5,:1].clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda a:loss(x,a,y,offsets=(0,1,3,5)),(p,)))

    def test_flat_inputs_and_mixed_precision_finite(self):
        for x,p,y in [(self.x,torch.ones_like(self.p),self.y),(self.x,self.p,torch.ones_like(self.y)),(self.x*0,self.p*0,self.y*0)]:
            p=p.clone().requires_grad_(); v=loss(x,p,y); v.backward(); self.assertTrue(torch.isfinite(v)); self.assertTrue(torch.isfinite(p.grad).all())
        for dtype in [torch.float16,torch.bfloat16]:
            p=self.p.to(dtype).requires_grad_(); v=loss(self.x.to(dtype),p,self.y.to(dtype)); v.backward()
            self.assertEqual(v.dtype,torch.float32); self.assertTrue(torch.isfinite(p.grad).all())

    def test_invalid_history_and_forecast(self):
        for x in [self.x[0],self.x[:,:0],self.x[:,:,:2],self.x.long(),self.x*float('nan')]:
            with self.assertRaises(ValueError): loss(x,self.p,self.y)
        with self.assertRaises(ValueError): loss(self.x,self.p.long(),self.y)
        with self.assertRaises(ValueError): loss(self.x,self.p,self.y[:,:48])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_gpu_full_shape(self):
        x=torch.randn(32,96,7,device='cuda'); y=torch.randn_like(x); p=torch.randn_like(x,requires_grad=True)
        with torch.autocast('cuda',dtype=torch.float16): value=(p-y).square().mean()+.001*loss(x,p,y)
        value.backward(); self.assertTrue(torch.isfinite(value)); self.assertTrue(torch.isfinite(p.grad).all())


if __name__=='__main__': unittest.main()
