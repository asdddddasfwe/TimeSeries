"""Independent nonlinear recurrence, stable cap gradients, and full H720 checks."""
import unittest
from pathlib import Path
import torch
from models.OscillatorPDMStatePatchMAFirstUnitResidualBounded import Model,rms_cap,lagged_scan


class BoundedResidualTests(unittest.TestCase):
    def test_cap_finite_large_zero_and_identity_region(self):
        zero=torch.zeros(2,8,2,requires_grad=True);rms_cap(zero).sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())
        value=torch.tensor([[[3e30,-3e30]]],dtype=torch.float32,requires_grad=True)
        capped=rms_cap(value);self.assertTrue(torch.isfinite(capped).all())
        torch.testing.assert_close(capped,torch.tensor([[[1.,-1.]]]))
        capped.sum().backward();self.assertTrue(torch.isfinite(value.grad).all())
        tiny=torch.randn(2,8,2)*.01;self.assertTrue(torch.equal(tiny,rms_cap(tiny)))

    def test_independent_equations_and_gradients(self):
        torch.set_num_threads(2)
        for lag in (1,8):
            torch.manual_seed(34)
            a=(torch.randn(2,19,3,2,dtype=torch.float64)*.3).requires_grad_()
            u=torch.randn_like(a,requires_grad=True);h=torch.randn(2,3,2,dtype=torch.float64,requires_grad=True)
            actual=lagged_scan(a,u,h,lag)
            states=[h]
            for k in range(1,20):
                p=states[-1];ak=a[:,k-1]
                v=torch.stack((ak[...,0]*p[...,0]-ak[...,1]*p[...,1],ak[...,1]*p[...,0]+ak[...,0]*p[...,1]),-1)+u[:,k-1]
                if k>=lag:v=v+states[k-lag]
                norm=v.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1.)
                states.append(v/norm)
            expected=torch.stack(states[1:],1);torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
            w=torch.randn_like(actual)
            ga=torch.autograd.grad((actual*w).sum(),(a,u,h),retain_graph=True)
            gb=torch.autograd.grad((expected*w).sum(),(a,u,h))
            for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)

    def test_long_unstable_input_is_bounded_without_detaching_feedback(self):
        a=torch.zeros(2,180,8,2);a[...,0]=.999
        a.requires_grad_();u=torch.full_like(a,.05,requires_grad=True);h=torch.ones(2,8,2,requires_grad=True)
        for lag in (1,8):
            states=lagged_scan(a,u,h,lag)
            self.assertTrue(torch.isfinite(states).all())
            self.assertLessEqual(float(states.square().mean((-2,-1)).max()),1.000001)
            g=torch.autograd.grad(states[:,-1].square().mean(),(a,u,h),retain_graph=True)
            self.assertTrue(all(torch.isfinite(v).all() for v in g))

    def test_full_h720_state_identity_finite_forward_and_all204_gradients(self):
        from tmp.run_mafirst_level_head_e2e_h192_seed2021_20261017 import config,control_reference
        torch.set_num_threads(2);cfg=config(Path(__file__).resolve().parents[1]/'dataset/ETT-small');cfg.pred_len=720;cfg.tensorboard=False
        original=control_reference(cfg)
        for lag in (1,8):
            cfg.unit_residual_lag=lag;model=Model(cfg);model.load_state_dict(original.state_dict(),strict=True)
            self.assertEqual(len(model.state_dict()),236);self.assertEqual(len(list(model.parameters())),204)
            self.assertEqual(sum(p.numel() for p in model.parameters()),320450)
            for n,v in original.state_dict().items():
                self.assertTrue(torch.equal(v,model.state_dict()[n]));self.assertNotEqual(v.data_ptr(),model.state_dict()[n].data_ptr())
            p=model(torch.randn(2,96,7));self.assertEqual(tuple(p.shape),(2,720,7));self.assertTrue(torch.isfinite(p).all())
            p.square().mean().backward();self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters()))


if __name__=='__main__':unittest.main()
