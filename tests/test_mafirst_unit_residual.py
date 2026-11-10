"""Validate exact unscaled equations, not presumed numerical stability."""
import unittest
from pathlib import Path
import torch
from models.OscillatorPDMStatePatchMAFirstUnitResidual import Model,lagged_scan
from layers.OscillatorTimeMix import complex_oscillator_scan


class UnitResidualTests(unittest.TestCase):
    def test_independent_forward_and_gradient_both_lags(self):
        torch.set_num_threads(2)
        for lag in (1,8):
            torch.manual_seed(71)
            a=(torch.randn(2,25,3,2,dtype=torch.float64)*.15).requires_grad_()
            u=torch.randn_like(a,requires_grad=True);h=torch.randn(2,3,2,dtype=torch.float64,requires_grad=True)
            actual=lagged_scan(a,u,h,lag)
            states=[h]
            for k in range(1,26):
                prev=states[-1];ak=a[:,k-1]
                rotated=torch.stack((ak[...,0]*prev[...,0]-ak[...,1]*prev[...,1],ak[...,1]*prev[...,0]+ak[...,0]*prev[...,1]),-1)
                states.append(rotated+u[:,k-1]+(states[k-lag] if k>=lag else 0))
            expected=torch.stack(states[1:],1);torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
            weight=torch.randn_like(actual)
            ga=torch.autograd.grad((actual*weight).sum(),(a,u,h),retain_graph=True)
            gb=torch.autograd.grad((expected*weight).sum(),(a,u,h))
            for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)

    def test_delay_boundary_no_padding(self):
        torch.manual_seed(44);a=torch.randn(1,19,2,2,dtype=torch.float64)*.1
        u,h=torch.randn_like(a),torch.randn(1,2,2,dtype=torch.float64)
        torch.testing.assert_close(lagged_scan(a,u,h,8)[:,:7],complex_oscillator_scan(a,u,h)[:,:7],atol=1e-14,rtol=1e-14)

    def test_no_hidden_half_scaling(self):
        a=torch.zeros(1,16,1,2,dtype=torch.float64);u=torch.zeros_like(a);h=torch.ones(1,1,2,dtype=torch.float64)
        self.assertTrue(torch.equal(lagged_scan(a,u,h,1),h[:,None].expand_as(a)))
        for coefficient in (.5,0.,2.):
            with self.assertRaises(ValueError):lagged_scan(a,u,h,1,coefficient)

    def test_identical_initial_state_and_parameter_counts(self):
        from tmp.run_mafirst_level_head_e2e_h192_seed2021_20261017 import config,control_reference
        torch.set_num_threads(2);cfg=config(Path(__file__).resolve().parents[1]/'dataset/ETT-small');cfg.pred_len=720;cfg.tensorboard=False
        original=control_reference(cfg)
        for lag in (1,8):
            cfg.unit_residual_lag=lag;model=Model(cfg);model.load_state_dict(original.state_dict(),strict=True)
            self.assertEqual(len(model.state_dict()),236);self.assertEqual(len(list(model.parameters())),204)
            self.assertEqual(sum(p.numel() for p in model.parameters()),320450)
            for name,v in original.state_dict().items():
                self.assertTrue(torch.equal(v,model.state_dict()[name]));self.assertNotEqual(v.data_ptr(),model.state_dict()[name].data_ptr())


if __name__=='__main__':unittest.main()
