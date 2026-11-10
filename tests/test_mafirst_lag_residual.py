"""Delay boundary, independent recurrence/adjoint, and full-model endpoints."""
import unittest
from pathlib import Path
import torch
from models.OscillatorPDMStatePatchMAFirstLagResidual import lagged_scan, Model
from models.OscillatorPDMStatePatchMAFirstTemporalResidual import residual_transition
from layers.OscillatorTimeMix import complex_oscillator_scan


class LagResidualScanTests(unittest.TestCase):
    def test_recurrence_and_all_gradients(self):
        torch.set_num_threads(2)
        for lag in (1, 8, 24, 50):
            torch.manual_seed(73)
            a = (torch.randn(2, 49, 3, 2, dtype=torch.float64) * .2).requires_grad_()
            u = torch.randn_like(a, requires_grad=True)
            h0 = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
            output = lagged_scan(a, u, h0, lag, .5)
            states = [h0]
            for k in range(1, 50):
                prev, ak = states[-1], a[:, k-1]
                real = ak[..., 0]*prev[..., 0] - ak[..., 1]*prev[..., 1]
                imag = ak[..., 1]*prev[..., 0] + ak[..., 0]*prev[..., 1]
                rotated = torch.stack((real, imag), -1)
                value = rotated + u[:, k-1] if k < lag else .5*rotated + .5*states[k-lag] + u[:, k-1]
                states.append(value)
            expected = torch.stack(states[1:], 1)
            torch.testing.assert_close(output, expected, atol=1e-12, rtol=1e-12)
            weights = torch.randn_like(output)
            actual_grad = torch.autograd.grad((output*weights).sum(), (a,u,h0), retain_graph=True)
            reference_grad = torch.autograd.grad((expected*weights).sum(), (a,u,h0))
            for actual, reference in zip(actual_grad, reference_grad):
                torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)

    def test_no_shortcut_before_lag_and_exact_baseline_endpoint(self):
        torch.manual_seed(29)
        a = torch.randn(1, 49, 2, 2, dtype=torch.float64)*.2
        u, h = torch.randn_like(a), torch.randn(1,2,2,dtype=torch.float64)
        base = complex_oscillator_scan(a,u,h)
        torch.testing.assert_close(lagged_scan(a,u,h,24)[:,:23], base[:,:23], atol=1e-15, rtol=1e-15)
        self.assertTrue(torch.equal(lagged_scan(a,u,h,24,1), base))
        self.assertTrue(torch.equal(lagged_scan(a,u,h,50,.5), base))

    def test_lag1_matches_adjacent_residual(self):
        torch.manual_seed(44)
        a = torch.randn(2,31,3,2,dtype=torch.float64)*.2
        u,h = torch.randn_like(a),torch.randn(2,3,2,dtype=torch.float64)
        torch.testing.assert_close(lagged_scan(a,u,h,1,.5),
                                   complex_oscillator_scan(residual_transition(a,.5),u,h),
                                   atol=1e-12,rtol=1e-12)

    def test_invalid_parameters(self):
        a,u,h = torch.zeros(1,2,1,2),torch.zeros(1,2,1,2),torch.zeros(1,1,2)
        for lag in (0,-1,1.5,True):
            with self.assertRaises(ValueError):
                lagged_scan(a,u,h,lag)
        for mix in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):
                lagged_scan(a,u,h,1,mix)


class LagResidualModelTests(unittest.TestCase):
    def test_model_shape_finite_gradients_initialization_and_endpoint(self):
        # Run only after predecessor completion and resource admission.
        from tmp.run_mafirst_level_head_e2e_h192_seed2021_20261017 import config,control_reference
        torch.set_num_threads(2)
        cfg=config(Path(__file__).resolve().parents[1]/'dataset/ETT-small')
        cfg.pred_len,cfg.tensorboard=720,False
        original=control_reference(cfg).eval()
        cfg.future_residual_lag,cfg.future_residual_mix=8,1.
        endpoint=Model(cfg).eval();endpoint.load_state_dict(original.state_dict(),strict=True)
        cfg.future_residual_mix=.5
        model=Model(cfg).eval();model.load_state_dict(original.state_dict(),strict=True)
        self.assertEqual(len(model.state_dict()),236)
        self.assertEqual(len(list(model.parameters())),204)
        self.assertEqual(sum(p.numel() for p in model.parameters()),320450)
        for name,value in original.state_dict().items():
            self.assertTrue(torch.equal(value,model.state_dict()[name]))
            self.assertNotEqual(value.data_ptr(),model.state_dict()[name].data_ptr())
        x=torch.randn(2,96,7)
        with torch.no_grad():
            p=original(x);self.assertTrue(torch.equal(p,endpoint(x)))
            out=model(x);self.assertEqual(tuple(out.shape),(2,720,7))
            self.assertTrue(torch.isfinite(out).all());self.assertFalse(torch.equal(p,out))
        model.train();model(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
