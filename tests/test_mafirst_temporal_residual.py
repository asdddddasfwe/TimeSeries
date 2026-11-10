"""Independent recurrence/gradient checks and matched MAFirst initialization."""
import copy
import unittest
from pathlib import Path

import torch

from layers.OscillatorTimeMix import complex_oscillator_scan
from models.OscillatorPDMStatePatchMAFirstTemporalResidual import Model, residual_transition
from tmp.run_mafirst_level_head_e2e_h192_seed2021_20261017 import config, control_reference


class TemporalResidualTests(unittest.TestCase):
    def test_invalid_mix(self):
        for mix in (-0.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                residual_transition(torch.zeros(2), mix)

    def test_spectral_bound(self):
        theta = torch.linspace(0, torch.pi, 100, dtype=torch.float64)
        radius = torch.linspace(0, 1, 101, dtype=torch.float64)[:, None]
        a = torch.stack((radius * theta.cos(), radius * theta.sin()), -1)
        for mix in (0.0, 0.1, 0.5, 1.0):
            result = residual_transition(a, mix)
            self.assertTrue((result.norm(dim=-1) <= 1 + 1e-15).all())
        self.assertIs(residual_transition(a, 1), a)

    def test_scan_matches_independent_residual_loop_and_gradient(self):
        torch.manual_seed(17)
        a = (torch.randn(2, 11, 3, 2, dtype=torch.float64) * 0.2).requires_grad_()
        u = torch.randn_like(a, requires_grad=True)
        h = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
        result = complex_oscillator_scan(residual_transition(a, 0.5), u, h)
        previous, steps = h, []
        for k in range(a.shape[1]):
            real = a[:, k, :, 0] * previous[..., 0] - a[:, k, :, 1] * previous[..., 1]
            imag = a[:, k, :, 1] * previous[..., 0] + a[:, k, :, 0] * previous[..., 1]
            previous = 0.5 * previous + 0.5 * torch.stack((real, imag), -1) + u[:, k]
            steps.append(previous)
        expected = torch.stack(steps, 1)
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
        upstream = torch.randn_like(result)
        actual_grad = torch.autograd.grad((result * upstream).sum(), (a, u, h), retain_graph=True)
        expected_grad = torch.autograd.grad((expected * upstream).sum(), (a, u, h))
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)

    def test_model_state_history_drive_and_baseline_endpoint(self):
        torch.set_num_threads(2)
        cfg = config(Path(__file__).resolve().parents[1] / 'dataset/ETT-small')
        cfg.pred_len, cfg.tensorboard = 720, False
        original = control_reference(cfg).eval()
        cfg.temporal_residual_mix = 1.0
        endpoint = Model(cfg).eval()
        endpoint.load_state_dict(original.state_dict(), strict=True)
        cfg_half = copy.copy(cfg)
        cfg_half.temporal_residual_mix = 0.5
        residual = Model(cfg_half).eval()
        residual.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(len(residual.state_dict()), 236)
        self.assertEqual(len(list(residual.parameters())), 204)
        self.assertEqual(sum(p.numel() for p in residual.parameters()), 320450)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, residual.state_dict()[key]), key)
            self.assertNotEqual(value.data_ptr(), residual.state_dict()[key].data_ptr())
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            baseline = original(x)
            self.assertTrue(torch.equal(baseline, endpoint(x)))
            changed = residual(x)
            self.assertEqual(tuple(changed.shape), (2, 720, 7))
            self.assertTrue(torch.isfinite(changed).all())
            self.assertFalse(torch.equal(baseline, changed))
            raw, (seasonal, trend) = original.encode_multiscale(original.downsample(x))
            for old_encoder, new_encoder, s, t in zip(original.seasonals, residual.seasonals, seasonal, trend):
                old = old_encoder.decoder.dynamics(s, t)
                new = new_encoder.decoder.dynamics(s, t)
                self.assertTrue(torch.equal(old[1], new[1]))
                torch.testing.assert_close(new[0], residual_transition(old[0], 0.5), rtol=0, atol=0)
                future = old_encoder.temporal_projection(s.transpose(1, 2)).transpose(1, 2)
                future = future + old_encoder.future_positions.to(future)
                future = future.reshape(future.shape[0], 180, 4, future.shape[-1]).mean(2)
                self.assertTrue(torch.equal(old_encoder.decoder.future_drive(future, s),
                                            new_encoder.decoder.future_drive(future, s)))
        residual.train()
        residual(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in residual.parameters()))


if __name__ == '__main__':
    unittest.main()
