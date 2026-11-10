"""Numerical time-unit tests and actual H720 model checks; no formal data/test."""
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
import torch
from models.OscillatorPDMStatePatchMAFirst import Model as Original
from models.OscillatorPDMStatePatchTimeConsistent import (
    Model, compose_future_patch, load_original_scratch)

R = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class TimeConsistentTests(unittest.TestCase):
    def test_block_composition_matches_four_point_loop(self):
        torch.manual_seed(7)
        a = torch.polar(torch.rand(2, 3, dtype=torch.float64) * .3 + .65,
                        torch.rand(2, 3, dtype=torch.float64) * .25)
        pair = torch.view_as_real(a)
        native = torch.randn(2, 12, 3, 2, dtype=torch.float64)
        for stride in (1, 2, 4, 8):
            a4, block = compose_future_patch(pair, native, stride)
            z = torch.randn(2, 3, dtype=torch.complex128)
            reference = z.clone()
            total = sum(a ** j for j in range(stride))
            drive = torch.view_as_complex(native) / total[:, None, :]
            actual = z.clone()
            for k in range(3):
                for j in range(4):
                    reference = a * reference + drive[:, k * 4 + j]
                actual = torch.view_as_complex(a4) * actual + torch.view_as_complex(block)[:, k]
                torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)

    def test_known_continuous_forcing_across_all_history_strides(self):
        generator = torch.tensor([[-.001 + .01j, -.2 + .2j]], dtype=torch.complex128)
        a = generator.exp()
        forcing = torch.tensor([[.2 + .1j, -.1 + .3j]], dtype=torch.complex128)
        phi1 = torch.expm1(generator) / generator
        phi4 = torch.expm1(4 * generator) / generator
        for stride in (1, 2, 4, 8):
            phi_native = torch.expm1(stride * generator) / generator
            native = (phi_native * forcing)[:, None, :].expand(-1, 8, -1).contiguous()
            a4, b4 = compose_future_patch(
                torch.view_as_real(a), torch.view_as_real(native), stride)
            torch.testing.assert_close(torch.view_as_complex(a4), (4 * generator).exp(), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(torch.view_as_complex(b4),
                                       (phi4 * forcing)[:, None, :].expand(-1, 2, -1), rtol=1e-12, atol=1e-12)
            total = sum(a ** j for j in range(stride))
            torch.testing.assert_close(phi_native / total, phi1, rtol=1e-12, atol=1e-12)

    def test_zero_generator_has_finite_exact_limit(self):
        a = torch.tensor([[[1., 0.]]], dtype=torch.float64)
        for stride in (1, 2, 4, 8):
            native = torch.tensor([[[[.5 * stride, -.25 * stride]]]],
                                  dtype=torch.float64).expand(-1, 8, -1, -1).contiguous()
            transition, drive = compose_future_patch(a, native, stride)
            torch.testing.assert_close(transition, a, rtol=0, atol=0)
            torch.testing.assert_close(drive, torch.tensor([[[[2., -1.]], [[2., -1.]]]],
                                      dtype=torch.float64), rtol=0, atol=0)

    def test_gradcheck_transition_and_forcing(self):
        a = torch.tensor([[[.96, .04], [.91, .07]]], dtype=torch.float64, requires_grad=True)
        u = torch.randn(1, 8, 2, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x, y: compose_future_patch(x, y, 8), (a, u), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_same_initial_parameters_history_and_trend(self):
        model, cfg = load_original_scratch(R, R / "tmp/time_consistent_unit_logs")
        old = Original(SimpleNamespace(**cfg))
        old.load_state_dict(model.state_dict(), strict=True)
        model.eval(); old.eval()
        torch.manual_seed(11)
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            a, (s, t) = model.encode_multiscale(model.downsample(x))
            b, (ss, tt) = old.encode_multiscale(old.downsample(x))
            for first, second in zip(a + s + t, b + ss + tt):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            for e, ee, h, trend in zip(model.seasonals, old.seasonals, s, t):
                for first, second in zip(e.decoder.dynamics(h, trend), ee.decoder.dynamics(h, trend)):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
            new_parts = model.predict_components(x)
            old_parts = old.predict_components(x)
            torch.testing.assert_close(new_parts[..., 4:], old_parts[..., 4:], rtol=0, atol=0)
            self.assertGreater(float((new_parts[..., :4] - old_parts[..., :4]).abs().max()), 1e-6)
        self.assertEqual(set(model.state_dict()), set(old.state_dict()))
        self.assertTrue(all(torch.equal(v, old.state_dict()[k]) for k, v in model.state_dict().items()))

    def test_actual_h720_forward_backward_all_parameters_finite(self):
        model, _ = load_original_scratch(R, R / "tmp/time_consistent_unit_logs")
        model.train()
        torch.manual_seed(19)
        prediction = model(torch.randn(2, 96, 7))
        self.assertEqual(tuple(prediction.shape), (2, 720, 7))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
