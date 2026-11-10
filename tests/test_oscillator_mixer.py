"""Parallel scan identities and complete four-scale forecasting contracts."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from layers.OscillatorMixer import (complex_product, parallel_oscillator_scan,
                                    OscillatorDecoder)
from models.OscillatorMixer import Model


def config(**kwargs):
    values = dict(task_name="long_term_forecast", seq_len=16, pred_len=11,
                  enc_in=2, c_out=2, features="M", moving_avg=5, d_model=4,
                  d_ff=8, e_layers=1, dropout=0., top_k=2, ssm_modes=3)
    values.update(kwargs)
    return SimpleNamespace(**values)


def serial_scan(a, b, initial):
    current, outputs = initial, []
    for h in range(b.shape[1]):
        current = complex_product(a[:, h], current) + b[:, h]
        outputs.append(current)
    return torch.stack(outputs, dim=1)


class OscillatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_parallel_values_and_gradients_equal_serial_for_odd_and_even_lengths(self):
        torch.manual_seed(11)
        for length in (1, 2, 3, 5, 12, 17, 96):
            a = (torch.randn(2, length, 3, 2, dtype=torch.double) * .15).requires_grad_()
            b = torch.randn_like(a, requires_grad=True)
            initial = torch.randn(2, 3, 2, dtype=torch.double, requires_grad=True)
            actual = parallel_oscillator_scan(a, b, initial)
            expected = serial_scan(a, b, initial)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            weights = torch.randn_like(actual)
            left = torch.autograd.grad((actual * weights).sum(), (a, b, initial), retain_graph=True)
            right = torch.autograd.grad((expected * weights).sum(), (a, b, initial))
            for x, y in zip(left, right):
                torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-11)

    def test_rotation_decay_and_same_layer_long_range_propagation(self):
        radius, omega, horizon = .9, .4, 19
        a = torch.tensor([radius * torch.cos(torch.tensor(omega)),
                          radius * torch.sin(torch.tensor(omega))]).reshape(1, 1, 1, 2)
        initial = torch.tensor([[[1., 0.]]])
        states = parallel_oscillator_scan(a.expand(1, horizon, 1, 2),
                                          torch.zeros(1, horizon, 1, 2), initial)
        t = torch.arange(1, horizon+1)
        expected = radius**t * torch.cos(omega*t)
        torch.testing.assert_close(states[0, :, 0, 0], expected, atol=1e-6, rtol=1e-5)
        drive = torch.zeros(1, horizon, 1, 2, requires_grad=True)
        states = parallel_oscillator_scan(a.expand(1, horizon, 1, 2), drive, initial * 0)
        gradient, = torch.autograd.grad(states[0, -1, 0, 0], drive)
        self.assertGreater(gradient[0, 0].abs().sum().item(), 0.)

    def test_no_future_drive_leakage(self):
        a = torch.tensor([.8, .1]).reshape(1, 1, 1, 2).expand(1, 13, 1, 2)
        drive = torch.randn(1, 13, 1, 2, requires_grad=True)
        state = parallel_oscillator_scan(a, drive, torch.zeros(1, 1, 2))
        gradient, = torch.autograd.grad(state[:, 4].square().sum(), drive)
        self.assertTrue((gradient[:, 5:] == 0).all())
        changed = drive.detach().clone()
        changed[:, 5:] += 100
        torch.testing.assert_close(parallel_oscillator_scan(a, changed, torch.zeros(1, 1, 2))[:, :5],
                                   state[:, :5], atol=0, rtol=0)

    def test_half_scan_accumulates_in_float32_and_long_horizon_gradients(self):
        a = torch.tensor([.99, .005]).reshape(1, 1, 1, 2).expand(2, 720, 3, 2).half()
        drive = torch.randn(2, 720, 3, 2).half().requires_grad_()
        initial = torch.zeros(2, 3, 2).half()
        output = parallel_oscillator_scan(a, drive, initial)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, serial_scan(a.float(), drive.float(), initial.float()),
                                   atol=2e-4, rtol=2e-4)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(drive.grad).all())

    def test_input_conditioned_stable_poles_and_state_only_readout(self):
        decoder = OscillatorDecoder(16, 11, 4, 3, 0.).eval()
        history, future = torch.randn(3, 16, 4), torch.randn(3, 11, 4)
        transition, initial, radius, frequency = decoder.dynamics(history)
        self.assertTrue(((radius > 0) & (radius < 1)).all())
        self.assertTrue(((frequency > 0) & (frequency < torch.pi)).all())
        self.assertGreater((transition - decoder.dynamics(history.flip(1))[0]).abs().max().item(), 0.)
        actual = decoder(history, future)
        no_memory = decoder(history, future, "no_memory")
        self.assertGreater((actual - no_memory).abs().max().item(), 1e-6)
        with patch.object(decoder, "states", return_value=torch.zeros(3, 11, 3, 2)):
            torch.testing.assert_close(decoder(history, future), torch.zeros(3, 11), atol=0, rtol=0)

    def test_four_scales_and_unchanged_trend_and_fusion_initialization(self):
        model = Model(config()).eval()
        self.assertEqual(model.lengths, [16, 8, 4, 2])
        self.assertEqual(model.horizons, [11, 6, 3, 2])
        self.assertEqual(len(model.seasonals), 4)
        for length, head in zip(model.lengths, model.trend_heads):
            torch.testing.assert_close(head.weight, torch.full_like(head.weight, 1/length))
            torch.testing.assert_close(head.bias, torch.zeros_like(head.bias))
        torch.testing.assert_close(model.scale_fusion.weight, torch.full((1, 4), .25))
        self.assertFalse(any("message_blocks" in name or "lag_embedding" in name
                             for name, _ in model.named_parameters()))

    def test_batch_channel_independence_ms_single_inverse_and_ignored_future(self):
        model = Model(config()).eval()
        x = torch.randn(3, 16, 2)
        y = model(x)
        self.assertEqual(y.shape, (3, 11, 2))
        torch.testing.assert_close(y, torch.cat([model(v[None]) for v in x]), atol=1e-6, rtol=1e-5)
        changed = x.clone()
        changed[..., 0] += torch.randn(3, 16) * 5
        torch.testing.assert_close(model(changed)[..., 1], y[..., 1], atol=1e-6, rtol=1e-5)
        nan = torch.full((3, 11, 2), float("nan"))
        torch.testing.assert_close(model(x, nan, nan, nan), y, atol=0, rtol=0)
        ms = Model(config(features="MS", c_out=1)).eval()
        ms.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(ms(x), y[..., -1:], atol=0, rtol=0)
        observed = []
        original = model.normalize.forward
        def record(x, mode):
            observed.append(mode)
            return original(x, mode)
        with patch.object(model.normalize, "forward", side_effect=record):
            model(x)
        self.assertEqual(observed, ["norm", "denorm"])

    def test_finite_gradients_and_all_dynamics_scales_update(self):
        model = Model(config()).train()
        groups = model.optimizer_parameter_groups(1e-4)
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        self.assertEqual(groups[-1]["weight_decay"], 0.)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(groups, lr=.001)
        loss = (model(torch.randn(3, 16, 2)) - torch.randn(3, 11, 2)).square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        for name, p in model.named_parameters():
            if any(key in name for key in ("decoder.log_decay", "decoder.frequency_logits",
                                          "decoder.initial_state.weight", "decoder.drive_projection.weight",
                                          "decoder.readout.weight", "trend_heads")):
                self.assertFalse(torch.equal(before[name], p), name)

    def test_checkpoint_roundtrip_constant_s_and_long_forecast(self):
        for horizon in (1, 193, 720):
            cfg = config(pred_len=horizon, enc_in=1, c_out=1, features="S",
                         task_name="short_term_forecast")
            model = Model(cfg).eval()
            restored = Model(cfg).eval()
            restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
            x = torch.full((2, 16, 1), 2.)
            output = model(x)
            self.assertEqual(output.shape, (2, horizon, 1))
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(restored(x), output, atol=0, rtol=0)

    def test_cpu_autocast_period_fusion_and_state_scan(self):
        model = Model(config()).train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prediction = model(torch.randn(2, 16, 2))
            loss = prediction.float().square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_invalid_config_and_interventions(self):
        for kwargs in (dict(ssm_modes=0), dict(ssm_dynamics_weight_decay=-1.),
                       dict(ssm_dynamics_weight_decay=float("nan")), dict(seq_len=17),
                       dict(c_out=3), dict(moving_avg=4)):
            with self.assertRaises(ValueError):
                Model(config(**kwargs))
        model = Model(config())
        with self.assertRaises(ValueError):
            model.forecast(torch.randn(2, 16, 2), zero_acf=True)
        with self.assertRaises(ValueError):
            model.forecast(torch.randn(2, 16, 2), intervention="unknown")


if __name__ == "__main__":
    unittest.main()
