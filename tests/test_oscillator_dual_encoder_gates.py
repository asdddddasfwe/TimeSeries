"""Gate semantics, native recurrence, common scratch tensors and gradients."""
import copy
import unittest

import torch
from torch.nn import functional as F

from test_pdm_direct_patch import cfg
from models.OscillatorDualEncoderTimeMixer import Model as Ungated, HistoryOscillator
from models.OscillatorDualEncoderGatedTimeMixer import Model, GatedHistoryOscillator
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts


class GateTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_common_initial_state_rng_and_independent_gates(self):
        base = Ungated(cfg())
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2021)
        gated = Model(cfg())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in base.parameters()), 199633)
        self.assertEqual(sum(p.numel() for p in gated.parameters()), 203857)
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, gated.state_dict()[name]), name)
        additional = set(gated.state_dict())-set(base.state_dict())
        self.assertEqual(len(additional), 32)
        self.assertTrue(all('.forget_gate.' in name or '.write_gate.' in name for name in additional))
        storages = [p.data_ptr() for oscillator in [*gated.season_oscillators, *gated.trend_oscillators]
                    for gate in (oscillator.forget_gate, oscillator.write_gate) for p in gate.parameters()]
        self.assertEqual(len(storages), len(set(storages)))
        groups = norm_bias_no_decay_groups(gated, 1e-4)
        ids = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in gated.parameters()})

    def test_formula_serial_and_gradient_reference(self):
        oscillator = GatedHistoryOscillator(HistoryOscillator(12, 8, 3, 0.)).double()
        with torch.no_grad():
            oscillator.forget_gate.weight.normal_(0, .15)
            oscillator.write_gate.weight.normal_(0, .15)
            oscillator.forget_gate.bias.fill_(1.)
            oscillator.write_gate.bias.fill_(.5)
        history = torch.randn(2, 12, 8, dtype=torch.double, requires_grad=True)
        base_a, base_b, initial = HistoryOscillator.scan_inputs(oscillator, history)
        normalized = F.layer_norm(history, (8,), oscillator.drive_norm.weight,
                                  oscillator.drive_norm.bias, oscillator.drive_norm.eps)
        forget = torch.sigmoid(F.linear(normalized, oscillator.forget_gate.weight, oscillator.forget_gate.bias))
        write = torch.sigmoid(F.linear(normalized, oscillator.write_gate.weight, oscillator.write_gate.bias))
        a, b, _ = oscillator.scan_inputs(history)
        torch.testing.assert_close(a, base_a*forget[..., None])
        torch.testing.assert_close(b, base_b*write[..., None])
        self.assertGreater(float(forget.std(1).mean()), 0.)
        self.assertTrue((a.square().sum(-1) <= base_a.square().sum(-1)+1e-14).all())
        states = []
        h = initial
        for t in range(12):
            ar, ai = base_a[:, t].unbind(-1)
            hr, hi = h.unbind(-1)
            h = torch.stack((ar*hr-ai*hi, ar*hi+ai*hr), -1)*forget[:, t, :, None]+base_b[:, t]*write[:, t, :, None]
            states.append(h)
        expected = oscillator.readout(torch.stack(states, 1).flatten(2))
        actual = oscillator(history)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
        parameters = (history, oscillator.forget_gate.weight, oscillator.write_gate.weight, oscillator.log_decay)
        left = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
        right = torch.autograd.grad(expected.square().sum(), parameters)
        for u, v in zip(left, right):
            torch.testing.assert_close(u, v, atol=1e-10, rtol=1e-9)
        # Forgetting independently clears propagated state; writing independently
        # clears the new drive. Neither changes rotation coordinates.
        with torch.no_grad():
            oscillator.forget_gate.weight.zero_(); oscillator.forget_gate.bias.fill_(-50)
            oscillator.write_gate.weight.zero_(); oscillator.write_gate.bias.fill_(50)
        a, b, _ = oscillator.scan_inputs(history)
        self.assertLess(float(a.abs().max()), 1e-20)
        torch.testing.assert_close(b, HistoryOscillator.scan_inputs(oscillator, history)[1])

    def check_full(self, device):
        torch.manual_seed(2021)
        base = Ungated(cfg()).to(device)
        torch.manual_seed(2021)
        gated = Model(cfg()).to(device)
        x, y = torch.randn(2, 96, 7, device=device), torch.randn(2, 96, 7, device=device, requires_grad=True)
        base.train(); gated.train()
        states, predictions = [], []
        for model in (base, gated):
            torch.manual_seed(77)
            if device == 'cuda':
                torch.cuda.manual_seed_all(77)
            predictions.append(model(x))
            states.append(torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
        self.assertTrue(torch.equal(*states))
        self.assertGreater(float((predictions[0]-predictions[1]).detach().abs().max()), 0.)
        gated.enable_gate_statistics(True)
        optimizer = torch.optim.Adam(norm_bias_no_decay_groups(gated, 1e-4), lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = gated(x)
            distance, _ = parts(prediction, y)
            loss = (prediction-y.detach()).square().mean()+distance
            loss.backward()
            self.assertIsNone(y.grad)
            for name, parameter in gated.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if '.forget_gate.' in name or '.write_gate.' in name:
                    self.assertGreater(float(parameter.grad.norm()), 0., name)
            for oscillator in [*gated.season_oscillators, *gated.trend_oscillators]:
                values = oscillator.last_gate_tensors
                self.assertTrue((values['forget'] >= 0).all() and (values['forget'] <= 1).all())
                self.assertTrue((values['write'] >= 0).all() and (values['write'] <= 1).all())
                self.assertTrue((values['effective_radius'] <= 1+1e-6).all())
            optimizer.step()
        gated.eval()
        gated.enable_gate_statistics(False)
        with torch.no_grad():
            reference = gated(x)
            gated.enable_gate_statistics(True)
            torch.testing.assert_close(gated(x), reference, rtol=0, atol=0)
            restored = Model(cfg()).to(device).eval()
            restored.load_state_dict(copy.deepcopy(gated.state_dict()), strict=True)
            torch.testing.assert_close(restored(x), reference, rtol=0, atol=0)

    def test_cpu_training(self):
        self.check_full('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_training(self):
        self.check_full('cuda')


if __name__ == '__main__':
    unittest.main()
