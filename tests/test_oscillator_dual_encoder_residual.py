"""Residual identity, serial recurrence/gradient, shared initialization and training."""
import copy
import unittest

import torch

from test_pdm_direct_patch import cfg
from models.OscillatorDualEncoderTimeMixer import HistoryOscillator, Model as Control
from models.OscillatorDualEncoderResidualTimeMixer import ResidualHistoryOscillator, Model
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts


class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_same_state_rng_and_optimizer(self):
        control = Control(cfg())
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2021)
        residual = Model(cfg())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(control.state_dict()), set(residual.state_dict()))
        self.assertEqual(sum(p.numel() for p in residual.parameters()), 199633)
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, residual.state_dict()[name]), name)
            self.assertNotEqual(value.data_ptr(), residual.state_dict()[name].data_ptr(), name)
        groups = norm_bias_no_decay_groups(residual, 1e-4)
        ids = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in residual.parameters()})
        self.assertFalse(any('gate' in n or 'trend_condition' in n for n in residual.state_dict()))

    def test_zero_readout_preserves_input_and_gradient(self):
        oscillator = ResidualHistoryOscillator(HistoryOscillator(12, 8, 3, 0.)).double()
        with torch.no_grad():
            oscillator.readout[-1].weight.zero_()
            oscillator.readout[-1].bias.zero_()
        x = torch.randn(2, 12, 8, dtype=torch.double, requires_grad=True)
        output = oscillator(x)
        torch.testing.assert_close(output, x, rtol=0, atol=0)
        gradient, = torch.autograd.grad(output.sum(), x)
        torch.testing.assert_close(gradient, torch.ones_like(x), rtol=0, atol=0)

    def test_serial_values_and_gradient(self):
        model = ResidualHistoryOscillator(HistoryOscillator(12, 8, 3, 0.)).double()
        x = torch.randn(2, 12, 8, dtype=torch.double, requires_grad=True)
        a, b, h = model.scan_inputs(x)
        sequence = []
        for t in range(12):
            ar, ai = a[:, t].unbind(-1)
            hr, hi = h.unbind(-1)
            h = torch.stack((ar*hr-ai*hi, ar*hi+ai*hr), -1)+b[:, t]
            sequence.append(h)
        expected = x + model.readout(torch.stack(sequence, 1).flatten(2))
        actual = model(x)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        variables = (x, model.log_decay, model.frequency_logits, model.drive_projection.weight)
        left = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
        right = torch.autograd.grad(expected.square().sum(), variables)
        for u, v in zip(left, right):
            torch.testing.assert_close(u, v, rtol=1e-9, atol=1e-10)

    def check_full(self, device):
        torch.manual_seed(2021)
        control = Control(cfg()).to(device)
        torch.manual_seed(2021)
        model = Model(cfg()).to(device)
        x = torch.randn(2, 96, 7, device=device)
        target = torch.randn_like(x, requires_grad=True)
        predictions, rngs = [], []
        for instance in (control, model):
            torch.manual_seed(71)
            predictions.append(instance(x))
            rngs.append(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state())
        self.assertTrue(torch.equal(*rngs))
        self.assertGreater(float((predictions[0]-predictions[1]).detach().abs().max()), 0.)
        optimizer = torch.optim.Adam(norm_bias_no_decay_groups(model, 1e-4), lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            self.assertEqual(prediction.shape, x.shape)
            distance, _ = parts(prediction, target)
            ((prediction-target.detach()).square().mean()+distance).backward()
            self.assertIsNone(target.grad)
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for modules in (model.season_oscillators, model.trend_oscillators):
                for module in modules:
                    self.assertGreater(float(module.drive_projection.weight.grad.norm()), 0.)
                    self.assertGreater(float(module.readout[-1].weight.grad.norm()), 0.)
            optimizer.step()
        model.eval()
        restored = Model(cfg()).to(device).eval()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_cpu_training(self):
        self.check_full('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_training(self):
        self.check_full('cuda')


if __name__ == '__main__':
    unittest.main()
