"""Ordered aggregation, unchanged recurrence, gradients and checkpoint loading."""
import io
import unittest
from unittest.mock import patch

import torch

from test_oscillator_mixer import config
from models.OscillatorPDMStatePatch import Model as MeanPoolModel
from models import OscillatorPDMStatePatchConcat as concat
from utils.patch_covariance_loss import parts
from utils.timemix_decay_study import parameter_groups


def cfg():
    return config(seq_len=96, pred_len=96, enc_in=7, c_out=7, moving_avg=25,
                  d_model=32, d_ff=64, ssm_modes=8, top_k=5, dropout=.2,
                  e_layers=1, down_sampling_layers=3, down_sampling_window=2,
                  seasonal_normalize=False)


class StatePatchConcatTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_position_order_and_patch_isolation(self):
        encoder = concat.Model(cfg()).seasonals[0]
        projection = encoder.patch_driver_projection
        # Route coordinate zero of each position to a distinct output channel.
        with torch.no_grad():
            projection.weight.zero_()
            projection.bias.zero_()
            for position in range(4):
                projection.weight[position, position * 32] = 1
        features = torch.randn(2, 96, 32, requires_grad=True)
        result = concat.project_patch_drivers(encoder, features)
        expected = features.reshape(2, 24, 4, 32)[..., 0]
        torch.testing.assert_close(result[..., :4], expected, rtol=0, atol=0)
        reversed_features = features.reshape(2, 24, 4, 32).flip(2).reshape(2, 96, 32)
        reversed_result = concat.project_patch_drivers(encoder, reversed_features)
        self.assertFalse(torch.equal(result, reversed_result))
        gradient, = torch.autograd.grad(result[0, 3, :4].sum(), features)
        self.assertEqual(int(torch.count_nonzero(gradient[0, 12:16])), 4)
        gradient[0, 12:16] = 0
        self.assertEqual(int(torch.count_nonzero(gradient)), 0)

    def check_model(self, device):
        torch.manual_seed(2021)
        old = MeanPoolModel(cfg())
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(2021)
        model = concat.Model(cfg())
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 153058)
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        model.load_mean_pool_state(old.state_dict())
        old, model = old.to(device), model.to(device)
        features = torch.randn(2, 96, 32, device=device)
        for encoder in model.seasonals:
            actual = concat.project_patch_drivers(encoder, features)
            torch.testing.assert_close(actual, features.reshape(2, 24, 4, 32).mean(2),
                                       rtol=1e-6, atol=3e-7)
        x = torch.randn(2, 96, 7, device=device)
        for training in (False, True):
            old.train(training)
            model.train(training)
            torch.manual_seed(123)
            expected = old(x)
            expected_rng = torch.get_rng_state().clone()
            expected_cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
            torch.manual_seed(123)
            actual = model(x)
            self.assertEqual(actual.shape, (2, 96, 7))
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            if device == 'cuda':
                self.assertTrue(torch.equal(expected_cuda_rng, torch.cuda.get_rng_state()))
        model.eval()
        captures = []
        scan = concat.complex_oscillator_scan

        def record(a, b, h):
            result = scan(a, b, h)
            captures.append((a.detach(), b.detach(), h.detach(), result.detach()))
            return result

        with patch.object(concat, 'complex_oscillator_scan', record):
            components = model.predict_components(x)
        self.assertEqual(components.shape, (2, 96, 7, 8))
        self.assertEqual(len(captures), 4)
        _, (seasons, trends) = model.encode_multiscale(model.downsample(x))
        for encoder, history, trend, (a, b, h, states) in zip(
                model.seasonals, seasons, trends, captures):
            original_a, _, _, _ = encoder.decoder.dynamics(history, trend)
            self.assertTrue(torch.equal(a, original_a[:, None].expand_as(a)))
            self.assertEqual(states.shape, (14, 24, 8, 2))
            serial = []
            for step in range(24):
                ar, ai = a[:, step].unbind(-1)
                hr, hi = h.unbind(-1)
                h = torch.stack((ar * hr - ai * hi, ar * hi + ai * hr), -1) + b[:, step]
                serial.append(h)
            torch.testing.assert_close(states, torch.stack(serial, 1), rtol=2e-5, atol=2e-6)
        model.train()
        prediction = model(x)
        target = torch.randn_like(prediction, requires_grad=True)
        distance, _ = parts(prediction, target, 1., 4)
        loss = (prediction - target.detach()).square().mean() + distance
        loss.backward()
        self.assertIsNone(target.grad)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            if '.patch_driver_projection.weight' in name:
                self.assertGreater(float(parameter.grad.norm()), 0.)
                self.assertFalse(torch.equal(parameter.grad[:, :32], parameter.grad[:, 32:64]))
        groups = parameter_groups(model, 'adam_norm_bias')
        parameters = [p for group in groups for p in group['params']]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters()})
        optimizer = torch.optim.Adam(groups, lr=.001)
        before = model.seasonals[0].patch_driver_projection.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.seasonals[0].patch_driver_projection.weight))
        model.eval()
        with torch.no_grad():
            expected = model(x)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = concat.Model(cfg()).to(device)
        restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(expected, restored(x)))

    def test_model_cpu(self):
        self.check_model('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_model_cuda(self):
        self.check_model('cuda')


if __name__ == '__main__':
    unittest.main()
