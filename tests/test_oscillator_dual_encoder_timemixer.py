"""Semantic contracts for recurrence-before-PDM and prediction-only FMM."""
import io
import unittest

import torch

from test_pdm_direct_patch import cfg
from models import OscillatorDualEncoderTimeMixer as architecture
from models.OscillatorPDMSegmentPE import encode_positioned_segments
from models.OscillatorPDMSeasonalFFT import select_seasonal_feature_periods
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts


class DualEncoderTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_patch_alignment_endpoints_and_gradients(self):
        encoder = architecture.SlidingPatchTrendEncoder(8, 16, 0., heads=2)
        for length in (2, 4, 5, 12, 24, 48, 96):
            x = torch.randn(2, length, 8, requires_grad=True)
            patches, padded = encoder.patchify(x)
            for j in range(patches.shape[1]):
                for k in range(4):
                    torch.testing.assert_close(patches[:, j, k], x[:, min(2*j+k, length-1)])
            restored = encoder.overlap_average(patches, padded, length)
            torch.testing.assert_close(restored, x)
            gradient, = torch.autograd.grad(restored.sum(), x)
            torch.testing.assert_close(gradient, torch.ones_like(x))
            output = encoder(x)
            self.assertEqual(output.shape, x.shape)
            self.assertTrue(torch.isfinite(output).all())

    def test_recurrence_matches_serial_values_and_gradients(self):
        oscillator = architecture.HistoryOscillator(12, 8, 3, 0.).double()
        history = torch.randn(2, 12, 8, dtype=torch.double, requires_grad=True)
        a, b, initial = oscillator.scan_inputs(history)
        states = []
        h = initial
        for t in range(12):
            ar, ai = a[:, t].unbind(-1)
            hr, hi = h.unbind(-1)
            h = torch.stack((ar * hr - ai * hi, ar * hi + ai * hr), -1) + b[:, t]
            states.append(h)
        reference = oscillator.readout(torch.stack(states, 1).flatten(2))
        actual = oscillator(history)
        torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-10)
        parameters = (history, oscillator.log_decay, oscillator.frequency_logits,
                      oscillator.drive_projection.weight)
        left = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
        right = torch.autograd.grad(reference.square().sum(), parameters)
        for l, r in zip(left, right):
            torch.testing.assert_close(l, r, atol=1e-10, rtol=1e-9)
        self.assertTrue((a.square().sum(-1) < 1).all())

    def test_seasonal_fft_unchanged_and_pdm_direction(self):
        model = architecture.Model(cfg()).eval()
        history = torch.randn(3, 24, 32)
        encoder = model.season_encoders[2]
        expected = encode_positioned_segments(
            encoder, history, *select_seasonal_feature_periods(history, 5))
        torch.testing.assert_close(encoder(history), expected, rtol=0, atol=0)
        seasons = [torch.randn(2, t, 32, requires_grad=True) for t in model.lengths]
        trends = [torch.randn(2, t, 32, requires_grad=True) for t in model.lengths]
        ms = model.pdm.season_mix([s.transpose(1, 2) for s in seasons])
        mt = model.pdm.trend_mix([t.transpose(1, 2) for t in trends])
        # Fine seasonal output cannot depend on coarser inputs; coarse trend
        # output cannot depend on finer inputs. Opposite paths must exist.
        self.assertTrue(all(v is None for v in torch.autograd.grad(
            ms[0].sum(), seasons[1:], allow_unused=True, retain_graph=True)))
        self.assertTrue(all(v is None for v in torch.autograd.grad(
            mt[-1].sum(), trends[:-1], allow_unused=True, retain_graph=True)))
        self.assertGreater(float(torch.autograd.grad(ms[-1].square().sum(), seasons[0],
                                                    retain_graph=True)[0].norm()), 0)
        self.assertGreater(float(torch.autograd.grad(mt[0].square().sum(), trends[-1],
                                                    retain_graph=True)[0].norm()), 0)
        merged = model.pdm(seasons, trends)
        for actual, s, t, a, b in zip(merged, seasons, trends, ms, mt):
            torch.testing.assert_close(actual, s + t + model.pdm.channel_mlp(a + b))

    def check_full_model(self, device):
        model = architecture.Model(cfg()).to(device)
        n = sum(p.numel() for p in model.parameters())
        self.assertGreater(n, 0)
        seasons = {p.data_ptr() for p in model.season_oscillators.parameters()}
        trends = {p.data_ptr() for p in model.trend_oscillators.parameters()}
        self.assertFalse(seasons & trends)
        forbidden = ('future_gate', 'future_positions', 'trend_condition', 'initial_state')
        self.assertFalse(any(any(s in k for s in forbidden) for k in model.state_dict()))
        x = torch.randn(2, 96, 7, device=device)
        events, handles = [], []
        for label, modules in (('season', model.season_oscillators), ('trend', model.trend_oscillators)):
            for i, module in enumerate(modules):
                def record(m, args, output, label=label, i=i):
                    events.append((label, output.shape[1]))
                    self.assertEqual(output.shape, (14, model.lengths[i], 32))
                handles.append(module.register_forward_hook(record))
        handles.append(model.pdm.register_forward_pre_hook(lambda m, args: events.append(('pdm', 0))))
        handles.append(model.predict_layers[0].register_forward_pre_hook(lambda m, args: events.append(('fmm', 0))))
        model.eval()
        with torch.no_grad():
            components = model.predict_components(x)
        for handle in handles:
            handle.remove()
        self.assertEqual(events[-2:], [('pdm', 0), ('fmm', 0)])
        self.assertEqual(len(events), 10)
        self.assertEqual(components.shape, (2, 96, 7, 4))
        expected = model.normalizers[0](components.sum(-1), 'denorm')
        with torch.no_grad():
            torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
            torch.testing.assert_close(model(x, None, torch.full_like(x, float('nan'))), expected)
        # Window and variable isolation are unaffected by batch packing.
        changed = x.clone()
        changed[1] += 8
        changed[0, :, 1:] *= 3
        with torch.no_grad():
            torch.testing.assert_close(model(changed)[0, :, 0], expected[0, :, 0], atol=5e-6, rtol=2e-5)
        optimizer = torch.optim.Adam(norm_bias_no_decay_groups(model, 1e-4), lr=1e-3)
        model.train()
        target = torch.randn_like(x, requires_grad=True)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            distance, _ = parts(prediction, target)
            loss = (prediction - target.detach()).square().mean() + distance
            loss.backward()
            self.assertIsNone(target.grad)
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for module in (model.trend_encoders, model.season_oscillators, model.trend_oscillators,
                           model.pdm, model.predict_layers):
                self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters()), 0.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            reference = model(x)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = architecture.Model(cfg()).to(device).eval()
        restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(x), reference, rtol=0, atol=0)

    def test_cpu_forward_backward_and_reload(self):
        self.check_full_model('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_backward_and_reload(self):
        self.check_full_model('cuda')

    def test_ms_and_horizon_contract(self):
        config = cfg()
        config.pred_len, config.features, config.c_out = 17, 'MS', 1
        model = architecture.Model(config).eval()
        with torch.no_grad():
            self.assertEqual(model(torch.randn(2, 96, 7)).shape, (2, 17, 1))
        config.e_layers = 2
        with self.assertRaisesRegex(ValueError, 'e_layers=1'):
            architecture.Model(config)


if __name__ == '__main__':
    unittest.main()
