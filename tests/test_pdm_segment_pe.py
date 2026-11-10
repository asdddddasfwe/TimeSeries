"""Segment coordinates, routing isolation and forecasting regression tests."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from torch import nn
from test_oscillator_mixer import config
from layers.LagARFormer import sinusoidal_positions, LagAxialBlock
from models.OscillatorFeaturePDMTimeMix import encode_one_fft_layer
from models.OscillatorPDMNoHistoryPE import Model as Control
from models.OscillatorPDMSegmentPE import Model, segment_positions, encode_positioned_segments
from models.OscillatorPDMSeasonalFFT import select_seasonal_feature_periods


class SegmentPETests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(2021)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def pair(self):
        cfg = config(seq_len=96, pred_len=96, moving_avg=25, e_layers=2,
                     dropout=0., down_sampling_layers=3, down_sampling_window=2)
        cfg.seasonal_normalize = False
        control, treatment = Control(cfg), Model(cfg)
        treatment.load_state_dict(control.state_dict(), strict=True)
        return control.eval(), treatment.eval()

    def test_two_coordinates_and_zero_padding(self):
        reference = torch.zeros(1, 7, 3, 4, dtype=torch.float64)
        pe = segment_positions(10, 4, 7, reference)[0].permute(1, 2, 0)
        self.assertEqual(pe.dtype, reference.dtype)
        for t in range(10):
            cycle, phase = divmod(t, 4)
            expected = torch.cat((sinusoidal_positions(3, 3)[cycle],
                                  sinusoidal_positions(4, 4)[phase])).double()
            torch.testing.assert_close(pe[cycle, phase], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(pe.reshape(-1, 7)[10:], torch.zeros(2, 7, dtype=pe.dtype)))
        self.assertFalse(torch.equal(pe[0, 0, :3], pe[1, 0, :3]))
        torch.testing.assert_close(pe[0, 0, 3:], pe[1, 0, 3:], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            segment_positions(10, 0, 7, reference)

    def test_grouped_duplicate_routes_match_per_series_reference(self):
        block = LagAxialBlock(8, 8, 0.)
        block.row.gamma.data.fill_(.2)
        block.column.gamma.data.fill_(.3)
        encoder = SimpleNamespace(blocks=nn.ModuleList([block]))
        tokens = torch.randn(3, 10, 8, requires_grad=True)
        periods = torch.tensor([[3, 3, 5], [2, 5, 5], [4, 3, 2]])
        logits = torch.randn(3, 3, requires_grad=True)
        weights = logits.softmax(-1)
        actual = encode_positioned_segments(encoder, tokens, periods, weights)
        expected = []
        for i in range(3):
            fused = torch.zeros_like(tokens[i:i+1])
            for j in range(3):
                period = int(periods[i, j])
                image = torch.nn.functional.pad(tokens[i:i+1], (0, 0, 0, (-10) % period))
                image = image.reshape(1, -1, period, 8).permute(0, 3, 1, 2).contiguous()
                image = block(image + segment_positions(10, period, 8, image))
                restored = image.permute(0, 2, 3, 1).reshape(1, -1, 8)[:, :10]
                fused = fused + restored * weights[i, j]
            expected.append(fused)
        torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-5, atol=1e-6)
        actual.square().mean().backward()
        for value in [tokens.grad, logits.grad]:
            self.assertIsNotNone(value)
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(value.abs().sum().item(), 0.)

    def test_zero_segment_pe_exactly_recovers_old_encoder(self):
        encoder = SimpleNamespace(blocks=nn.ModuleList([LagAxialBlock(8, 8, 0.)]))
        tokens = torch.randn(3, 10, 8)
        periods = torch.tensor([[3, 3], [2, 5], [4, 3]])
        weights = torch.randn(3, 2).softmax(-1)
        def zeros(length, period, width, reference):
            return reference.new_zeros(1, width, (length + period - 1) // period, period)
        with patch('models.OscillatorPDMSegmentPE.segment_positions', side_effect=zeros):
            actual = encode_positioned_segments(encoder, tokens, periods, weights)
        expected = encode_one_fft_layer(encoder, tokens, periods, weights, 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_no_change_to_ma_inputs_fft_routes_or_trend_stream(self):
        control, treatment = self.pair()
        x = torch.randn(2, 96, 2)
        records = []
        for model, target in [(control, 'models.OscillatorPDMSeasonalFFT.select_seasonal_feature_periods'),
                              (treatment, 'models.OscillatorPDMSegmentPE.select_seasonal_feature_periods')]:
            rec = {'ma': [], 'routes': []}
            def route(s, k):
                p, w = select_seasonal_feature_periods(s, k)
                rec['routes'].append((s.detach().clone(), p.clone(), w.detach().clone()))
                return p, w
            handle = model.initial_decomposition.register_forward_pre_hook(
                lambda m, args: rec['ma'].append(args[0].detach().clone()))
            with torch.no_grad(), patch(target, side_effect=route):
                _, (rec['season'], rec['trend']) = model.encode_multiscale(model.downsample(x))
            handle.remove()
            records.append(rec)
        for field in ['ma', 'trend']:
            self.assertEqual(len(records[0][field]), 4)
            for a, b in zip(records[0][field], records[1][field]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(len(records[0]['routes']), 4)
        for a, b in zip(records[0]['routes'], records[1]['routes']):
            for aa, bb in zip(a, b):
                torch.testing.assert_close(aa, bb, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(records[0]['season'], records[1]['season'])))

    def test_state_compatibility_finite_forward_backward_and_future_pe(self):
        control, treatment = self.pair()
        self.assertEqual(sum(p.numel() for p in control.parameters()),
                         sum(p.numel() for p in treatment.parameters()))
        self.assertEqual(set(control.state_dict()), set(treatment.state_dict()))
        x, y = torch.randn(2, 96, 2), torch.randn(2, 96, 2)
        prediction = treatment(x)
        self.assertEqual(prediction.shape, y.shape)
        (prediction-y).square().mean().backward()
        for p in treatment.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        with torch.no_grad():
            before = treatment(x)
            for encoder in treatment.seasonals:
                encoder.positions.normal_(0, 100)
            torch.testing.assert_close(treatment(x), before, rtol=0, atol=0)
            for encoder in treatment.seasonals:
                encoder.future_positions.zero_()
            self.assertFalse(torch.equal(treatment(x), before))


if __name__ == '__main__':
    unittest.main()
