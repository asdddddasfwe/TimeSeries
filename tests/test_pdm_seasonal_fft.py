"""Routing contracts for MA decomposition followed by seasonal FFT encoding."""
import unittest
from unittest.mock import patch

import torch

from test_oscillator_mixer import config
from models.OscillatorPDMNoRawBypass import Model as Control
from models.OscillatorPDMSeasonalFFT import Model, select_seasonal_feature_periods
from models.OscillatorFeaturePDMTimeMix import encode_one_fft_layer


class SeasonalFFTTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.cfg = config(seq_len=96, pred_len=24, moving_avg=25,
                          e_layers=2, dropout=0.)
        self.cfg.seasonal_normalize = False

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_remaining_weights_rng_preserved_and_extra_fft_parameters_removed(self):
        torch.manual_seed(2021)
        control = Control(self.cfg)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2021)
        model = Model(self.cfg)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        removed_keys = {k for k in control.state_dict() if ".blocks.1." in k}
        self.assertTrue(removed_keys)
        self.assertEqual(set(control.state_dict()) - removed_keys, set(model.state_dict()))
        removed_count = sum(p.numel() for e in control.seasonals for p in e.blocks[1].parameters())
        self.assertEqual(sum(p.numel() for p in control.parameters()) -
                         sum(p.numel() for p in model.parameters()), removed_count)
        self.assertEqual(len(model.pdm_blocks), 2)
        self.assertTrue(all(len(e.blocks) == 1 for e in model.seasonals))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(any(k.startswith("raw_trend_") for k in model.state_dict()))

    def test_one_split_per_scale_and_direct_streams_to_decoder(self):
        model = Model(self.cfg).eval()
        splits, fft_inputs, last_outputs, decoder_inputs = [], [], [], []
        handles = [model.initial_decomposition.register_forward_hook(
            lambda module, args, output: splits.append(output))]
        handles.append(model.pdm_blocks[-1].register_forward_hook(
            lambda module, args, output: last_outputs.append(output)))
        for block in model.pdm_blocks:
            self.assertFalse(hasattr(block, "decompsition"))
        self.assertFalse(hasattr(model, "readout_decomposition"))
        for head in model.feature_trend_heads:
            handles.append(head.register_forward_pre_hook(
                lambda module, args: decoder_inputs.append(args[0])))
        def spy(encoder, tokens, periods, weights, index):
            fft_inputs.append((tokens, index))
            return encode_one_fft_layer(encoder, tokens, periods, weights, index)
        with patch("models.OscillatorPDMSeasonalFFT.encode_one_fft_layer", side_effect=spy):
            with torch.no_grad():
                result = model(torch.randn(2, 96, 2))
        for handle in handles:
            handle.remove()
        self.assertEqual(result.shape, (2, 24, 2))
        self.assertEqual(len(splits), 4)  # Four scales, ONE decomposition stage.
        self.assertEqual(len(fft_inputs), 4)  # One FFT encoding per scale.
        for (season, _), (tokens, layer) in zip(splits, fft_inputs[:4]):
            self.assertEqual(layer, 0)
            torch.testing.assert_close(tokens, season, rtol=0, atol=0)
        for actual, expected in zip(decoder_inputs, last_outputs[0][1]):
            self.assertIs(actual, expected)
        self.assertIs(model.decoder_histories(last_outputs[0]), last_outputs[0])

    def test_trend_perturbation_cannot_enter_any_layer_fft(self):
        model = Model(self.cfg).eval()
        seasons = [torch.randn(2, length, self.cfg.d_model) for length in model.lengths]
        trends = [torch.randn_like(h) for h in seasons]
        traces = []
        def spy(encoder, tokens, periods, weights, index):
            traces.append((tokens.clone(), periods.clone(), weights.clone()))
            return encode_one_fft_layer(encoder, tokens, periods, weights, index)
        def run(season, trend):
            season = model.encode_seasonal_features(season)
            for block in model.pdm_blocks:
                season, trend = block(season, trend)
            return season, trend
        with patch("models.OscillatorPDMSeasonalFFT.encode_one_fft_layer", side_effect=spy):
            with torch.no_grad():
                before_s, before_t = run(seasons, trends)
                after_s, after_t = run(seasons, [t + 10. for t in trends])
        for first, second in zip(traces[:4], traces[4:]):
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(before_s, after_s):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_t, after_t)))
        # Also check the reverse direction at fixed parameters.
        with torch.no_grad():
            _, changed_t = run([s + 10. for s in seasons], trends)
        for a, b in zip(before_t, changed_t):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_periods_from_feature_spectrum_without_channel_cancellation(self):
        t = torch.arange(96, dtype=torch.float64)
        wave = torch.cos(2 * torch.pi * t / 12)
        season = torch.stack((wave, -wave), -1)[None].requires_grad_()
        periods, weights = select_seasonal_feature_periods(season, 2)
        self.assertEqual(periods[0, 0].item(), 12)
        weights[0, 0].backward()
        self.assertTrue(torch.isfinite(season.grad).all())
        p, w = select_seasonal_feature_periods(torch.zeros(2, 12, 4), 100)
        self.assertEqual(p.shape, (2, 6))
        self.assertTrue(torch.isfinite(w).all())
        torch.testing.assert_close(w.sum(-1), torch.ones(2))

    def test_reconstruction_gradients_optimizer_and_ms_output(self):
        torch.manual_seed(2021)
        model = Model(self.cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        x, target = torch.randn(2, 96, 2), torch.randn(2, 24, 2)
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        for _ in range(2):
            optimizer.zero_grad()
            parts = model.predict_components(x)
            self.assertEqual(parts.shape, (2, 24, 2, 8))
            expected = model.normalize(parts[..., :4].sum(-1) + parts[..., 4:].sum(-1), "denorm")
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
            loss = (expected - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        for prefix in ("seasonals.", "feature_trend_heads.", "pdm_blocks."):
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.named_parameters()
                                if k.startswith(prefix)))
        expected = model(x)
        model.features, model.c_out = "MS", 1
        torch.testing.assert_close(model(x), expected[..., -1:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
