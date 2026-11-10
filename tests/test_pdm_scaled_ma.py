import unittest

import torch

from layers.Autoformer_EncDec import series_decomp
from models.OscillatorPDMScaleAllocation import Model as Baseline
from models.OscillatorPDMScaledMA import (
    Model, ScaleAwareDecomposition, scaled_odd_kernel,
)
from test_oscillator_mixer import config


class ScaledMATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2021)

    def test_kernel_rule_generalizes_and_handles_ties(self):
        self.assertEqual(
            [scaled_odd_kernel(25, 96, n) for n in [96, 48, 24, 12]],
            [25, 13, 7, 3],
        )
        self.assertEqual(
            [scaled_odd_kernel(17, 128, n) for n in [128, 64, 32, 16]],
            [17, 9, 5, 3],
        )
        self.assertEqual(scaled_odd_kernel(9, 9, 4), 5)
        self.assertEqual(scaled_odd_kernel(3, 96, 1), 1)
        for base in [1, 3, 9, 25, 101]:
            for length in range(1, 97):
                k = scaled_odd_kernel(base, 96, length)
                self.assertGreaterEqual(k, 1)
                self.assertEqual(k % 2, 1)
                self.assertLessEqual(abs(k - base * length / 96), 1)

    def test_invalid_configuration_rejected(self):
        for args in [(24, 96, 48), (0, 96, 48), (25, 0, 1),
                     (25, 96, 0), (25, 96, 97), (25.0, 96, 48),
                     (True, 96, 48)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                scaled_odd_kernel(*args)

    def test_decomposition_shape_reconstruction_and_gradient(self):
        module = ScaleAwareDecomposition(25, 96, [96, 48, 24, 12])
        for length in [96, 48, 24, 12]:
            x = torch.randn(2, length, 3, dtype=torch.float64, requires_grad=True)
            season, trend = module(x)
            self.assertEqual(season.shape, x.shape)
            torch.testing.assert_close(season + trend, x, atol=1e-14, rtol=0)
            kernel = min(range(1, 100, 2),
                         key=lambda k: (abs(k - 25 * length / 96), -k))
            expected = series_decomp(kernel)(x)
            for value, reference in zip((season, trend), expected):
                torch.testing.assert_close(value, reference, atol=0, rtol=0)
            (season.square().mean() + trend.square().mean()).backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        with self.assertRaises(ValueError):
            module(torch.randn(2, 11, 3))

    def test_model_only_changes_feature_decomposition(self):
        cfg = config(seq_len=96, pred_len=96, moving_avg=25, dropout=.2)
        cfg.e_layers = 2
        cfg.seasonal_normalize = False
        torch.manual_seed(2021)
        baseline = Baseline(cfg, variant="bypass")
        rng = torch.get_rng_state()
        torch.manual_seed(2021)
        model = Model(cfg)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(baseline.state_dict()), set(model.state_dict()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
        for block in model.pdm_blocks:
            self.assertEqual(block.decompsition.kernels, model.ma_kernels)
        self.assertEqual(model.raw_trend_decomposition.moving_avg.kernel_size, 25)
        x = torch.randn(2, 96, cfg.enc_in)
        torch.testing.assert_close(model.predict_raw_trend(x),
                                   baseline.predict_raw_trend(x), atol=0, rtol=0)
        prediction = model(x)
        self.assertEqual(prediction.shape, (2, 96, cfg.enc_in))
        prediction.square().mean().backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all()
                            for v in model.parameters()))
        model.eval()
        other = Model(cfg).eval()
        other.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(model(x), other(x), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
