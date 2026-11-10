"""Verify actual branch removal against the unchanged twelve-path control."""
import unittest

import torch

from test_oscillator_mixer import config
from models.OscillatorPDMReadoutSplit import Model as Control
from models.OscillatorPDMNoRawBypass import Model


class NoRawBypassTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.cfg = config(seq_len=96, pred_len=24, moving_avg=25,
                          e_layers=2, dropout=.2)
        self.cfg.seasonal_normalize = False

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def pair(self):
        torch.manual_seed(2021)
        control = Control(self.cfg, variant="seasonal")
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2021)
        model = Model(self.cfg)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        return control, model

    def test_removal_preserves_remaining_initial_weights(self):
        control, model = self.pair()
        for name in ("raw_trend_decomposition", "raw_trend_mix", "raw_trend_heads"):
            self.assertFalse(hasattr(model, name))
        self.assertFalse(model.raw_trend_bypass)
        self.assertFalse(any(k.startswith("raw_trend_") for k in model.state_dict()))
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[k], rtol=0, atol=0)
        removed = sum(p.numel() for p in control.raw_trend_mix.parameters())
        removed += sum(p.numel() for p in control.raw_trend_heads.parameters())
        self.assertEqual(sum(p.numel() for p in control.parameters()) -
                         sum(p.numel() for p in model.parameters()), removed)

    def test_components_and_forecast_equal_control_without_raw(self):
        control, model = self.pair()
        x = torch.randn(2, 96, 2)
        for training in (False, True):
            control.train(training)
            model.train(training)
            with torch.no_grad():
                torch.manual_seed(97)
                old = control.predict_components(x)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(97)
                new = model.predict_components(x)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(new.shape, (2, 24, 2, 8))
                torch.testing.assert_close(new, old[..., :8], rtol=0, atol=0)
                expected = control.normalize(old[..., :4].sum(-1) +
                                             old[..., 4:8].sum(-1), "denorm")
                torch.manual_seed(97)
                torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        model.eval()
        with torch.no_grad():
            expected = model(x)
        model.features, model.c_out = "MS", 1
        with torch.no_grad():
            torch.testing.assert_close(model(x), expected[..., -1:], rtol=0, atol=0)

    def test_finite_backward_and_both_retained_paths_update(self):
        _, model = self.pair()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        x, target = torch.randn(2, 96, 2), torch.randn(2, 24, 2)
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        for _ in range(2):
            optimizer.zero_grad()
            prediction = model(x)
            loss = (prediction - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())
            optimizer.step()
        for prefix in ("seasonals.", "feature_trend_heads."):
            self.assertTrue(any(not torch.equal(p, before[k])
                                for k, p in model.named_parameters()
                                if k.startswith(prefix)))


if __name__ == "__main__":
    unittest.main()
