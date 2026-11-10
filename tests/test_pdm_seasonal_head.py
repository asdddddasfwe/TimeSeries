import unittest

import torch

from models.OscillatorPDMScaledMA import Model as Baseline
from models.OscillatorPDMSeasonalHead import Model
from test_oscillator_mixer import config


class SeasonalHeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2021)
        self.cfg = config(seq_len=96, pred_len=96, moving_avg=25, e_layers=2)

    def test_common_weights_and_auto_windows_preserved(self):
        torch.manual_seed(2021)
        old = Baseline(self.cfg)
        torch.manual_seed(2021)
        model = Model(self.cfg)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(model.ma_kernels, {96: 25, 48: 13, 24: 7, 12: 3})
        with self.assertRaises(RuntimeError):
            model.load_state_dict(old.state_dict(), strict=True)

    def test_final_trend_mixer_cannot_change_seasonal_predictions(self):
        model = Model(self.cfg).eval()
        x = torch.randn(2, 96, 2)
        with torch.no_grad():
            before = model.predict_components(x)
            for parameter in model.pdm_blocks[-1].mixing_multi_scale_trend.parameters():
                parameter.add_(.5)
            after = model.predict_components(x)
        torch.testing.assert_close(before[..., :4], after[..., :4], rtol=0, atol=0)
        torch.testing.assert_close(before[..., 8:], after[..., 8:], rtol=0, atol=0)
        self.assertGreater((before[..., 4:8] - after[..., 4:8]).abs().max().item(), 1e-6)

    def test_seasonal_loss_has_no_gradient_to_final_trend_mixer(self):
        model = Model(self.cfg)
        components = model.predict_components(torch.randn(2, 96, 2))
        components[..., :4].square().mean().backward()
        block = model.pdm_blocks[-1]
        # stack creates zero gradients for unused components; both zero/None
        # represent no computational contribution to the seasonal forecast.
        self.assertTrue(all(v.grad is None or not v.grad.any()
                            for v in block.mixing_multi_scale_trend.parameters()))
        self.assertTrue(any(v.grad is not None and v.grad.any()
                            for v in block.mixing_multi_scale_season.parameters()))
        self.assertTrue(any(v.grad is not None and v.grad.any()
                            for v in model.seasonals[0].decoder.parameters()))

    def test_components_reconstruct_and_raw_bypass_is_unchanged(self):
        model = Model(self.cfg).eval()
        x = torch.randn(2, 96, 2)
        with torch.no_grad():
            parts = model.predict_components(x)
            expected = model.normalize(parts.sum(-1), 'denorm')
            normalized = model.normalize(x, 'norm')
            raw = model.predict_raw_trend(normalized)
            torch.testing.assert_close(raw, parts[..., 8:].sum(-1), rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        self.assertEqual(parts.shape, (2, 96, 2, 12))

    def test_finite_backward_optimizer_coverage_and_reload(self):
        model = Model(self.cfg)
        groups = model.optimizer_parameter_groups(1e-4)
        ids = [id(v) for group in groups for v in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(v) for v in model.parameters()})
        optimizer = torch.optim.Adam(groups, lr=.001)
        x = torch.randn(2, 96, 2)
        loss = (model(x) - torch.randn(2, 96, 2)).square().mean()
        loss.backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all()
                            for v in model.parameters()))
        optimizer.step()
        restored = Model(self.cfg).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)

    def test_no_future_input_ms_and_one_inverse_normalization(self):
        model = Model(self.cfg).eval()
        x = torch.randn(2, 96, 2)
        events = []
        handle = model.normalize.register_forward_pre_hook(lambda module, args: events.append(args[1]))
        with torch.no_grad():
            output = model(x)
        handle.remove()
        self.assertEqual(events, ['norm', 'denorm'])
        nan = torch.full((2, 96, 2), float('nan'))
        with torch.no_grad():
            torch.testing.assert_close(model(x, nan, nan, nan), output, rtol=0, atol=0)
            model.features, model.c_out = 'MS', 1
            torch.testing.assert_close(model(x), output[..., -1:], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
