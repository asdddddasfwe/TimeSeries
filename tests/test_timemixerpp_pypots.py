"""Regression coverage for the retained best PyPOTS forecast path."""

import io
from argparse import Namespace
import unittest

import torch

from layers.pypots_timemixerpp.layers import FFT_for_Period, RowAttention, ColAttention, MixerBlock
from models.TimeMixerPPPyPOTS import Model


def configuration(**kwargs):
    values = dict(task_name="long_term_forecast", seq_len=32, pred_len=5,
                  enc_in=3, c_out=3, features="M", e_layers=2, d_model=8,
                  d_ff=16, n_heads=2, num_kernels=2, top_k=2, dropout=0.0,
                  down_sampling_layers=2, down_sampling_window=2,
                  down_sampling_method="avg", channel_independence=1,
                  use_norm=1, freq="h")
    values.update(kwargs)
    return Namespace(**values)


class PyPOTSTimeMixerPPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(2021)

    def test_both_forecast_tasks_have_finite_outputs_and_active_gradients(self):
        for task in ("long_term_forecast", "short_term_forecast"):
            for use_norm in (0, 1):
                with self.subTest(task=task, use_norm=use_norm):
                    model = Model(configuration(task_name=task, use_norm=use_norm))
                    output = model(torch.randn(2, 32, 3))
                    self.assertEqual(tuple(output.shape), (2, 5, 3))
                    output.square().mean().backward()
                    self.assertTrue(torch.isfinite(output).all())
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertIsNotNone(model.backbone.predict_layers[-1].weight.grad)
                    self.assertIsNotNone(model.backbone.enc_embedding.value_embedding.tokenConv.weight.grad)


    def test_axial_attention_gates_start_as_identity(self):
        x = torch.randn(2, 8, 4, 3)
        for cls in (RowAttention, ColAttention):
            layer = cls(8, 16)
            torch.testing.assert_close(layer(x), x, atol=0, rtol=0)
            self.assertEqual(layer.gamma.item(), 0.0)


    def test_ensemble_trains_both_attention_branches_and_all_heads(self):
        for task in ("long_term_forecast", "short_term_forecast"):
            with self.subTest(task=task):
                model = Model(configuration(task_name=task))
                for block in model.backbone.encoder_model:
                    with torch.no_grad():
                        block.row_attn_2d_trend.gamma.fill_(0.5)
                        block.col_attn_2d_season.gamma.fill_(0.5)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                x, target = torch.randn(2, 32, 3), torch.randn(2, 5, 3)
                before = model(x).detach()
                loss = (model(x) - target).square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                active = [head.weight for head in model.backbone.predict_layers]
                for block in model.backbone.encoder_model:
                    for branch in (block.row_attn_2d_trend, block.col_attn_2d_season):
                        active.extend([branch.query_conv.weight, branch.key_conv.weight,
                                       branch.value_conv.weight, branch.gamma])
                saved = [parameter.detach().clone() for parameter in active]
                for parameter in active:
                    self.assertIsNotNone(parameter.grad)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                optimizer.step()
                for old, parameter in zip(saved, active):
                    self.assertFalse(torch.equal(old, parameter))
                after = model(x)
                self.assertTrue(torch.isfinite(after).all())
                self.assertFalse(torch.allclose(before, after))


    def test_constant_input_fft_excludes_dc_and_clamps_top_k(self):
        x = torch.zeros(2, 12, 8, requires_grad=True)
        periods, weights, indices = FFT_for_Period(x, k=100)
        self.assertEqual(len(periods), 6)
        self.assertTrue((periods > 0).all())
        self.assertTrue((indices > 0).all())
        weights.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        model = Model(configuration(top_k=100))
        output = model(torch.ones(2, 32, 3))
        self.assertTrue(torch.isfinite(output).all())


    def test_checkpoint_reload_ms_target_and_no_calendar_or_future_inputs(self):
        model = Model(configuration()).eval()
        x = torch.randn(2, 32, 3)
        expected = model(x)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        target_model = Model(configuration(c_out=1, features="MS")).eval()
        target_model.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        actual = target_model(x, torch.full((2, 32, 4), float("nan")),
                              torch.full((2, 5, 3), float("nan")),
                              torch.full((2, 5, 4), float("nan")))
        torch.testing.assert_close(actual, expected[..., -1:], atol=0, rtol=0)

    def test_adam_update_changes_prediction_with_finite_gradients(self):
        model = Model(configuration())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x, target = torch.randn(2, 32, 3), torch.randn(2, 5, 3)
        before = model(x).detach()
        for _ in range(2):
            optimizer.zero_grad()
            loss = (model(x) - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        self.assertFalse(torch.allclose(model(x), before))

    def test_mean_aggregation_matches_scale_predictions_and_one_inverse(self):
        for use_norm in (0, 1):
            with self.subTest(use_norm=use_norm):
                model = Model(configuration(
                    use_norm=use_norm)).eval()
                x = torch.randn(2, 32, 3) * 3 + torch.tensor([10., -5., 100.])
                projected = []
                hook = model.backbone.projection_layer.register_forward_hook(
                    lambda module, inputs, output: projected.append(output))
                actual = model(x)
                hook.remove()
                self.assertEqual(len(projected), 3)
                outputs = [value.reshape(2, 3, 5).permute(0, 2, 1) for value in projected]
                expected = torch.stack(outputs).mean(0)
                if use_norm:
                    norm = model.backbone.revin_layers[0]
                    expected = (expected - norm.affine_bias) / (norm.affine_weight + norm.eps)
                    expected = expected * norm.stdev + norm.mean
                torch.testing.assert_close(actual, expected)

    def test_mean_aggregation_updates_every_scale_head(self):
        for task in ("long_term_forecast", "short_term_forecast"):
            with self.subTest(task=task):
                model = Model(configuration(task_name=task))
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                before = [head.weight.detach().clone() for head in model.backbone.predict_layers]
                loss = (model(torch.randn(2, 32, 3)) - torch.randn(2, 5, 3)).square().mean()
                loss.backward()
                for head in model.backbone.predict_layers:
                    self.assertIsNotNone(head.weight.grad)
                    self.assertTrue(torch.isfinite(head.weight.grad).all())
                    self.assertGreater(head.weight.grad.abs().sum().item(), 0)
                optimizer.step()
                for saved, head in zip(before, model.backbone.predict_layers):
                    self.assertFalse(torch.equal(saved, head.weight))


    def test_axial_attention_routes_season_and_trend(self):
        sequence = torch.tensor([1., 2., 3., 4., 10., 20., 30., 40.]).view(1, 8, 1)
        block = MixerBlock(8, 2, 1, 1, 1, 1, 2)
        with torch.no_grad():
            for layer in (block.row_attn_2d_trend, block.col_attn_2d_season):
                layer.query_conv.weight.zero_()
                layer.query_conv.bias.zero_()
                layer.key_conv.weight.zero_()
                layer.key_conv.bias.zero_()
                layer.value_conv.weight.fill_(1)
                layer.value_conv.bias.zero_()
                layer.gamma.fill_(1)
        image = block.time_imaging(sequence, period=4)
        season, trend = block.dual_axis_attn(image)
        torch.testing.assert_close(season, image + torch.tensor([2.5, 25.]).view(1, 1, 2, 1))
        torch.testing.assert_close(trend, image + torch.tensor([5.5, 11., 16.5, 22.]).view(1, 1, 1, 4))

    def test_periods_are_computed_on_coarsest_grid(self):
        wave = torch.cos(2 * torch.pi * 5 * torch.arange(12) / 12)
        periods, _, frequencies = FFT_for_Period(wave.view(1, 12, 1), k=1)
        self.assertEqual(frequencies.tolist(), [5])
        self.assertEqual(periods.tolist(), [2])

    def test_invalid_or_removed_configuration_fails_before_forward(self):
        for overrides in (
            {"task_name": "imputation"}, {"seq_len": 31}, {"seq_len": 4},
            {"down_sampling_layers": 0}, {"down_sampling_method": "max"},
            {"down_sampling_method": "conv"},
            {"channel_independence": 0},
            {"channel_independence": 2}, {"n_heads": 100}, {"top_k": 0},
            {"num_kernels": 0}, {"d_model": 7}, {"c_out": 2},
            {"pypots_period_basis": "original"}, {"pypots_period_basis": "coarsest"},
            {"pypots_period_basis": "invalid"}, {"pypots_padding_mask": "all"},
            {"pypots_scale_aggregation": "last"}, {"pypots_attention_routing": "upstream"},
            {"pypots_conv_channels": "joint"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                Model(configuration(**overrides))

    def test_retained_downsampling_is_average_pooling_at_every_scale(self):
        model = Model(configuration())
        self.assertIsInstance(model.backbone.down_pool, torch.nn.AvgPool1d)
        self.assertEqual(list(model.backbone.down_pool.parameters()), [])
        x = torch.arange(2 * 32 * 3).float().reshape(2, 32, 3)
        scales = model.backbone._multi_scale_inputs(x)
        self.assertEqual([v.shape[1] for v in scales], [32, 16, 8])
        for fine, coarse in zip(scales, scales[1:]):
            expected = fine.reshape(2, -1, 2, 3).mean(2)
            torch.testing.assert_close(coarse, expected)


if __name__ == "__main__":
    unittest.main()
