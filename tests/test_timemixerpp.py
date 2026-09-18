"""TimeMixer++ architecture and TSLib interface regression tests."""

import io
import unittest
from argparse import Namespace

import torch

from layers.TimeMixerPP import (
    AxialAttention, MultiScaleImageMixing, from_time_image,
    select_periods, to_time_image,
)
from models.TimeMixerPP import Model


def configuration(**overrides):
    values = dict(
        task_name="long_term_forecast", seq_len=17, pred_len=5, enc_in=3,
        c_out=3, d_model=8, d_ff=16, n_heads=2, e_layers=2, dropout=0.0,
        top_k=3, down_sampling_layers=2, down_sampling_window=2,
        down_sampling_method="conv", use_norm=1, features="M",
        embed="timeF", freq="h", num_class=4,
    )
    values.update(overrides)
    return Namespace(**values)


class TimeMixerPPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(2021)

    def test_time_image_chronology_and_padding_roundtrip(self):
        x = torch.arange(1, 11, dtype=torch.float32).reshape(1, 10, 1)
        image = to_time_image(x, 4)
        expected = torch.tensor([[1, 5, 9], [2, 6, 10], [3, 7, 0], [4, 8, 0.]])
        torch.testing.assert_close(image[0, 0], expected)
        torch.testing.assert_close(from_time_image(image, 10), x)

    def test_fft_period_uses_original_length_excludes_dc_and_keeps_gradients(self):
        t = torch.arange(16, dtype=torch.float32)
        signal = (100 + torch.sin(2 * torch.pi * 2 * t / 16)
                  + 0.2 * torch.sin(2 * torch.pi * 3 * t / 16))
        x = signal.reshape(1, 16, 1).requires_grad_()
        periods, weights = select_periods(x, full_length=64, top_k=2)
        self.assertEqual(periods, [32, 22])
        self.assertGreater(weights[0, 0], weights[0, 1])
        weights[0, 0].backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        periods, weights = select_periods(torch.zeros(2, 3, 4), 17, 100)
        self.assertEqual(periods, [17])
        torch.testing.assert_close(weights, torch.ones(2, 1))

    def test_axial_attention_semantics_and_padding_keys(self):
        image = torch.tensor([[[[1., 10., 100.], [3., 30., 300.]]]])
        valid = torch.ones(1, 1, 2, 3, dtype=torch.bool)
        for axis in ["season", "trend"]:
            with self.subTest(axis=axis):
                layer = AxialAttention(1, 1, axis, 0)
                with torch.no_grad():
                    layer.qkv.weight.zero_()
                    layer.qkv.bias.zero_()
                    layer.qkv.weight[2, 0, 1, 1] = 1
                    layer.projection.weight.fill_(1)
                    layer.projection.bias.zero_()
                expected = (torch.tensor([[[[2., 20., 200.], [2., 20., 200.]]]])
                            if axis == "season" else
                            torch.tensor([[[[37., 37., 37.], [111., 111., 111.]]]]))
                torch.testing.assert_close(layer(image, valid), expected)
                empty = torch.zeros_like(valid)
                torch.testing.assert_close(layer(image, empty), torch.zeros_like(image))

    def test_cross_scale_information_flows_in_paper_directions(self):
        mixing = MultiScaleImageMixing(3, 4, 8)
        seasons = [torch.randn(1, 4, 3, width) for width in (5, 3, 2)]
        trends = [torch.randn_like(x) for x in seasons]
        masks = [torch.ones(1, 1, 3, x.shape[-1], dtype=torch.bool) for x in seasons]
        base = mixing(seasons, trends, masks)
        fine_season = [seasons[0] + 1, *seasons[1:]]
        changed = mixing(fine_season, trends, masks)
        self.assertFalse(torch.allclose(changed[-1], base[-1]))
        coarse_season = [*seasons[:-1], seasons[-1] + 1]
        torch.testing.assert_close(mixing(coarse_season, trends, masks)[0], base[0])
        coarse_trend = [*trends[:-1], trends[-1] + 1]
        self.assertFalse(torch.allclose(mixing(seasons, coarse_trend, masks)[0], base[0]))
        fine_trend = [trends[0] + 1, *trends[1:]]
        torch.testing.assert_close(mixing(seasons, fine_trend, masks)[-1], base[-1])

    def test_all_five_task_interfaces_and_finite_gradients(self):
        for task in ["long_term_forecast", "short_term_forecast", "imputation",
                     "anomaly_detection", "classification"]:
            with self.subTest(task=task):
                model = Model(configuration(task_name=task))
                x = torch.randn(2, 17, 3)
                mark = (torch.ones(2, 17) if task == "classification"
                        else torch.randn(2, 17, 4))
                mask = torch.ones_like(x) if task == "imputation" else None
                output = model(x, mark, None, None, mask=mask)
                expected = ((2, 4) if task == "classification" else
                            (2, 5, 3) if "forecast" in task else (2, 17, 3))
                self.assertEqual(tuple(output.shape), expected)
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(output).all())
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_downsampling_is_trainable_and_checkpoint_reload_is_exact(self):
        config = configuration()
        model = Model(config)
        x, mark = torch.randn(2, 17, 3), torch.randn(2, 17, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = model.downsample[0].weight.detach().clone()
        model(x, mark).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.downsample[0].weight))
        model.eval()
        output = model(x, mark)
        torch.testing.assert_close(model(x, mark), output, rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = Model(config).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored(x, mark), output, rtol=0, atol=0)

    def test_imputation_ignores_missing_values_and_handles_empty_channels(self):
        model = Model(configuration(task_name="imputation")).eval()
        x = torch.randn(2, 17, 3)
        mask = torch.rand_like(x) > 0.4
        mask[..., 1] = False
        clean = x.masked_fill(~mask, 0)
        poisoned = x.masked_fill(~mask, float("nan"))
        expected = model(clean, mask=mask)
        actual = model(poisoned, mask=mask)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(model(poisoned, mask=torch.zeros_like(mask))).all())

    def test_classification_padding_does_not_leak_and_short_inputs_work(self):
        model = Model(configuration(task_name="classification")).eval()
        x = torch.randn(2, 17, 3)
        mask = torch.ones(2, 17, dtype=torch.bool)
        mask[:, 11:] = False
        expected = model(x, mask)
        altered = x.clone()
        altered[:, 11:] = 1e6
        torch.testing.assert_close(model(altered, mask), expected)
        torch.testing.assert_close(model(x[:, :11], mask[:, :11]), expected)
        self.assertTrue(torch.isfinite(model(x, torch.zeros_like(mask))).all())

    def test_single_scale_and_multivariate_target_denormalization(self):
        model = Model(configuration(
            down_sampling_layers=0, features="MS", c_out=1)).eval()
        x = torch.randn(2, 17, 3)
        shift = torch.tensor([3., -7., 11.])
        expected = model(x)
        self.assertEqual(tuple(expected.shape), (2, 5, 1))
        torch.testing.assert_close(model(x + shift), expected + 11, atol=2e-5, rtol=2e-5)

    def test_invalid_configuration_and_missing_mask_fail_clearly(self):
        invalid = [
            {"down_sampling_layers": 5}, {"d_model": 7}, {"top_k": 0},
            {"down_sampling_window": 3}, {"down_sampling_method": "avg"},
            {"task_name": "zero_shot_forecast"}, {"c_out": 2},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Model(configuration(**values))
        with self.assertRaisesRegex(ValueError, "observed-value mask"):
            Model(configuration(task_name="imputation"))(torch.randn(2, 17, 3))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_mixed_precision_with_odd_lengths(self):
        model = Model(configuration()).cuda()
        x = torch.randn(2, 17, 3, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(x)
            loss = output.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(output).all())
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
