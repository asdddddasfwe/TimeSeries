"""Architecture contracts for the independent SeasonTrendMixer model."""
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from layers.SeasonTrendMixer import (
    PeriodPatchEncoder, TemporalScaleMixing, select_periods,
)
from models.SeasonTrendMixer import Model


def config(**changes):
    values = dict(task_name="long_term_forecast", seq_len=32, pred_len=8,
                  enc_in=3, c_out=3, features="M", moving_avg=5, d_model=8,
                  d_ff=16, n_heads=2, e_layers=2, dropout=0.0, top_k=3,
                  down_sampling_layers=3, down_sampling_window=2,
                  down_sampling_method="avg", channel_independence=1, use_norm=1)
    values.update(changes)
    return SimpleNamespace(**values)


class SeasonTrendMixerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(2021)

    def test_decomposition_reconstruction_and_four_scales(self):
        model = Model(config(seq_len=96))
        x = torch.randn(2, 96, 3)
        season, trend = model.decomposition(x)
        torch.testing.assert_close(season + trend, x)
        for component in (season, trend):
            scales = model.downsample(component)
            self.assertEqual([v.shape[1] for v in scales], [96, 48, 24, 12])
            for upper, lower in zip(scales, scales[1:]):
                torch.testing.assert_close(lower, upper.reshape(2, -1, 2, 3).mean(2))

    def test_fft_known_periods_per_series_and_excludes_dc(self):
        t = torch.arange(96, dtype=torch.float32)
        x = torch.stack([torch.sin(2 * torch.pi * t / period) + 100
                         for period in (12, 24, 8, 16)])
        periods, weights = select_periods(x, 1)
        self.assertEqual(periods[:, 0].tolist(), [12, 24, 8, 16])
        torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_constant_top_k_cap_and_low_precision_fft(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.ones(2, 24, dtype=dtype)
            periods, weights = select_periods(x, 100)
            self.assertEqual(periods.shape, (2, 12))
            self.assertEqual(periods[0].tolist(), [24 // f for f in range(1, 13)])
            self.assertTrue(torch.isfinite(weights).all())
            torch.testing.assert_close(weights.sum(-1), torch.ones(2, dtype=dtype))
        output = Model(config())(torch.full((2, 32, 3), 7.0))
        self.assertTrue(torch.isfinite(output).all())

    def test_patch_padding_and_decode_preserve_exact_time_order(self):
        encoder = PeriodPatchEncoder(16, 16, 2, 32, 1, 0.0, 3)
        with torch.no_grad():
            encoder.local_encoder[-1].weight.zero_()
            encoder.local_encoder[-1].bias.zero_()
            nn.init.eye_(encoder.patch_embedding.weight)
            nn.init.zeros_(encoder.patch_embedding.bias)
            nn.init.eye_(encoder.patch_decoder.weight)
            nn.init.zeros_(encoder.patch_decoder.bias)
            encoder.position_encoding.zero_()
        encoder.blocks = nn.ModuleList([nn.Identity()])
        seen = []
        encoder.patch_embedding.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        sequence = torch.arange(32).reshape(2, 16).float()
        restored = encoder.encode_period(sequence, 5)
        self.assertEqual(tuple(seen[0].shape), (2, 4, 16))
        torch.testing.assert_close(seen[0][:, 0, :5], sequence[:, :5])
        self.assertTrue((seen[0][:, :, 5:] == 0).all())
        torch.testing.assert_close(seen[0][:, -1, 0], sequence[:, -1])
        self.assertTrue((seen[0][:, -1, 1:] == 0).all())
        torch.testing.assert_close(restored, sequence, atol=0, rtol=0)

    def test_transformer_attends_across_period_tokens_without_masks(self):
        encoder = PeriodPatchEncoder(16, 16, 2, 32, 1, 0.0, 3)
        with torch.no_grad():
            encoder.local_encoder[-1].weight.zero_()
            encoder.local_encoder[-1].bias.zero_()
            nn.init.eye_(encoder.patch_embedding.weight)
            nn.init.zeros_(encoder.patch_embedding.bias)
            nn.init.eye_(encoder.patch_decoder.weight)
            nn.init.zeros_(encoder.patch_decoder.bias)
            encoder.position_encoding.zero_()
        seen = []
        self_test = self
        class AcrossPatches(nn.Module):
            def forward(self, tokens, **kwargs):
                self_test.assertEqual(kwargs, {})
                seen.append(tokens.clone())
                return tokens + tokens.mean(dim=1, keepdim=True)
        encoder.blocks = nn.ModuleList([AcrossPatches()])
        x = torch.arange(32).float().reshape(2, 16)
        y = encoder.encode_period(x, 4)
        expected_patches = x.reshape(2, 4, 4)
        self.assertEqual(seen[0].shape, (2, 4, 16))
        torch.testing.assert_close(seen[0][..., :4], expected_patches)
        expected = expected_patches + expected_patches.mean(dim=1, keepdim=True)
        torch.testing.assert_close(y, expected.reshape(2, 16))

    def test_duplicate_period_weights_coalesce_and_remain_differentiable(self):
        encoder = PeriodPatchEncoder(16, 2, 1, 4, 1, 0.0, 3)
        periods = torch.tensor([[2, 2, 4], [4, 2, 4]])
        weights = torch.tensor([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], requires_grad=True)
        calls = []
        def encode(sequence, period):
            calls.append((period, sequence.shape[0]))
            return sequence * period
        with patch("layers.SeasonTrendMixer.select_periods", return_value=(periods, weights)):
            with patch.object(encoder, "encode_period", side_effect=encode):
                output = encoder(torch.ones(1, 16, 2))
        self.assertEqual(calls, [(2, 2), (4, 2)])
        torch.testing.assert_close(output[0, 0], torch.tensor([3.0, 3.4]))
        output.sum().backward()
        self.assertTrue(torch.isfinite(weights.grad).all())
        self.assertTrue((weights.grad != 0).all())

    def test_pre_ln_patch_transformers_and_independent_scales(self):
        model = Model(config())
        parameter_sets = []
        for length, encoder in zip(model.lengths[:2], model.season_encoders):
            self.assertEqual(len(encoder.blocks), 2)
            self.assertEqual(encoder.patch_embedding.in_features, length)
            self.assertEqual(encoder.patch_decoder.out_features, length)
            parameter_sets.append({id(p) for p in encoder.parameters()})
            for layer in encoder.blocks:
                self.assertTrue(layer.norm_first)
                self.assertEqual(layer.self_attn.num_heads, 2)
                self.assertEqual(layer.linear1.out_features, 16)
                self.assertEqual(layer.activation.__name__, "gelu")
        self.assertFalse(parameter_sets[0] & parameter_sets[1])
        self.assertEqual(sum(isinstance(m, nn.TransformerEncoderLayer)
                             for m in model.modules()), 4)

    def test_only_fine_seasons_select_fft_once_and_decompose_once(self):
        model = Model(config())
        with patch("layers.SeasonTrendMixer.select_periods", wraps=select_periods) as fft:
            with patch.object(model.decomposition, "forward",
                              wraps=model.decomposition.forward) as decomp:
                model(torch.randn(2, 32, 3))
        self.assertEqual(decomp.call_count, 1)
        self.assertEqual(fft.call_count, 2)
        self.assertEqual([tuple(call.args[0].shape) for call in fft.call_args_list],
                         [(6, 32), (6, 16)])

    def test_six_identity_preprocessors_and_eight_uniform_heads(self):
        model = Model(config())
        preprocessors = list(model.season_linears) + list(model.trend_linears)
        heads = list(model.season_heads) + list(model.trend_heads)
        self.assertEqual(len(preprocessors), 6)
        self.assertEqual(len(heads), 8)
        self.assertEqual(len({id(m.weight) for m in preprocessors + heads}), 14)
        for layer in preprocessors:
            torch.testing.assert_close(layer.weight, torch.eye(layer.in_features))
            torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))
        for layer in heads:
            torch.testing.assert_close(layer.weight,
                                       torch.full_like(layer.weight, 1 / layer.in_features))
            torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))

    def test_all_linear_and_attention_paths_receive_gradients_and_update(self):
        model = Model(config())
        layers = list(model.season_linears) + list(model.trend_linears)
        layers += list(model.season_heads) + list(model.trend_heads)
        counts = [0] * len(layers)
        handles = []
        for i, layer in enumerate(layers):
            def hook(module, args, output, index=i):
                counts[index] += 1
            handles.append(layer.register_forward_hook(hook))
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = (model(torch.randn(4, 32, 3)) - torch.randn(4, 8, 3)).square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(counts, [1] * 14)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        checked = [f"{group}.{i}.weight" for group, size in
                   (("season_linears", 2), ("trend_linears", 4),
                    ("season_heads", 4), ("trend_heads", 4)) for i in range(size)]
        checked += [name for name, _ in model.named_parameters()
                    if any(part in name for part in (".self_attn.in_proj_weight",
                                                     ".linear1.weight", ".linear2.weight",
                                                     ".patch_embedding.weight", ".patch_decoder.weight",
                                                     ".local_encoder.0.weight", ".local_encoder.2.weight"))]
        optimizer.step()
        for name in checked:
            self.assertGreater(dict(model.named_parameters())[name].grad.abs().sum().item(), 0, name)
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)
        for handle in handles:
            handle.remove()

    def test_cross_scale_propagation_directions(self):
        lengths = [32, 16, 8, 4]
        scales = [torch.full((1, length, 1), float(i)) for i, length in enumerate(lengths)]
        class MeanMap(nn.Module):
            def __init__(self, length):
                super().__init__()
                self.length = length
            def forward(self, value):
                return value.mean(-1, keepdim=True).expand(*value.shape[:-1], self.length)
        for direction, expected in ((True, [0, 1, 3, 6]), (False, [6, 6, 5, 3])):
            mixer = TemporalScaleMixing(lengths, direction)
            for mapping in mixer.mappings:
                self.assertIsInstance(mapping[0], nn.Linear)
                self.assertIsInstance(mapping[1], nn.GELU)
                self.assertIsInstance(mapping[2], nn.Linear)
            mixer.mappings = nn.ModuleList([MeanMap(lengths[i]) for i in mixer.order[1:]])
            output = mixer(scales)
            self.assertEqual([value[0, 0, 0].item() for value in output], expected)

    def test_sum_fusion_and_single_shared_normalization(self):
        model = Model(config())
        modes, head_values, denorm_inputs = [], [], []
        def norm_hook(module, args):
            modes.append(args[1])
            if args[1] == "denorm":
                denorm_inputs.append(args[0].detach())
        model.normalize.register_forward_pre_hook(norm_hook)
        for head in list(model.season_heads) + list(model.trend_heads):
            head.register_forward_hook(lambda m, a, y: head_values.append(y.detach()))
        x = torch.randn(2, 32, 3) * 5 + 10
        y = model(x)
        self.assertEqual(modes, ["norm", "denorm"])
        expected = sum(head_values).transpose(1, 2)
        torch.testing.assert_close(denorm_inputs[0], expected)
        expected = expected * (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        expected = expected + x.mean(1, keepdim=True)
        torch.testing.assert_close(y, expected)

    def test_eval_batch_order_and_companion_independence(self):
        model = Model(config()).eval()
        x = torch.randn(4, 32, 3)
        with torch.no_grad():
            batched = model(x)
            alone = torch.cat([model(sample[None]) for sample in x], 0)
            permuted = model(x[[3, 0, 2, 1]])
        torch.testing.assert_close(batched, alone, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(permuted, batched[[3, 0, 2, 1]], atol=2e-5, rtol=2e-5)

    def test_channel_independence(self):
        model = Model(config()).eval()
        x = torch.randn(2, 32, 3)
        changed = x.clone()
        changed[:, :, 1] = torch.randn(2, 32) * 50
        with torch.no_grad():
            baseline, output = model(x), model(changed)
        torch.testing.assert_close(baseline[:, :, [0, 2]], output[:, :, [0, 2]])

    def test_m_s_ms_shapes_and_future_input_isolation(self):
        for task in ("long_term_forecast", "short_term_forecast"):
            for features, channels, c_out in (("M", 3, 3), ("S", 1, 1),
                                             ("MS", 3, 1), ("MS", 3, 3)):
                with self.subTest(task=task, features=features, c_out=c_out):
                    model = Model(config(task_name=task, features=features,
                                         enc_in=channels, c_out=c_out)).eval()
                    x = torch.randn(2, 32, channels)
                    poison = torch.full((2, 8, channels), float("nan"))
                    with torch.no_grad():
                        y = model(x, poison, poison, poison, mask=poison)
                        reference = model(x)
                    self.assertEqual(y.shape, (2, 8, c_out))
                    self.assertTrue(torch.isfinite(y).all())
                    torch.testing.assert_close(y, reference)

    def test_ms_slices_after_inverse_normalization(self):
        full = Model(config()).eval()
        single = Model(config(features="MS", c_out=1)).eval()
        single.load_state_dict(full.state_dict(), strict=True)
        x = torch.randn(2, 32, 3) * torch.tensor([1.0, 10.0, 100.0])
        with torch.no_grad():
            torch.testing.assert_close(single(x), full(x)[:, :, -1:])

    def test_all_script_horizons_and_strict_checkpoint_round_trip(self):
        for horizon in (96, 192, 336, 720):
            cfg = config(seq_len=96, pred_len=horizon)
            original = Model(cfg).eval()
            x = torch.randn(1, 96, 3)
            stream = io.BytesIO()
            torch.save(original.state_dict(), stream)
            stream.seek(0)
            restored = Model(cfg).eval()
            restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
            with torch.no_grad():
                reference, output = original(x), restored(x)
            self.assertEqual(output.shape, (1, horizon, 3))
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output, reference, atol=0, rtol=0)


    def test_variable_periods_share_registered_weights_and_use_one_token_per_patch(self):
        encoder = PeriodPatchEncoder(96, 8, 2, 16, 2, 0.0, 3)
        names_and_ids = [(name, id(p)) for name, p in encoder.named_parameters()]
        calls = []
        def capture(module, args, kwargs):
            self.assertEqual(kwargs, {})
            calls.append(tuple(args[0].shape))
        handle = encoder.blocks[0].register_forward_pre_hook(capture, with_kwargs=True)
        loss = 0
        for period in (96, 48, 24, 5, 2):
            x = torch.randn(2, 96)
            y = encoder.encode_period(x, period)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(torch.isfinite(y).all())
            self.assertEqual(calls[-1], (2, (96 + period - 1) // period, 8))
            loss = loss + y.square().mean()
        handle.remove()
        self.assertEqual(names_and_ids, [(name, id(p)) for name, p in encoder.named_parameters()])
        loss.backward()
        for name, parameter in encoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for parameter in (encoder.patch_embedding.weight, encoder.patch_decoder.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_fixed_position_encoding_indexes_patches(self):
        encoder = PeriodPatchEncoder(16, 8, 2, 16, 1, 0.0, 3)
        with torch.no_grad():
            encoder.patch_embedding.weight.zero_()
            encoder.patch_embedding.bias.zero_()
        seen = []
        class Capture(nn.Module):
            def forward(self, tokens):
                seen.append(tokens.detach().clone())
                return tokens
        encoder.blocks = nn.ModuleList([Capture()])
        encoder.encode_period(torch.randn(2, 16), 4)
        torch.testing.assert_close(seen[0], encoder.position_encoding[:, :4].expand(2, -1, -1))
        self.assertFalse(torch.equal(seen[0][:, 0], seen[0][:, 1]))


    def test_patch_local_convolution_models_neighbors_without_cross_patch_leakage(self):
        encoder = PeriodPatchEncoder(16, 8, 2, 16, 1, 0.0, 3)
        with torch.no_grad():
            for layer in (encoder.local_encoder[0], encoder.local_encoder[2]):
                layer.weight.fill_(1)
                layer.bias.zero_()
        before_embedding = []
        encoder.patch_embedding.register_forward_pre_hook(
            lambda module, args: before_embedding.append(args[0].detach().clone()))
        x = torch.zeros(2, 16)
        x[0, 3] = 1  # Last point in the first patch.
        encoder.encode_period(x, 4)
        features = before_embedding[0]
        self.assertEqual(features.shape, (2, 4, 16))
        # A size-3 kernel reaches its within-patch neighbor, but not farther points.
        self.assertGreater(features[0, 0, 2].item(), 0)
        self.assertGreater(features[0, 0, 3].item(), 1)
        self.assertEqual(features[0, 0, 1].item(), 0)
        self.assertTrue((features[0, 1:] == 0).all())  # No adjacent-patch leakage.
        self.assertTrue((features[1] == 0).all())  # No sample/channel mixing.
        self.assertTrue((features[..., 4:] == 0).all())  # Width padding happens last.

    def test_local_convolution_uses_actual_period_width_and_independent_scale_parameters(self):
        model = Model(config())
        encoders = model.season_encoders
        self.assertIsInstance(encoders[0].local_encoder[0], nn.Conv1d)
        self.assertIsInstance(encoders[0].local_encoder[1], nn.GELU)
        self.assertEqual(encoders[0].local_encoder[0].kernel_size, (3,))
        self.assertEqual(encoders[0].local_encoder[2].kernel_size, (1,))
        self.assertFalse({id(p) for p in encoders[0].local_encoder.parameters()} &
                         {id(p) for p in encoders[1].local_encoder.parameters()})
        seen = []
        encoders[0].local_encoder.register_forward_pre_hook(
            lambda module, args: seen.append(tuple(args[0].shape)))
        encoders[0].encode_period(torch.randn(2, 32), 5)
        self.assertEqual(seen, [(14, 1, 5)])  # 2 series x ceil(32/5) separate patches.

    def test_identical_patches_receive_distinct_positions_before_transformer(self):
        encoder = PeriodPatchEncoder(16, 8, 2, 16, 1, 0.0, 3)
        embedded, transformer_inputs = [], []
        encoder.patch_embedding.register_forward_hook(
            lambda module, args, out: embedded.append(out.detach().clone()))
        encoder.blocks[0].register_forward_pre_hook(
            lambda module, args: transformer_inputs.append(args[0].detach().clone()))
        repeated_patch = torch.tensor([1., -1., 2., -2.]).repeat(2, 4)
        encoder.encode_period(repeated_patch, 4)
        torch.testing.assert_close(embedded[0], embedded[0][:, :1].expand(-1, 4, -1))
        positions = encoder.position_encoding[:, :4]
        torch.testing.assert_close(transformer_inputs[0], embedded[0] + positions)
        for index in range(1, 4):
            self.assertFalse(torch.equal(transformer_inputs[0][:, 0],
                                         transformer_inputs[0][:, index]))
        self.assertFalse(encoder.position_encoding.requires_grad)

    def test_invalid_configuration_and_input_fail_early(self):
        invalid = (dict(seq_len=95), dict(seq_len=8), dict(pred_len=0),
                   dict(d_model=7), dict(n_heads=0), dict(e_layers=0),
                   dict(top_k=0), dict(moving_avg=4), dict(dropout=1),
                   dict(down_sampling_layers=2), dict(down_sampling_window=3),
                   dict(down_sampling_method="conv"), dict(channel_independence=0),
                   dict(use_norm=0), dict(features="S"), dict(c_out=2),
                   dict(features="unknown"), dict(task_name="classification"))
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                Model(config(**change))
        with self.assertRaisesRegex(ValueError, "Expected input"):
            Model(config())(torch.randn(2, 16, 3))


if __name__ == "__main__":
    unittest.main()
