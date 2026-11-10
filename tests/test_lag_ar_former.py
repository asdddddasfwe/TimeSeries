"""Contracts for FFT encoding and synchronous lag message forecasting."""
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from layers.LagARFormer import autocorrelation, lag_statistics, LagMessageBlock, LagAxialBlock, select_fft_periods
from layers.pypots_timemixerpp.layers import RowAttention, ColAttention
from models.LagARFormer import Model


def config(**changes):
    values = dict(task_name="long_term_forecast", seq_len=32, pred_len=12,
                  enc_in=3, c_out=3, features="M", moving_avg=5, d_model=8,
                  d_ff=16, n_heads=2, e_layers=2, dropout=0.0, top_k=4,
                  down_sampling_layers=3, down_sampling_window=2, down_sampling_method="avg",
                  channel_independence=1, use_norm=1,
                  lag_max=0, lag_layers=2, lag_chunk_size=4)
    values.update(changes)
    return SimpleNamespace(**values)


class LagARFormerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(2021)

    def test_acf_exact_normalization_and_no_circular_wrap(self):
        x = torch.tensor([[1., -1., 1., -1., 1., -1., 1., -1.]])
        expected = torch.tensor([[-7/8, 6/8, -5/8, 4/8, -3/8, 2/8, -1/8]])
        torch.testing.assert_close(autocorrelation(x + 100, 7), expected)
        lags, features = lag_statistics(x, 7)
        self.assertEqual(lags.tolist(), [list(range(1, 8))])
        torch.testing.assert_close(features[0, :3, 1], expected[0, :3])
        torch.testing.assert_close(features[0, :3, 5], torch.tensor([7/8, 6/8, 5/8]))
        torch.testing.assert_close(features[0, :3, 6], torch.tensor([3/4, 2/4, 1/4]))

    def test_known_period_and_window_change_statistics(self):
        t = torch.arange(96).float()
        x = torch.cos(2 * torch.pi * t / 12)[None]
        acf = autocorrelation(x, 24)
        self.assertAlmostEqual(acf[0, 11].item(), 84 / 96, places=5)
        self.assertAlmostEqual(acf[0, 23].item(), 72 / 96, places=5)
        changed = torch.cat((x[:, :48], torch.cos(2 * torch.pi * t[:48] / 8)[None]), -1)
        _, features = lag_statistics(changed, 24)
        self.assertGreater(features[..., 4].abs().max().item(), 0.5)

    def test_all_lags_constant_and_short_window_reliability(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            lags, features = lag_statistics(torch.ones(2, 8, dtype=dtype), 7)
            self.assertEqual(lags[0].tolist(), list(range(1, 8)))
            self.assertTrue(torch.isfinite(features).all())
            self.assertTrue((features[..., 1:5] == 0).all())
            self.assertTrue((features[:, 3:, 6] == 0).all())


    def test_one_decomposition_four_scales_and_linear_trends(self):
        model = Model(config())
        self.assertEqual(model.lengths,[32,16,8,4])
        self.assertEqual(model.horizons,[12,6,3,2])
        self.assertEqual(len(model.seasonals),4)
        for length,horizon,head in zip(model.lengths,model.horizons,model.trend_heads):
            self.assertIsInstance(head,nn.Linear)
            torch.testing.assert_close(head.weight,torch.full((horizon,length),1/length))
            self.assertTrue((head.bias==0).all())
        x = torch.randn(2,32,3)
        season,trend = model.decomposition(x)
        torch.testing.assert_close(season+trend,x)
        with patch.object(model.decomposition,"forward",wraps=model.decomposition.forward) as decomp:
            with patch.object(model.normalize,"forward",wraps=model.normalize.forward) as norm:
                result=model(x)
        self.assertEqual(decomp.call_count,1)
        self.assertEqual([c.args[1] for c in norm.call_args_list],["norm","denorm"])
        self.assertFalse(any(isinstance(m,nn.TransformerEncoderLayer) for m in model.modules()))
        self.assertEqual(tuple(result.shape),(2,12,3))

    def test_all_learned_branches_receive_finite_gradients_and_update(self):
        model = Model(config())
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Upstream gamma=0 intentionally blocks Q/K/V gradients at step zero.
        for step in range(2):
            optimizer.zero_grad()
            output = model(torch.randn(3, 32, 3))
            loss = (output - torch.randn_like(output)).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if step == 0 and "_conv." in name:
                    self.assertEqual(parameter.grad.abs().max().item(), 0, name)
                if step == 1 and name.endswith("weight"):
                    self.assertGreater(parameter.grad.abs().max().item(), 0, name)
            optimizer.step()
        for name, parameter in model.named_parameters():
            if name.endswith("weight") or name.endswith("gamma"):
                self.assertFalse(torch.equal(before[name], parameter), name)

    def test_eval_batch_and_channel_independence(self):
        model = Model(config()).eval()
        with torch.no_grad():
            for block in model.seasonals[0].blocks:
                block.row.gamma.fill_(0.2)
                block.column.gamma.fill_(0.3)
        x = torch.randn(3, 32, 3)
        expected = model(x)
        separate = torch.cat([model(row[None]) for row in x], 0)
        torch.testing.assert_close(expected, separate, rtol=1e-5, atol=1e-6)
        order = [2, 0, 1]
        torch.testing.assert_close(model(x[:, :, order]), expected[:, :, order], rtol=1e-5, atol=1e-6)

    def test_constant_and_short_term_s_ms_and_future_isolation(self):
        constant = torch.tensor([1., 4., -10.])[None, None].expand(2, 32, 3)
        model = Model(config()).eval()
        self.assertTrue(torch.isfinite(model(constant)).all())
        single = Model(config(task_name="short_term_forecast", enc_in=1, c_out=1, features="S"))
        self.assertEqual(tuple(single(torch.randn(2, 32, 1)).shape), (2, 12, 1))
        ms = Model(config(features="MS", c_out=1)).eval()
        ms.load_state_dict(model.state_dict(), strict=True)
        x = torch.randn(2, 32, 3) * torch.tensor([1., 2., 10.]) + torch.tensor([0., 8., -30.])
        torch.testing.assert_close(ms(x), model(x)[:, :, -1:], rtol=0, atol=0)
        expected = model(x)
        actual = model(x, torch.full((2, 32, 4), float("nan")),
                       torch.full((2, 12, 3), float("nan")),
                       torch.full((2, 12, 4), float("nan")))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_strict_checkpoint_roundtrip(self):
        model = Model(config()).eval()
        x = torch.randn(2, 32, 3)
        expected = model(x)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = Model(config()).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        torch.testing.assert_close(expected, restored(x), atol=0, rtol=0)


    def test_upstream_row_and_column_directions(self):
        image = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()
        for cls, axis in ((RowAttention, -1), (ColAttention, -2)):
            attention = cls(3, 4)
            with torch.no_grad():
                for module in (attention.query_conv, attention.key_conv, attention.value_conv):
                    module.weight.zero_()
                    module.bias.zero_()
                for channel in range(3):
                    attention.value_conv.weight[channel, channel, 0, 0] = 1.
                attention.gamma.fill_(1.)
            expected = image + image.mean(axis, keepdim=True)
            torch.testing.assert_close(attention(image), expected)

    def test_axial_branches_are_parallel_with_one_residual(self):
        block = LagAxialBlock(8, 16, 0.)
        self.assertIsInstance(block.row, RowAttention)
        self.assertIsInstance(block.column, ColAttention)
        self.assertEqual(block.row.gamma.item(), 0.)
        self.assertEqual(block.column.gamma.item(), 0.)
        image = torch.randn(2, 8, 7, 5)
        with torch.no_grad():
            block.row.gamma.fill_(0.2)
            block.column.gamma.fill_(0.3)
        expected = block.norm(
            (image + (block.row(image) - image) + (block.column(image) - image))
            .permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        torch.testing.assert_close(block(image), expected, atol=1e-6, rtol=1e-5)

    def test_fused_axial_matches_original_values_and_all_gradients(self):
        for width, qk in ((8, 16), (7, 4), (8, 8)):
            block = LagAxialBlock(width, qk, 0.).double()
            with torch.no_grad():
                block.row.gamma.fill_(0.4)
                block.column.gamma.fill_(-0.2)
            x = torch.randn(2, width, 3, 5, dtype=torch.double, requires_grad=True)
            actual = block(x)
            expected = block.norm(
                (x + (block.row(x) - x) + (block.column(x) - x))
                .permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
            parameters = (x,) + tuple(block.parameters())
            ga = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
            gb = torch.autograd.grad(expected.square().sum(), parameters)
            for a, b in zip(ga, gb):
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-7)

    def test_fft_acf_matches_direct_values_and_gradients(self):
        for length in (1, 7, 24, 96):
            x = (torch.randn(3, length, dtype=torch.double) + 13).requires_grad_()
            max_lag = length + 5
            z = x - x.mean(-1, keepdim=True)
            energy = z.square().sum(-1).clamp_min(1e-8)
            expected = torch.stack([
                (z[:, lag:] * z[:, :-lag]).sum(-1) / energy
                if lag < length else energy * 0
                for lag in range(1, max_lag + 1)
            ], -1)
            actual = autocorrelation(x, max_lag)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
            a = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
            b = torch.autograd.grad(expected.square().sum(), x)[0]
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
            self.assertTrue((actual[:, length - 1:] == 0).all())


    def test_fft_image_padding_restore(self):
        encoder = Model(config()).seasonals[0]
        encoder.blocks = nn.ModuleList([nn.Identity()])
        time = torch.arange(2 * 32 * 8).reshape(2, 32, 8).float()
        for period in (2, 5, 16, 32):
            seen = []
            handle = encoder.blocks[0].register_forward_pre_hook(
                lambda m, args: seen.append(args[0].detach().clone()))
            actual = encoder.encode_image(time, period)
            handle.remove()
            self.assertEqual(tuple(seen[0].shape), (2, 8, (32 + period - 1) // period, period))
            restored_padded = seen[0].permute(0, 2, 3, 1).reshape(2, -1, 8)
            self.assertTrue((restored_padded[:, 32:] == 0).all())
            torch.testing.assert_close(actual, time)

    def test_grouped_fft_periods_merge_duplicates_and_preserve_gradients(self):
        encoder = Model(config()).seasonals[0].eval()
        with torch.no_grad():
            for block in encoder.blocks:
                block.row.gamma.fill_(0.2)
                block.column.gamma.fill_(0.3)
        time = torch.randn(2, 32, 8, requires_grad=True)
        weights = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], requires_grad=True)
        periods = torch.tensor([[5, 5, 16], [16, 3, 3]])
        with patch.object(encoder, "encode_image", wraps=encoder.encode_image) as imaging:
            grouped = encoder.encode_periods(time, periods, weights)
        # Three distinct widths, each sample processed once per width.
        self.assertEqual(imaging.call_count, 3)
        sizes = {call.args[1]: call.args[0].shape[0] for call in imaging.call_args_list}
        self.assertEqual(sizes, {3: 1, 5: 1, 16: 2})
        separate = torch.stack([
            sum(encoder.encode_image(time[n:n+1], int(periods[n, k]))[0] * weights[n, k]
                for k in range(3))
            for n in range(2)])
        torch.testing.assert_close(grouped, separate, rtol=1e-5, atol=1e-6)
        grouped[..., 0].sum().backward()
        self.assertTrue(torch.isfinite(time.grad).all())
        self.assertTrue(torch.isfinite(weights.grad).all())
        self.assertTrue((weights.grad.abs() > 0).all())

    def test_fft_peaks_period_definition_dc_constants_and_topk(self):
        t = torch.arange(96).float()
        x = torch.stack([torch.cos(2 * torch.pi * t * frequency / 96) + 50
                         for frequency in (4, 5, 12)])
        periods, weights = select_fft_periods(x, 1)
        self.assertEqual(periods[:, 0].tolist(), [24, 19, 8])
        torch.testing.assert_close(weights, torch.ones_like(weights))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            periods, weights = select_fft_periods(torch.ones(2, 32, dtype=dtype), 100)
            self.assertEqual(tuple(periods.shape), (2, 16))
            self.assertEqual(periods[0].tolist(), [32 // f for f in range(1, 17)])
            self.assertTrue(torch.isfinite(weights).all())
            torch.testing.assert_close(weights.sum(-1), torch.ones(2))


    def test_axial_depth_shared_parameters_and_no_multihead_option(self):
        model = Model(config(d_model=7, n_heads=4, e_layers=3))
        self.assertEqual(len(model.seasonals[0].blocks), 3)
        self.assertEqual(sum(isinstance(m, RowAttention) for m in model.modules()), 12)
        self.assertEqual(sum(isinstance(m, ColAttention) for m in model.modules()), 12)
        self.assertEqual(model.seasonals[0].blocks[0].row.query_conv.out_channels, 16)
        self.assertTrue(torch.isfinite(model(torch.randn(2, 32, 3))).all())
        # n_heads is accepted by the framework but does not change this encoder.
        torch.manual_seed(1)
        first = Model(config(n_heads=1))
        torch.manual_seed(1)
        second = Model(config(n_heads=7))
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)


    def test_full_time_features_all_lags_and_future_positions(self):
        model = Model(config(top_k=1))
        captured = []
        handles = [
            model.seasonals[0].temporal_projection.register_forward_pre_hook(
                lambda m, args: captured.append(("projection", args[0].shape))),
            model.seasonals[0].message_blocks[0].register_forward_pre_hook(
                lambda m, args: captured.append(("messages", tuple(v.shape for v in args)))),
        ]
        with patch("torch.linalg.solve_triangular", side_effect=AssertionError("AR solve removed")):
            output = model(torch.randn(2, 32, 3))
        for handle in handles:
            handle.remove()
        self.assertEqual(captured[0], ("projection", torch.Size([6, 8, 32])))
        self.assertEqual(captured[1][1], (torch.Size([6,32,8]), torch.Size([6,12,8]),
                                         torch.Size([6,31,8])))
        self.assertEqual(output.shape, (2,12,3))
        names = list(model.state_dict())
        self.assertFalse(any("forcing" in name or "coefficient" in name for name in names))
        positions = model.seasonals[0].future_positions
        self.assertFalse(torch.equal(positions[0], positions[1]))

    def test_message_indices_include_history_and_previous_future(self):
        block = LagMessageBlock(4,4,1,2,3,2,0.)
        expected_indices = torch.tensor([[3,2,1], [4,3,2], [5,4,3], [6,5,4]])
        torch.testing.assert_close(block.source_indices, expected_indices)
        source = torch.arange(8.).reshape(1,8,1)
        target, lag = torch.zeros(1,4,1), torch.zeros(1,3,1)
        output = block.message_sum(target, source, lag, block.source_indices)
        expected = F.gelu(source[:, expected_indices, :]).sum(2)
        torch.testing.assert_close(output, expected)
        # Sources are strictly before the target in concatenated history/future time.
        self.assertTrue((block.source_indices < 4 + torch.arange(4)[:,None]).all())

    def test_synchronous_updates_have_one_hop_per_layer(self):
        class TakeMessages(nn.Module):
            def forward(self, x):
                return x[...,1:]
        block = LagMessageBlock(2,4,1,2,1,1,0.).eval()
        block.state_norm = nn.Identity()
        block.message_norm = nn.Identity()
        block.update = TakeMessages()
        with torch.no_grad():
            block.target_projection.weight.zero_()
            block.target_projection.bias.zero_()
            block.source_projection.weight.fill_(1.)
            block.lag_projection.weight.zero_()
        history, tokens = torch.zeros(1,2,1), torch.zeros(1,1,1)
        initial = torch.tensor([[[2.],[0.],[0.],[0.]]])
        original = initial.clone()
        first = block(history, initial, tokens)
        second = block(history, first, tokens)
        torch.testing.assert_close(initial, original, atol=0, rtol=0)
        self.assertGreater(first[0,1,0].item(), 0)
        self.assertEqual(first[0,2,0].item(), 0.)
        self.assertGreater(second[0,2,0].item(), 0)
        self.assertEqual(second[0,3,0].item(), 0.)

    def test_chunked_checkpoint_matches_scalar_reference_outputs_and_gradients(self):
        block = LagMessageBlock(6,5,4,8,5,2,0.).double().train()
        history = torch.randn(1,6,4,dtype=torch.double,requires_grad=True)
        future = torch.randn(1,5,4,dtype=torch.double,requires_grad=True)
        lag = torch.randn(1,5,4,dtype=torch.double,requires_grad=True)
        actual = block(history,future,lag)
        normalized = block.state_norm(future)
        target = block.target_projection(normalized)
        source = block.source_projection(torch.cat((block.state_norm(history),normalized),1))
        relation = block.lag_projection(lag)
        summed = []
        for h in range(5):
            message = sum(F.gelu(target[:,h] + source[:,6+h-tau] + relation[:,tau-1])
                          for tau in range(1,6))
            summed.append(message / (5 ** 0.5))
        messages = torch.stack(summed,1)
        expected = future + block.update(torch.cat((normalized,block.message_norm(messages)),-1))
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        inputs = (history,future,lag) + tuple(block.parameters())
        ga = torch.autograd.grad(actual.square().sum(),inputs,retain_graph=True)
        gb = torch.autograd.grad(expected.square().sum(),inputs)
        for a,b in zip(ga,gb):
            torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-8)

    def test_chunk_size_does_not_select_or_discard_lags(self):
        first = LagMessageBlock(6,5,4,8,5,1,0.).eval()
        second = LagMessageBlock(6,5,4,8,5,20,0.).eval()
        second.load_state_dict(first.state_dict(),strict=True)
        history,future,lag = torch.randn(2,6,4),torch.randn(2,5,4),torch.randn(2,5,4)
        torch.testing.assert_close(first(history,future,lag),second(history,future,lag),
                                   atol=1e-6,rtol=1e-5)

    def test_lag_statistics_affect_predictions_and_all_receive_gradients(self):
        model = Model(config(seq_len=16,lag_chunk_size=4)).eval()
        x = torch.randn(2,16)
        lags,stats = lag_statistics(x,15)
        stats = stats.detach().requires_grad_()
        with patch("layers.LagARFormer.lag_statistics",return_value=(lags,stats)):
            original = model.seasonals[0].forecast_features(x)
            original[...,0].sum().backward()
        self.assertTrue(torch.isfinite(stats.grad).all())
        self.assertTrue((stats.grad.abs().sum(-1)>0).all())
        changed = stats.detach().clone()
        changed[...,1:5] = -changed[...,1:5]
        with patch("layers.LagARFormer.lag_statistics",return_value=(lags,changed)):
            other = model.seasonals[0].forecast_features(x)
        self.assertGreater((original-other).abs().max().item(),1e-6)

    def test_period_imaging_is_independent_of_message_lag_range(self):
        x = torch.cos(2*torch.pi*torch.arange(32).float()/8)[None]
        encoder = Model(config(top_k=1,lag_max=3)).seasonals[0]
        with patch("layers.LagARFormer.select_fft_periods", wraps=select_fft_periods) as fft:
            with patch.object(encoder,"encode_image",wraps=encoder.encode_image) as image:
                output = encoder.forecast_features(x)
        self.assertEqual(fft.call_count,1)
        self.assertEqual([call.args[1] for call in image.call_args_list],[8])
        self.assertEqual(encoder.message_blocks[0].max_lag,3)
        self.assertEqual(output.shape,(1,12,8))

    def test_long_horizons_and_backward_remain_finite(self):
        for horizon in (96,192,336,720):
            model = Model(config(seq_len=16,pred_len=horizon,enc_in=1,c_out=1,top_k=2))
            x = torch.randn(1,16,1)
            output = model(x)
            self.assertEqual(output.shape,(1,horizon,1))
            loss = output.square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_single_step_and_arbitrary_width(self):
        model = Model(config(pred_len=1,d_model=7,n_heads=4))
        self.assertTrue(torch.isfinite(model(torch.randn(2,32,3))).all())

    def test_v4_checkpoint_rejects_previous_decoder_state(self):
        model = Model(config())
        state = model.state_dict()
        self.assertEqual(state["seasonals.0.lag_message_encoder_version"].item(),4)
        del state["seasonals.0.lag_message_encoder_version"]
        state["seasonal.fft_period_encoder_version"] = torch.tensor(3)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state,strict=True)

    def test_aligned_pooling_reconstructs_each_downsampled_input(self):
        model=Model(config(seq_len=96))
        x=torch.arange(96.).reshape(1,96,1)
        season,trend=model.decomposition(x)
        seasons,trends=model.downsample(season),model.downsample(trend)
        for stride,length,s,t in zip(model.strides,model.lengths,seasons,trends):
            expected=x.reshape(1,length,stride,1).mean(2)
            torch.testing.assert_close(s+t,expected)
            torch.testing.assert_close(model.downsample(x)[model.strides.index(stride)],expected)
        self.assertEqual(model.lengths,[96,48,24,12])

    def test_native_forecast_lengths_and_lag_units(self):
        model=Model(config(seq_len=96,pred_len=13,enc_in=1,c_out=1))
        self.assertEqual(model.horizons,[13,7,4,2])
        self.assertEqual(model.lag_limits,[95,47,23,11])
        seen=[]
        handles=[]
        for scale,encoder in enumerate(model.seasonals):
            def capture(module,args,output,scale=scale):
                seen.append((scale,args[0].shape[1],output.shape[1]))
            handles.append(encoder.register_forward_hook(capture))
        self.assertEqual(model(torch.randn(1,96,1)).shape,(1,13,1))
        for h in handles:
            h.remove()
        self.assertEqual(seen,[(0,96,13),(1,48,7),(2,24,4),(3,12,2)])
        for stride,length,encoder in zip(model.strides,model.lengths,model.seasonals):
            indices=encoder.message_blocks[0].source_indices
            self.assertEqual(indices[0,0].item(),length-1)
            self.assertEqual(indices[1,0].item(),length)
            # Local future bin 1 reads bin 0 at lag 1: one stride apart.
            self.assertEqual((length+1-indices[1,0].item())*stride,stride)

    def test_restore_resolution_repeats_aligned_bins_and_crops_partial_tail(self):
        model=Model(config(pred_len=13))
        for stride,horizon in zip(model.strides,model.horizons):
            native=torch.arange(horizon,dtype=torch.float32).reshape(1,horizon,1)
            restored=model.restore_resolution(native,stride)
            expected=(torch.arange(13)//stride).float().reshape(1,13,1)
            torch.testing.assert_close(restored,expected,atol=0,rtol=0)

    def test_scale_fusion_starts_as_mean_and_is_learnable(self):
        model=Model(config()).eval()
        torch.testing.assert_close(model.scale_fusion.weight,torch.full((1,4),0.25))
        x=torch.randn(2,32,3)
        with torch.no_grad():
            season,trend=model.decomposition(model.normalize(x,"norm"))
            scales=model.predict_scales(season,trend)
            expected=model.normalize(scales.mean(-1),"denorm")
            torch.testing.assert_close(model(x),expected,atol=1e-6,rtol=1e-5)
            model.scale_fusion.weight.zero_()
            model.scale_fusion.weight[0,2]=1.
            expected=model.normalize(scales[...,2],"denorm")
            torch.testing.assert_close(model(x),expected,atol=1e-6,rtol=1e-5)
        self.assertTrue(model.scale_fusion.weight.requires_grad)

    def test_scale_parameters_independent_and_local_lag_cap_clamped(self):
        model=Model(config(lag_max=10))
        self.assertEqual(model.lag_limits,[10,10,7,3])
        self.assertEqual(len({id(s.value_embedding.weight) for s in model.seasonals}),4)
        self.assertEqual(len({id(h.weight) for h in model.trend_heads}),4)
        self.assertEqual(model.multiresolution_version.item(),5)
        for changes in (dict(seq_len=24,lag_max=0),dict(pred_len=1)):
            other=Model(config(**changes))
            self.assertTrue(torch.isfinite(other(torch.randn(1,other.seq_len,3))).all())
        for changes in (dict(seq_len=17),dict(down_sampling_window=3),
                        dict(down_sampling_method="conv")):
            with self.assertRaises(ValueError):
                Model(config(**changes))

    def test_invalid_settings(self):
        for changes in (dict(seq_len=3),dict(pred_len=0),dict(enc_in=0),
                        dict(moving_avg=4),dict(d_model=0),dict(top_k=0),
                        dict(lag_max=32),dict(lag_max=-1),dict(lag_layers=0),
                        dict(lag_chunk_size=0),dict(ar_rho=0.98),
                        dict(down_sampling_layers=0),dict(channel_independence=0),
                        dict(use_norm=0),dict(features="S"),dict(c_out=2),
                        dict(task_name="classification"),dict(dropout=1.)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                Model(config(**changes))


if __name__ == "__main__":
    unittest.main()
