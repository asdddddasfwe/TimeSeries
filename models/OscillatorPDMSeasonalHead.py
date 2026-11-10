"""Final PDM seasonal features feed oscillators; trend features use linear heads.

Earlier FFT/PDM layers still exchange context. This separates the prediction
inputs, not the entire encoder or independently supervised signal components.
"""
import torch
from torch import nn

from layers.LagARFormer import select_fft_periods
from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorTimeMix import complex_oscillator_scan
from models.OscillatorFeaturePDMTimeMix import encode_one_fft_layer
from models.OscillatorPDMScaledMA import Model as ScaledMA
from utils.logger import get_logger


class SeparateFinalPDM(nn.Module):
    """Reuse the last block's weights, but never merge its two decoder inputs.

    Split the original residual too: S=s+MLP(mix_s), T=t+MLP(mix_t).
    The channel MLP is shared between branches. Its nonlinear evaluations are
    separate, so their sum is not the old MLP(mix_s+mix_t) output.
    """

    def __init__(self, block):
        super().__init__()
        if not block.channel_independence:
            raise ValueError("Separate final PDM requires channel independence")
        self.decompsition = block.decompsition
        self.mixing_multi_scale_season = block.mixing_multi_scale_season
        self.mixing_multi_scale_trend = block.mixing_multi_scale_trend
        self.out_cross_layer = block.out_cross_layer

    def forward(self, histories):
        pairs = [self.decompsition(x) for x in histories]
        seasons, trends = zip(*pairs)
        mixed_s = self.mixing_multi_scale_season([x.transpose(1, 2) for x in seasons])
        mixed_t = self.mixing_multi_scale_trend([x.transpose(1, 2) for x in trends])
        season_out = [s + self.out_cross_layer(m) for s, m in zip(seasons, mixed_s)]
        trend_out = [t + self.out_cross_layer(m) for t, m in zip(trends, mixed_t)]
        return season_out, trend_out


class LinearFeatureTrendHead(nn.Module):
    """[B*C,T,D] -> [B*C,H] through two affine maps, without recurrence."""

    def __init__(self, length, horizon, width):
        super().__init__()
        self.temporal = nn.Linear(length, horizon)
        self.readout = nn.Linear(width, 1)
        nn.init.constant_(self.temporal.weight, 1. / length)
        nn.init.zeros_(self.temporal.bias)
        nn.init.zeros_(self.readout.bias)

    def forward(self, history):
        future = self.temporal(history.transpose(1, 2)).transpose(1, 2)
        return self.readout(future).squeeze(-1)


class Model(ScaledMA):
    """Four seasonal recurrent heads + four feature-trend linear heads + bypass."""

    def __init__(self, configs):
        super().__init__(configs)
        if not self.pdm_blocks:
            raise ValueError("Seasonal-head PDM requires at least one encoder layer")
        self.pdm_blocks[-1] = SeparateFinalPDM(self.pdm_blocks[-1])
        self.feature_trend_heads = nn.ModuleList([
            LinearFeatureTrendHead(length, self.pred_len, configs.d_model)
            for length in self.lengths
        ])
        self.register_buffer('seasonal_head_version', torch.tensor(1, dtype=torch.long))
        get_logger(__name__).info(
            "OscillatorPDMSeasonalHead overrides inherited combined-PDM routing: "
            "final PDM S=s+MLP(mix_s) feeds four oscillator heads; "
            "T=t+MLP(mix_t) feeds four affine feature-trend heads; "
            "MLP weights shared but evaluations separate; earlier PDM retains "
            "mixed context; original raw-trend bypass retained; 12 outputs, "
            "one inverse Normalize; old 8-component diagnostics incompatible")

    def encode_separated_histories(self, raw_scales):
        """Return normalized inputs and four [B*C,T_i,D] tensors per branch."""
        normalized = [self.normalize(raw_scales[0], 'norm')]
        normalized += [norm(x, 'norm') for norm, x in zip(self.coarse_normalizers, raw_scales[1:])]
        batch, _, channels = normalized[0].shape
        histories, routing = [], []
        for encoder, x in zip(self.seasonals, normalized):
            flat = x.transpose(1, 2).reshape(batch * channels, encoder.length)
            routing.append(select_fft_periods(flat, encoder.period_top_k))
            tokens = encoder.value_embedding(flat[..., None])
            histories.append(tokens + encoder.positions.to(tokens))
        for index, block in enumerate(self.pdm_blocks):
            histories = [encode_one_fft_layer(e, h, *route, index)
                         for e, h, route in zip(self.seasonals, histories, routing)]
            if index == len(self.pdm_blocks) - 1:
                seasons, trends = block(histories)
                return normalized, seasons, trends
            histories = block(histories)
        raise RuntimeError("Missing final PDM layer")

    def predict_components(self, x_enc):
        """Normalized [B,H,C,12]: seasonal 4, feature-trend 4, raw-trend 4.

        Computed in the same forward, without retaining tensor references on
        the model. Sum and inverse-normalize exactly once to obtain forecasts.
        """
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected observed history [B,seq_len,enc_in]")
        scales, seasons, trends = self.encode_separated_histories(self.downsample(x_enc))
        batch, _, channels = x_enc.shape
        alpha, beta = self.branch_gains()
        seasonal = [calibrated_seasonal_forecast(e, x, alpha, beta, encoded_history=s,
                    scan_function=complex_oscillator_scan)
                    for e, x, s in zip(self.seasonals, scales, seasons)]
        feature_trend = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                         for head, t in zip(self.feature_trend_heads, trends)]
        _, raw_trend = self.raw_trend_decomposition(scales[0])
        raw_scales = self.raw_trend_mix([x.transpose(1, 2) for x in self.downsample(raw_trend)])
        raw_predictions = [head(x.transpose(1, 2)).transpose(1, 2)
                           for head, x in zip(self.raw_trend_heads, raw_scales)]
        return torch.stack(seasonal + feature_trend + raw_predictions, dim=-1)

    def forecast(self, x_enc, zero_acf=False, intervention='normal'):
        if zero_acf or intervention != 'normal':
            raise ValueError("Seasonal-head PDM supports the normal forecasting path only")
        prediction = self.normalize(self.predict_components(x_enc).sum(-1), 'denorm')
        return prediction[..., -1:] if self.features == 'MS' and self.c_out == 1 else prediction
