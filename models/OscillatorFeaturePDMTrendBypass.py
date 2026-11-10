"""Feature PDM with an optional original raw-trend direct prediction bypass.

The PDM encoder, normalization, routing and recurrence are unchanged. The sole
added path decomposes the finest normalized observed history, downsamples its
MA trend, applies the original coarse-to-fine mixer and four direct heads, and
adds their normalized forecasts before the existing single inverse Normalize.
"""
import copy
import torch
from torch import nn
from layers.Autoformer_EncDec import series_decomp
from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorTimeMix import complex_oscillator_scan
from models.TimeMixer import MultiScaleTrendMixing
from models.OscillatorFeaturePDMTimeMix import Model as FeaturePDM


class Model(FeaturePDM):
    def __init__(self, configs, enabled=None):
        super().__init__(configs, enabled=True)
        self.raw_trend_bypass = bool(getattr(configs, 'raw_trend_bypass', False) if enabled is None else enabled)
        if self.raw_trend_bypass:
            cfg = copy.copy(configs)
            cfg.down_sampling_layers, cfg.down_sampling_window = 3, 2
            self.raw_trend_decomposition = series_decomp(configs.moving_avg)
            self.raw_trend_mix = MultiScaleTrendMixing(cfg)
            self.raw_trend_heads = nn.ModuleList([nn.Linear(t, self.pred_len) for t in self.lengths])
            for length, head in zip(self.lengths, self.raw_trend_heads):
                nn.init.constant_(head.weight, 1. / length)
                nn.init.zeros_(head.bias)

    def predict_raw_trend(self, normalized_history):
        """[B,96,C] -> [B,96,C]; exactly the old trend bypass equations."""
        _, trend = self.raw_trend_decomposition(normalized_history)
        scales = self.raw_trend_mix([x.transpose(1, 2) for x in self.downsample(trend)])
        return torch.stack([head(x.transpose(1, 2)).transpose(1, 2)
                            for head, x in zip(self.raw_trend_heads, scales)], -1).sum(-1)

    def forecast(self, x_enc, zero_acf=False, intervention='normal'):
        if not self.raw_trend_bypass:
            return super().forecast(x_enc, zero_acf, intervention)
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected observed history [B,seq_len,enc_in]')
        if zero_acf or intervention != 'normal':
            raise ValueError('Trend bypass study supports normal forecasting only')
        scales, histories = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        outputs = [calibrated_seasonal_forecast(e, x, alpha, beta, encoded_history=h,
                   scan_function=complex_oscillator_scan) for e, x, h in zip(self.seasonals, scales, histories)]
        prediction = torch.stack(outputs, -1).sum(-1) + self.predict_raw_trend(scales[0])
        prediction = self.normalize(prediction, 'denorm')
        return prediction[..., -1:] if self.features == 'MS' and self.c_out == 1 else prediction
