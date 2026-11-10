"""Controlled PDM readout interventions with the unchanged raw-trend bypass."""
import torch
from torch import nn
from models.OscillatorFeaturePDMTrendBypass import Model as TrendBypass
from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorTimeMix import complex_oscillator_scan


class Model(TrendBypass):
    def __init__(self, configs, variant=None):
        super().__init__(configs, enabled=True)
        self.scale_variant = variant or getattr(configs, 'scale_variant', 'bypass')
        if self.scale_variant not in ('bypass', 'drop_pdm12', 'channel_weights'):
            raise ValueError('Unknown scale intervention')
        if self.scale_variant == 'channel_weights':
            self.scale_gate_logits = nn.Parameter(torch.zeros(4, self.enc_in))

    def scale_weights(self):
        # Static channel-specific gains, shared over windows and forecast time.
        if self.scale_variant == 'channel_weights':
            return 2 * torch.sigmoid(self.scale_gate_logits)
        weight = next(self.parameters()).new_ones(4, self.enc_in)
        if self.scale_variant == 'drop_pdm12':
            weight[3] = 0
        return weight

    def forecast(self, x_enc, zero_acf=False, intervention='normal'):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected observed history [B,seq_len,enc_in]')
        if zero_acf or intervention != 'normal':
            raise ValueError('Scale study supports normal forecasting only')
        scales, histories = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        outputs = [calibrated_seasonal_forecast(e, x, alpha, beta, encoded_history=h,
                   scan_function=complex_oscillator_scan)
                   for e, x, h in zip(self.seasonals, scales, histories)]
        # Each output is [B,96,C]; weights are [scale,C]. Keep all encoders/PDM.
        weights = self.scale_weights()
        prediction = torch.stack([out * weights[i] for i, out in enumerate(outputs)], -1).sum(-1)
        prediction = prediction + self.predict_raw_trend(scales[0])
        prediction = self.normalize(prediction, 'denorm')
        return prediction[..., -1:] if self.features == 'MS' and self.c_out == 1 else prediction
