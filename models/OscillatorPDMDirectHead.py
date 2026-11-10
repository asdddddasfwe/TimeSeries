"""Replace the seasonal oscillator decoder by direct affine feature readout.

Input normalization, sole MA/FFT encoding, PDM and feature-trend heads stay
unchanged. Preserve temporal projection and future PE; remove the complete
oscillator decoder, including its internal drive norm, gates, trend condition
and history/drive gain parameters. This is a decoder-family comparison.
"""
import torch
from torch import nn
from models.OscillatorPDMTrendCondition import Model as OscillatorModel
from utils.logger import get_logger


class Model(OscillatorModel):
    def __init__(self, configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            width = encoder.value_embedding.out_features
            del encoder.decoder
            encoder.direct_readout = nn.Linear(width, 1)
        del self.history_gain_offset
        del self.drive_gain_offset
        get_logger(__name__).info(
            'OscillatorPDMDirectHead OVERRIDES inherited decoder logs: no oscillator decoder, '
            'history scan, future gate, drive norm or trend-conditioned dynamics; '
            'retained temporal projection plus future PE -> Linear(D,1); '
            'input normalization, one MA/FFT, PDM and feature-trend heads unchanged')

    def optimizer_parameter_groups(self, weight_decay):
        # No oscillator dynamics remain. External grouping retains norm/bias exemptions.
        return [dict(params=list(self.parameters()), weight_decay=weight_decay,
                     group_name='regular')]

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected history [B,seq_len,enc_in]')
        _, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        batch, _, channels = x_enc.shape
        seasonal = []
        for encoder, history in zip(self.seasonals, seasons):
            # [B*C,T,D] -> [B*C,H,D] -> [B,H,C]. No recurrent state or drive.
            future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
            future = future + encoder.future_positions.to(future)
            output = encoder.direct_readout(future).squeeze(-1)
            seasonal.append(output.reshape(batch, channels, self.pred_len).transpose(1, 2))
        trend = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                 for head, t in zip(self.feature_trend_heads, trends)]
        return torch.stack(seasonal + trend, -1)
