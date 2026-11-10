"""Disconnected seasonal dynamics plus an independent recurrent trend head."""
import copy

import torch
from torch import nn

from models.OscillatorPDMStatePatch import state_patch_forecast
from models.OscillatorPDMStatePatchDisconnected import Model as Disconnected


class TrendPatchHead(nn.Module):
    """Same decoder as season: project96, mean4, original A24, state16->32->4.

    Input is the final same-scale trend history [B*C,T,D]. No extra embedding,
    FFT, attention or PDM is applied. Parameters are independent, initialized
    from the seasonal scratch decoder, with a zero final readout to preserve
    the original zero trend output at initialization.
    """

    def __init__(self, encoder):
        super().__init__()
        self.temporal_projection = copy.deepcopy(encoder.temporal_projection)
        self.register_buffer('future_positions', encoder.future_positions.clone())
        self.decoder = copy.deepcopy(encoder.decoder)
        self.horizon = encoder.horizon
        nn.init.zeros_(self.decoder.readout.output_projection.weight)
        nn.init.zeros_(self.decoder.readout.output_projection.bias)


class Model(Disconnected):
    def __init__(self, configs):
        super().__init__(configs)
        self.feature_trend_heads = nn.ModuleList([TrendPatchHead(e) for e in self.seasonals])
        self.trend_history_gain_offset = nn.Parameter(torch.zeros(self.enc_in))
        self.trend_drive_gain_offset = nn.Parameter(torch.zeros(self.enc_in))

    def optimizer_parameter_groups(self, weight_decay):
        protected = set()
        for encoder in [*self.seasonals, *self.feature_trend_heads]:
            decoder = encoder.decoder
            protected.update((id(decoder.log_decay), id(decoder.frequency_logits)))
            protected.update(id(p) for p in decoder.condition.parameters())
        return [dict(params=[p for p in self.parameters() if id(p) not in protected],
                     weight_decay=weight_decay, group_name='regular'),
                dict(params=[p for p in self.parameters() if id(p) in protected],
                     weight_decay=self.dynamics_weight_decay, group_name='ssm_dynamics')]

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected observed history [B,seq_len,enc_in]')
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        seasonal = [state_patch_forecast(e, x, alpha, beta, s, None)
                    for e, x, s in zip(self.seasonals, scales, seasons)]
        alpha_t = 1 + self.trend_history_gain_offset
        beta_t = 1 + self.trend_drive_gain_offset
        trend = [state_patch_forecast(e, x, alpha_t, beta_t, t, None)
                 for e, x, t in zip(self.feature_trend_heads, scales, trends)]
        return torch.stack(seasonal + trend, -1)
