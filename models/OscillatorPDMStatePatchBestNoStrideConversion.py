"""Original best S gate; remove only future radius-root and angle/stride conversion.
Downsampling, historical dynamics/scan, positions, mean4 drive and readout retained.
"""
import math
import torch
from torch.nn import functional as F
from models.OscillatorPDMTrendCondition import TrendConditionedDecoder
from models.OscillatorPDMStatePatchBestGateContext import Model as SRoute

class NoStrideConversionDecoder(TrendConditionedDecoder):
    def dynamics(self, history, trend_history=None):
        if trend_history is None or trend_history.shape != history.shape:
            raise ValueError('Expected same-scale seasonal/trend histories [N,T,D]')
        context = self.context_norm(torch.cat((history.mean(1), history[:, -1]), -1))
        trend = self.trend_context_norm(torch.cat((trend_history.mean(1), trend_history[:, -1]), -1))
        dtype = torch.float64 if context.dtype == torch.float64 else torch.float32
        shifts = self.condition(context).to(dtype) + self.trend_condition(trend).to(dtype)
        (decay_shift, frequency_shift) = shifts.chunk(2, -1)
        rate = F.softplus(self.log_decay.to(dtype) + decay_shift)
        radius = torch.exp(-rate)
        frequency = math.pi * torch.sigmoid(self.frequency_logits.to(dtype) + frequency_shift)
        transition = torch.stack((radius * frequency.cos(), radius * frequency.sin()), -1)
        initial = self.encode_history_states(history, transition)[:, -1]
        return (transition, initial, radius, frequency)

class Model(SRoute):
    def __init__(self,configs):
        super().__init__(configs)
        if self.seq_len!=96 or self.pred_len!=720 or self.gate_context_policy!='season':
            raise ValueError('Registered ablation requires original S gate 96/H720')
        # Class-only substitution adds no state, parameters, buffers or RNG calls.
        for encoder in self.seasonals:
            if type(encoder.decoder) is not TrendConditionedDecoder:
                raise TypeError('Expected the original trend-conditioned decoder')
            encoder.decoder.__class__=NoStrideConversionDecoder

    def predict_components(self,x_enc):
        result=super().predict_components(x_enc)
        for diag,encoder in zip(self.last_gate_context_diagnostics,self.seasonals):
            diag.update(stride_conversion_enabled=False,historical_transition_used_directly=True,
                        history_dynamics_unchanged=True,history_stride=encoder.decoder.stride)
        return result
