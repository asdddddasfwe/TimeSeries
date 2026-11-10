"""MAFirst S-gate ablation using one four-sample clock tick per future block."""
import torch

from models.OscillatorPDMStatePatchBestGateContext import Model as SRoute
from models.OscillatorPDMStatePatchBestNoStrideConversion import NoStrideConversionDecoder
from models.OscillatorPDMTrendCondition import TrendConditionedDecoder


class BlockClockDecoder(NoStrideConversionDecoder):
    def dynamics(self, history, trend_history=None):
        transition, initial, radius, angle = super().dynamics(history, trend_history)
        if self.stride == 4:
            return transition, initial, radius, angle
        exponent = 4.0 / self.stride
        radius = radius.pow(exponent)
        angle = angle * exponent
        transition = torch.stack((radius * angle.cos(), radius * angle.sin()), dim=-1)
        return transition, initial, radius, angle


class Model(SRoute):
    def __init__(self, configs):
        super().__init__(configs)
        if (self.seq_len, self.pred_len) != (96, 720) or self.gate_context_policy != 'season':
            raise ValueError('Expected MAFirst S gate with 96 observed and 720 forecast points')
        for encoder in self.seasonals:
            if type(encoder.decoder) is not TrendConditionedDecoder:
                raise TypeError('Expected the original trend-conditioned decoder')
            encoder.decoder.__class__ = BlockClockDecoder

    def predict_components(self, x_enc):
        components = super().predict_components(x_enc)
        for diag, encoder in zip(self.last_gate_context_diagnostics, self.seasonals):
            stride = encoder.decoder.stride
            diag.update(history_stride=stride, future_block_points=4, transition_exponent=4.0 / stride)
        return components
