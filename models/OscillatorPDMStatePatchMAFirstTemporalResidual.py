"""MAFirst with an identity shortcut between adjacent future hidden states.

Only the four future decoder transitions change. For complex mode a, use
a_res = (1 - mix) + mix * a, retaining the original drive u and history state.
Thus h[k] = h[k-1] + mix * (A - I) h[k-1] + u[k]. This is a bounded
transition mixture, not I + A, not an output bypass, and not output feedback.
It changes both damping and phase; it is not an isolated gradient intervention.
"""
import math

import torch
from torch import nn

from models.OscillatorPDMStatePatchMAFirst import Model as Original
from models.OscillatorPDMTrendCondition import TrendConditionedDecoder


def residual_transition(transition, mix):
    """Mix identity with complex rotations represented by [..., real/imag]."""
    mix = float(mix)
    if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
        raise ValueError('temporal_residual_mix must be finite and in [0, 1]')
    if mix == 1.0:
        return transition
    return torch.stack(((1.0 - mix) + mix * transition[..., 0],
                        mix * transition[..., 1]), dim=-1)


class TemporalResidualDecoder(TrendConditionedDecoder):
    def __init__(self, source, mix):
        nn.Module.__init__(self)
        residual_transition(torch.zeros(2), mix)
        self.temporal_residual_mix = float(mix)
        for name in ('length', 'horizon', 'modes', 'stride'):
            setattr(self, name, getattr(source, name))
        for name, module in source.named_children():
            self.add_module(name, module)
        for name, parameter in source.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, value in source.named_buffers(recurse=False):
            self.register_buffer(name, value,
                                 persistent=name not in source._non_persistent_buffers_set)

    def dynamics(self, history, trend_history=None):
        # Original dynamics computes the historical scan before this replacement.
        transition, initial, radius, frequency = super().dynamics(history, trend_history)
        if self.temporal_residual_mix == 1.0:
            return transition, initial, radius, frequency
        mixed = residual_transition(transition, self.temporal_residual_mix)
        return mixed, initial, mixed.norm(dim=-1), torch.atan2(mixed[..., 1], mixed[..., 0])


class Model(Original):
    def __init__(self, configs):
        super().__init__(configs)
        mix = float(getattr(configs, 'temporal_residual_mix', 0.5))
        for encoder in self.seasonals:
            encoder.decoder = TemporalResidualDecoder(encoder.decoder, mix)
