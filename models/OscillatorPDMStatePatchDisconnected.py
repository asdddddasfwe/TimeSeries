"""Mean-pool A24 model with the trend-to-seasonal dynamics edge removed."""
from torch import nn

from layers.OscillatorTimeMix import FullResolutionGatedDecoder
from models.OscillatorPDMStatePatch import Model as MeanStatePatch


class DisconnectedDecoder(FullResolutionGatedDecoder):
    """Preserve seasonal dynamics and remove all trend-condition parameters."""

    def __init__(self, source):
        nn.Module.__init__(self)
        for name in ('length', 'horizon', 'modes', 'stride'):
            setattr(self, name, getattr(source, name))
        for name, module in source.named_children():
            if name not in ('trend_condition', 'trend_context_norm'):
                self.add_module(name, module)
        for name, value in source.named_parameters(recurse=False):
            self.register_parameter(name, value)
        for name, value in source.named_buffers(recurse=False):
            self.register_buffer(name, value)

    def dynamics(self, history, trend_history=None):
        # The compatibility argument is deliberately never read.
        return super().dynamics(history)


class Model(MeanStatePatch):
    def __init__(self, configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            encoder.decoder = DisconnectedDecoder(encoder.decoder)

    def optimizer_parameter_groups(self, weight_decay):
        # Skip the removed trend-condition-specific optimizer override.
        from models.OscillatorPDMSegmentPE import Model as SegmentPE
        return SegmentPE.optimizer_parameter_groups(self, weight_decay)
