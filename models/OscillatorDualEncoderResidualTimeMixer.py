"""Parameter-free input skip around each native-resolution history oscillator."""
from torch import nn

from models.OscillatorDualEncoderTimeMixer import HistoryOscillator, Model as BaseModel


class ResidualHistoryOscillator(HistoryOscillator):
    """Return x + Readout(scan(x)); preserve state16 and the original transition.

    The input is the branch attention output [N,T,D], not raw observations.
    No gate, learned scale, normalization or extra activation is introduced.
    """

    def __init__(self, source):
        nn.Module.__init__(self)
        self.length, self.width, self.modes = source.length, source.width, source.modes
        for name, module in source.named_children():
            self.add_module(name, module)
        for name, parameter in source.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, buffer in source.named_buffers(recurse=False):
            self.register_buffer(name, buffer)

    def forward(self, history):
        return history + super().forward(history)


class Model(BaseModel):
    """Both branches/four scales receive the same fixed unit input skip."""

    def __init__(self, configs):
        super().__init__(configs)
        self.season_oscillators = nn.ModuleList([
            ResidualHistoryOscillator(module) for module in self.season_oscillators])
        self.trend_oscillators = nn.ModuleList([
            ResidualHistoryOscillator(module) for module in self.trend_oscillators])
