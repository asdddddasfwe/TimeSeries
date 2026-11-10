"""Input-dependent forget/write gates on native-resolution history scans."""
import math

import torch
from torch import nn

from models.OscillatorDualEncoderTimeMixer import HistoryOscillator, Model as Ungated


class GatedHistoryOscillator(HistoryOscillator):
    """h_t = f(x_t) * A(context) * h_(t-1) + i(x_t) * b(x_t).

Each mode has one real scalar gate, shared by its real/imaginary coordinates.
Sigmoid gates never increase the base transition radius. Gates depend only on
current encoded inputs, preserving affine parallel scanning (not on h_(t-1)).
Zero weights and logit(.99) biases give two .99 gates at initialization.
The initial function is close to, but not identical to, the ungated encoder.
"""

    def __init__(self, source, initial_gate=.99):
        nn.Module.__init__(self)
        if not 0 < initial_gate < 1:
            raise ValueError('Initial sigmoid gate must be strictly between zero and one')
        self.length, self.width, self.modes = source.length, source.width, source.modes
        for name, module in source.named_children():
            self.add_module(name, module)
        for name, value in source.named_parameters(recurse=False):
            self.register_parameter(name, value)
        # Gate construction consumes no caller RNG. It does not share storage
        # with either the opposite branch or another scale.
        with torch.random.fork_rng(devices=[]):
            self.forget_gate = nn.Linear(self.width, self.modes)
            self.write_gate = nn.Linear(self.width, self.modes)
        for gate in (self.forget_gate, self.write_gate):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, math.log(initial_gate/(1-initial_gate)))
        self.capture_gate_statistics = False
        self.last_gate_tensors = None

    def gates(self, history):
        normalized = self.drive_norm(history)
        return torch.sigmoid(self.forget_gate(normalized)), torch.sigmoid(self.write_gate(normalized))

    def scan_inputs(self, history):
        transition, drive, initial = super().scan_inputs(history)
        forget, write = self.gates(history)
        gated_transition = transition * forget.unsqueeze(-1)
        gated_drive = drive * write.unsqueeze(-1)
        self.last_gate_tensors = None
        if self.capture_gate_statistics:
            self.last_gate_tensors = dict(forget=forget.detach(), write=write.detach(),
                effective_radius=gated_transition.detach().square().sum(-1).sqrt())
        return gated_transition, gated_drive, initial


class Model(Ungated):
    """Identical backbone/FMM, with eight independent pairs of history gates."""

    def __init__(self, configs):
        super().__init__(configs)
        initial = getattr(configs, 'history_gate_initial', .99)
        self.season_oscillators = nn.ModuleList([
            GatedHistoryOscillator(source, initial) for source in self.season_oscillators])
        self.trend_oscillators = nn.ModuleList([
            GatedHistoryOscillator(source, initial) for source in self.trend_oscillators])

    def enable_gate_statistics(self, enabled):
        for module in [*self.season_oscillators, *self.trend_oscillators]:
            module.capture_gate_statistics = enabled
            if not enabled:
                module.last_gate_tensors = None
