"""History-initialized oscillators sharing dynamics and drive with forecasting."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from layers.OscillatorMixer import OscillatorDecoder, parallel_oscillator_scan


class HistoryInitializedDecoder(OscillatorDecoder):
    """Replace the summary initial-state head with a scan of encoded history.

    Reuse the original decoder's modules so the same seed initializes every
    common parameter identically. The discarded initial_state head is neither
    registered nor optimized here. The history encoder remains bidirectional;
    these historical states are NOT causal one-step predictions of observations.
    """

    def __init__(self, decoder):
        nn.Module.__init__(self)
        self.length, self.horizon, self.modes = decoder.length, decoder.horizon, decoder.modes
        for name in ("context_norm", "condition", "log_decay", "frequency_logits",
                     "drive_norm", "drive_projection", "drive_dropout", "readout"):
            setattr(self, name, getattr(decoder, name))
        self.register_buffer("history_initialization_version", torch.tensor(1, dtype=torch.long))

    def encode_history_states(self, history, transition):
        """Scan [N,T,D] features from zero with the SAME A and input map as future.

        transition is [N,J,2]; output is [N,T,J,2]. The last state remains
        attached to autograd when used to initialize the future scan.
        Dropout shares its rate/module, with independent masks on each call.
        """
        drive = self.drive_projection(self.drive_dropout(F.gelu(self.drive_norm(history))))
        drive = drive.reshape(history.shape[0], self.length, self.modes, 2)
        initial = transition.new_zeros(history.shape[0], self.modes, 2)
        return parallel_oscillator_scan(
            transition[:, None].expand(-1, self.length, -1, -1), drive, initial)

    def dynamics(self, history):
        """Condition A once on the observed window, then carry its history state.

        A is constant across both the history and forecast time axes for each
        window/channel/scale. Only the initialization differs from OscillatorMixer.
        """
        context = self.context_norm(torch.cat((history.mean(1), history[:, -1]), dim=-1))
        dtype = torch.float64 if context.dtype == torch.float64 else torch.float32
        decay_shift, frequency_shift = self.condition(context).to(dtype).chunk(2, dim=-1)
        rate = F.softplus(self.log_decay.to(dtype) + decay_shift)
        radius = torch.exp(-rate)
        frequency = math.pi * torch.sigmoid(self.frequency_logits.to(dtype) + frequency_shift)
        transition = torch.stack((radius * frequency.cos(), radius * frequency.sin()), dim=-1)
        initial = self.encode_history_states(history, transition)[:, -1]
        return transition, initial, radius, frequency
