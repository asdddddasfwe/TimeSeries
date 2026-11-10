"""Nonlinear mixing of oscillator modes after the parallel state scan."""
import torch
from torch import nn


class GELUMLPReadout(nn.Module):
    """Map [...,2J] -> hidden32 -> GELU -> scalar, independently per position.

    Four scales own independent readouts; weights are shared across time and
    variables within a scale. No activation is fed into the recurrence.
    """

    def __init__(self, in_features, hidden_features=32):
        super().__init__()
        self.in_features, self.out_features = in_features, 1
        self.input_projection = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_features, 1)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.output_projection.bias)
        self.register_buffer("gelu_mlp_readout_version", torch.tensor(1, dtype=torch.long))

    @property
    def weight(self):
        # Existing decoder/diagnostics use .weight.dtype at the scan boundary.
        # A property avoids registering the same Parameter under two names.
        return self.input_projection.weight

    def forward(self, states):
        return self.output_projection(self.activation(self.input_projection(states)))
