"""Pointwise Tanh applied after state scanning and before linear readout."""
import torch
from torch import nn
from torch.nn import functional as F


class TanhReadout(nn.Module):
    """Keep the original Linear weights; apply Tanh to each scanned state.

    Input [...,2J] -> Tanh -> Linear(2J,1). No time or channel mixing,
    additional parameter, output activation or state feedback is introduced.
    Keeping .weight supports the existing float32-scan/autocast readout boundary.
    """

    def __init__(self, linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight
        self.register_parameter("bias", linear.bias)
        self.activation = nn.Tanh()
        self.register_buffer("tanh_readout_version", torch.tensor(1, dtype=torch.long))

    def forward(self, states):
        return F.linear(self.activation(states), self.weight, self.bias)
