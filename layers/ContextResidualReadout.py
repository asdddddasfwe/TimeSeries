"""History-statistics-conditioned residual after a frozen GELU readout."""
import torch
from torch import nn


class ContextResidualReadout(nn.Module):
    """Context is train-standardized [history mean, log(history std)].

    Set context [B,C,2] before forward. Delta [C,3*(hidden+1)] contains
    unconditional, mean-conditioned, log-std-conditioned coefficients.
    The context must be computed from observed inputs, never labels.
    """
    def __init__(self, base, channels):
        super().__init__()
        self.base = base
        self.channels = channels
        self.context = None
        self.delta = nn.Parameter(base.output_projection.weight.new_zeros(
            channels, 3*(base.output_projection.in_features+1)))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, states):
        if states.ndim != 3 or states.shape[0] % self.channels:
            raise ValueError('Expected [B*C,H,F]')
        b = states.shape[0] // self.channels
        if self.context is None or self.context.shape != (b,self.channels,2):
            raise ValueError('Set observed-history context [B,C,2]')
        hidden = self.base.activation(self.base.input_projection(states))
        original = self.base.output_projection(hidden)
        z = hidden.reshape(b,self.channels,states.shape[1],-1)
        z = torch.cat([z,torch.ones_like(z[...,:1])],-1)
        q = self.context.detach().to(z)
        design = torch.cat([z,z*q[:,:,None,0:1],z*q[:,:,None,1:2]],-1)
        correction = (design*self.delta[None,:,None,:]).sum(-1)
        return original+correction.reshape_as(original)
