"""A small shared history/horizon head for Gaussian innovations."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DirectedGaussianHead(nn.Module):
    def __init__(self, context_width, horizon, lags=(1, 2, 4, 8, 12, 24),
                 hidden=32, variance_floor=1e-4, initial_variance=1., row_bound=.95):
        super().__init__()
        if context_width < 1 or hidden < 1 or horizon < 2:
            raise ValueError('Invalid head dimensions')
        if len(set(lags)) != len(lags) or any(type(k) is not int or not 0 < k < horizon for k in lags):
            raise ValueError('Invalid positive lags')
        if not 0 < variance_floor < initial_variance or not 0 < row_bound < 1:
            raise ValueError('Invalid variance floor/initialization or row bound')
        self.lags, self.horizon = tuple(lags), horizon
        self.variance_floor, self.row_bound = variance_floor, row_bound
        h = torch.arange(horizon, dtype=torch.float32) / max(horizon-1, 1)
        positions = torch.stack((h, h.square(), (2*math.pi*h).sin(), (2*math.pi*h).cos()), -1)
        self.register_buffer('horizon_features', positions)
        mask = torch.arange(horizon)[:, None] >= torch.tensor(self.lags, dtype=torch.long)[None, :]
        self.register_buffer('lag_mask', mask)
        self.context = nn.Linear(context_width+4, hidden)
        self.variance = nn.Linear(hidden, 1)
        self.edges = nn.Linear(hidden, len(lags)) if lags else None
        nn.init.zeros_(self.variance.weight)
        nn.init.constant_(self.variance.bias, math.log(math.expm1(initial_variance-variance_floor)))
        if self.edges is not None:
            nn.init.zeros_(self.edges.weight)
            nn.init.zeros_(self.edges.bias)

    def forward(self, context):
        """Observed-history context [B,C,F] -> coefficients [B,H,C,K], d [B,H,C]."""
        if context.ndim != 3:
            raise ValueError('Expected context [B,C,F]')
        batch, channels, _ = context.shape
        pos = self.horizon_features.to(context)[None, None].expand(batch, channels, -1, -1)
        features = context[:, :, None].expand(-1, -1, self.horizon, -1)
        hidden = F.gelu(self.context(torch.cat((features, pos), -1)))
        variance = F.softplus(self.variance(hidden).squeeze(-1)) + self.variance_floor
        if self.edges is None:
            edges = hidden.new_zeros(batch, channels, self.horizon, 0)
        else:
            raw = self.edges(hidden) * self.lag_mask[None, None].to(hidden)
            # Smooth bound: ||row||_1 <= row_bound. Finite-horizon stability
            # safeguard, not a causal-identification or stationarity claim.
            edges = self.row_bound * raw / (math.sqrt(len(self.lags)) * (1+raw.square().sum(-1, keepdim=True)).sqrt())
        return edges.permute(0, 2, 1, 3), variance.transpose(1, 2)
