"""Keep original mean-patch seasonal model; predict trend in four-point blocks."""
import torch
from torch import nn
from models.OscillatorPDMStatePatch import Model as Original


class PatchMLPTrendHead(nn.Module):
    """[B*C,T,D] -> [B*C,24,D] -> [B*C,24,4] -> [B*C,96]."""
    def __init__(self, length, horizon, width):
        super().__init__()
        if horizon % 4:
            raise ValueError('Trend horizon must be divisible by four')
        self.length, self.horizon, self.width = length, horizon, width
        self.temporal = nn.Sequential(nn.Linear(length, horizon), nn.GELU(),
                                      nn.Linear(horizon, horizon//4))
        self.readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.length, self.width):
            raise ValueError('Expected trend history [B*C,T,D]')
        future = self.temporal(history.transpose(1, 2)).transpose(1, 2)
        # Contiguous groups: patch k predicts positions 4*k through 4*k+3.
        return self.readout(future).reshape(history.shape[0], self.horizon)


class Model(Original):
    def __init__(self, configs):
        super().__init__(configs)
        # New heads use standard Linear initialization without consuming the
        # caller RNG. Common parameters are copied from fresh control by runner.
        with torch.random.fork_rng(devices=[]):
            self.feature_trend_heads = nn.ModuleList([
                PatchMLPTrendHead(length, self.pred_len, configs.d_model)
                for length in self.lengths])
