"""Original MAFirst recurrence plus a zero-start nonlinear trend readout."""
import torch
from torch import nn
from models.OscillatorPDMStatePatchMAFirst import Model as Original

class FlexibleFeatureTrendHead(nn.Module):
    """Keep temporal projection and linear readout; add shared 32->64->1 MLP."""
    def __init__(self, original, scale):
        super().__init__()
        self.temporal = original.temporal
        self.readout = original.readout
        width = self.readout.in_features
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20211020 + scale)
            self.nonlinear_readout = nn.Sequential(
                nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        # Only the last layer is zero: initial function is exactly unchanged.
        nn.init.zeros_(self.nonlinear_readout[-1].weight)
        nn.init.zeros_(self.nonlinear_readout[-1].bias)

    def forward(self, history):
        # [B*C,T,D] -> [B*C,H,D] -> [B*C,H].
        future = self.temporal(history.transpose(1, 2)).transpose(1, 2)
        return (self.readout(future) + self.nonlinear_readout(future)).squeeze(-1)

class Model(Original):
    def __init__(self, configs):
        super().__init__(configs)
        self.feature_trend_heads = nn.ModuleList([
            FlexibleFeatureTrendHead(head, scale)
            for scale, head in enumerate(self.feature_trend_heads)])
        # Inherit original predict_components/forecast: retain recurrent means.
