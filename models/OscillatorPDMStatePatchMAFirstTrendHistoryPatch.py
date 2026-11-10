"""Original recurrence with historical-patch features for a joint horizon head."""
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorPDMStatePatchMAFirst import Model as Original

class HistoryPatchTrendHead(nn.Module):
    """[N,T,D] -> local four-step tokens -> MLP -> [N,H], no H*D future tokens."""
    def __init__(self, original, scale):
        super().__init__()
        self.temporal=original.temporal
        self.readout=original.readout
        self.length=self.temporal.in_features
        self.horizon=self.temporal.out_features
        self.width=self.readout.in_features
        if self.length%4:raise ValueError('History length must be divisible by four')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20211020+scale)
            self.segment_encoder=nn.Sequential(nn.Linear(4*self.width,self.width),nn.GELU())
            self.nonlinear_readout=nn.Sequential(
                nn.Linear((self.length//4)*self.width,64),nn.GELU(),
                nn.Linear(64,self.horizon))
        nn.init.zeros_(self.nonlinear_readout[-1].weight)
        nn.init.zeros_(self.nonlinear_readout[-1].bias)

    def linear_forecast(self,history):
        # Algebraically identical to old temporal-then-feature readout.
        # Read the history scalar first to avoid materializing [N,H,D].
        # FP32 operation ordering changes: validate tolerance, not bit equality.
        scalar=F.linear(history,self.readout.weight,None).squeeze(-1)
        bias=self.temporal.bias*self.readout.weight.sum()
        return F.linear(scalar,self.temporal.weight,bias)+self.readout.bias

    def forward(self,history):
        if history.ndim!=3 or history.shape[1:]!=(self.length,self.width):
            raise ValueError('Expected [N,history_length,feature_width]')
        tokens=history.reshape(history.shape[0],self.length//4,4*self.width)
        features=self.segment_encoder(tokens)
        return self.linear_forecast(history)+self.nonlinear_readout(features.flatten(1))

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        self.feature_trend_heads=nn.ModuleList([
            HistoryPatchTrendHead(head,scale)
            for scale,head in enumerate(self.feature_trend_heads)])
        # Inherit original recurrence, its mean, and the eight-component forecast.
