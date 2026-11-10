"""Preserve original trend heads; add eight-knot global forecast corrections."""
import math
import torch
from torch import nn
from models.OscillatorPDMStatePatch import Model as Original


def history_pool_matrix(length, knots=8):
    """Fixed ordered interval means, with adaptive-average pooling boundaries."""
    result=torch.zeros(knots,length)
    for i in range(knots):
        start=i*length//knots
        stop=math.ceil((i+1)*length/knots)
        result[i,start:stop]=1/(stop-start)
    return result


def interpolation_matrix(horizon, knots=8):
    """Piecewise linear basis with knots at equally spaced horizon endpoints."""
    if horizon<2 or not 2<=knots<=horizon:
        raise ValueError('Require 2<=knots<=horizon')
    coordinate=torch.arange(horizon,dtype=torch.float64)*(knots-1)/(horizon-1)
    left=coordinate.floor().long().clamp(max=knots-2)
    fraction=coordinate-left
    basis=torch.zeros(horizon,knots,dtype=torch.float64)
    basis[torch.arange(horizon),left]=1-fraction
    basis[torch.arange(horizon),left+1]=fraction
    return basis.float()


class InterpolatedTrendCorrection(nn.Module):
    """[N,T,D] -> original [N,H] + correction from eight global future knots."""
    def __init__(self, original, length, horizon, width, knots=8, hidden=32):
        super().__init__()
        self.length,self.horizon,self.width=length,horizon,width
        self.temporal=original.temporal
        self.readout=original.readout
        self.register_buffer('history_pool',history_pool_matrix(length,knots))
        self.register_buffer('future_basis',interpolation_matrix(horizon,knots))
        self.coefficient_net=nn.Sequential(nn.Linear((knots+2)*width,hidden),nn.GELU(),nn.Linear(hidden,knots))
        nn.init.zeros_(self.coefficient_net[2].weight)
        nn.init.zeros_(self.coefficient_net[2].bias)

    def forward(self, history):
        if history.ndim!=3 or history.shape[1:]!=(self.length,self.width):
            raise ValueError('Expected trend history [N,T,D]')
        future=self.temporal(history.transpose(1,2)).transpose(1,2)
        base=self.readout(future).squeeze(-1)
        pooled=torch.matmul(self.history_pool,history)
        # Ordered features plus recent encoded level and end-to-start change.
        # These are observed encoded features, not physical trend labels.
        context=torch.cat((pooled.flatten(1),history[:,-1],history[:,-1]-history[:,0]),dim=-1)
        coefficients=self.coefficient_net(context)
        correction=torch.matmul(coefficients,self.future_basis.transpose(0,1))
        return base+correction


class Model(Original):
    def __init__(self, configs):
        super().__init__(configs)
        with torch.random.fork_rng(devices=[]):
            self.feature_trend_heads=nn.ModuleList([
                InterpolatedTrendCorrection(head,length,self.pred_len,configs.d_model)
                for head,length in zip(self.feature_trend_heads,self.lengths)])
