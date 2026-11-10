"""MA-first baseline with a trend-only causal temporal residual encoder.

Four independent native-scale encoders preserve [N,T,32]. Three internally residual depthwise
kernel3 layers at dilations1/2/4 (left endpoint replication), each GELU,
followed by a zero-initialized bias-free pointwise32->32 projection. No new
normalization, dropout, gate, alpha, output skip or prediction head.
"""
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorPDMStatePatchMAFirst import Model as MAFirst


class TrendTemporalInnerResidual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.dilations=(1,2,4)
        self.depthwise=nn.ModuleList([
            nn.Conv1d(width,width,kernel_size=3,dilation=d,groups=width,bias=False)
            for d in self.dilations])
        self.output=nn.Conv1d(width,width,kernel_size=1,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,history):
        if history.ndim!=3:raise ValueError('Expected [N,T,D] trend history')
        x=history.transpose(1,2)
        for conv,dilation in zip(self.depthwise,self.dilations):
            x=x+F.gelu(conv(F.pad(x,(2*dilation,0),mode='replicate')))
        return history+self.output(x).transpose(1,2)


class Model(MAFirst):
    def __init__(self,configs):
        super().__init__(configs)
        with torch.random.fork_rng(devices=[]):
            self.trend_temporal_encoders=nn.ModuleList([
                TrendTemporalInnerResidual(configs.d_model) for _ in self.seasonals])

    def encode_multiscale(self,raw_scales):
        normalized=[self.normalize(raw_scales[0],'norm')]
        normalized += [norm(x,'norm') for norm,x in zip(self.coarse_normalizers,raw_scales[1:])]
        batch,_,channels=normalized[0].shape
        seasons,trends=[],[]
        for encoder,temporal,x in zip(self.seasonals,self.trend_temporal_encoders,normalized):
            observed=x.transpose(1,2).reshape(batch*channels,encoder.length,1)
            seasonal,trend=self.initial_decomposition(observed)
            seasons.append(F.linear(seasonal,encoder.value_embedding.weight,None))
            trends.append(temporal(encoder.value_embedding(trend)))
        seasons=self.encode_seasonal_features(seasons)
        for pdm in self.pdm_blocks:seasons,trends=pdm(seasons,trends)
        return normalized,(seasons,trends)
