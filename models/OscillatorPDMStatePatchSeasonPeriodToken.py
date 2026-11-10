"""Raw seasonal MA residuals: an ENTIRE FFT period is one attention token.

FFT chooses observed-history routes, not spectral coefficient tokens. At each
native scale T, right-pad a period P to T and project T->D (shared across all
period routes at that scale). Attend across ceil(T/P) cycles using the existing
column QKV/gamma/dropout/LN. Remove row attention. Decode D->P*D using the first
P*D rows of a learned D->T*D matrix, restore observed order, crop last-cycle
padding, and amplitude-weight routes. Original trend encoder/heads/PDM and
trend->season conditioning remain. This is a codec/geometry/capacity redesign,
not a parameter-free removal of the external scalar embedding.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from layers.LagARFormer import sinusoidal_positions
from models.OscillatorPDMSeasonalFFT import select_seasonal_feature_periods
from models.OscillatorPDMStatePatchMAFirst import Model as MAFirst


class PeriodAttention(nn.Module):
    """Only the existing column axis remains: [N,number_of_cycles,D]."""
    def __init__(self, original):
        super().__init__()
        self.column = original.column
        self.dropout = original.dropout
        self.norm = original.norm

    def forward(self, tokens):
        projections = [self.column.query_conv,self.column.key_conv,self.column.value_conv]
        weight = torch.cat([p.weight.flatten(1) for p in projections])
        bias = torch.cat([p.bias for p in projections])
        q,k,v = F.linear(tokens,weight,bias).split([p.out_channels for p in projections],-1)
        width = v.shape[-1]
        if width < q.shape[-1]:v = F.pad(v,(0,q.shape[-1]-width))
        update = F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],dropout_p=0.,scale=1.)[:,0,:,:width]
        return self.norm(tokens+self.dropout(self.column.gamma*update))


class PeriodCodec(nn.Module):
    def __init__(self,length,width):
        super().__init__()
        self.length,self.width = length,width
        self.token_projection = nn.Linear(length,width,bias=False)
        self.period_readout = nn.Linear(width,length*width,bias=False)

    def forward(self,cycles,attention):
        # cycles [N,ceil(T/P),P], including zero pad of the last partial cycle.
        period = cycles.shape[-1]
        if not 1 <= period <= self.length:raise ValueError('Invalid observed period')
        tokens = self.token_projection(F.pad(cycles,(0,self.length-period)))
        tokens = tokens+sinusoidal_positions(tokens.shape[1],self.width).to(tokens)
        tokens = attention(tokens)
        decoded = F.linear(tokens,self.period_readout.weight[:period*self.width])
        return decoded.reshape(*cycles.shape,self.width)


def period_token_features(encoder,observed,codec):
    periods,weights = select_seasonal_feature_periods(observed,encoder.period_top_k)
    groups = {}
    for series,row in enumerate(periods.detach().cpu().tolist()):
        for period in set(row):groups.setdefault(period,[]).append(series)
    n,length,_ = observed.shape
    output = observed.new_zeros(n,length,codec.width)
    for period in sorted(groups):
        selected = torch.tensor(groups[period],device=observed.device,dtype=torch.long)
        count = (length+period-1)//period
        cycles = F.pad(observed[selected].squeeze(-1),(0,count*period-length)).reshape(-1,count,period)
        decoded = codec(cycles,encoder.blocks[0])
        restored = decoded.reshape(len(selected),count*period,codec.width)[:,:length]
        merged = (weights[selected]*(periods[selected]==period)).sum(-1).to(restored)
        output = output.index_add(0,selected,restored*merged[:,None,None])
    return output


class Model(MAFirst):
    def __init__(self,configs):
        super().__init__(configs)
        # Relocate the old scalar projection to TREND only, without adding trend capacity.
        self.trend_embeddings = nn.ModuleList([copy.deepcopy(e.value_embedding) for e in self.seasonals])
        for encoder in self.seasonals:
            del encoder.value_embedding
            encoder.blocks = nn.ModuleList([PeriodAttention(encoder.blocks[0])])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20211015)
            self.season_period_codecs = nn.ModuleList([PeriodCodec(e.length,configs.d_model) for e in self.seasonals])

    def clone_embeddings(self,source):
        for encoder,trend in zip(source.seasonals,self.trend_embeddings):
            trend.load_state_dict(encoder.value_embedding.state_dict(),strict=True)

    def encode_multiscale(self,raw_scales):
        normalized = [self.normalize(raw_scales[0],'norm')]
        normalized += [norm(x,'norm') for norm,x in zip(self.coarse_normalizers,raw_scales[1:])]
        batch,_,channels = normalized[0].shape
        seasons,trends = [],[]
        for encoder,embedding,codec,x in zip(self.seasonals,self.trend_embeddings,self.season_period_codecs,normalized):
            observed = x.transpose(1,2).reshape(batch*channels,encoder.length,1)
            seasonal,trend = self.initial_decomposition(observed)
            seasons.append(period_token_features(encoder,seasonal,codec))
            trends.append(embedding(trend))
        for pdm in self.pdm_blocks:seasons,trends = pdm(seasons,trends)
        return normalized,(seasons,trends)
