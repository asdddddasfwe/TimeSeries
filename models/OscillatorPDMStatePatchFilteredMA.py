"""Observed-only adaptive low-pass preprocessing followed by unchanged MA25."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import Model as Original

class ObservedLowPass(nn.Module):
    """[B*C,T,1], physical cutoff [B*C,1], reflect only observed points.

    Direct mode removes the entire soft high-pass residual. It is not a
    non-differentiable brick-wall bin cutoff. Preserve the cropped window DC.
    """
    def __init__(self, temperature=1/96):
        super().__init__()
        self.temperature=float(temperature)
        if self.temperature<=0:raise ValueError('temperature must be positive')

    def forward(self,x,cutoff,spacing=1):
        if x.ndim!=3 or x.shape[-1]!=1 or x.shape[1]<2:raise ValueError('expected [N,T>=2,1]')
        if cutoff.shape!=(len(x),1) or spacing<=0:raise ValueError('invalid cutoff or spacing')
        length=x.shape[1]
        padded=F.pad(x[...,0][:,None],(length-1,length-1),mode='reflect')[:,0]
        freq=torch.fft.rfftfreq(padded.shape[1],d=float(spacing),device=x.device,dtype=x.dtype)
        mask=torch.sigmoid((cutoff-freq[None])/self.temperature)
        mask=torch.cat([torch.ones_like(mask[:,:1]),mask[:,1:]],1)
        low=torch.fft.irfft(torch.fft.rfft(padded,dim=1)*mask,n=padded.shape[1],dim=1)[:,length-1:2*length-1,None]
        return low-low.mean(1,keepdim=True)+x.mean(1,keepdim=True)

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        if self.seq_len!=96 or self.lengths!=[96,48,24,12]:raise ValueError('requires original four scales and observed96')
        self.filter_mode=getattr(configs,'prefilter_mode','direct')
        if self.filter_mode not in ['direct','weak']:raise ValueError('invalid prefilter mode')
        self.cutoff_min=1/96;self.cutoff_max=1/2;self.initial_cutoff=1/4
        self.lowpass=ObservedLowPass(1/96)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20211030)
            self.cutoff_predictor=nn.Sequential(nn.Linear(96,16),nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.cutoff_predictor[-1].weight)
        position=(self.initial_cutoff-self.cutoff_min)/(self.cutoff_max-self.cutoff_min)
        nn.init.constant_(self.cutoff_predictor[-1].bias,math.log(position/(1-position)))
        if self.filter_mode=='weak':self.strength_logit=nn.Parameter(torch.tensor(math.log(.01/.99)))

    def predict_cutoff(self,x):
        batch,length,channels=x.shape
        if length!=96:raise ValueError('expected observed96 only')
        flat=x.transpose(1,2).reshape(batch*channels,length)
        return self.cutoff_min+(self.cutoff_max-self.cutoff_min)*torch.sigmoid(self.cutoff_predictor(flat))

    def strength(self,x):
        return x.new_ones(()) if self.filter_mode=='direct' else self.strength_logit.sigmoid()

    def encode_multiscale(self,raw_scales):
        normalized=[self.normalize(raw_scales[0],'norm')]
        normalized += [norm(x,'norm') for norm,x in zip(self.coarse_normalizers,raw_scales[1:])]
        batch,_,channels=normalized[0].shape
        cutoff=self.predict_cutoff(normalized[0]);alpha=self.strength(normalized[0])
        seasons,trends,energies,filter_energies=[],[],[],[]
        for index,(encoder,x) in enumerate(zip(self.seasonals,normalized)):
            observed=x.transpose(1,2).reshape(batch*channels,encoder.length,1)
            low=self.lowpass(observed,cutoff,2**index)
            cleaned=observed-alpha*(observed-low)
            seasonal,trend=self.initial_decomposition(cleaned)
            seasons.append(F.linear(seasonal,encoder.value_embedding.weight,None))
            trends.append(encoder.value_embedding(trend))
            energies.append(torch.stack([trend.detach().square().mean(),seasonal.detach().square().mean()]))
            filter_energies.append(torch.stack([z.detach().square().mean() for z in [observed,low,cleaned,observed-cleaned]]))
        self.last_spectral_diagnostics=dict(cutoff=cutoff.detach(),alpha=alpha.detach().expand_as(cutoff),energy=torch.stack(energies),filter_energy=torch.stack(filter_energies))
        seasons=self.encode_seasonal_features(seasons)
        for pdm in self.pdm_blocks:seasons,trends=pdm(seasons,trends)
        return normalized,(seasons,trends)