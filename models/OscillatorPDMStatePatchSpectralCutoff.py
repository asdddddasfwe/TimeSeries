"""Complementary soft spectral entrance split for the MAFirst sliding model."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import Model as Original

class SoftSpectralSplit(nn.Module):
    """[N,T,1] observed history, [N,1] cutoff in cycles/original sample.

    Reflect T-1 observations at both ends; preserve phase and the entire DC.
    Return seasonal, trend and the real low-pass mask. No future is read.
    """
    def __init__(self, temperature=1.0/96.0):
        super().__init__()
        if temperature<=0: raise ValueError('temperature must be positive')
        self.temperature=float(temperature)

    def forward(self, history, cutoff, sample_spacing=1):
        if history.ndim!=3 or history.shape[-1]!=1 or history.shape[1]<2:
            raise ValueError('Expected [N,T>=2,1] history')
        if cutoff.shape!=(len(history),1) or sample_spacing<=0:
            raise ValueError('Expected one cutoff per history and positive spacing')
        length=history.shape[1]
        padded=F.pad(history[...,0][:,None],(length-1,length-1),mode='reflect')[:,0]
        frequencies=torch.fft.rfftfreq(padded.shape[1],d=float(sample_spacing),device=history.device,dtype=history.dtype)
        mask=torch.sigmoid((cutoff-frequencies[None])/self.temperature)
        mask=torch.cat((torch.ones_like(mask[:,:1]),mask[:,1:]),dim=1)
        spectrum=torch.fft.rfft(padded,dim=1)
        low=torch.fft.irfft(spectrum*mask,n=padded.shape[1],dim=1)
        trend=low[:,length-1:2*length-1,None]
        return history-trend,trend,mask

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        if self.seq_len!=96 or self.lengths!=[96,48,24,12]:
            raise ValueError('This trial requires the original four scales')
        self.cutoff_mode=getattr(configs,'spectral_cutoff_mode','dynamic')
        if self.cutoff_mode not in ('fixed','dynamic'):raise ValueError('Invalid cutoff mode')
        self.cutoff_min=1.0/96.0
        self.cutoff_max=1.0/16.0
        self.fixed_cutoff=1.0/25.0
        self.spectral_split=SoftSpectralSplit()
        if self.cutoff_mode=='dynamic':
            # Preserve the external initialization RNG stream of the base model.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20211029)
                self.cutoff_predictor=nn.Sequential(nn.Linear(96,16),nn.GELU(),nn.Linear(16,1))
            nn.init.zeros_(self.cutoff_predictor[-1].weight)
            position=(self.fixed_cutoff-self.cutoff_min)/(self.cutoff_max-self.cutoff_min)
            nn.init.constant_(self.cutoff_predictor[-1].bias,math.log(position/(1.0-position)))

    def predict_cutoff(self,normalized_history):
        """[B,96,C] -> [B*C,1], sharing the small network across variables."""
        batch,length,channels=normalized_history.shape
        if length!=96:raise ValueError('Expected 96 observed points')
        flat=normalized_history.transpose(1,2).reshape(batch*channels,length)
        if self.cutoff_mode=='fixed':return flat.new_full((len(flat),1),self.fixed_cutoff)
        return self.cutoff_min+(self.cutoff_max-self.cutoff_min)*torch.sigmoid(self.cutoff_predictor(flat))

    def encode_multiscale(self,raw_scales):
        normalized=[self.normalize(raw_scales[0],'norm')]
        normalized += [norm(x,'norm') for norm,x in zip(self.coarse_normalizers,raw_scales[1:])]
        batch,_,channels=normalized[0].shape
        cutoff=self.predict_cutoff(normalized[0]);seasons,trends,energy=[],[],[]
        for index,(encoder,x) in enumerate(zip(self.seasonals,normalized)):
            observed=x.transpose(1,2).reshape(batch*channels,encoder.length,1)
            seasonal,trend,_=self.spectral_split(observed,cutoff,2**index)
            seasons.append(F.linear(seasonal,encoder.value_embedding.weight,None))
            trends.append(encoder.value_embedding(trend))
            energy.append(torch.stack((trend.detach().square().mean(),seasonal.detach().square().mean())))
        # Detached last batch only: no epoch tensor history or retained graph.
        self.last_spectral_diagnostics={'cutoff':cutoff.detach(),'energy':torch.stack(energy)}
        seasons=self.encode_seasonal_features(seasons)
        for pdm in self.pdm_blocks:seasons,trends=pdm(seasons,trends)
        return normalized,(seasons,trends)