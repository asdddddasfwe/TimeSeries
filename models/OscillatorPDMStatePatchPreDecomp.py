"""Raw MA -> raw FFT/sliding patch gathering -> patch projection -> features.

The 0.318 independent-PDM-MLP architecture supplies PDM and forecast heads.
FFT selects periods on scalar seasonal residuals BEFORE learned projection.
This redesign adds patch encoding capacity; it is not an order-only control.
"""
import torch
from torch import nn
from torch.nn import functional as F
from layers.LagARFormer import sinusoidal_positions
from models.OscillatorPDMSeasonalFFT import select_seasonal_feature_periods
from models.OscillatorPDMStatePatchSeparateMLP import Model as SeparateMLP


class PatchCodec(nn.Module):
    """Gather scalar four-point patches before Linear(4,D); decode D->4D."""
    def __init__(self, width, bias=True):
        super().__init__()
        self.width = width
        self.patch_projection = nn.Linear(4,width,bias=bias)
        self.patch_readout = nn.Linear(width,4*width,bias=bias)

    def project(self, patches, use_bias=True):
        return F.linear(patches,self.patch_projection.weight,
                        self.patch_projection.bias if use_bias else None)

    def decode(self, tokens, use_bias=True):
        return F.linear(tokens,self.patch_readout.weight,
                        self.patch_readout.bias if use_bias else None)

    def sliding(self, observed):
        n,length,_ = observed.shape
        padded_length = max(4,length)
        padded_length += (-(padded_length-4)) % 2
        padded = F.pad(observed.transpose(1,2),(0,padded_length-length),mode='replicate')
        patches = padded.unfold(-1,4,2).squeeze(1)
        tokens = self.project(patches)
        decoded = self.decode(tokens).reshape(n,-1,4,self.width)
        count = decoded.shape[1]
        columns = decoded.permute(0,3,2,1).reshape(n,4*self.width,count)
        values = F.fold(columns,(1,padded_length),(1,4),stride=(1,2))
        counts = F.fold(observed.new_ones(1,4,count),(1,padded_length),(1,4),stride=(1,2))
        return (values/counts).squeeze(2).transpose(1,2)[:,:length]


def fft_patch_features(encoder, observed, codec):
    """FFT on [N,T,1], period images -> phase patches -> axial attention.

    Each period row is right-zero-padded to a multiple of four. Patch PE marks
    cycle and phase-patch indices. Native feature positions are restored before
    PDM and the original history scan. No FFT reads projected or PE tokens.
    """
    periods,weights = select_seasonal_feature_periods(observed,encoder.period_top_k)
    groups = {}
    for series,row in enumerate(periods.detach().cpu().tolist()):
        for period in set(row): groups.setdefault(period,[]).append(series)
    n,length,_ = observed.shape
    output = observed.new_zeros(n,length,codec.width)
    block = encoder.blocks[0]
    parameters = block.projection_parameters()
    for period in sorted(groups):
        selected = torch.tensor(groups[period],device=observed.device,dtype=torch.long)
        cycles = (length+period-1)//period
        raw = F.pad(observed[selected].squeeze(-1),(0,cycles*period-length)).reshape(-1,cycles,period)
        padded_period = 4*((period+3)//4)
        raw = F.pad(raw,(0,padded_period-period))
        patches = raw.reshape(len(selected),cycles,padded_period//4,4)
        tokens = codec.project(patches,use_bias=False)
        half = codec.width//2
        cpos = sinusoidal_positions(cycles,half).to(tokens)
        ppos = sinusoidal_positions(padded_period//4,codec.width-half).to(tokens)
        positions = torch.cat((cpos[:,None,:].expand(-1,padded_period//4,-1),
                               ppos[None,:,:].expand(cycles,-1,-1)),dim=-1)
        valid = torch.arange(cycles,device=observed.device)[:,None]*period + torch.arange(padded_period//4,device=observed.device)[None,:]*4 < length
        image = (tokens+positions*valid[...,None]).permute(0,3,1,2).contiguous()
        encoded = block(image,parameters).permute(0,2,3,1)
        decoded = codec.decode(encoded,use_bias=False).reshape(len(selected),cycles,padded_period,codec.width)
        restored = decoded[:,:,:period].reshape(len(selected),cycles*period,codec.width)[:,:length]
        merged = (weights[selected]*(periods[selected]==period)).sum(-1).to(restored)
        output = output.index_add(0,selected,restored*merged[:,None,None])
    return output


class Model(SeparateMLP):
    def __init__(self, configs):
        super().__init__(configs)
        self.shared_patch_codecs = nn.ModuleList([PatchCodec(configs.d_model) for _ in self.seasonals])
        for encoder in self.seasonals:
            del encoder.value_embedding

    def codecs(self,index):
        return self.shared_patch_codecs[index],self.shared_patch_codecs[index]

    def encode_multiscale(self,raw_scales):
        normalized = [self.normalize(raw_scales[0],'norm')]
        normalized += [norm(x,'norm') for norm,x in zip(self.coarse_normalizers,raw_scales[1:])]
        batch,_,channels = normalized[0].shape
        seasons,trends = [],[]
        for index,(encoder,x) in enumerate(zip(self.seasonals,normalized)):
            observed = x.transpose(1,2).reshape(batch*channels,encoder.length,1)
            seasonal,trend = self.initial_decomposition(observed)
            sc,tc = self.codecs(index)
            seasons.append(fft_patch_features(encoder,seasonal,sc))
            trends.append(tc.sliding(trend))
        for pdm in self.pdm_blocks: seasons,trends = pdm(seasons,trends)
        return normalized,(seasons,trends)
