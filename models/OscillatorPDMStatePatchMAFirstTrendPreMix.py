"""Historical patch Transformer BEFORE PDM, horizon MLP AFTER PDM."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from models.OscillatorPDMStatePatchMAFirst import Model as Original
from models.OscillatorPDMStatePatchMAFirstTrendPatchTransformer import PatchTransformerTrendHead

def restore_patch_residual(delta,length):
    """Repeat each token correction over its four covered times, then average.
    Parameter-free overlap-add: [N,L,D] -> [N,T,D], patch4/stride2.
    """
    n,l,d=delta.shape
    columns=delta[:,:,None,:].expand(n,l,4,d).permute(0,3,2,1).reshape(n,4*d,l)
    output=F.fold(columns,(1,length),(1,4),stride=(1,2)).squeeze(2).transpose(1,2)
    count=F.fold(delta.new_ones(1,4,l),(1,length),(1,4),stride=(1,2)).reshape(1,length,1)
    if not bool((count>0).all()):raise ValueError('Uncovered historical positions')
    return output/count

class PreMixTrendHead(PatchTransformerTrendHead):
    def encode_before_mix(self,history):
        patches=history.unfold(1,4,2).permute(0,1,3,2).reshape(history.shape[0],self.patch_count,4*self.width)
        tokens=self.patch_projection(patches)+self.patch_positions.to(history)
        with sdpa_kernel(SDPBackend.MATH):encoded=self.transformer(tokens)
        return history+restore_patch_residual(encoded-tokens,self.length)

    def forward(self,history):
        # Fixed readout pooling preserves the post-mix arm's exact MLP shapes.
        pooled=history.unfold(1,4,2).mean(-1)
        return self.linear_forecast(history)+self.nonlinear_readout(pooled.flatten(1))

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        self.feature_trend_heads=nn.ModuleList([
            PreMixTrendHead(head,i) for i,head in enumerate(self.feature_trend_heads)])

    def encode_multiscale(self,raw_scales):
        normalized=[self.normalize(raw_scales[0],'norm')]
        normalized += [norm(x,'norm') for norm,x in zip(self.coarse_normalizers,raw_scales[1:])]
        batch,_,channels=normalized[0].shape
        seasons,trends=[],[]
        for encoder,x in zip(self.seasonals,normalized):
            observed=x.transpose(1,2).reshape(batch*channels,encoder.length,1)
            seasonal,trend=self.initial_decomposition(observed)
            seasons.append(F.linear(seasonal,encoder.value_embedding.weight,None))
            trends.append(encoder.value_embedding(trend))
        seasons=self.encode_seasonal_features(seasons)
        trends=[head.encode_before_mix(t) for head,t in zip(self.feature_trend_heads,trends)]
        for pdm in self.pdm_blocks:seasons,trends=pdm(seasons,trends)
        return normalized,(seasons,trends)
