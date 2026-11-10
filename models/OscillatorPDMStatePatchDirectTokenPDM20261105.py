"""Direct compact seasonal-token PDM, with residual MLP + LayerNorm."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from models.OscillatorPDMStatePatchBestCombination20261104 import InheritedCombination
from models.OscillatorPDMStatePatchBestSpectralContinuousRNG20261103 import spectral_dynamics, Model as SpectralModel
from models.OscillatorPDMStatePatchBestTimeVaryingRotation import position_transitions
from models.OscillatorDualEncoderTimeMixer import sinusoidal_positions
from layers.OscillatorTimeMix import complex_oscillator_scan

def compact_linear(layer, output_factor=1):
    # Deterministic scratch-reference conversion: sum grouped input columns,
    # average grouped output rows. No trained checkpoint is used.
    w=layer.weight.detach(); out_n,in_n=w.shape
    with torch.random.fork_rng(devices=[]):
        result=nn.Linear(in_n//4,out_n//output_factor,bias=layer.bias is not None)
    with torch.no_grad():
        result.weight.copy_(w.reshape(out_n//output_factor,output_factor,in_n//4,4).sum(-1).mean(1))
        if layer.bias is not None:result.bias.copy_(layer.bias.reshape(-1,output_factor).mean(1))
    return result

class Model(InheritedCombination):
    def __init__(self,configs):
        super().__init__(configs)
        for e,p in zip(self.seasonals,self.season_patch_encoders):
            p.stride=4
            del p.patch_readout
            e.decoder.length//=4
            if isinstance(e.temporal_projection,nn.Sequential):
                e.temporal_projection[0]=compact_linear(e.temporal_projection[0])
            else:
                e.temporal_projection=compact_linear(e.temporal_projection)
        for pdm in self.pdm_blocks:
            for stage in pdm.mixing_multi_scale_season.down_sampling_layers:
                stage[0]=compact_linear(stage[0],4)
                stage[2]=compact_linear(stage[2],4)
            pdm.token_residual_norms=nn.ModuleList([nn.LayerNorm(configs.d_model) for _ in self.seasonals])

    def load_original_reference(self,state):
        current=self.state_dict();converted={}
        for key,target in current.items():
            if key not in state:
                assert '.token_residual_norms.' in key,key
                converted[key]=target.clone();continue
            source=state[key]
            if source.shape==target.shape:converted[key]=source.clone();continue
            if source.ndim==2:
                fo=source.shape[0]//target.shape[0];fi=source.shape[1]//target.shape[1]
                assert fi==4 and fo in (1,4),key
                converted[key]=source.reshape(target.shape[0],fo,target.shape[1],fi).sum(-1).mean(1)
            else:
                assert source.ndim==1 and source.numel()==target.numel()*4,key
                converted[key]=source.reshape(-1,4).mean(1)
        self.load_state_dict(converted,strict=True)
        return converted

    def encode_direct_tokens(self,raw_scales):
        normalized=[self.normalize(raw_scales[0],'norm')]
        normalized += [norm(x,'norm') for norm,x in zip(self.coarse_normalizers,raw_scales[1:])]
        batch,_,channels=normalized[0].shape
        seasons,trends=[],[]
        for e,p,x in zip(self.seasonals,self.season_patch_encoders,normalized):
            observed=x.transpose(1,2).reshape(batch*channels,e.length,1)
            season,trend=self.initial_decomposition(observed)
            history=F.linear(season,e.value_embedding.weight,None)
            patches,_=p.patchify(history);n,k,size,d=patches.shape
            embedded=p.patch_embedding(patches.reshape(n,k,size*d))
            with sdpa_kernel(SDPBackend.MATH):
                tokens=p.attention(embedded+sinusoidal_positions(k,d).to(embedded))
            # Token-level skip connection; no token decoding or time expansion.
            seasons.append(p.output_norm(history.reshape(n,k,4,d).mean(2)+tokens))
            trends.append(e.value_embedding(trend))
        for pdm in self.pdm_blocks:
            mixed=pdm.mixing_multi_scale_season([s.transpose(1,2) for s in seasons])
            seasons=[norm(s+pdm.out_cross_layer(m)) for s,m,norm in zip(seasons,mixed,pdm.token_residual_norms)]
            mt=pdm.mixing_multi_scale_trend([t.transpose(1,2) for t in trends])
            trends=[t+pdm.trend_out_cross_layer(m) for t,m in zip(trends,mt)]
        return normalized,seasons,trends

    def predict_components(self,x_enc):
        scales,seasons,trends=self.encode_direct_tokens(self.downsample(x_enc))
        spectrum=self.spectral_encoder(SpectralModel.spectrum_features(self,scales[0]))
        alpha,beta=self.branch_gains();batch,_,channels=x_enc.shape
        recurrent=[];diagnostics=[]
        for i,(e,history,trend) in enumerate(zip(self.seasonals,seasons,trends)):
            decoder=e.decoder
            future=e.temporal_projection(history.transpose(1,2)).transpose(1,2)+e.future_positions.to(history)
            n,h,d=future.shape;steps=h//4;future=future.reshape(n,steps,4,d).mean(2)
            compact_trend=trend.reshape(n,history.shape[1],4,d).mean(2)
            transition,initial,_,angle=spectral_dynamics(decoder,history,compact_trend,self.spectral_conditions[i](spectrum))
            sequence,diag=position_transitions(decoder,history,transition,angle,self.position_frequency_heads[i],self.position_frequency_basis)
            drive=decoder.future_drive(future,history).reshape(n,steps,decoder.modes,2)*beta.repeat(batch)[:,None,None,None]
            initial=initial*alpha.repeat(batch)[:,None,None]
            states=complex_oscillator_scan(sequence,drive,initial)
            outputs=decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
            recurrent.append(outputs.reshape(batch,channels,h).transpose(1,2))
            diagnostics.append(dict(context_error=0.,trend_delta_energy=0.,is_season_tensor=True,history_length=history.shape[1],direct_token_pdm=True,frequency_delta_max_abs=diag['frequency_delta_max_abs']))
        self.last_gate_context_diagnostics=diagnostics
        feature=[head(t).reshape(batch,channels,self.pred_len).transpose(1,2) for head,t in zip(self.feature_trend_heads,trends)]
        return torch.stack(recurrent+feature,-1)
