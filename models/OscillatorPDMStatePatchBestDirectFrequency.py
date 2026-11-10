"""Original best S gate with direct S/T/position-conditioned future angles.
No addition of base omega, delta, margin or tanh. History and future radii retained.
"""
import math
import torch
from torch import nn
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import Model as Original
from layers.OscillatorTimeMix import complex_oscillator_scan

def direct_position_transitions(decoder,history,trend,radius,head,basis):
    # Flattened sample-variable axis N, same-scale history [N,L,32].
    cs=decoder.context_norm(torch.cat((history.mean(1),history[:,-1]),-1))
    ct=decoder.trend_context_norm(torch.cat((trend.mean(1),trend[:,-1]),-1))
    n=history.shape[0];steps=basis.shape[0]
    summary=torch.cat((cs,ct),-1)[:,None,:].expand(-1,steps,-1)
    positions=basis.to(cs)[None,:,:].expand(n,-1,-1)
    inputs=torch.cat((summary,positions),-1)  # [N,180,132]
    logits=head(inputs)
    angle=(math.pi/decoder.stride)*torch.sigmoid(logits.to(radius.dtype))
    sequence=torch.stack((radius[:,None]*angle.cos(),radius[:,None]*angle.sin()),-1)
    with torch.no_grad():
        error=(sequence.double().square().sum(-1)-radius.double().square()[:,None]).abs().max()
        diag=dict(context_error=0.,trend_delta_energy=0.,is_season_tensor=True,history_length=history.shape[1],
                  time_varying_rotation=True,direct_frequency_generation=True,direct_frequency_uses_season_and_trend=True,
                  direct_frequency_uses_position=True,no_base_frequency_or_delta_addition=True,
                  history_dynamics_unchanged=True,radius_preservation_max_abs=error,
                  frequency_time_std_max=angle.double().std(1,unbiased=False).max(),
                  direct_frequency_min=angle.min(),direct_frequency_max=angle.max(),
                  direct_frequency_upper_bound=math.pi/decoder.stride)
    return sequence,diag

def direct_forecast(encoder,x,history_gain,drive_gain,history,trend,head,basis):
    batch,_,channels=x.shape
    future=encoder.temporal_projection(history.transpose(1,2)).transpose(1,2)
    future=future+encoder.future_positions.to(future)
    n,h,d=future.shape
    if h%4:raise ValueError('Forecast horizon must be divisible by4')
    future=future.reshape(n,h//4,4,d).mean(2)
    decoder=encoder.decoder
    transition,initial,radius,frequency=decoder.dynamics(history,trend)
    drive=decoder.future_drive(future,history).reshape(n,h//4,decoder.modes,2)
    initial=initial*history_gain.repeat(batch)[:,None,None]
    drive=drive*drive_gain.repeat(batch)[:,None,None,None]
    sequence,diag=direct_position_transitions(decoder,history,trend,radius,head,basis)
    states=complex_oscillator_scan(sequence,drive,initial)
    outputs=decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    if outputs.shape!=(n,h//4,4):raise RuntimeError('Expected four values per state')
    return outputs.reshape(batch,channels,h).transpose(1,2),diag

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        if self.seq_len!=96 or self.pred_len!=720:raise ValueError('Registered experiment fixes input96/H720')
        # Standard random heads; preserve the constructor RNG stream of common parameters.
        with torch.random.fork_rng(devices=[]):
            self.direct_frequency_heads=nn.ModuleList([
                nn.Sequential(nn.Linear(4*e.decoder.drive_projection.in_features+4,32),nn.GELU(),nn.Linear(32,e.decoder.modes))
                for e in self.seasonals])
        steps=self.pred_len//4
        basis=torch.cos(math.pi*(torch.arange(steps,dtype=torch.float32)[:,None]+.5)*torch.arange(1,5,dtype=torch.float32)[None,:]/steps)
        self.register_buffer('direct_frequency_basis',basis)
        self.gate_context_policy='season'

    def predict_components(self,x_enc):
        if x_enc.ndim!=3 or x_enc.shape[1:]!=(self.seq_len,self.enc_in):raise ValueError('Expected B,96,C history')
        scales,(seasons,trends)=self.encode_multiscale(self.downsample(x_enc))
        alpha,beta=self.branch_gains();batch,_,channels=x_enc.shape
        recurrent=[];diagnostics=[]
        for e,x,s,t,head in zip(self.seasonals,scales,seasons,trends,self.direct_frequency_heads):
            result,diag=direct_forecast(e,x,alpha,beta,s,t,head,self.direct_frequency_basis)
            recurrent.append(result);diagnostics.append(diag)
        feature=[head(t).reshape(batch,channels,self.pred_len).transpose(1,2) for head,t in zip(self.feature_trend_heads,trends)]
        self.last_gate_context_diagnostics=diagnostics
        return torch.stack(recurrent+feature,-1)
