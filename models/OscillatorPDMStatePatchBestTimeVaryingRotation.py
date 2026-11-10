"""Original best S-gate baseline with smooth future-position rotation only.
Historical dynamics, radius, future drive and endpoint readout are unchanged.
"""
import math
import torch
from torch import nn
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import Model as Original
from layers.OscillatorTimeMix import complex_oscillator_scan

def position_transitions(decoder,history,transition,frequency,head,basis):
    context=decoder.context_norm(torch.cat((history.mean(1),history[:,-1]),-1))
    coefficients=head(context).reshape(history.shape[0],decoder.modes,4)
    raw=torch.einsum('nmk,tk->ntm',coefficients.to(transition.dtype),basis.to(transition))
    # Quarter of distance to either valid angular boundary, with original stride convention.
    upper=math.pi/decoder.stride
    margin=.25*torch.minimum(frequency,upper-frequency)
    delta=margin[:,None,:]*raw.tanh()
    co,si=delta.cos(),delta.sin();real,imag=transition.unbind(-1)
    sequence=torch.stack((real[:,None]*co-imag[:,None]*si,real[:,None]*si+imag[:,None]*co),-1)
    with torch.no_grad():
        diag=dict(context_error=0.,trend_delta_energy=0.,is_season_tensor=True,history_length=history.shape[1],time_varying_rotation=True,history_dynamics_unchanged=True,frequency_delta_max_abs=delta.abs().max(),frequency_delta_bound_max=margin.max(),frequency_time_std_max=delta.double().std(1,unbiased=False).max(),radius_preservation_max_abs=(sequence.double().square().sum(-1)-transition.double().square().sum(-1)[:,None]).abs().max())
    return sequence,diag

def timevary_forecast(encoder, x, history_gain, drive_gain, history, trend, head, basis):
    (batch, _, channels) = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    (n, h, d) = future.shape
    if h % 4:
        raise ValueError('Forecast horizon must be divisible by4')
    future = future.reshape(n, h // 4, 4, d).mean(2)
    decoder = encoder.decoder
    (transition, initial, _, frequency) = decoder.dynamics(history, trend)
    drive = decoder.future_drive(future, history).reshape(n, h // 4, decoder.modes, 2)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    drive = drive * drive_gain.repeat(batch)[:, None, None, None]
    (sequence, diag) = position_transitions(decoder, history, transition, frequency, head, basis)
    states = complex_oscillator_scan(sequence, drive, initial)
    outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    if outputs.shape != (n, h // 4, 4):
        raise RuntimeError('Expected four values per state')
    return (outputs.reshape(batch, channels, h).transpose(1, 2), diag)

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        if self.seq_len!=96 or self.pred_len!=720:raise ValueError('Registered experiment fixes input96/H720')
        with torch.random.fork_rng(devices=[]):
            self.position_frequency_heads=nn.ModuleList([nn.Linear(2*e.decoder.drive_projection.in_features,4*e.decoder.modes,bias=False) for e in self.seasonals])
        for head in self.position_frequency_heads:nn.init.zeros_(head.weight)
        steps=self.pred_len//4
        basis=torch.cos(math.pi*(torch.arange(steps,dtype=torch.float32)[:,None]+.5)*torch.arange(1,5,dtype=torch.float32)[None,:]/steps)
        self.register_buffer('position_frequency_basis',basis)
        self.gate_context_policy='season'

    def predict_components(self,x_enc):
        if x_enc.ndim!=3 or x_enc.shape[1:]!=(self.seq_len,self.enc_in):raise ValueError('Expected B,96,C history')
        scales,(seasons,trends)=self.encode_multiscale(self.downsample(x_enc))
        alpha,beta=self.branch_gains();batch,_,channels=x_enc.shape
        recurrent=[];diagnostics=[]
        for e,x,s,t,head in zip(self.seasonals,scales,seasons,trends,self.position_frequency_heads):
            result,diag=timevary_forecast(e,x,alpha,beta,s,t,head,self.position_frequency_basis)
            recurrent.append(result);diagnostics.append(diag)
        feature=[head(t).reshape(batch,channels,self.pred_len).transpose(1,2) for head,t in zip(self.feature_trend_heads,trends)]
        self.last_gate_context_diagnostics=diagnostics
        return torch.stack(recurrent+feature,-1)
