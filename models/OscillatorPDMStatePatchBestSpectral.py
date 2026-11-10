"""Historical best S-gate model plus input-spectrum-conditioned rotary dynamics.
The spectrum alters decay/frequency BEFORE the historical scan. No forecast bypass.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import Model as Original
from layers.OscillatorTimeMix import complex_oscillator_scan

def spectral_dynamics(decoder, history, trend_history, spectral_shift):
    if trend_history is None or trend_history.shape != history.shape:
        raise ValueError('Expected same-scale seasonal/trend histories [N,T,D]')
    context = decoder.context_norm(torch.cat((history.mean(1), history[:, -1]), -1))
    trend = decoder.trend_context_norm(torch.cat((trend_history.mean(1), trend_history[:, -1]), -1))
    dtype = torch.float64 if context.dtype == torch.float64 else torch.float32
    shifts = decoder.condition(context).to(dtype) + decoder.trend_condition(trend).to(dtype) + spectral_shift.to(dtype)
    (decay_shift, frequency_shift) = shifts.chunk(2, -1)
    rate = F.softplus(decoder.log_decay.to(dtype) + decay_shift)
    radius = torch.exp(-rate)
    frequency = math.pi * torch.sigmoid(decoder.frequency_logits.to(dtype) + frequency_shift)
    transition = torch.stack((radius * frequency.cos(), radius * frequency.sin()), -1)
    initial = decoder.encode_history_states(history, transition)[:, -1]
    if decoder.stride != 1:
        radius = radius.pow(1.0 / decoder.stride)
        frequency = frequency / decoder.stride
        transition = torch.stack((radius * frequency.cos(), radius * frequency.sin()), -1)
    return (transition, initial, radius, frequency)

def spectral_forecast(encoder, x, history_gain, drive_gain, history, trend, spectral_shift):
    (batch, _, channels) = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    (n, h, d) = future.shape
    if h % 4:
        raise ValueError('Forecast horizon must be divisible by4')
    future = future.reshape(n, h // 4, 4, d).mean(2)
    decoder = encoder.decoder
    (transition, initial, _, _) = spectral_dynamics(decoder, history, trend, spectral_shift)
    drive = decoder.future_drive(future, history).reshape(n, h // 4, decoder.modes, 2)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    drive = drive * drive_gain.repeat(batch)[:, None, None, None]
    states = complex_oscillator_scan(transition[:, None].expand(-1, h // 4, -1, -1), drive, initial)
    outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    if outputs.shape != (n, h // 4, 4):
        raise RuntimeError('Expected four values per state')
    return outputs.reshape(batch, channels, h).transpose(1, 2)

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        if self.seq_len!=96:raise ValueError('This registered experiment fixes history=96')
        # Extra initialization does not advance the baseline CPU RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.spectral_encoder=nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.LayerNorm(32))
            self.spectral_conditions=nn.ModuleList([nn.Linear(32,16) for _ in range(4)])
        for head in self.spectral_conditions:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.gate_context_policy='season'

    def spectrum_features(self,normalized):
        # [B,96,C] -> [B*C,49] complex -> [B*C,96] real then imaginary.
        b,l,c=normalized.shape
        if l!=96:raise ValueError('Expected exactly 96 observed history points')
        spectrum=torch.fft.rfft(normalized.transpose(1,2).reshape(b*c,l),dim=-1,norm='ortho')[:,1:]
        return torch.cat((spectrum.real,spectrum.imag),-1)

    def predict_components(self,x_enc):
        if x_enc.ndim!=3 or x_enc.shape[1:]!=(self.seq_len,self.enc_in):raise ValueError('Expected B,96,C history')
        scales,(seasons,trends)=self.encode_multiscale(self.downsample(x_enc))
        q=self.spectral_encoder(self.spectrum_features(scales[0]))
        alpha,beta=self.branch_gains();batch,_,channels=x_enc.shape
        recurrent=[];diagnostics=[]
        for e,x,s,t,head in zip(self.seasonals,scales,seasons,trends,self.spectral_conditions):
            shift=head(q)
            recurrent.append(spectral_forecast(e,x,alpha,beta,s,t,shift))
            with torch.no_grad():
                diagnostics.append(dict(context_error=0.,trend_delta_energy=0.,is_season_tensor=True,history_length=s.shape[1],spectral_shift_max_abs=shift.abs().max(),spectral_shift_mean_square=shift.double().square().mean(),spectral_before_history_scan=True))
        feature=[head(t).reshape(batch,channels,self.pred_len).transpose(1,2) for head,t in zip(self.feature_trend_heads,trends)]
        self.last_gate_context_diagnostics=diagnostics
        return torch.stack(recurrent+feature,-1)
