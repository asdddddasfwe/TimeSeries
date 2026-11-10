"""Original MA-first sliding / mean4 / endpoint model; only future gate context changes."""
import torch
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import Model as Original
from layers.OscillatorTimeMix import complex_oscillator_scan

def gate_context(season,trend,policy):
    if season.shape!=trend.shape or policy not in ('season','sum'):
        raise ValueError('Matched final PDM feature tensors and explicit gate policy required')
    return season if policy=='season' else season+trend

def gate_forecast(encoder,x,history_gain,drive_gain,history,trend,context):
    batch,_,channels=x.shape
    future=encoder.temporal_projection(history.transpose(1,2)).transpose(1,2)
    future=future+encoder.future_positions.to(future)
    n,h,d=future.shape
    if h%4:raise ValueError('Forecast horizon must be divisible by4')
    future=future.reshape(n,h//4,4,d).mean(2)
    decoder=encoder.decoder
    transition,initial,_,_=decoder.dynamics(history,trend)
    drive=decoder.future_drive(future,context).reshape(n,h//4,decoder.modes,2)
    initial=initial*history_gain.repeat(batch)[:,None,None]
    drive=drive*drive_gain.repeat(batch)[:,None,None,None]
    states=complex_oscillator_scan(transition[:,None].expand(-1,h//4,-1,-1),drive,initial)
    outputs=decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    if outputs.shape!=(n,h//4,4):raise RuntimeError('Expected four values per state')
    return outputs.reshape(batch,channels,h).transpose(1,2)

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        self.gate_context_policy=getattr(configs,'gate_context_policy','sum')
        if self.gate_context_policy not in ('season','sum'):raise ValueError('Explicit gate policy required')

    def predict_components(self,x_enc):
        if x_enc.ndim!=3 or x_enc.shape[1:]!=(self.seq_len,self.enc_in):
            raise ValueError('Expected history B,seq_len,enc_in')
        scales,(seasons,trends)=self.encode_multiscale(self.downsample(x_enc))
        alpha,beta=self.branch_gains();batch,_,channels=x_enc.shape
        recurrent=[];diagnostics=[]
        for e,x,s,t in zip(self.seasonals,scales,seasons,trends):
            context=gate_context(s,t,self.gate_context_policy)
            recurrent.append(gate_forecast(e,x,alpha,beta,s,t,context))
            with torch.no_grad():
                expected=s if self.gate_context_policy=='season' else s+t
                diagnostics.append(dict(context_error=(context-expected).abs().max(),trend_delta_energy=(context-s).double().square().mean(),is_season_tensor=context is s,history_length=s.shape[1]))
        feature=[head(t).reshape(batch,channels,self.pred_len).transpose(1,2) for head,t in zip(self.feature_trend_heads,trends)]
        self.last_gate_context_diagnostics=diagnostics
        return torch.stack(recurrent+feature,-1)
