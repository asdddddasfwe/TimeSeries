"""One recurrent hidden state directly reads out four disjoint future values.
Keep the original history encoder, transition A, and future feature projection.
Pool each group of four projected future features (including PE), scan24 times,
and decode each state16 through GELU MLP16->32->4. No scalar output bypass.
"""
import torch
from torch import nn
from models.OscillatorPDMTrendCondition import Model as PointwiseModel
from layers.OscillatorTimeMix import complex_oscillator_scan


def convert_pointwise_state(state):
    converted={}
    for k,v in state.items():
        if '.decoder.readout.output_projection.' in k:
            converted[k]=v.repeat(4,1) if v.ndim==2 else v.repeat(4)
        else:converted[k]=v.clone()
    return converted


def state_patch_forecast(encoder,x,history_gain,drive_gain,history,trend):
    batch,_,channels=x.shape
    future=encoder.temporal_projection(history.transpose(1,2)).transpose(1,2)
    future=future+encoder.future_positions.to(future)
    # Fixed mean pooling of four derived features; no observations from future.
    n,h,d=future.shape
    if h%4:raise ValueError('Forecast horizon must be divisible by4')
    future=future.reshape(n,h//4,4,d).mean(2)
    decoder=encoder.decoder
    transition,initial,_,_=decoder.dynamics(history,trend)
    drive=decoder.future_drive(future,history).reshape(n,h//4,decoder.modes,2)
    initial=initial*history_gain.repeat(batch)[:,None,None]
    drive=drive*drive_gain.repeat(batch)[:,None,None,None]
    # One ORIGINAL A per recurrent patch index; not A**4 and no feedback of outputs.
    states=complex_oscillator_scan(transition[:,None].expand(-1,h//4,-1,-1),drive,initial)
    outputs=decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    if outputs.shape!=(n,h//4,4):raise RuntimeError('Expected four values per state')
    return outputs.reshape(batch,channels,h).transpose(1,2)


class Model(PointwiseModel):
    def __init__(self,configs):
        super().__init__(configs)
        if self.pred_len%4:raise ValueError('Forecast horizon must be divisible by4')
        for encoder in self.seasonals:
            head=encoder.decoder.readout
            old=head.output_projection
            with torch.random.fork_rng(devices=[]):replacement=nn.Linear(old.in_features,4)
            with torch.no_grad():
                replacement.weight.copy_(old.weight.repeat(4,1))
                replacement.bias.copy_(old.bias.repeat(4))
            head.output_projection=replacement
            head.out_features=4
            encoder.decoder.horizon=self.pred_len//4

    def load_pointwise_initial_state(self,state):
        self.load_state_dict(convert_pointwise_state(state),strict=True)

    def predict_components(self,x_enc):
        if x_enc.ndim!=3 or x_enc.shape[1:]!=(self.seq_len,self.enc_in):
            raise ValueError('Expected history B,seq_len,enc_in')
        scales,(seasons,trends)=self.encode_multiscale(self.downsample(x_enc))
        alpha,beta=self.branch_gains();batch,_,channels=x_enc.shape
        recurrent=[state_patch_forecast(e,x,alpha,beta,s,t) for e,x,s,t in zip(self.seasonals,scales,seasons,trends)]
        feature=[head(t).reshape(batch,channels,self.pred_len).transpose(1,2) for head,t in zip(self.feature_trend_heads,trends)]
        return torch.stack(recurrent+feature,-1)
