"""Two separate interventions on original MAFirst; no residual/RMS-cap stacking."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorPDMStatePatchMAFirst import Model as Original
from layers.OscillatorTimeMix import complex_oscillator_scan

def correction_scan(transition,drive,initial,anchor=None,gate=None):
    """N,K,J,2 affine scan. Gate is historical-context/position conditioned only."""
    if anchor is None:
        if gate is not None:raise ValueError('Gate requires anchor')
        return complex_oscillator_scan(transition,drive,initial)
    if anchor.shape!=drive.shape or gate.shape!=drive.shape[:-1]+(1,):
        raise ValueError('Anchor/gate shape mismatch')
    return complex_oscillator_scan((1.-gate)*transition,(1.-gate)*drive+gate*anchor,initial)

def forecast(encoder,x,history_gain,drive_gain,history,trend,arm):
    b,_,channels=x.shape
    future=encoder.temporal_projection(history.transpose(1,2)).transpose(1,2)
    future=future+encoder.future_positions.to(future)
    n,h,width=future.shape
    future=future.reshape(n,h//4,4,width).mean(2)
    decoder=encoder.decoder
    transition,initial,_,_=decoder.dynamics(history,trend)
    drive=decoder.future_drive(future,history).reshape(n,h//4,decoder.modes,2)
    initial=initial*history_gain.repeat(b)[:,None,None]
    drive=drive*drive_gain.repeat(b)[:,None,None,None]
    anchor=gate=None
    if arm=='gated_anchor':
        context=F.layer_norm(future,(width,))
        anchor=decoder.correction_anchor(context).reshape_as(drive)
        gate=torch.sigmoid(decoder.correction_gate(context)).unsqueeze(-1)
    states=correction_scan(transition[:,None].expand_as(drive),drive,initial,anchor,gate)
    output=decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    return output.reshape(b,channels,h).transpose(1,2),gate

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        self.correction_arm=configs.correction_arm
        if self.correction_arm not in ('centered_recurrence','gated_anchor'):raise ValueError('Unknown arm')
        self.last_gates=[]
        if self.correction_arm=='gated_anchor':
            for scale,encoder in enumerate(self.seasonals):
                decoder=encoder.decoder
                width=decoder.drive_projection.in_features
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(20211020+scale)
                    decoder.correction_anchor=nn.Linear(width,2*decoder.modes)
                    decoder.correction_gate=nn.Linear(width,decoder.modes)
                nn.init.zeros_(decoder.correction_anchor.weight)
                nn.init.zeros_(decoder.correction_anchor.bias)
                nn.init.zeros_(decoder.correction_gate.weight)
                nn.init.constant_(decoder.correction_gate.bias,math.log(.05/.95))

    def predict_components(self,x_enc):
        if x_enc.ndim!=3 or x_enc.shape[1:]!=(self.seq_len,self.enc_in):
            raise ValueError('Expected B,96,7')
        scales,(seasons,trends)=self.encode_multiscale(self.downsample(x_enc))
        alpha,beta=self.branch_gains()
        outputs=[forecast(e,x,alpha,beta,s,t,self.correction_arm)
                 for e,x,s,t in zip(self.seasonals,scales,seasons,trends)]
        recurrent=torch.stack([v[0] for v in outputs],-1)
        self.last_gates=[g.detach() for _,g in outputs if g is not None]
        if self.correction_arm=='centered_recurrence':
            # Per sample/channel/scale, over the complete generated horizon only.
            recurrent=recurrent-recurrent.mean(1,keepdim=True)
        b,_,channels=x_enc.shape
        feature=torch.stack([head(t).reshape(b,channels,self.pred_len).transpose(1,2)
                   for head,t in zip(self.feature_trend_heads,trends)],-1)
        return torch.cat((recurrent,feature),-1)
