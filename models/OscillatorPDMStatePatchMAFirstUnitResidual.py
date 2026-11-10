"""Both hidden-state coefficients exactly one, as explicitly requested.

Adjacent: h[k] = A h[k-1] + h[k-1] + u[k].
Lag8: original recurrence before k=8, then A h[k-1] + h[k-8] + u[k].
No normalization, clipping, coefficient mixing, learned gate, or extra states.
"""
import torch
from layers.OscillatorTimeMix import complex_oscillator_scan as base_scan
from models.OscillatorPDMStatePatchMAFirst import Model as Original


def lagged_scan(transition, drive, initial, lag=8, mix=1.0):
    if isinstance(lag,bool) or int(lag)!=lag or int(lag)<1:
        raise ValueError('lag must be a positive integer')
    if float(mix)!=1.0:
        raise ValueError('This experiment requires both coefficients exactly1')
    lag=int(lag)
    if transition.shape!=drive.shape or drive.ndim!=4 or drive.shape[-1]!=2:
        raise ValueError('Expected transition/drive [N,K,J,2]')
    if initial.shape!=(drive.shape[0],drive.shape[2],2) or drive.shape[1]<1:
        raise ValueError('Invalid initial/horizon')
    if lag==1:
        added=torch.stack((transition[...,0]+1.0,transition[...,1]),-1)
        return base_scan(added,drive,initial)
    horizon=drive.shape[1]
    if lag>horizon:return base_scan(transition,drive,initial)
    prefix=lag-1
    states=base_scan(transition[:,:prefix],drive[:,:prefix],initial)
    blocks=[states];previous=states[:,-1];delayed=torch.cat((initial[:,None],states),1)
    for begin in range(prefix,horizon,lag):
        end=min(begin+lag,horizon)
        states=base_scan(transition[:,begin:end],drive[:,begin:end]+delayed[:,:end-begin],previous)
        blocks.append(states);previous,delayed=states[:,-1],states
    return torch.cat(blocks,1)


def forecast(encoder,x,history_gain,drive_gain,history,trend,lag):
    batch,_,channels=x.shape
    future=encoder.temporal_projection(history.transpose(1,2)).transpose(1,2)
    future=future+encoder.future_positions.to(future)
    n,horizon,width=future.shape
    future=future.reshape(n,horizon//4,4,width).mean(2)
    decoder=encoder.decoder
    transition,initial,_,_=decoder.dynamics(history,trend)
    drive=decoder.future_drive(future,history).reshape(n,horizon//4,decoder.modes,2)
    initial=initial*history_gain.repeat(batch)[:,None,None]
    drive=drive*drive_gain.repeat(batch)[:,None,None,None]
    states=lagged_scan(transition[:,None].expand_as(drive),drive,initial,lag,1.)
    output=decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    return output.reshape(batch,channels,horizon).transpose(1,2)


class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        self.unit_residual_lag=int(getattr(configs,'unit_residual_lag',1))
        if self.unit_residual_lag not in (1,8):raise ValueError('Only predeclared lag1 or lag8')

    def predict_components(self,x_enc):
        if x_enc.ndim!=3 or x_enc.shape[1:]!=(self.seq_len,self.enc_in):raise ValueError('Expected B,96,C')
        scales,(seasons,trends)=self.encode_multiscale(self.downsample(x_enc))
        alpha,beta=self.branch_gains();batch,_,channels=x_enc.shape
        recurrent=[forecast(e,x,alpha,beta,s,t,self.unit_residual_lag) for e,x,s,t in zip(self.seasonals,scales,seasons,trends)]
        feature=[head(t).reshape(batch,channels,self.pred_len).transpose(1,2) for head,t in zip(self.feature_trend_heads,trends)]
        return torch.stack(recurrent+feature,-1)
