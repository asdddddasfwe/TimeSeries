"""Unit-coefficient residuals with an explicit per-step hidden-state RMS cap.

v[k] = A h[k-1] + h[k-lag] + u[k] (shortcut only when k>=lag).
h[k] = v[k] / max(1, RMS(v[k])), RMS over the 8 complex modes.
The original historical scan/initialization is preserved. This is a nonlinear
state-normalized model, not the original unbounded recurrence or a 0.5 mixture.
"""
import torch
from models.OscillatorPDMStatePatchMAFirstUnitResidual import Model as UnitOriginal


def rms_cap(value):
    """Bound RMS by1 without squaring a huge FP32 number or dividing by zero."""
    # Detaching a purely algebraic scaling factor retains the exact RMS gradient.
    scale=value.detach().abs().amax(dim=(-2,-1),keepdim=True).clamp_min(1.)
    normalized=value/scale
    rms=scale*normalized.square().mean(dim=(-2,-1),keepdim=True).clamp_min(1e-12).sqrt()
    return value/rms.clamp_min(1.)


def lagged_scan(transition,drive,initial,lag=8,mix=1.):
    if isinstance(lag,bool) or int(lag)!=lag or int(lag)<1:raise ValueError('Positive integer lag required')
    if float(mix)!=1.:raise ValueError('Both summation coefficients must remain1')
    if transition.shape!=drive.shape or drive.ndim!=4 or drive.shape[-1]!=2:raise ValueError('Expected N,K,J,2')
    if initial.shape!=(drive.shape[0],drive.shape[2],2) or drive.shape[1]<1:raise ValueError('Invalid initial/horizon')
    lag=int(lag);states=[initial]
    for k in range(drive.shape[1]):
        previous=states[-1];a=transition[:,k]
        rotated=torch.stack((a[...,0]*previous[...,0]-a[...,1]*previous[...,1],
                             a[...,1]*previous[...,0]+a[...,0]*previous[...,1]),-1)
        value=rotated+drive[:,k]
        if k+1>=lag:value=value+states[k+1-lag]
        # Cap at each step, before feedback, never after the whole exploding scan.
        states.append(rms_cap(value))
    return torch.stack(states[1:],1)


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


class Model(UnitOriginal):
    def predict_components(self,x_enc):
        if x_enc.ndim!=3 or x_enc.shape[1:]!=(self.seq_len,self.enc_in):raise ValueError('Expected B,96,C')
        scales,(seasons,trends)=self.encode_multiscale(self.downsample(x_enc))
        alpha,beta=self.branch_gains();batch,_,channels=x_enc.shape
        recurrent=[forecast(e,x,alpha,beta,s,t,self.unit_residual_lag) for e,x,s,t in zip(self.seasonals,scales,seasons,trends)]
        feature=[head(t).reshape(batch,channels,self.pred_len).transpose(1,2) for head,t in zip(self.feature_trend_heads,trends)]
        return torch.stack(recurrent+feature,-1)
