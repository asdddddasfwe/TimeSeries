"""All-direct with explicit historical radius/angle inputs to the future head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorPDMStatePatchBestAllDirectParameters import Model as Original, parameter_context
from layers.OscillatorTimeMix import complex_oscillator_scan

def direct_future_transitions(decoder,history,trend,head,basis):
    context=parameter_context(decoder,history,trend)
    raw_r,raw_w=decoder.history_parameter_head(context).chunk(2,-1)
    rh=torch.exp(-F.softplus(raw_r));wh=math.pi*torch.sigmoid(raw_w)
    n=history.shape[0];steps=basis.shape[0]
    summary=context[:,None].expand(-1,steps,-1)
    positions=basis.to(context)[None].expand(n,-1,-1)
    # Read actual history parameters, preserving gradient paths into their head.
    parameters=torch.cat((rh,wh),-1)[:,None].expand(-1,steps,-1)
    raw_decay,raw_angle=head(torch.cat((summary,positions,parameters),-1)).chunk(2,-1)
    radius=torch.exp(-F.softplus(raw_decay)/decoder.stride)
    angle=(math.pi/decoder.stride)*torch.sigmoid(raw_angle)
    sequence=torch.stack((radius*angle.cos(),radius*angle.sin()),-1)
    with torch.no_grad():
        diag=dict(context_error=0.,trend_delta_energy=0.,is_season_tensor=True,
          historical_parameter_conditioning=True,historical_radius_and_angle_inputs=True,
          historical_inputs_detached=False,all_direct_parameter_generation=True,
          direct_history_decay_angle=True,direct_future_decay_angle=True,
          no_base_parameter_or_delta_addition=True,no_historical_radius_or_angle_input=False,
          direct_calibration_gains=True,historical_scan_algorithm_unchanged=True,
          history_dynamics_unchanged=False,fixed_stride_units_retained=True,
          future_radius_min=radius.min(),future_radius_max=radius.max(),
          future_radius_time_std_max=radius.double().std(1,unbiased=False).max(),
          frequency_time_std_max=angle.double().std(1,unbiased=False).max(),
          direct_frequency_min=angle.min(),direct_frequency_max=angle.max(),
          direct_frequency_upper_bound=math.pi/decoder.stride)
    return sequence,diag

def all_direct_forecast(encoder,x,history_gain,drive_gain,history,trend,head,basis):
    batch,_,channels=x.shape
    future=encoder.temporal_projection(history.transpose(1,2)).transpose(1,2)
    future=future+encoder.future_positions.to(future)
    n,h,width=future.shape
    future=future.reshape(n,h//4,4,width).mean(2)
    decoder=encoder.decoder
    _,initial,_,_=decoder.dynamics(history,trend)
    drive=decoder.future_drive(future,history).reshape(n,h//4,decoder.modes,2)
    initial=initial*history_gain.repeat(batch)[:,None,None]
    drive=drive*drive_gain.repeat(batch)[:,None,None,None]
    sequence,diag=direct_future_transitions(decoder,history,trend,head,basis)
    states=complex_oscillator_scan(sequence,drive,initial)
    outputs=decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    return outputs.reshape(batch,channels,h).transpose(1,2),diag

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        # Preserve all original 132 input columns and downstream initialization.
        # Extra columns have independent untrained random weights, not zero gates.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20211103)
            for e,head in zip(self.seasonals,self.direct_future_parameter_heads):
                old=head[0];expanded=nn.Linear(old.in_features+2*e.decoder.modes,old.out_features)
                with torch.no_grad():
                    expanded.weight[:,:old.in_features].copy_(old.weight)
                    expanded.bias.copy_(old.bias)
                head[0]=expanded
    def predict_components(self,x_enc):
        scales,(seasons,trends)=self.encode_multiscale(self.downsample(x_enc))
        alpha,beta=self.branch_gains();batch,_,channels=x_enc.shape
        recurrent=[];diagnostics=[]
        for e,x,s,t,head in zip(self.seasonals,scales,seasons,trends,self.direct_future_parameter_heads):
            result,diag=all_direct_forecast(e,x,alpha,beta,s,t,head,self.direct_parameter_position_basis)
            recurrent.append(result);diagnostics.append(diag)
        feature=[head(t).reshape(batch,channels,self.pred_len).transpose(1,2) for head,t in zip(self.feature_trend_heads,trends)]
        self.last_gate_context_diagnostics=diagnostics
        return torch.stack(recurrent+feature,-1)
