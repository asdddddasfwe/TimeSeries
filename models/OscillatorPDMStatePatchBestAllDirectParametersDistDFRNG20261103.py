"""Original best S gate with directly generated historical/future decay and angles.
No base-plus-delta dynamics, inherited-radius future head or 1+offset calibration.
Encoder/PDM residuals and the existing direct 2*sigmoid future gate are retained.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import Model as Original
from models.OscillatorPDMTrendCondition import TrendConditionedDecoder
from layers.OscillatorTimeMix import complex_oscillator_scan

def parameter_context(decoder,history,trend):
    if history.shape!=trend.shape:raise ValueError('Expected matching final S/T history features')
    cs=decoder.context_norm(torch.cat((history.mean(1),history[:,-1]),-1))
    ct=decoder.trend_context_norm(torch.cat((trend.mean(1),trend[:,-1]),-1))
    return torch.cat((cs,ct),-1)

def parameter_head(inputs,modes,prior):
    head=nn.Sequential(nn.Linear(inputs,32),nn.GELU(),nn.Linear(32,2*modes))
    nn.init.normal_(head[-1].weight,std=.01)
    # Initialization only: these are independent trainable output biases at runtime.
    with torch.no_grad():head[-1].bias.copy_(torch.cat(prior))
    return head

class DirectHistoryDecoder(TrendConditionedDecoder):
    def dynamics(self,history,trend_history=None):
        context=parameter_context(self,history,trend_history)
        dtype=torch.float64 if context.dtype==torch.float64 else torch.float32
        raw_decay,raw_angle=self.history_parameter_head(context).to(dtype).chunk(2,-1)
        radius=torch.exp(-F.softplus(raw_decay))
        angle=math.pi*torch.sigmoid(raw_angle)
        transition=torch.stack((radius*angle.cos(),radius*angle.sin()),-1)
        initial=self.encode_history_states(history,transition)[:,-1]
        return transition,initial,radius,angle

def direct_future_transitions(decoder,history,trend,head,basis):
    context=parameter_context(decoder,history,trend)
    n=history.shape[0];steps=basis.shape[0]
    summary=context[:,None,:].expand(-1,steps,-1)
    positions=basis.to(context)[None].expand(n,-1,-1)
    dtype=torch.float64 if context.dtype==torch.float64 else torch.float32
    raw_decay,raw_angle=head(torch.cat((summary,positions),-1)).to(dtype).chunk(2,-1)
    radius=torch.exp(-F.softplus(raw_decay)/decoder.stride)
    angle=(math.pi/decoder.stride)*torch.sigmoid(raw_angle)
    sequence=torch.stack((radius*angle.cos(),radius*angle.sin()),-1)
    with torch.no_grad():
        diag=dict(context_error=0.,trend_delta_energy=0.,is_season_tensor=True,history_length=history.shape[1],
                  all_direct_parameter_generation=True,direct_history_decay_angle=True,
                  direct_future_decay_angle=True,no_base_parameter_or_delta_addition=True,
                  no_historical_radius_or_angle_input=True,direct_calibration_gains=True,
                  existing_future_gate_already_direct=True,historical_scan_algorithm_unchanged=True,
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
    n,h,d=future.shape
    if h%4:raise ValueError('Forecast horizon must be divisible by4')
    future=future.reshape(n,h//4,4,d).mean(2)
    decoder=encoder.decoder
    _,initial,_,_=decoder.dynamics(history,trend)
    drive=decoder.future_drive(future,history).reshape(n,h//4,decoder.modes,2)
    initial=initial*history_gain.repeat(batch)[:,None,None]
    drive=drive*drive_gain.repeat(batch)[:,None,None,None]
    sequence,diag=direct_future_transitions(decoder,history,trend,head,basis)
    states=complex_oscillator_scan(sequence,drive,initial)
    outputs=decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    if outputs.shape!=(n,h//4,4):raise RuntimeError('Expected four values per state')
    return outputs.reshape(batch,channels,h).transpose(1,2),diag

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        if self.seq_len!=96 or self.pred_len!=720:raise ValueError('Registered experiment fixes input96/H720')
        priors=[(e.decoder.log_decay.detach().clone(),e.decoder.frequency_logits.detach().clone()) for e in self.seasonals]
        self.history_gain=nn.Parameter((1+self.history_gain_offset).detach().clone())
        self.drive_gain=nn.Parameter((1+self.drive_gain_offset).detach().clone())
        del self.history_gain_offset,self.drive_gain_offset
        for e,prior in zip(self.seasonals,priors):
            e.decoder.__class__=DirectHistoryDecoder
            del e.decoder.log_decay,e.decoder.frequency_logits,e.decoder.condition,e.decoder.trend_condition
            e.decoder.history_parameter_head=parameter_head(4*e.decoder.drive_projection.in_features,e.decoder.modes,prior)
        self.direct_future_parameter_heads=nn.ModuleList([
            parameter_head(4*e.decoder.drive_projection.in_features+4,e.decoder.modes,prior)
            for e,prior in zip(self.seasonals,priors)])
        steps=self.pred_len//4
        basis=torch.cos(math.pi*(torch.arange(steps,dtype=torch.float32)[:,None]+.5)*torch.arange(1,5,dtype=torch.float32)[None,:]/steps)
        self.register_buffer('direct_parameter_position_basis',basis)
        self.gate_context_policy='season'

    def branch_gains(self):return self.history_gain,self.drive_gain

    def optimizer_parameter_groups(self,weight_decay):
        return [dict(params=list(self.parameters()),weight_decay=weight_decay,group_name='direct_parameter_generation')]

    def predict_components(self,x_enc):
        if x_enc.ndim!=3 or x_enc.shape[1:]!=(self.seq_len,self.enc_in):raise ValueError('Expected B,96,C history')
        scales,(seasons,trends)=self.encode_multiscale(self.downsample(x_enc))
        alpha,beta=self.branch_gains();batch,_,channels=x_enc.shape
        recurrent=[];diagnostics=[]
        for e,x,s,t,head in zip(self.seasonals,scales,seasons,trends,self.direct_future_parameter_heads):
            result,diag=all_direct_forecast(e,x,alpha,beta,s,t,head,self.direct_parameter_position_basis)
            recurrent.append(result);diagnostics.append(diag)
        feature=[head(t).reshape(batch,channels,self.pred_len).transpose(1,2) for head,t in zip(self.feature_trend_heads,trends)]
        self.last_gate_context_diagnostics=diagnostics
        return torch.stack(recurrent+feature,-1)
