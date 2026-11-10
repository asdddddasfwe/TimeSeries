"""Fixed-grid angular correction; no dependence on forecast horizon in features."""
import math
import torch
PERIODS=(96.,384.)
def learned_periods(offsets):
    return offsets.new_tensor(PERIODS)*torch.exp(offsets)

def features(t,periods):
    return torch.stack([f(2*math.pi*t/period) for period in periods for f in (torch.sin,torch.cos)],-1)
def angular_delta(decoder,history,frequency,head,steps,phase_bounded=False,period_offsets=None):
    periods=learned_periods(period_offsets)
    context=decoder.context_norm(torch.cat((history.mean(1),history[:,-1]),-1))
    coefficients=head(context).reshape(history.shape[0],decoder.modes,4).to(frequency)
    margin=.25*torch.minimum(frequency,math.pi/decoder.stride-frequency)
    if phase_bounded:
        t=4*torch.arange(steps+1,device=frequency.device,dtype=frequency.dtype)
        basis=features(t,periods);raw=torch.einsum('nmk,tk->ntm',coefficients,basis)
        potential=.5*margin[:,None,:]*torch.tanh(raw-raw[:,:1])
        delta=potential[:,1:]-potential[:,:-1]
    else:
        t=4*(torch.arange(steps,device=frequency.device,dtype=frequency.dtype)+.5)
        raw=torch.einsum('nmk,tk->ntm',coefficients,features(t,periods))
        delta=margin[:,None,:]*torch.tanh(raw)
    return delta,margin

def position_transitions(decoder,history,transition,frequency,head,basis,period_offsets,phase_bounded=False):
    delta,margin=angular_delta(decoder,history,frequency,head,len(basis),phase_bounded,period_offsets)
    co,si=delta.cos(),delta.sin();real,imag=transition.unbind(-1)
    sequence=torch.stack((real[:,None]*co-imag[:,None]*si,real[:,None]*si+imag[:,None]*co),-1)
    diag=dict(frequency_delta_max_abs=delta.detach().abs().max(),frequency_delta_bound_max=margin.detach().max(),cumulative_phase_max_abs=delta.detach().cumsum(1).abs().max(),fixed_periods=PERIODS,phase_bounded=phase_bounded)
    return sequence,diag
