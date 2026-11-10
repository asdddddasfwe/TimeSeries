"""History feature observations assimilated by an isotropic oscillator filter.

Only history state estimation changes. Existing drive features are learned
pseudo-observations of all real/imag state coordinates (H=I), not raw physical
observations. Q/R ratios are trained by forecast MSE, not noise likelihood.
This does not assert calibrated output uncertainty or causal online filtering:
encoder features and dynamics already use the entire observed history window.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorPDMTrendCondition import Model as BaseModel, TrendConditionedDecoder
from layers.OscillatorTimeMix import complex_oscillator_scan


def filter_states(observations, transition, process_variance, stride=1):
    """[N,T,J,2], [N,J,2], [J] -> filtered states and scalar covariance/gain.

    A=r*rotation, H=I, R=I and P0=I keep each 2x2 covariance isotropic.
    Q is per original time step. History Q sums stride independent increments.
    The scalar Riccati recursion is followed by the exact parallel affine scan.
    No observation is detached; gradients flow through Q, A and observations.
    """
    if observations.ndim!=4 or observations.shape[-1]!=2 or observations.shape[1]<1:
        raise ValueError('Expected observations [N,T,J,2]')
    if transition.shape!=(observations.shape[0],observations.shape[2],2):
        raise ValueError('Transition shape mismatch')
    if process_variance.shape!=(observations.shape[2],) or not isinstance(stride,int) or stride<1:
        raise ValueError('Process variance/stride mismatch')
    dtype=torch.float64 if observations.dtype==torch.float64 else torch.float32
    obs=observations.to(dtype);a=transition.to(dtype);q=process_variance.to(dtype)
    r2=a.square().sum(-1)
    # a is the coarse-history transition; convert each original noise increment.
    q_history=q*sum(r2.clamp_min(1e-12).pow(i/stride) for i in range(stride))
    covariance=torch.ones_like(r2)
    gains=[];covariances=[]
    for _ in range(obs.shape[1]):
        prior=r2*covariance+q_history
        gain=prior/(prior+1.)
        # Joseph form; positive terms avoid subtraction-induced negative P.
        covariance=(1.-gain).square()*prior+gain.square()
        gains.append(gain);covariances.append(covariance)
    gains=torch.stack(gains,1);covariances=torch.stack(covariances,1)
    effective=(1.-gains[...,None])*a[:,None]
    states=complex_oscillator_scan(effective,gains[...,None]*obs,torch.zeros_like(a))
    return states,gains,covariances


class InnovationDecoder(TrendConditionedDecoder):
    def __init__(self, decoder):
        nn.Module.__init__(self)
        for name in ('length','horizon','modes','stride'):
            setattr(self,name,getattr(decoder,name))
        for name,module in decoder.named_children():self.add_module(name,module)
        for name,value in decoder.named_parameters(recurse=False):self.register_parameter(name,value)
        for name,value in decoder.named_buffers(recurse=False):self.register_buffer(name,value)
        # R=1 fixes the otherwise unidentifiable common noise scale under MSE.
        self.log_process_variance=nn.Parameter(torch.full((self.modes,),math.log(math.expm1(.01-1e-6))))
        self.filter_observer=None

    def encode_history_states(self, history, transition):
        observation=self.drive_projection(self.drive_dropout(F.gelu(self.drive_norm(history))))
        observation=observation.reshape(history.shape[0],self.length,self.modes,2)
        q=F.softplus(self.log_process_variance)+1e-6
        states,gains,covariances=filter_states(observation,transition,q,self.stride)
        if self.filter_observer is not None:
            self.filter_observer(observation,transition,states,gains,covariances,q)
        return states


class Model(BaseModel):
    def __init__(self,configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            encoder.decoder=InnovationDecoder(encoder.decoder)

    def optimizer_parameter_groups(self,weight_decay):
        protected={id(e.decoder.log_process_variance) for e in self.seasonals}
        groups=[]
        for group in super().optimizer_parameter_groups(weight_decay):
            regular=[p for p in group['params'] if id(p) not in protected]
            noise=[p for p in group['params'] if id(p) in protected]
            if regular:groups.append({**group,'params':regular})
            if noise:groups.append({**group,'params':noise,'weight_decay':0.,'group_name':'filter_noise_ratio'})
        return groups
