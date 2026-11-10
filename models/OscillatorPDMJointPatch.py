"""Disjoint four-step joint seasonal readout, preserving the 96-step oscillator.
Each scale retains its pointwise GELU MLP and adds a shared zero-initialized
linear residual from four consecutive 2J states to four outputs. No recurrence
or normalization change; no previous predicted patch is fed back.
"""
import torch
from torch import nn
from models.OscillatorPDMTrendCondition import Model as PointwiseModel


class JointPatchReadout(nn.Module):
    def __init__(self, pointwise, patch_length=4):
        super().__init__()
        if patch_length != 4:
            raise ValueError('This experiment fixes disjoint length4 patches')
        self.pointwise=pointwise
        self.patch_length=patch_length
        self.in_features=pointwise.in_features
        self.out_features=1
        with torch.random.fork_rng(devices=[]):
            self.joint_projection=nn.Linear(patch_length*self.in_features,patch_length)
        nn.init.zeros_(self.joint_projection.weight)
        nn.init.zeros_(self.joint_projection.bias)
        self.register_buffer('joint_patch_readout_version',torch.tensor(1,dtype=torch.long))

    @property
    def weight(self):return self.pointwise.weight

    def forward(self, states):
        # states [B*C,H,2J]; preserve [B*C,H,1] for existing branch captures.
        if states.ndim!=3 or states.shape[-1]!=self.in_features or states.shape[1]%self.patch_length:
            raise ValueError('Expected B*C,H,2J states with H divisible by4')
        b,h,d=states.shape
        patches=states.reshape(b,h//self.patch_length,self.patch_length*d)
        correction=self.joint_projection(patches).reshape(b,h,1)
        return self.pointwise(states)+correction


class Model(PointwiseModel):
    def __init__(self,configs):
        super().__init__(configs)
        if self.pred_len%4:raise ValueError('Forecast horizon must be divisible by4')
        for encoder in self.seasonals:
            encoder.decoder.readout=JointPatchReadout(encoder.decoder.readout)

    def load_pointwise_initial_state(self,state):
        """Function-preserving scratch conversion, not pretrained fine-tuning."""
        own=self.state_dict();mapped={}
        for key,value in state.items():
            newkey=key.replace('.decoder.readout.','.decoder.readout.pointwise.')
            if newkey not in own or own[newkey].shape!=value.shape:raise ValueError('Incompatible pointwise state: '+key)
            mapped[newkey]=value
        extra=set(own)-set(mapped)
        expected={f'seasonals.{i}.decoder.readout.{s}' for i in range(4) for s in ('joint_projection.weight','joint_projection.bias','joint_patch_readout_version')}
        if extra!=expected:raise ValueError('Unexpected joint head state keys')
        for k in extra:
            own[k]=torch.ones_like(own[k]) if k.endswith('_version') else torch.zeros_like(own[k])
        own.update(mapped);self.load_state_dict(own,strict=True)
