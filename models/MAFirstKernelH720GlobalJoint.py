"""C backbone plus one shared residual dense720 seasonal output map.
All720 outputs are jointly decoded from local future states. Trends untouched.
"""
import torch
from torch import nn
from torch.nn import functional as F
from models.MAFirstKernelH720DirectGram import Model as FrozenC
class Model(FrozenC):
    def __init__(self, configs):
        super().__init__(configs)
        self.global_seasonal_delta=nn.Parameter(torch.zeros(720,720))
        self.global_joint_output=True
    def mix_seasonal(self, seasonal):
        # [B,H,C,S=4] -> [B,C,S,H]; same time map for each variable and scale.
        if seasonal.ndim!=4 or seasonal.shape[1:]!=(720,7,4):
            raise ValueError('Expected seasonal components [B,720,7,4]')
        z=seasonal.permute(0,2,3,1)
        return (z+F.linear(z,self.global_seasonal_delta)).permute(0,3,1,2)
    def predict_components(self, x_enc):
        components=super().predict_components(x_enc)
        return torch.cat((self.mix_seasonal(components[...,:4]),components[...,4:]),dim=-1)
