"""Original recurrence; direct historical sliding-patch Transformer trend head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from models.OscillatorPDMStatePatchMAFirst import Model as Original

class PatchTransformerTrendHead(nn.Module):
    """[N,T,D] -> overlapping historical tokens -> Transformer -> joint [N,H]."""
    def __init__(self,original,scale):
        super().__init__()
        self.temporal=original.temporal
        self.readout=original.readout
        self.length=self.temporal.in_features
        self.horizon=self.temporal.out_features
        self.width=self.readout.in_features
        if self.length<4 or self.width%4:raise ValueError('Need history>=4, width divisible by4')
        self.patch_count=(self.length-4)//2+1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20211020+scale)
            self.patch_projection=nn.Linear(4*self.width,self.width)
            layer=nn.TransformerEncoderLayer(self.width,4,dim_feedforward=64,
                dropout=0.,activation='gelu',batch_first=True,norm_first=True)
            self.transformer=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
            self.nonlinear_readout=nn.Sequential(
                nn.Linear(self.patch_count*self.width,64),nn.GELU(),
                nn.Linear(64,self.horizon))
        position=torch.arange(self.patch_count,dtype=torch.float32)[:,None]
        frequency=torch.exp(torch.arange(0,self.width,2,dtype=torch.float32)*(-math.log(10000.)/self.width))
        encoding=torch.zeros(self.patch_count,self.width)
        encoding[:,0::2]=torch.sin(position*frequency)
        encoding[:,1::2]=torch.cos(position*frequency)
        self.register_buffer('patch_positions',encoding[None])
        nn.init.zeros_(self.nonlinear_readout[-1].weight)
        nn.init.zeros_(self.nonlinear_readout[-1].bias)

    def linear_forecast(self,history):
        # Same affine function without forming [N,H,D] future feature positions.
        scalar=F.linear(history,self.readout.weight,None).squeeze(-1)
        bias=self.temporal.bias*self.readout.weight.sum()
        return F.linear(scalar,self.temporal.weight,bias)+self.readout.bias

    def encode_history(self,history):
        if history.ndim!=3 or history.shape[1:]!=(self.length,self.width):
            raise ValueError('Expected [N,history_length,feature_width]')
        # unfold gives [N,patch,D,4]; keep original time-major within each patch.
        patches=history.unfold(1,4,2).permute(0,1,3,2).reshape(history.shape[0],self.patch_count,4*self.width)
        tokens=self.patch_projection(patches)+self.patch_positions.to(history)
        # Fixed math SDP for the declared new encoder; no Flash-only behavior.
        with sdpa_kernel(SDPBackend.MATH):
            return self.transformer(tokens)

    def forward(self,history):
        encoded=self.encode_history(history)
        return self.linear_forecast(history)+self.nonlinear_readout(encoded.flatten(1))

class Model(Original):
    def __init__(self,configs):
        super().__init__(configs)
        self.feature_trend_heads=nn.ModuleList([
            PatchTransformerTrendHead(head,scale)
            for scale,head in enumerate(self.feature_trend_heads)])
        # Original recurrence, drive, readout and recurrent means are inherited.
