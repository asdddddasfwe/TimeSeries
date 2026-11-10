"""Mean-pool state-to-four decoder with history-only two-axis Q/K RoPE.
No history additive PE. Retain FFT routing, unmasked zero padding, V,
residual/gates/norm, future additive PE, mean drivers and original A24.
"""
import math
import torch
from torch.nn import functional as F
from layers.LagARFormer import LagAxialBlock
from models.OscillatorPDMStatePatch import Model as MeanModel
from models.OscillatorPDMSeasonalFFT import select_seasonal_feature_periods


def rotate_positions(x, positions):
    """Rotate adjacent Q/K channel pairs; x [...,L,D], positions [L]."""
    width=x.shape[-1]
    if width%2 or positions.ndim!=1 or positions.numel()!=x.shape[-2]:
        raise ValueError('RoPE requires even Q/K width and one position per token')
    dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
    rates=torch.exp(torch.arange(0,width,2,device=x.device,dtype=dtype)*(-math.log(10000.)/width))
    angles=positions.to(device=x.device,dtype=dtype)[:,None]*rates
    cosine,sine=angles.cos().to(x),angles.sin().to(x)
    pairs=x.reshape(*x.shape[:-1],width//2,2)
    a,b=pairs.unbind(-1)
    return torch.stack((a*cosine-b*sine,a*sine+b*cosine),-1).flatten(-2)


class RotaryAxialBlock(LagAxialBlock):
    def forward(self,image,projections=None):
        weight,bias,sizes=self.projection_parameters() if projections is None else projections
        tokens=image.permute(0,2,3,1)
        rq,rk,rv,cq,ck,cv=F.linear(tokens,weight,bias).split(sizes,dim=-1)
        phase=torch.arange(tokens.shape[2],device=image.device)
        cycles=torch.arange(tokens.shape[1],device=image.device)
        rq,rk=rotate_positions(rq,phase),rotate_positions(rk,phase)
        cq,ck=rotate_positions(cq.transpose(1,2),cycles),rotate_positions(ck.transpose(1,2),cycles)
        def attend(q,k,v):
            width=v.shape[-1]
            if width<q.shape[-1]:v=F.pad(v,(0,q.shape[-1]-width))
            return F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=1.)[...,:width]
        row=attend(rq,rk,rv)
        column=attend(cq,ck,cv.transpose(1,2)).transpose(1,2)
        update=self.row.gamma*row+self.column.gamma*column
        mixed=image+self.dropout(update.permute(0,3,1,2))
        return self.norm(mixed.permute(0,2,3,1)).permute(0,3,1,2).contiguous()


def encode_rotary_segments(encoder,tokens,periods,weights):
    if tokens.dtype in (torch.float16,torch.bfloat16):tokens=tokens.float()
    groups={}
    for series,row in enumerate(periods.detach().cpu().tolist()):
        for period in set(row):groups.setdefault(period,[]).append(series)
    block=encoder.blocks[0];projections=block.projection_parameters()
    fused=torch.zeros_like(tokens)
    for period in sorted(groups):
        selected=torch.tensor(groups[period],dtype=torch.long,device=tokens.device)
        group=tokens[selected];count,length,width=group.shape
        image=F.pad(group,(0,0,0,(-length)%period))
        image=image.reshape(count,-1,period,width).permute(0,3,1,2).contiguous()
        # No additive cycle/phase PE. Padding remains unmasked as in control.
        image=block(image,projections)
        encoded=image.permute(0,2,3,1).reshape(count,-1,width)[:,:length]
        merged=(weights[selected]*(periods[selected]==period)).sum(-1).to(tokens)
        fused=fused.index_add(0,selected,encoded*merged[:,None,None])
    return fused


class Model(MeanModel):
    def __init__(self,configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            for i,old in enumerate(encoder.blocks):
                width=old.norm.normalized_shape[0]
                qdim=old.row.query_conv.out_channels
                if qdim%2:raise ValueError('Even Q/K width required')
                with torch.random.fork_rng(devices=[]):
                    block=RotaryAxialBlock(width,qdim,old.dropout.p)
                block.load_state_dict(old.state_dict(),strict=True)
                encoder.blocks[i]=block

    def encode_seasonal_features(self,seasons):
        if len(seasons)!=len(self.seasonals):raise ValueError('Expected one history per scale')
        return [encode_rotary_segments(e,s,*select_seasonal_feature_periods(s,e.period_top_k))
                for e,s in zip(self.seasonals,seasons)]
