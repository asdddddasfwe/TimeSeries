"""Ordinary biased value MMD squared on raw four-point patches, W identically one.
No patch normalization, centering, learned bandwidth, time feature or mass correction.
"""
import math
import torch
import numpy as np

def patch_parts(prediction, target, sigma_value=1.):
    if prediction.ndim!=4 or prediction.shape!=target.shape or not all(prediction.shape) or prediction.shape[-2]<2:
        raise ValueError('Expected B,C,N,L with N>=2')
    if prediction.device!=target.device or not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError('Matching floating inputs on same device required')
    if not math.isfinite(sigma_value) or sigma_value<=0:raise ValueError('Positive finite sigma required')
    dtype=prediction.dtype;p=prediction.double();y=target.detach().double()
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite inputs')
    def gram(a,b):return torch.exp(-.5*((a.unsqueeze(-2)-b.unsqueeze(-3))/sigma_value).square().sum(-1))
    pp=gram(p,p).mean((-1,-2))
    with torch.no_grad():yy=gram(y,y).mean((-1,-2))
    py=gram(p,y).mean((-1,-2));raw=pp+yy-2*py
    if not torch.isfinite(raw).all() or raw.min() < -1e-10:raise RuntimeError('Invalid value MMD squared')
    value=raw.clamp_min(0).mean().to(dtype)
    return value,dict(value_mmd=float(value.detach()),mmd_prediction_self=float(pp.detach().mean()),mmd_target_self=float(yy.mean()),mmd_cross=float(py.detach().mean()),minimum_unclipped_mmd=float(raw.detach().min()),patch_count=p.shape[-2],sigma_value=sigma_value,time_weight='identically one',normalization='N squared')

def parts(prediction,target,sigma_value=1.,length=4):
    if prediction.ndim!=3 or prediction.shape!=target.shape:raise ValueError('Expected B,H,C')
    if isinstance(length,bool) or not isinstance(length,int) or length<1 or prediction.shape[1]%length or prediction.shape[1]//length<2:raise ValueError('Invalid patch length')
    return patch_parts(prediction.transpose(1,2).unfold(-1,length,length),target.transpose(1,2).unfold(-1,length,length),sigma_value)

def numpy_parts(prediction,target,sigma_value=1.,length=4):
    totals=np.zeros(4);count=len(prediction)
    for start in range(0,count,32):
        p=np.asarray(prediction[start:start+32],dtype=float).transpose(0,2,1).reshape(-1,prediction.shape[2],prediction.shape[1]//length,length)
        y=np.asarray(target[start:start+32],dtype=float).transpose(0,2,1).reshape(p.shape)
        def term(a,b):return np.exp(-((a[..., :,None,:]-b[...,None,:,:])**2).sum(-1)/(2*sigma_value**2)).mean((-1,-2))
        pp,yy,py=term(p,p),term(y,y),term(p,y);raw=pp+yy-2*py
        assert np.isfinite(raw).all() and raw.min()>=-1e-10
        totals+=np.array([np.maximum(raw,0).mean(),pp.mean(),yy.mean(),py.mean()])*len(p)
    return dict(zip(('value_mmd','mmd_prediction_self','mmd_target_self','mmd_cross'),map(float,totals/count)))
