"""FP64 per-sequence/variable biased MMD², time×amplitude RBF; no learned scales."""
import math
import numpy as np
import torch

def kernel(a,b,sigma_value=1.,sigma_time=2.,amplitude=True):
    if a.shape!=b.shape or a.ndim<2 or a.shape[-1]!=4:raise ValueError('Matching [...,N,4] required')
    if any(not math.isfinite(v) or v<=0 for v in (sigma_value,sigma_time)):raise ValueError('Positive finite bandwidth required')
    a,b=a.double(),b.double();n=a.shape[-2];i=torch.arange(n,device=a.device,dtype=torch.float64)
    time=torch.exp(-.5*((i[:,None]-i[None,:])/sigma_time).square())
    base=torch.exp(-.5*((a.unsqueeze(-2)-b.unsqueeze(-3))/sigma_value).square().sum(-1))
    if amplitude:
        ha,hb=torch.logaddexp(a,torch.zeros_like(a)),torch.logaddexp(b,torch.zeros_like(b))
        base=base*(ha@hb.transpose(-1,-2))/4
    return time*base

def parts(prediction,target,sigma_value=1.,sigma_time=2.,amplitude=True):
    if prediction.ndim!=3 or prediction.shape!=target.shape or prediction.shape[1]%4 or prediction.shape[1]<8 or not all(prediction.shape):raise ValueError('Matching nonempty B,H,C divisible by4 H>=8 required')
    if not prediction.is_floating_point() or not target.is_floating_point() or prediction.device!=target.device:raise ValueError('Matching floating tensors required')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():raise ValueError('Nonfinite input')
    p=prediction.double().transpose(1,2).unfold(-1,4,4);y=target.detach().double().transpose(1,2).unfold(-1,4,4)
    pp=kernel(p,p,sigma_value,sigma_time,amplitude).mean((-1,-2))
    with torch.no_grad():yy=kernel(y,y,sigma_value,sigma_time,amplitude).mean((-1,-2))
    py=kernel(p,y,sigma_value,sigma_time,amplitude).mean((-1,-2));raw=pp+yy-2*py
    if not torch.isfinite(raw).all() or raw.min() < -1e-10:raise RuntimeError('Invalid amplitude MMD squared')
    value=raw.clamp_min(0).mean().to(prediction.dtype)
    return value,dict(time_position_mmd=float(value.detach()),mmd_prediction_self=float(pp.detach().mean()),mmd_target_self=float(yy.mean()),mmd_cross=float(py.detach().mean()),minimum_unclipped_mmd=float(raw.detach().min()),patch_count=p.shape[-2],sigma_value=sigma_value,sigma_time_patches=sigma_time,amplitude=amplitude)

def numpy_parts(prediction,target,sigma_value=1.,sigma_time=2.,amplitude=True,chunk=8):
    """Independent saved-array NumPy audit including PP,YY,PY and diagonals."""
    total=np.zeros(4);count=len(prediction);n=prediction.shape[1]//4
    time=np.exp(-.5*((np.arange(n)[:,None]-np.arange(n)[None,:])/sigma_time)**2)
    def term(a,b):
        k=np.exp(-.5*(((a[..., :,None,:]-b[...,None,:,:])/sigma_value)**2).sum(-1))
        if amplitude:k*=np.einsum('...ir,...jr->...ij',np.logaddexp(a,0),np.logaddexp(b,0))/4
        return (k*time).mean((-1,-2))
    for s in range(0,count,chunk):
        p=np.asarray(prediction[s:s+chunk],dtype=np.float64).transpose(0,2,1).reshape(-1,prediction.shape[2],n,4)
        y=np.asarray(target[s:s+chunk],dtype=np.float64).transpose(0,2,1).reshape(p.shape)
        pp,yy,py=term(p,p),term(y,y),term(p,y);raw=pp+yy-2*py
        assert np.isfinite(raw).all() and raw.min()>=-1e-10
        total+=np.array([np.maximum(raw,0).mean(),pp.mean(),yy.mean(),py.mean()])*len(p)
    return dict(zip(('time_position_mmd','mmd_prediction_self','mmd_target_self','mmd_cross'),map(float,total/count)))
