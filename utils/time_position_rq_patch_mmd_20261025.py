"""Time-position biased MMD² with fixed isotropic Rational Quadratic value kernel.

The same PSD kernel is used for PP,YY,PY, including diagonals and /N².
FP64 internal computation, target detached, independent B,C sequences.
RQ remains a distance kernel: self-Gram ambiguity is not eliminated.
"""
import math
import numpy as np
import torch

def validate(sigma_value,sigma_time,alpha,family):
    if any(not math.isfinite(v) or v<=0 for v in (sigma_value,sigma_time,alpha)):raise ValueError('Positive finite parameters required')
    if family not in ('rq','rbf'):raise ValueError('rq or rbf required')

def kernel(a,b,sigma_value=1.,sigma_time=2.,alpha=1.,family='rq'):
    if a.shape!=b.shape or a.ndim<2 or a.shape[-1]!=4:raise ValueError('Matching [...,N,4] required')
    validate(sigma_value,sigma_time,alpha,family)
    a,b=a.double(),b.double();n=a.shape[-2];i=torch.arange(n,device=a.device,dtype=torch.float64)
    time=torch.exp(-.5*((i[:,None]-i[None,:])/sigma_time).square())
    d2=((a.unsqueeze(-2)-b.unsqueeze(-3))/sigma_value).square().sum(-1)
    value=(1+d2/(2*alpha)).pow(-alpha) if family=='rq' else torch.exp(-.5*d2)
    return time*value

def parts(prediction,target,sigma_value=1.,sigma_time=2.,alpha=1.,family='rq'):
    if prediction.ndim!=3 or prediction.shape!=target.shape or prediction.shape[1]%4 or prediction.shape[1]<8 or not all(prediction.shape):raise ValueError('Matching nonempty B,H,C divisible by4 H>=8 required')
    if not prediction.is_floating_point() or not target.is_floating_point() or prediction.device!=target.device:raise ValueError('Matching floating tensors required')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():raise ValueError('Nonfinite input')
    p=prediction.double().transpose(1,2).unfold(-1,4,4);y=target.detach().double().transpose(1,2).unfold(-1,4,4)
    pp=kernel(p,p,sigma_value,sigma_time,alpha,family).mean((-1,-2))
    with torch.no_grad():yy=kernel(y,y,sigma_value,sigma_time,alpha,family).mean((-1,-2))
    py=kernel(p,y,sigma_value,sigma_time,alpha,family).mean((-1,-2));raw=pp+yy-2*py
    if not torch.isfinite(raw).all() or raw.min() < -1e-10:raise RuntimeError('Invalid RQ MMD squared')
    value=raw.clamp_min(0).mean().to(prediction.dtype)
    return value,dict(time_position_mmd=float(value.detach()),mmd_prediction_self=float(pp.detach().mean()),mmd_target_self=float(yy.mean()),mmd_cross=float(py.detach().mean()),minimum_unclipped_mmd=float(raw.detach().min()),patch_count=p.shape[-2],sigma_value=sigma_value,sigma_time_patches=sigma_time,alpha=alpha,kernel_family=family,no_amplitude_weighting=True)

def numpy_parts(prediction,target,sigma_value=1.,sigma_time=2.,alpha=1.,family='rq',chunk=16):
    """Independent saved-array NumPy: squared norm/matmul distances, FP64."""
    validate(sigma_value,sigma_time,alpha,family)
    total=np.zeros(4);count=len(prediction);n=prediction.shape[1]//4
    time=np.exp(-.5*((np.arange(n)[:,None]-np.arange(n)[None,:])/sigma_time)**2)
    def term(a,b):
        aa=(a*a).sum(-1);bb=(b*b).sum(-1)
        d2=np.maximum(aa[..., :,None]+bb[...,None,:]-2*(a@b.swapaxes(-1,-2)),0)/(sigma_value*sigma_value)
        k=np.power(1+d2/(2*alpha),-alpha) if family=='rq' else np.exp(-.5*d2)
        return (k*time).mean((-1,-2))
    for s in range(0,count,chunk):
        p=np.asarray(prediction[s:s+chunk],dtype=np.float64).transpose(0,2,1).reshape(-1,prediction.shape[2],n,4)
        y=np.asarray(target[s:s+chunk],dtype=np.float64).transpose(0,2,1).reshape(p.shape)
        pp,yy,py=term(p,p),term(y,y),term(p,y);raw=pp+yy-2*py
        assert np.isfinite(raw).all() and raw.min()>=-1e-10
        total+=np.array([np.maximum(raw,0).mean(),pp.mean(),yy.mean(),py.mean()])*len(p)
    return dict(zip(('time_position_mmd','mmd_prediction_self','mmd_target_self','mmd_cross'),map(float,total/count)))
