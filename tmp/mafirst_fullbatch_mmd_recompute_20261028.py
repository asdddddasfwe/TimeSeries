"""Biased empirical MMD² on (patch index, raw four-point value) pairs.

Product Gaussian kernel, no Gram centering/square or patch normalization.\nFull batch retained; PP and PY each recompute their exact unchanged Gram graph during backward.
Each B,C sequence separately owns N time-stamped patches; no batch/cross-channel
mixing. sigma_time is in PATCH INDEX units, not normalized [0,1] time. Both
self terms and the cross term use the identical fixed time Gram and /N².
Compute FP64 then return input dtype, target detached. No trainable parameters/RNG.
"""
import math
import torch
import numpy as np


def patch_parts(prediction,target,sigma_value=2.,sigma_time=24.):
    if prediction.ndim!=4 or prediction.shape!=target.shape or not all(prediction.shape) or prediction.shape[-2]<2:
        raise ValueError('Expected matching nonempty B,C,N,L with N>=2')
    if prediction.device!=target.device or not prediction.is_floating_point() or not target.is_floating_point():raise ValueError('Matching device floating inputs required')
    if any(not math.isfinite(v) or v<=0 for v in (sigma_value,sigma_time)):raise ValueError('Positive finite bandwidths required')
    dtype=prediction.dtype;p=prediction.double();y=target.detach().double()
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite patch input')
    n=p.shape[-2];index=torch.arange(n,device=p.device,dtype=torch.float64)
    time=torch.exp(-.5*((index[:,None]-index[None,:])/sigma_time).square())
    def raw_gram(a,b):return torch.exp(-.5*((a.unsqueeze(-2)-b.unsqueeze(-3))/sigma_value).square().sum(-1))
    def gram(a,b):
        if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(raw_gram,a,b,use_reentrant=False,preserve_rng_state=True)
        return raw_gram(a,b)
    pp=(time*gram(p,p)).mean((-1,-2))
    with torch.no_grad():yy=(time*gram(y,y)).mean((-1,-2))
    py=(time*gram(p,y)).mean((-1,-2));raw=pp+yy-2*py
    if not torch.isfinite(raw).all() or raw.min() < -1e-10:raise RuntimeError('Invalid time-position MMD²')
    value=raw.clamp_min(0).mean().to(dtype)
    return value,dict(time_position_mmd=float(value.detach()),mmd_prediction_self=float(pp.detach().mean()),mmd_target_self=float(yy.mean()),mmd_cross=float(py.detach().mean()),minimum_unclipped_mmd=float(raw.detach().min()),patch_count=n,sigma_value=sigma_value,sigma_time_patches=sigma_time)


def parts(prediction,target,sigma_value=2.,sigma_time=24.,length=4):
    if prediction.ndim!=3 or prediction.shape!=target.shape:raise ValueError('Expected matching B,H,C')
    if isinstance(length,bool) or not isinstance(length,int) or length<1 or prediction.shape[1]%length or prediction.shape[1]//length<2:raise ValueError('H must be divisible by positive patch length and N>=2')
    return patch_parts(prediction.transpose(1,2).unfold(-1,length,length),target.transpose(1,2).unfold(-1,length,length),sigma_value,sigma_time)


def numpy_parts(prediction,target,sigma_value=2.,sigma_time=24.,length=4):
    """Independent saved-array audit [B,H,C], no torch/model/forward."""
    count=len(prediction);totals=np.zeros(4,dtype=np.float64)
    for start in range(0,count,64):
        p=np.asarray(prediction[start:start+64],dtype=np.float64).transpose(0,2,1).reshape(-1,prediction.shape[2],prediction.shape[1]//length,length)
        y=np.asarray(target[start:start+64],dtype=np.float64).transpose(0,2,1).reshape(p.shape)
        n=p.shape[-2];time=np.exp(-((np.arange(n)[:,None]-np.arange(n)[None,:])**2)/(2*sigma_time**2))
        def term(a,b):return (time*np.exp(-((a[..., :,None,:]-b[...,None,:,:])**2).sum(-1)/(2*sigma_value**2))).mean((-1,-2))
        pp,yy,py=term(p,p),term(y,y),term(p,y);raw=pp+yy-2*py
        assert np.isfinite(raw).all() and raw.min()>=-1e-10
        totals+=np.array([np.maximum(raw,0).mean(),pp.mean(),yy.mean(),py.mean()])*len(p)
    return dict(zip(('time_position_mmd','mmd_prediction_self','mmd_target_self','mmd_cross'),map(float,totals/count)))
