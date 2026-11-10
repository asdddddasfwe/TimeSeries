"""Per-sequence centered offdiagonal Gram alignment, not double-centered CKA.

No trained parameters, per-row probabilities, diagonal terms, or future inputs.
The target is detached. Computation uses FP64 and returns the prediction dtype.
"""
import numpy as np
import torch
STD_FLOOR=1e-4

def parts(prediction,target,std_floor=STD_FLOOR):
    if prediction.ndim<2 or prediction.shape!=target.shape or prediction.shape[-1]!=prediction.shape[-2] or prediction.shape[-1]<3:
        raise ValueError('Matching square matrices with N>=3 required')
    if not prediction.is_floating_point() or not target.is_floating_point() or prediction.device!=target.device:
        raise ValueError('Matching floating device inputs required')
    if not np.isfinite(std_floor) or std_floor<=0:raise ValueError('Positive finite standard deviation floor required')
    dtype=prediction.dtype;n=prediction.shape[-1];mask=~torch.eye(n,device=prediction.device,dtype=torch.bool)
    p=prediction.double()[...,mask];y=target.detach().double()[...,mask]
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite offdiagonal values')
    pc=p-p.mean(-1,keepdim=True);yc=y-y.mean(-1,keepdim=True)
    vp=pc.square().mean(-1);vy=yc.square().mean(-1);usable=vy>std_floor**2
    correlation=(pc*yc).mean(-1)/(vp.clamp_min(std_floor**2)*vy.clamp_min(std_floor**2)).sqrt()
    raw=torch.where(usable,1-correlation,torch.zeros_like(correlation))
    if not torch.isfinite(raw).all() or raw.min() < -1e-10 or raw.max()>2+1e-10:raise RuntimeError('Invalid centered alignment')
    result=raw.clamp_min(0).mean().to(dtype)
    return result,dict(alignment_loss=float(result.detach()),mean_correlation=float(torch.where(usable,correlation,torch.zeros_like(correlation)).detach().mean()),target_degenerate_fraction=float((~usable).double().mean()),prediction_degenerate_fraction=float((vp<=std_floor**2).double().mean()),prediction_offdiagonal_std=float(vp.detach().sqrt().mean()),target_offdiagonal_std=float(vy.sqrt().mean()),std_floor=std_floor,diagonal_excluded=True,per_matrix_centering=True,target_detached=True)

def numpy_parts(prediction,target,std_floor=STD_FLOOR):
    """Independent NumPy formula for saved matrix diagnostics only."""
    n=prediction.shape[-1];mask=~np.eye(n,dtype=bool)
    p=np.asarray(prediction,dtype=float)[...,mask];y=np.asarray(target,dtype=float)[...,mask]
    pc=p-p.mean(-1,keepdims=True);yc=y-y.mean(-1,keepdims=True)
    vp=(pc*pc).mean(-1);vy=(yc*yc).mean(-1)
    correlation=(pc*yc).mean(-1)/np.sqrt(np.maximum(vp,std_floor**2)*np.maximum(vy,std_floor**2));usable=vy>std_floor**2
    return dict(alignment_loss=float(np.maximum(np.where(usable,1-correlation,0),0).mean()),mean_correlation=float(np.where(usable,correlation,0).mean()),target_degenerate_fraction=float((~usable).mean()),prediction_degenerate_fraction=float((vp<=std_floor**2).mean()),prediction_offdiagonal_std=float(np.sqrt(vp).mean()),target_offdiagonal_std=float(np.sqrt(vy).mean()))
