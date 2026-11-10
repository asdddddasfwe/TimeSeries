"""Lag-zero RKHS covariance matching of disjoint raw-value patches.
Inputs [B,H,C], patches [B,C,N,L], H=N*L. No patch centering or rescaling.
C = Phi(patches).T J Phi(patches) / N; compare operators in one Gaussian RKHS.
"""
import math
import torch


def patch_covariance_parts(p, y, bandwidth=1.):
    """Inputs [B,C,N,L]; returns differentiable distance and scalar diagnostics."""
    if p.ndim != 4 or p.shape != y.shape or not all(p.shape) or p.shape[-2] < 2:
        raise ValueError('Matching nonempty B,C,N,L patches, N>=2 required')
    if p.device != y.device or not p.is_floating_point() or not y.is_floating_point():
        raise ValueError('Floating tensors on same device required')
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Finite positive bandwidth required')
    dtype=p.dtype;p=p.double();y=y.detach().double()
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite patch covariance input')
    def gram(a,b):
        # Standard vector RBF: SUM of coordinate squares, not their mean.
        k=torch.exp(-.5*((a.unsqueeze(-2)-b.unsqueeze(-3))/bandwidth).square().sum(-1))
        return k-k.mean(-1,keepdim=True)-k.mean(-2,keepdim=True)+k.mean((-1,-2),keepdim=True)
    pp=gram(p,p).square().mean((-1,-2))
    with torch.no_grad(): yy=gram(y,y).square().mean((-1,-2))
    py=gram(p,y).square().mean((-1,-2))
    raw=pp+yy-2*py
    if not torch.isfinite(raw).all() or raw.min() < -1e-10:
        raise RuntimeError('Invalid patch covariance distance')
    d=raw.clamp_min(0).mean().to(dtype)
    diag=dict(patch_covariance=float(d.detach()),prediction_covariance_norm2=float(pp.detach().mean()),
              target_covariance_norm2=float(yy.mean()),cross_covariance_inner_product=float(py.detach().mean()),
              minimum_unclipped_distance=float(raw.detach().min()),patch_count=p.shape[-2])
    return d,diag


def parts(prediction,target,bandwidth=1.,length=4):
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError('Matching B,H,C tensors required')
    if isinstance(length,bool) or not isinstance(length,int) or length < 1 or prediction.shape[1] % length or prediction.shape[1]//length < 2:
        raise ValueError('H must be divisible by patch length and contain >=2 disjoint patches')
    # unfold with stride=length: no overlap and no discarded horizon samples.
    return patch_covariance_parts(prediction.transpose(1,2).unfold(-1,length,length),
                                  target.transpose(1,2).unfold(-1,length,length),bandwidth)
