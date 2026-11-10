"""Masked local contrast-structure loss for an independently predicted Gram.

Uniform12x12 windows, stride6 including final boundary, C=.03**2.
Exclude the ORIGINAL matrix diagonal from ALL local statistics. Population
moments; FP64 arithmetic; detached targets; no target-flat skipping. This is
SSIM's contrast-structure component adapted to masked matrices, not full SSIM.
Global Pearson/std fields are diagnostics only, never the optimized local loss.
"""
import numpy as np
import torch
from torch.nn import functional as F
WINDOW_SIZE=12
STRIDE=6
STABILIZER_C=.03**2
STD_FLOOR=1e-4

def _pool(x,w,s):
    base=F.avg_pool2d(x,w,s)
    if (x.shape[-1]-w)%s:
        row=F.avg_pool2d(x[...,-w:,:],w,s)
        col=F.avg_pool2d(x[...,:,-w:],w,s)
        corner=x[...,-w:,-w:].mean((-2,-1),keepdim=True)
        base=torch.cat((torch.cat((base,col),-1),torch.cat((row,corner),-1)),-2)
    return base

def parts(prediction,target):
    if prediction.ndim<2 or prediction.shape!=target.shape or prediction.shape[-1]!=prediction.shape[-2] or prediction.shape[-1]<3:
        raise ValueError('Matching square matrices with N>=3 required')
    if not prediction.is_floating_point() or not target.is_floating_point() or prediction.device!=target.device:
        raise ValueError('Matching floating device inputs required')
    dtype=prediction.dtype;n=prediction.shape[-1];w=min(WINDOW_SIZE,n);s=min(STRIDE,max(1,w//2))
    p=prediction.double().reshape(-1,n,n);y=target.detach().double().reshape(-1,n,n)
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite matrices')
    mask=(~torch.eye(n,device=p.device,dtype=torch.bool)).double().expand_as(p)
    packet=torch.stack((mask,p*mask,y*mask,p.square()*mask,y.square()*mask,p*y*mask),1)
    values=_pool(packet,w,s);mass=values[:,0];assert (mass>0).all()
    mp=values[:,1]/mass;my=values[:,2]/mass
    vp=(values[:,3]/mass-mp.square()).clamp_min(0);vy=(values[:,4]/mass-my.square()).clamp_min(0)
    cov=values[:,5]/mass-mp*my
    similarity=(2*cov+STABILIZER_C)/(vp+vy+STABILIZER_C)
    raw=1-similarity
    if not torch.isfinite(raw).all() or raw.min() < -1e-9 or raw.max()>2+1e-9:raise RuntimeError('Invalid localCS loss')
    loss=raw.clamp_min(0).mean().to(dtype)
    with torch.no_grad():
        q=p[...,mask[0].bool()];t=y[...,mask[0].bool()];qc=q-q.mean(-1,keepdim=True);tc=t-t.mean(-1,keepdim=True)
        gv=qc.square().mean(-1);tv=tc.square().mean(-1)
        gc=(qc*tc).mean(-1)/(gv.clamp_min(STD_FLOOR**2)*tv.clamp_min(STD_FLOOR**2)).sqrt()
        gc=torch.where(tv>STD_FLOOR**2,gc,torch.zeros_like(gc))
        lc=cov/(vp.clamp_min(STD_FLOOR**2)*vy.clamp_min(STD_FLOOR**2)).sqrt()
        lc=torch.where(vy>STD_FLOOR**2,lc,torch.zeros_like(lc))
        meta=dict(alignment_loss=float(loss.detach()),local_cs_loss=float(loss.detach()),local_cs_similarity=float(similarity.mean()),local_mean_correlation=float(lc.mean()),local_prediction_std=float(vp.sqrt().mean()),local_target_std=float(vy.sqrt().mean()),local_target_degenerate_fraction=float((vy<=STD_FLOOR**2).double().mean()),local_prediction_degenerate_fraction=float((vp<=STD_FLOOR**2).double().mean()),mean_correlation=float(gc.mean()),target_degenerate_fraction=float((tv<=STD_FLOOR**2).double().mean()),prediction_degenerate_fraction=float((gv<=STD_FLOOR**2).double().mean()),prediction_offdiagonal_std=float(gv.sqrt().mean()),target_offdiagonal_std=float(tv.sqrt().mean()),window_size=w,stride=s,local_windows_per_matrix=int(mass.shape[-1]*mass.shape[-2]),stabilizer_C=STABILIZER_C,diagonal_excluded=True,target_flat_windows_included=True,target_detached=True,not_full_SSIM=True)
    return loss,meta

def numpy_parts(prediction,target):
    """Independent explicit-window implementation; no pooling/moment subtraction."""
    p=np.asarray(prediction,dtype=float);y=np.asarray(target,dtype=float)
    if p.shape!=y.shape or p.ndim<2 or p.shape[-1]!=p.shape[-2] or p.shape[-1]<3:raise ValueError('Matching square matrices required')
    n=p.shape[-1];p=p.reshape(-1,n,n);y=y.reshape(-1,n,n);w=min(WINDOW_SIZE,n);s=min(STRIDE,max(1,w//2));starts=list(range(0,n-w+1,s))
    if starts[-1]!=n-w:starts.append(n-w)
    mask=~np.eye(n,dtype=bool);losses=[];similarities=[];corr=[];ps=[];ys=[];td=[];pd=[]
    for i in starts:
        for j in starts:
            keep=mask[i:i+w,j:j+w];a=p[:,i:i+w,j:j+w][:,keep];b=y[:,i:i+w,j:j+w][:,keep]
            ac=a-a.mean(-1,keepdims=True);bc=b-b.mean(-1,keepdims=True);av=(ac*ac).mean(-1);bv=(bc*bc).mean(-1);co=(ac*bc).mean(-1)
            sim=(2*co+STABILIZER_C)/(av+bv+STABILIZER_C);similarities.append(sim);losses.append(np.maximum(1-sim,0))
            corr.append(np.where(bv>STD_FLOOR**2,co/np.sqrt(np.maximum(av,STD_FLOOR**2)*np.maximum(bv,STD_FLOOR**2)),0));ps.append(np.sqrt(av));ys.append(np.sqrt(bv));td.append(bv<=STD_FLOOR**2);pd.append(av<=STD_FLOOR**2)
    a=p[:,mask];b=y[:,mask];ac=a-a.mean(-1,keepdims=True);bc=b-b.mean(-1,keepdims=True);av=(ac*ac).mean(-1);bv=(bc*bc).mean(-1);gc=(ac*bc).mean(-1)/np.sqrt(np.maximum(av,STD_FLOOR**2)*np.maximum(bv,STD_FLOOR**2))
    return dict(alignment_loss=float(np.mean(losses)),local_cs_loss=float(np.mean(losses)),local_cs_similarity=float(np.mean(similarities)),local_mean_correlation=float(np.mean(corr)),local_prediction_std=float(np.mean(ps)),local_target_std=float(np.mean(ys)),local_target_degenerate_fraction=float(np.mean(td)),local_prediction_degenerate_fraction=float(np.mean(pd)),mean_correlation=float(np.mean(np.where(bv>STD_FLOOR**2,gc,0))),target_degenerate_fraction=float(np.mean(bv<=STD_FLOOR**2)),prediction_degenerate_fraction=float(np.mean(av<=STD_FLOOR**2)),prediction_offdiagonal_std=float(np.mean(np.sqrt(av))),target_offdiagonal_std=float(np.mean(np.sqrt(bv))),window_size=w,stride=s,local_windows_per_matrix=len(starts)**2,stabilizer_C=STABILIZER_C,diagonal_excluded=True,target_flat_windows_included=True,target_detached=True,not_full_SSIM=True)
