"""Positive multi-band time kernel; total time mass matches original sigma2."""
import math
import numpy as np
import torch
SIGMAS=(2.,24.,90.)
WEIGHTS=(1/3,1/3,1/3)
def time_kernel(n,device=None,sigmas=SIGMAS,weights=WEIGHTS):
 if n<2 or len(sigmas)!=len(weights) or not sigmas or any(not math.isfinite(s) or s<=0 for s in sigmas) or any(not math.isfinite(w) or w<0 for w in weights) or abs(sum(weights)-1)>1e-12:raise ValueError('Invalid bands/weights')
 i=torch.arange(n,device=device,dtype=torch.float64);d=i[:,None]-i[None,:]
 base=torch.exp(-.5*(d/2.).square());mass=base.mean()
 terms=torch.stack([torch.exp(-.5*(d/s).square()) for s in sigmas]);masses=terms.mean((-1,-2),keepdim=True)
 w=torch.tensor(weights,device=device,dtype=torch.float64)[:,None,None]
 return (w*terms*(mass/masses)).sum(0)
def kernel(a,b,sigmas=SIGMAS,weights=WEIGHTS):
 if a.shape!=b.shape or a.shape[-1]!=4:raise ValueError('Matching four point patches required')
 a,b=a.double(),b.double()
 return time_kernel(a.shape[-2],a.device,sigmas,weights)*torch.exp(-.5*(a.unsqueeze(-2)-b.unsqueeze(-3)).square().sum(-1))
def parts(prediction,target,sigmas=SIGMAS,weights=WEIGHTS):
 if prediction.ndim!=3 or prediction.shape!=target.shape or not all(prediction.shape) or prediction.shape[1]%4 or prediction.shape[1]<8 or not prediction.is_floating_point() or not target.is_floating_point() or prediction.device!=target.device:raise ValueError('Matching finite B,H,C required')
 if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():raise ValueError('Nonfinite')
 p=prediction.double().transpose(1,2).unfold(-1,4,4);y=target.detach().double().transpose(1,2).unfold(-1,4,4)
 pp=kernel(p,p,sigmas,weights).mean((-1,-2))
 with torch.no_grad():yy=kernel(y,y,sigmas,weights).mean((-1,-2))
 py=kernel(p,y,sigmas,weights).mean((-1,-2));raw=pp+yy-2*py
 if not torch.isfinite(raw).all() or raw.min()<-1e-10:raise RuntimeError('Invalid MMD')
 v=raw.clamp_min(0).mean().to(prediction.dtype)
 return v,dict(time_position_mmd=float(v.detach()),mmd_prediction_self=float(pp.detach().mean()),mmd_target_self=float(yy.mean()),mmd_cross=float(py.detach().mean()),minimum_unclipped_mmd=float(raw.detach().min()),patch_count=p.shape[-2],time_sigmas=list(sigmas),time_weights=list(weights),mass_matched_sigma2=True,value_sigma=1.,same_kernel_all_three_terms=True,raw_not_Adam_update=True)
def numpy_parts(prediction,target,chunk=16,sigmas=SIGMAS,weights=WEIGHTS):
 n=prediction.shape[1]//4;i=np.arange(n);d=i[:,None]-i[None,:];base=np.exp(-d*d/8);terms=[np.exp(-d*d/(2*s*s)) for s in sigmas];time=sum(w*t*(base.mean()/t.mean()) for w,t in zip(weights,terms));total=np.zeros(4)
 def term(a,b):
  aa=(a*a).sum(-1);bb=(b*b).sum(-1);d2=np.maximum(aa[..., :,None]+bb[...,None,:]-2*(a@b.swapaxes(-1,-2)),0)
  return (time*np.exp(-d2/2)).mean((-1,-2))
 for s in range(0,len(prediction),chunk):
  p=np.asarray(prediction[s:s+chunk],dtype=np.float64).transpose(0,2,1).reshape(-1,prediction.shape[2],n,4);y=np.asarray(target[s:s+chunk],dtype=np.float64).transpose(0,2,1).reshape(p.shape)
  pp,yy,py=term(p,p),term(y,y),term(p,y);raw=pp+yy-2*py
  assert np.isfinite(raw).all() and raw.min()>=-1e-10
  total+=np.array([np.maximum(raw,0).mean(),pp.mean(),yy.mean(),py.mean()])*len(p)
 return dict(zip(('time_position_mmd','mmd_prediction_self','mmd_target_self','mmd_cross'),map(float,total/len(prediction))))
