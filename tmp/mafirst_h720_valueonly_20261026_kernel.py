"""Larger Gaussian time/value bandwidths; original full biased MMD."""
import numpy as np
from utils.time_position_patch_mmd import parts as original_parts
SIGMA_TIME=2.0
SIGMA_VALUE=2.0
def parts(prediction,target):
 return original_parts(prediction,target,sigma_value=SIGMA_VALUE,sigma_time=SIGMA_TIME)
def numpy_parts(prediction,target,family='wide'):
 sv,st=(SIGMA_VALUE,SIGMA_TIME) if family=='wide' else (1.,2.)
 assert family in ('wide','rbf')
 totals=np.zeros(4);n=prediction.shape[1]//4;i=np.arange(n,dtype=np.float64);t=np.exp(-.5*((i[:,None]-i[None,:])/st)**2)
 def term(a,b):
  aa=(a*a).sum(-1);bb=(b*b).sum(-1);d2=np.maximum(aa[..., :,None]+bb[...,None,:]-2*(a@b.swapaxes(-1,-2)),0)
  return (t*np.exp(-d2/(2*sv*sv))).mean((-1,-2))
 for s in range(0,len(prediction),16):
  p=np.asarray(prediction[s:s+16],dtype=np.float64).transpose(0,2,1).reshape(-1,prediction.shape[2],n,4)
  y=np.asarray(target[s:s+16],dtype=np.float64).transpose(0,2,1).reshape(p.shape)
  pp,yy,py=term(p,p),term(y,y),term(p,y);raw=pp+yy-2*py
  assert np.isfinite(raw).all() and raw.min()>=-1e-10
  totals+=np.array([np.maximum(raw,0).mean(),pp.mean(),yy.mean(),py.mean()])*len(p)
 return dict(zip(('time_position_mmd','mmd_prediction_self','mmd_target_self','mmd_cross'),map(float,totals/len(prediction))))
