"""Shared helpers extracted from retired experiment entry points."""
import numpy as np
import torch


def history_statistics(values, windows, length=96):
    """Only input rows i:i+length; no future target rows enter statistics."""
    a=np.asarray(values,dtype=np.float32).astype(np.float64)
    cs=np.concatenate([np.zeros((1,a.shape[1])),np.cumsum(a,axis=0)],axis=0)
    sq=np.concatenate([np.zeros((1,a.shape[1])),np.cumsum(a*a,axis=0)],axis=0)
    mean=(cs[length:length+windows]-cs[:windows])/length
    variance=np.maximum((sq[length:length+windows]-sq[:windows])/length-mean*mean,0)
    return np.stack([mean,.5*np.log(variance+1e-5)],axis=-1)


def design_with_context(z,q):
    """z=[B,H,C,66], q=[B,C,2], result [B,H,C,198]."""
    return torch.cat([z,z*q[:,None,:,0:1],z*q[:,None,:,1:2]],-1)


def shuffled_indices(ntrain,nval,seed):
    """Joint mu/sigma and variable vector permutation; val draws train bank."""
    rng=np.random.default_rng(seed)
    permutation=rng.permutation(ntrain)
    if ntrain>1:
        fixed=np.flatnonzero(permutation==np.arange(ntrain))
        if len(fixed)>1:permutation[fixed]=np.roll(permutation[fixed],1)
        elif len(fixed)==1:
            i=int(fixed[0]);j=(i+1)%ntrain;permutation[i],permutation[j]=permutation[j],permutation[i]
        assert np.all(permutation!=np.arange(ntrain))
    return permutation,rng.integers(0,ntrain,size=nval)
