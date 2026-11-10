"""Independent NumPy trace reference for the boundary-inclusive covariance."""
import numpy as np


def numpy_boundary_kernel(prediction,target,history,lags=(1,4,12,24),bandwidth=1.,chunk=16):
    sums=np.zeros(len(lags));count=0
    for start in range(0,len(prediction),chunk):
        p=np.asarray(prediction[start:start+chunk],dtype=np.float64).transpose(0,2,1)
        y=np.asarray(target[start:start+chunk],dtype=np.float64).transpose(0,2,1)
        x=np.asarray(history[start:start+chunk],dtype=np.float64).transpose(0,2,1)
        n=p.shape[-1];h=np.eye(n)-np.ones((n,n))/n
        kernel=lambda a,b:np.exp(-((a[..., :,None]-b[...,None,:])**2)/(2*bandwidth**2))
        tr=lambda a,b:np.einsum('...ij,...ji->...',a@h,b@h)/(n*n)
        for j,k in enumerate(lags):
            # Explicit integer indices in the concatenated history/future sequences.
            idx=np.arange(x.shape[-1],x.shape[-1]+n)
            pc=np.concatenate([x,p],axis=-1);yc=np.concatenate([x,y],axis=-1)
            pa,pb=pc[...,idx-k],pc[...,idx];ya,yb=yc[...,idx-k],yc[...,idx]
            d=tr(kernel(pa,pa),kernel(pb,pb))+tr(kernel(ya,ya),kernel(yb,yb))-2*tr(kernel(pa,ya),kernel(yb,pb))
            assert np.isfinite(d).all() and d.min()>-1e-10
            sums[j]+=np.maximum(d,0).sum()
        count+=p.shape[0]*p.shape[1]
    values=sums/count
    return dict(kernel_distance=float(values.mean()),distance_by_lag={str(k):float(v) for k,v in zip(lags,values)})
