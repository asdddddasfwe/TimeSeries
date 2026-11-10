"""Independent NumPy lagged covariance distance: explicit H and matrix traces."""
import numpy as np


def numpy_kernel_autocovariance(prediction, target, lags=(1,4,12,24), bandwidth=1., chunk=16):
    sums=np.zeros(len(lags));count=0
    for start in range(0,len(prediction),chunk):
        p=np.asarray(prediction[start:start+chunk],dtype=np.float64).transpose(0,2,1)
        y=np.asarray(target[start:start+chunk],dtype=np.float64).transpose(0,2,1)
        for j,k in enumerate(lags):
            n=p.shape[-1]-k;h=np.eye(n)-np.ones((n,n))/n
            def kernel(a,b):return np.exp(-((a[..., :,None]-b[...,None,:])**2)/(2*bandwidth**2))
            ka=kernel(p[...,:-k],p[...,:-k]);kb=kernel(p[...,k:],p[...,k:])
            la=kernel(y[...,:-k],y[...,:-k]);lb=kernel(y[...,k:],y[...,k:])
            cross_a=kernel(p[...,:-k],y[...,:-k]);cross_b=kernel(y[...,k:],p[...,k:])
            # tr(K H L H), cross second kernel has target->prediction orientation.
            tr=lambda a,b:np.einsum('...ij,...ji->...',a@h,b@h)/(n*n)
            d=tr(ka,kb)+tr(la,lb)-2*tr(cross_a,cross_b)
            assert np.isfinite(d).all() and d.min()>-1e-10
            sums[j]+=np.maximum(d,0).sum()
        count+=p.shape[0]*p.shape[1]
    values=sums/count
    return dict(kernel_distance=float(values.mean()),distance_by_lag={str(k):float(v) for k,v in zip(lags,values)})
