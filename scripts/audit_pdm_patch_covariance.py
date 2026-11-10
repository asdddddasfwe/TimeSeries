"""Independent NumPy integer indexing, explicit centering J and matrix traces."""
import numpy as np


def numpy_covariance_parts(prediction,truth,bandwidth=1.,length=4):
    p=np.asarray(prediction,dtype=np.float64);y=np.asarray(truth,dtype=np.float64)
    if p.ndim!=3 or p.shape!=y.shape or not all(p.shape) or length<1 or p.shape[1]%length or p.shape[1]//length<2:
        raise ValueError('Matching B,H,C with disjoint length-L patches required')
    if not np.isfinite(bandwidth) or bandwidth<=0 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid bandwidth or nonfinite input')
    n=p.shape[1]//length;j=np.eye(n)-np.ones((n,n))/n
    totals=np.zeros(4);count=0
    for start in range(0,len(p),32):
        # Independent integer indexing, no torch/unfold or shared Gram helper.
        indices=np.arange(n)[:,None]*length+np.arange(length)[None,:]
        a=p[start:start+32,indices,:].transpose(0,3,1,2)
        b=y[start:start+32,indices,:].transpose(0,3,1,2)
        def gram(u,v):
            sq=np.zeros((*u.shape[:-2],n,n))
            for c in range(length):sq+=(u[...,c,None]-v[...,None,:,c])**2
            return np.exp(-sq/(2*bandwidth**2))
        aa,bb,ab=gram(a,a),gram(b,b),gram(a,b)
        trace=lambda z:np.trace(z,axis1=-2,axis2=-1)/(n*n)
        pp=trace(aa@j@aa@j);yy=trace(bb@j@bb@j)
        py=trace(ab@j@ab.swapaxes(-1,-2)@j)
        raw=pp+yy-2*py
        if not np.isfinite(raw).all() or raw.min() < -1e-10:raise RuntimeError('Invalid covariance trace distance')
        totals+=np.array([raw.clip(0).sum(),pp.sum(),yy.sum(),py.sum()]);count+=pp.size
    return dict(zip(('patch_covariance','prediction_covariance_norm2','target_covariance_norm2','cross_covariance_inner_product'),map(float,totals/count)))
