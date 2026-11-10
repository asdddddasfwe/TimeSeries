"""Independent NumPy reference: history-only AR4 posterior quadratic score."""
import numpy as np

def numpy_quadratic(prediction,target,history,chunk=64):
    total=np.zeros(2,dtype=np.float64);count=0
    for start in range(0,len(prediction),chunk):
        x=history[start:start+chunk].astype(np.float64).transpose(0,2,1)
        e=(prediction[start:start+chunk].astype(np.float64)-target[start:start+chunk]).transpose(0,2,1)
        z=np.stack([x[...,4-j:96-j] for j in range(1,5)]+[np.ones_like(x[...,4:])],axis=-1)
        precision=np.einsum('bcti,bctj->bcij',z,z)+np.eye(5)
        covariance=np.linalg.solve(precision,np.broadcast_to(np.eye(5),precision.shape))
        rhs=np.einsum('bcti,bct->bci',z,x[...,4:])
        mean=np.linalg.solve(precision,rhs[...,None])[...,0]
        ep=np.stack([e[...,4-j:96-j] for j in range(1,5)],axis=-1)
        residual=e[...,4:]-np.einsum('bcti,bci->bct',ep,mean[...,:4])
        q=.5*residual**2;u=.5*np.einsum('bcti,bcij,bctj->bct',ep,covariance[...,:4,:4],ep)
        assert np.isfinite(q).all() and np.isfinite(u).all() and u.min()>-1e-12
        total+=np.array([q.sum(),u.sum()]);count+=q.size
    q,u=total/count
    return dict(mean_quadratic=float(q),uncertainty_quadratic=float(u),quadratic=float(q+u))
