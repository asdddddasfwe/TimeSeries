"""Gaussian RKHS lag covariance including observed-history/future boundary pairs."""
import math
import torch
from utils.kernel_autocovariance_loss import centered_gram


def boundary_kernel_autocovariance_loss(prediction,target,history,lags=(1,4,12,24),bandwidth=1.):
    """[B,H,C] futures and [B,L,C] history; H pairs per lag.

    Left endpoints are concat(history[-k:], future[:-k]); right endpoints future.
    Shared detached history anchors both predicted and target covariance operators.
    Center all H pairs TOGETHER, not the k boundary pairs in isolation.
    """
    if prediction.ndim!=3 or prediction.shape!=target.shape or history.ndim!=3:
        raise ValueError('Expected [B,H,C] futures and [B,L,C] history')
    if (history.shape[0],history.shape[2])!=(prediction.shape[0],prediction.shape[2]):raise ValueError('History B/C mismatch')
    if any(v.device!=prediction.device or not v.is_floating_point() or not all(v.shape) for v in (prediction,target,history)):raise ValueError('Nonempty floating inputs on same device required')
    lags=tuple(lags)
    if not lags or len(set(lags))!=len(lags) or any(not isinstance(k,int) or isinstance(k,bool) or not 0<k<prediction.shape[1] or k>history.shape[1] for k in lags):raise ValueError('Invalid lags')
    if not math.isfinite(bandwidth) or bandwidth<=0:raise ValueError('Invalid bandwidth')
    p=prediction.double().transpose(1,2);y=target.detach().double().transpose(1,2);x=history.detach().double().transpose(1,2)
    if any(not torch.isfinite(v).all() for v in (p,y,x)):raise ValueError('Nonfinite boundary kernel input')
    distances=[];pp_values=[];yy_values=[];py_values=[]
    for k in lags:
        pa=torch.cat((x[...,-k:],p[...,:-k]),dim=-1);ya=torch.cat((x[...,-k:],y[...,:-k]),dim=-1)
        pp=(centered_gram(pa,pa,bandwidth)*centered_gram(p,p,bandwidth)).mean((-1,-2))
        with torch.no_grad():yy=(centered_gram(ya,ya,bandwidth)*centered_gram(y,y,bandwidth)).mean((-1,-2))
        py=(centered_gram(pa,ya,bandwidth)*centered_gram(p,y,bandwidth)).mean((-1,-2))
        raw=pp+yy-2*py
        if not torch.isfinite(raw).all() or raw.min() < -1e-10:raise RuntimeError('Invalid boundary kernel covariance distance')
        distances.append(raw.clamp_min(0).mean());pp_values.append(pp.detach().mean());yy_values.append(yy.mean());py_values.append(py.detach().mean())
    values=torch.stack(distances)
    return values.mean().to(prediction.dtype),dict(kernel_distance=float(values.detach().mean()),distance_by_lag={str(k):float(v.detach()) for k,v in zip(lags,values)},prediction_operator_norm2=[float(v) for v in pp_values],target_operator_norm2=[float(v) for v in yy_values],cross_operator_inner_product=[float(v) for v in py_values],bandwidth=bandwidth,lags=list(lags),pairs_per_lag=prediction.shape[1],boundary_pairs_per_lag=list(lags))
