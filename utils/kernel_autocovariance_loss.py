"""Within-window lagged RKHS covariance operator matching; no learned features.

Inputs [batch, horizon, channel], already in the SAME dataset scaler units.
C_k(z) = Phi(z[:-k])^T H Phi(z[k:]) / (horizon-k).
The distance is ||C_k(pred)-C_k(target)||_HS^2, not a difference of HSICs.
"""
import math
import torch


def centered_gram(a, b, bandwidth):
    gram = torch.exp(-0.5 * ((a.unsqueeze(-1)-b.unsqueeze(-2))/bandwidth).square())
    return gram-gram.mean(-1, keepdim=True)-gram.mean(-2, keepdim=True)+gram.mean((-1,-2), keepdim=True)


def kernel_autocovariance_loss(prediction, target, lags=(1,4,12,24), bandwidth=1.):
    """Average squared operator distance over lags, windows and variables.

    FP64 Gram algebra prevents cancellation at near-identical predictions.
    Target is detached; prediction enters BOTH lag endpoints and cross-Grams.
    Tiny negative roundoff (>= -1e-10) is clipped; larger negatives fail.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError('Expected matching [B,H,C] prediction/target')
    if prediction.device != target.device or not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError('Floating tensors on the same device required')
    if not all(prediction.shape) or not math.isfinite(bandwidth) or bandwidth<=0:
        raise ValueError('Nonempty inputs and finite positive bandwidth required')
    lags=tuple(lags)
    if not lags or len(set(lags))!=len(lags) or any(not isinstance(k,int) or isinstance(k,bool) or not 0<k<prediction.shape[1] for k in lags):
        raise ValueError('Lags must be distinct positive integers below horizon')
    p=prediction.double().transpose(1,2)
    y=target.detach().double().transpose(1,2)
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite kernel loss input')
    distances=[]; pp_values=[]; yy_values=[]; py_values=[]
    for k in lags:
        pa,pb=p[...,:-k],p[...,k:];ya,yb=y[...,:-k],y[...,k:]
        # Elementwise product of centered Grams equals the HS inner product.
        pp=(centered_gram(pa,pa,bandwidth)*centered_gram(pb,pb,bandwidth)).mean((-1,-2))
        with torch.no_grad():
            yy=(centered_gram(ya,ya,bandwidth)*centered_gram(yb,yb,bandwidth)).mean((-1,-2))
        py=(centered_gram(pa,ya,bandwidth)*centered_gram(pb,yb,bandwidth)).mean((-1,-2))
        raw=pp+yy-2*py
        if not torch.isfinite(raw).all() or raw.min() < -1e-10:
            raise RuntimeError('Invalid kernel covariance distance')
        distances.append(raw.clamp_min(0).mean())
        pp_values.append(pp.detach().mean());yy_values.append(yy.mean());py_values.append(py.detach().mean())
    values=torch.stack(distances)
    diagnostics=dict(kernel_distance=float(values.detach().mean()),
        distance_by_lag={str(k):float(v.detach()) for k,v in zip(lags,values)},
        prediction_operator_norm2=[float(v) for v in pp_values],
        target_operator_norm2=[float(v) for v in yy_values],
        cross_operator_inner_product=[float(v) for v in py_values],bandwidth=bandwidth,lags=list(lags))
    return values.mean().to(prediction.dtype),diagnostics
