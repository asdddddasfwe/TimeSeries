"""Finite-window covariance and history-boundary moments for [B,T,C] forecasts.

Only loss statistics are centered; no model normalization is modified. Historical
inputs and targets are detached. Biased covariance uses /H at every lag, with
no wraparound; boundary moments use the k actual cross-boundary pairs.
"""
import math
import torch
import torch.nn.functional as F

COV_LAGS = (0, 1, 2, 4, 8, 12, 24, 32, 48)
BOUNDARY_LAGS = (8, 16, 24, 48, 96)


def history_anchored_autocov_loss(history, prediction, target, variance_floor,
                                 cov_lags=COV_LAGS, boundary_lags=BOUNDARY_LAGS):
    if prediction.ndim != 3 or target.shape != prediction.shape or history.ndim != 3:
        raise ValueError('Expected history/prediction/target [B,T,C]')
    if history.shape[0] != prediction.shape[0] or history.shape[2] != prediction.shape[2]:
        raise ValueError('History batch and channels must match prediction')
    if history.device != prediction.device or target.device != prediction.device:
        raise ValueError('All inputs must share a device')
    if not prediction.is_floating_point():
        raise ValueError('Prediction must be floating point')
    h, length = prediction.shape[1], history.shape[1]
    if not cov_lags or not boundary_lags or len(set(cov_lags)) != len(cov_lags) or len(set(boundary_lags)) != len(boundary_lags):
        raise ValueError('Lags must be nonempty and unique')
    if any(not isinstance(k, int) or k < 0 or k >= h for k in cov_lags):
        raise ValueError('Covariance lags must be integers in [0,H)')
    if any(not isinstance(k, int) or k < 1 or k > min(length, h) for k in boundary_lags):
        raise ValueError('Boundary lags must be integers in [1,min(L,H)]')
    dtype = torch.float32 if prediction.dtype in (torch.float16, torch.bfloat16) else prediction.dtype
    p = prediction.to(dtype)
    y, x = target.detach().to(dtype), history.detach().to(dtype)
    eps = torch.as_tensor(variance_floor, device=p.device, dtype=dtype).detach()
    if eps.ndim > 1 or eps.numel() not in (1, p.shape[2]) or not bool(torch.isfinite(eps).all() and (eps > 0).all()):
        raise ValueError('Positive finite scalar or per-channel variance floor required')
    pc, yc, xc = p-p.mean(1, keepdim=True), y-y.mean(1, keepdim=True), x-x.mean(1, keepdim=True)
    vy, vx = yc.square().mean(1), xc.square().mean(1)
    gy, gx = vy/(vy+eps), vx/(vx+eps)
    cov_terms = []
    for k in cov_lags:
        cp = (pc[:, :h-k]*pc[:, k:]).sum(1)/h
        cy = (yc[:, :h-k]*yc[:, k:]).sum(1)/h
        error = (cp-cy)/(vy+eps)
        cov_terms.append(F.smooth_l1_loss(error, torch.zeros_like(error), beta=1., reduction='none'))
    w = p.new_tensor([h-k for k in cov_lags]); w = w/w.sum()
    covariance = (gy*(torch.stack(cov_terms, -1)*w).sum(-1)).mean()
    boundary_terms = []
    for k in boundary_lags:
        bp = (xc[:, length-k:]*pc[:, :k]).mean(1)
        by = (xc[:, length-k:]*yc[:, :k]).mean(1)
        error = (bp-by)/torch.sqrt((vx+eps)*(vy+eps))
        boundary_terms.append(F.smooth_l1_loss(error, torch.zeros_like(error), beta=1., reduction='none'))
    v = p.new_tensor(boundary_lags); v = v/v.sum()
    anchor = (gx*gy*(torch.stack(boundary_terms, -1)*v).sum(-1)).mean()
    return dict(covariance=covariance, anchor=anchor)
