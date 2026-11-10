"""Aligned four-increment patches, not kernel autocovariance. No history input."""
import math
import torch

def parts(p, y, bandwidth, length=4):
    if p.shape != y.shape or p.ndim != 3 or p.shape[1] <= length:
        raise ValueError('Matching B,H,C tensors with H > patch length required')
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Positive fixed training-only bandwidth required')
    e = p.double() - y.detach().double()
    increments = e[:, 1:] - e[:, :-1]
    # q is squared Euclidean patch error / (2*m*s^2), not raw level error.
    q = increments.unfold(1, length, 1).square().mean(-1) / (2*bandwidth**2)
    quadratic = q.mean()
    kernel = (-torch.expm1(-q)).mean()
    return quadratic.to(p.dtype), kernel.to(p.dtype), q
