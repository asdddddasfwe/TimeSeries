"""Aligned raw-value Gaussian patches in existing dataset-standardized units."""
import math
import torch


def parts(p, y, bandwidth=1., length=4):
    """B,H,C -> mean squared patch distance, mean 1-k, B,H-L+1,C q.

    No differences, centering, new normalization, history or cross-variable pairs.
    This is a nonlinear transformation of aligned patch MSE, not autocovariance.
    """
    if p.shape != y.shape or p.ndim != 3 or not isinstance(length, int) or length < 1 or p.shape[1] < length:
        raise ValueError('Matching B,H,C tensors and 1 <= length <= H required')
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Finite positive bandwidth required')
    e = p.double() - y.detach().double()
    if not torch.isfinite(e).all():
        raise ValueError('Nonfinite raw patch error')
    q = e.unfold(1, length, 1).square().mean(-1) / (2 * bandwidth**2)
    return q.mean().to(p.dtype), (-torch.expm1(-q)).mean().to(p.dtype), q
