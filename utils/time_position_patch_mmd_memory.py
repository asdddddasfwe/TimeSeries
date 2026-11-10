"""Memory bounded full biased time-position patch MMD.

Tile only independent batch sequences of temporary Gram tensors. Every patch
pair, diagonal, sample and channel is retained. Reduce inside each checkpoint
so autograd saves inputs rather than Gram outputs. Model/optimizer batch size,
FP64 kernel arithmetic, and the final global reduction are unchanged.
"""
import math

import torch
from torch.utils.checkpoint import checkpoint


def _terms(prediction, target, time_kernel, sigma_value):
    def term(a, b):
        distance = (a.unsqueeze(-2) - b.unsqueeze(-3)) / sigma_value
        gram = torch.exp(-0.5 * distance.square().sum(-1))
        return (time_kernel * gram).mean((-1, -2))

    pp = term(prediction, prediction)
    with torch.no_grad():
        yy = term(target, target)
    py = term(prediction, target)
    return pp, yy, py


def patch_parts(prediction, target, sigma_value=2.0, sigma_time=24.0,
                batch_tile_size=4):
    """MMD and diagnostics for [B,C,N,L]; targets are detached.

    Tiling affects temporary loss tensors, not model forward/optimizer updates.
    """
    if (prediction.ndim != 4 or prediction.shape != target.shape
            or not all(prediction.shape) or prediction.shape[-2] < 2):
        raise ValueError('Expected matching nonempty B,C,N,L with N>=2')
    if (prediction.device != target.device or not prediction.is_floating_point()
            or not target.is_floating_point()):
        raise ValueError('Matching device floating inputs required')
    if any(not math.isfinite(v) or v <= 0 for v in (sigma_value, sigma_time)):
        raise ValueError('Positive finite bandwidths required')
    if (isinstance(batch_tile_size, bool) or not isinstance(batch_tile_size, int)
            or batch_tile_size < 1):
        raise ValueError('batch_tile_size must be a positive integer')
    dtype = prediction.dtype
    p, y = prediction.double(), target.detach().double()
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite patch input')
    n = p.shape[-2]
    index = torch.arange(n, device=p.device, dtype=torch.float64)
    time_kernel = torch.exp(-0.5 * ((index[:, None] - index[None, :])
                                   / sigma_time).square())
    terms = []
    for start in range(0, len(p), batch_tile_size):
        a, b = p[start:start + batch_tile_size], y[start:start + batch_tile_size]
        if torch.is_grad_enabled() and a.requires_grad:
            values = checkpoint(_terms, a, b, time_kernel, sigma_value,
                                use_reentrant=False, preserve_rng_state=False)
        else:
            values = _terms(a, b, time_kernel, sigma_value)
        terms.append(values)
    pp, yy, py = [torch.cat([values[i] for values in terms], dim=0)
                  for i in range(3)]
    raw = pp + yy - 2 * py
    if not torch.isfinite(raw).all() or raw.min() < -1e-10:
        raise RuntimeError('Invalid time-position MMD squared')
    value = raw.clamp_min(0).mean().to(dtype)
    return value, dict(
        time_position_mmd=float(value.detach()),
        mmd_prediction_self=float(pp.detach().mean()),
        mmd_target_self=float(yy.mean()),
        mmd_cross=float(py.detach().mean()),
        minimum_unclipped_mmd=float(raw.detach().min()),
        patch_count=n, sigma_value=sigma_value, sigma_time_patches=sigma_time)


def parts(prediction, target, sigma_value=2.0, sigma_time=24.0, length=4,
          batch_tile_size=4):
    """Complete nonoverlapping length-L patches from [B,H,C] forecasts."""
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError('Expected matching B,H,C')
    if (isinstance(length, bool) or not isinstance(length, int) or length < 1
            or prediction.shape[1] % length or prediction.shape[1] // length < 2):
        raise ValueError('H must be divisible by positive patch length and N>=2')
    return patch_parts(prediction.transpose(1, 2).unfold(-1, length, length),
                       target.transpose(1, 2).unfold(-1, length, length),
                       sigma_value, sigma_time, batch_tile_size)
