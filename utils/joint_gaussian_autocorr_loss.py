"""Regularized joint lag-correlation Gaussian KL; only forecasts receive gradients."""
import math
import torch
from utils.gaussian_autocov_loss import _autocovariance_matrix
from utils.joint_gaussian_autocov_loss import DEFAULT_OFFSETS


def correlation_matrix(sequence, offsets=DEFAULT_OFFSETS, variance_floor=.01):
    """[B,T,C] -> [B,C,D,D], biased zero-padded lag Gram, unit diagonal.

    Above the floor this is C_ij/sqrt(C_ii*C_jj). Below it, diagonal
    repair shrinks toward identity and preserves positive semidefiniteness.
    Prediction normalization intentionally remains differentiable.
    """
    centered = sequence - sequence.mean(dim=1, keepdim=True)
    covariance = _autocovariance_matrix(centered, offsets)
    variance = covariance.diagonal(dim1=-2, dim2=-1)
    scale = variance.clamp_min(variance_floor)
    result = covariance / (scale.sqrt().unsqueeze(-1) * scale.sqrt().unsqueeze(-2))
    return result + torch.diag_embed(1 - variance / scale)


def joint_gaussian_autocorr_kl_loss(history, prediction, target,
                                   offsets=DEFAULT_OFFSETS, variance_floor=.01,
                                   jitter=.01, reduction='mean'):
    """KL(N(0,R_target+jI) || N(0,R_prediction+jI))/D.

    Separate centering and correlation normalization of [history,target]
    and [history,prediction]. Fixed history means scaling only the future
    is not an invariance. This is not conditional Gaussian KL.
    """
    if history.ndim != 3 or prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError('Expected history [B,L,C] and matching forecasts [B,H,C]')
    if min(history.shape) < 1 or min(prediction.shape) < 1 or history.shape[0] != prediction.shape[0] or history.shape[2] != prediction.shape[2]:
        raise ValueError('Nonempty matching batch/channel dimensions required')
    if history.device != prediction.device or target.device != prediction.device:
        raise ValueError('Inputs must share a device')
    if not all(t.is_floating_point() for t in (history, prediction, target)):
        raise ValueError('Floating point inputs required')
    offsets = tuple(offsets)
    if len(offsets) < 2 or any(type(v) is not int for v in offsets) or offsets[0] != 0 or tuple(sorted(set(offsets))) != offsets or offsets[-1] >= history.shape[1] + prediction.shape[1]:
        raise ValueError('Offsets must be unique increasing integers starting at zero and below total length')
    if any(not math.isfinite(v) or v <= 0 for v in (variance_floor, jitter)):
        raise ValueError('Positive finite floor and jitter required')
    if reduction not in ('mean', 'none'):
        raise ValueError('Invalid reduction')
    dtype = torch.promote_types(torch.promote_types(history.dtype, prediction.dtype), target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        fixed = history.detach().to(dtype)
        pred = torch.cat((fixed, prediction.to(dtype)), dim=1)
        truth = torch.cat((fixed, target.detach().to(dtype)), dim=1)
        if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
            raise ValueError('Nonfinite input')
        d = len(offsets)
        identity = torch.eye(d, dtype=dtype, device=prediction.device)
        sp = correlation_matrix(pred, offsets, variance_floor) + jitter * identity
        st = correlation_matrix(truth, offsets, variance_floor) + jitter * identity
        lp, lt = torch.linalg.cholesky(sp), torch.linalg.cholesky(st)
        whitened = torch.linalg.solve_triangular(lp, lt, upper=False)
        trace = whitened.square().sum(dim=(-2, -1))
        logdet = 2 * (lp.diagonal(dim1=-2, dim2=-1).log().sum(-1) - lt.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        values = ((trace - d + logdet) / (2 * d)).clamp_min(0)
        return values.mean() if reduction == 'mean' else values
