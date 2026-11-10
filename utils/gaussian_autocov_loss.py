"""Pure-network temporal covariance matching through zero-mean Gaussian KL.

No training entry point is changed by this standalone module. The auxiliary
loss must supplement pointwise MSE; it cannot identify offsets or sign flips.
"""
import math

import torch
import torch.nn.functional as F


DEFAULT_OFFSETS = (0, 1, 2, 4, 8, 12, 24, 48)


def _autocovariance_matrix(centered, offsets):
    """Build [B,C,D,D] PSD Gram matrices from centered [B,H,C] series.

    Full zero-padded shifts ensure entry (i,j) equals the biased lag
    autocovariance C[abs(offsets[i]-offsets[j])], with denominator H.
    Padding prevents circular wraparound and retains equal diagonal energies.
    """
    series = centered.transpose(1, 2)
    max_offset = offsets[-1]
    shifted = torch.stack([
        F.pad(series, (offset, max_offset - offset)) for offset in offsets
    ], dim=-2)
    return (shifted @ shifted.transpose(-1, -2)) / centered.shape[1]


def gaussian_autocov_kl_loss(prediction, target, offsets=DEFAULT_OFFSETS,
                            variance_floor=0.01, jitter=0.01,
                            reduction='mean'):
    """Compute KL(N(0,S_target) || N(0,S_prediction)) / D.

    Inputs: matching nonempty [B,H,C] tensors in the same output units.
    Default offsets require H >= 49; pass shorter offsets for shorter horizons.
    Each batch item and channel receives its own temporal covariance matrix.

    Both matrices share detached target-variance normalization:
        S_z = C_z / (variance(target) + variance_floor) + jitter * I.
    variance_floor has squared-output units; jitter is dimensionless. The
    defaults assume dataset-standardized outputs after any inverse RevIN.
    Do not normalize prediction and target independently to unit variance.

    Return [B,C] for reduction='none', otherwise the batch/channel mean scalar.
    Targets never receive gradients. Half/BFloat16 accumulate in Float32,
    Float64 inputs retain Float64. Cholesky triangular solves avoid explicit
    inverse/determinant. Insufficient jitter raises rather than silently
    changing the objective. Add this auxiliary loss to MSE with a small weight.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError('Expected matching [batch, time, channel] tensors')
    if min(prediction.shape) < 1 or prediction.device != target.device:
        raise ValueError('Inputs must be nonempty and on the same device')
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError('Inputs must be floating point')
    offsets = tuple(offsets)
    horizon = prediction.shape[1]
    if (len(offsets) < 2 or any(type(k) is not int for k in offsets)
            or offsets[0] != 0 or tuple(sorted(set(offsets))) != offsets
            or offsets[-1] >= horizon):
        raise ValueError('Offsets must start at zero and increase strictly below horizon')
    for name, value in [('variance_floor', variance_floor), ('jitter', jitter)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if reduction not in ('mean', 'none'):
        raise ValueError("reduction must be 'mean' or 'none'")
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        pred = prediction.to(dtype)
        truth = target.detach().to(dtype)
        if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
            raise ValueError('Inputs contain non-finite values')
        pred = pred - pred.mean(dim=1, keepdim=True)
        truth = truth - truth.mean(dim=1, keepdim=True)
        scale = (truth.square().mean(dim=1, keepdim=True) + variance_floor).sqrt()
        dimension = len(offsets)
        identity = torch.eye(dimension, device=pred.device, dtype=dtype)
        pred_cov = _autocovariance_matrix(pred / scale, offsets) + jitter * identity
        true_cov = _autocovariance_matrix(truth / scale, offsets) + jitter * identity
        if not torch.isfinite(pred_cov).all() or not torch.isfinite(true_cov).all():
            raise ValueError('Covariance overflow; check input scale')
        pred_chol = torch.linalg.cholesky(pred_cov)
        true_chol = torch.linalg.cholesky(true_cov)
        whitened = torch.linalg.solve_triangular(pred_chol, true_chol, upper=False)
        # ||L_pred^-1 L_true||_F^2 = trace(S_pred^-1 S_true).
        trace = whitened.square().sum(dim=(-2, -1))
        log_det_ratio = 2 * (
            pred_chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            - true_chol.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        # Numerical roundoff can yield tiny negative KL near identical inputs.
        values = ((trace - dimension + log_det_ratio) / (2 * dimension)).clamp_min(0)
        return values if reduction == 'none' else values.mean()
