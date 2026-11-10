"""Gaussian NLL with externally supplied label covariance.

Covariance describes label coordinates, not autocovariance of the predicted
curve. A fitted covariance is a statistical approximation specified by the
experiment, independently of this exact Gaussian likelihood calculation.
"""
import math

import torch


def label_gaussian_nll(prediction, target, covariance):
    """Mean per-scalar NLL: prediction/target [B,H,C], covariance [C,H,H].

    Target and externally estimated covariance are detached. Cholesky whitening
    implements the full multi-point conditional likelihood, including correlated
    preceding coordinates; no explicit inverse or pairwise shortcut is used.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError('Expected matching [B,H,C] prediction and target')
    h, c = prediction.shape[1:]
    if covariance.shape != (c, h, h):
        raise ValueError('Expected covariance [C,H,H]')
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    cov = covariance.detach().to(device=prediction.device, dtype=dtype)
    error = (target.detach().to(prediction.device, dtype) - prediction.to(dtype)).permute(2, 1, 0)
    if not torch.isfinite(cov).all() or not torch.isfinite(error).all():
        raise ValueError('Nonfinite covariance or error')
    if not torch.allclose(cov, cov.transpose(-1, -2), atol=1e-7, rtol=1e-6):
        raise ValueError('Covariance must be symmetric')
    chol = torch.linalg.cholesky(cov)
    white = torch.linalg.solve_triangular(chol, error, upper=False)
    quadratic = white.square().sum(dim=1)
    logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    loss = ((quadratic + logdet[:, None] + h * math.log(2 * math.pi)) / (2 * h)).mean()
    return loss, dict(quadratic_per_scalar=float(quadratic.detach().mean()/h),
                     logdet_per_scalar=float(logdet.detach().mean()/h),
                     cholesky_diagonal_min=float(chol.diagonal(dim1=-2, dim2=-1).min()),
                     whitened_residual_rms=float(white.detach().square().mean().sqrt()))


def fit_label_covariance(training_rows, history=96, horizon=96, diagonal_floor=.01, block_size=512):
    """Fit per-channel across-window covariance from training rows only.

    Rows [T,C] are already in existing StandardScaler coordinates. Labels are
    rows[n+history:n+history+horizon], n=0..T-history-horizon, with stride 1.
    Center each horizon coordinate across training windows (not within a curve).
    Use population denominator N and blockwise centered moment combination.
    """
    import numpy as np
    rows = np.asarray(training_rows, dtype=np.float64)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError('Expected finite training rows [T,C]')
    if min(history, horizon, block_size) < 1 or not math.isfinite(diagonal_floor) or diagonal_floor <= 0:
        raise ValueError('Invalid window, block size, or covariance floor')
    windows = np.lib.stride_tricks.sliding_window_view(rows, horizon, axis=0)[history:]
    if len(windows) < 2:
        raise ValueError('At least two training label windows are required')
    c = rows.shape[1]
    mean, moment, count = np.zeros((c, horizon)), np.zeros((c, horizon, horizon)), 0
    for start in range(0, len(windows), block_size):
        block = windows[start:start+block_size]
        n = len(block)
        local_mean = block.mean(0)
        centered = block - local_mean
        local_moment = np.einsum('nci,ncj->cij', centered, centered, optimize=True)
        delta = local_mean - mean
        moment += local_moment + np.einsum('ci,cj->cij', delta, delta) * (count*n/(count+n))
        mean += delta * (n/(count+n))
        count += n
    raw = moment / count
    raw = (raw + raw.transpose(0, 2, 1)) * .5
    covariance = raw + diagonal_floor*np.eye(horizon)[None]
    np.linalg.cholesky(covariance)
    return dict(covariance=covariance, raw_covariance=raw, label_mean=mean,
                count=np.array(count), diagonal_floor=np.array(diagonal_floor))
