"""Gaussian-form score with Toeplitz covariance from each observed target curve.

This target-dependent covariance yields a supervised weighted-error score, not
in general a normalized conditional predictive density. It is not an estimate
of the true conditional covariance, and is never passed into model inference.
All temporal lags use denominator H (biased autocovariance), preserving PSD.
"""
import math
import torch


def target_autocovariance(target, diagonal_floor=.01):
    """[B,H,C] -> covariance [B,C,H,H], autocovariance [B,C,H]."""
    if target.ndim != 3 or target.shape[1] < 2 or not target.is_floating_point():
        raise ValueError('Expected floating target [B,H,C], H>=2')
    if not math.isfinite(diagonal_floor) or diagonal_floor <= 0:
        raise ValueError('Diagonal floor must be finite and positive')
    if not bool(torch.isfinite(target).all()):
        raise ValueError('Nonfinite target')
    dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
    p = target.detach().to(dtype).transpose(1, 2)
    centered = p - p.mean(-1, keepdim=True)
    horizon = p.shape[-1]
    # Zero padding gives linear autocorrelation, never circular wraparound.
    spectrum = torch.fft.rfft(centered, n=2*horizon, dim=-1)
    gamma = torch.fft.irfft(spectrum.conj()*spectrum, n=2*horizon, dim=-1)[..., :horizon] / horizon
    indices = torch.arange(horizon, device=p.device)
    covariance = gamma[..., (indices[:, None]-indices[None, :]).abs()]
    covariance = covariance + diagonal_floor * torch.eye(horizon, device=p.device, dtype=dtype)
    return covariance, gamma


def target_gaussian_nll(mean, target, diagonal_floor=.01, return_diagnostics=False):
    """Gaussian-form score / H averaged over B,C; covariance uses only target.

    Error is uncentered target - prediction. Target and its covariance detach.
    For fixed labels, logdet has zero model gradient; this cannot reduce loss by
    manipulating predicted covariance. No branch supervision or additional head.
    """
    if target.shape != mean.shape or target.device != mean.device:
        raise ValueError('Target must match prediction')
    covariance, gamma = target_autocovariance(target, diagonal_floor)
    target = target.detach().to(covariance.dtype)
    if not bool(torch.isfinite(target).all()):
        raise ValueError('Nonfinite target')
    chol, info = torch.linalg.cholesky_ex(covariance)
    if bool((info != 0).any()):
        raise RuntimeError('Target covariance Cholesky failed; no automatic jitter change')
    residual = (target-mean.to(covariance.dtype)).transpose(1, 2).unsqueeze(-1)
    whitened = torch.linalg.solve_triangular(chol, residual, upper=False).squeeze(-1)
    horizon = mean.shape[1]
    quadratic = whitened.square().sum(-1)/horizon
    logdet = 2*chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)/horizon
    loss = .5*(quadratic+logdet+math.log(2*math.pi)).mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite target-autocovariance score')
    if not return_diagnostics:
        return loss
    diagnostics = dict(quadratic_per_scalar=float(quadratic.detach().mean()),
        logdet_per_scalar=float(logdet.detach().mean()),
        target_variance_mean=float(gamma[..., 0].detach().mean()),
        target_variance_min=float(gamma[..., 0].detach().min()),
        target_variance_max=float(gamma[..., 0].detach().max()),
        chol_diagonal_min=float(chol.diagonal(dim1=-2, dim2=-1).detach().min()),
        whitened_rms=float(whitened.detach().square().mean().sqrt()),
        lag1_covariance_mean=float(gamma[..., 1].detach().mean()),
        predicted_variance_mean=float(mean.detach().var(1, unbiased=False).mean()),
        target_covariance_detached=not covariance.requires_grad)
    return loss, diagnostics
