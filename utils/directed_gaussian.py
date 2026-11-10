"""History-conditioned Gaussian innovations with strictly past-to-future edges.

mean/variance: [B,H,C]; coefficients: [B,H,C,K]. Each lag k>0 maps
residual[h-k] to residual[h]. Channels are independent conditional on history.
These are predictive conditional relations, not identified intervention effects.
"""
import math
import torch


def _check(mean, coefficients, variance, lags):
    if mean.ndim != 3 or not mean.is_floating_point():
        raise ValueError('Expected floating mean [B,H,C]')
    if variance.shape != mean.shape or coefficients.shape != (*mean.shape, len(lags)):
        raise ValueError('Incompatible Gaussian parameter shapes')
    if len(set(lags)) != len(lags) or any(type(k) is not int or not 0 < k < mean.shape[1] for k in lags):
        raise ValueError('Lags must be unique positive integers smaller than H')
    if any(v.device != mean.device or v.dtype != mean.dtype for v in (coefficients, variance)):
        raise ValueError('Parameters must share device and dtype')
    if not all(bool(torch.isfinite(v).all()) for v in (mean, coefficients, variance)) or not bool((variance > 0).all()):
        raise ValueError('Expected finite parameters and positive innovation variance')


def innovations(mean, target, coefficients, lags):
    """Whiten residuals using only strictly earlier target values; [B,H,C]."""
    if target.shape != mean.shape or target.device != mean.device:
        raise ValueError('Target must match mean')
    residual = target.detach() - mean
    terms = []
    for i, lag in enumerate(lags):
        shifted = torch.cat((torch.zeros_like(residual[:, :lag]), residual[:, :-lag]), 1)
        terms.append(coefficients[..., i] * shifted)
    return residual - sum(terms) if terms else residual


def gaussian_nll(mean, target, coefficients, variance, lags, reduction='mean'):
    """Exact joint NLL, including log(2*pi), normalized per scalar by default.

    det(I-B)=1, so no further Jacobian term is required. Variance is the
    innovation variance, not the marginal predictive variance.
    """
    _check(mean, coefficients, variance, lags)
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError('Unknown reduction')
    # FP32 likelihood under autocast, while retaining double for gradcheck.
    dtype = torch.float64 if mean.dtype == torch.float64 else torch.float32
    m, b, d = (v.to(dtype) for v in (mean, coefficients, variance))
    e = innovations(m, target.to(dtype), b, lags)
    terms = .5 * (e.square() / d + d.log() + math.log(2 * math.pi))
    return terms if reduction == 'none' else getattr(terms, reduction)()


def transition_matrix(coefficients, lags):
    """Return B as [B,C,H,H], strictly lower triangular; diagnostic utility."""
    batch, horizon, channels, _ = coefficients.shape
    matrix = coefficients.new_zeros(batch, channels, horizon, horizon)
    for i, lag in enumerate(lags):
        values = coefficients[:, lag:, :, i].transpose(1, 2)
        matrix = matrix + torch.diag_embed(values, offset=-lag)
    return matrix


def predictive_covariance(mean, coefficients, variance, lags):
    """Full [B,C,H,H] covariance; only needed for diagnostics/short horizons."""
    _check(mean, coefficients, variance, lags)
    dtype = torch.float64 if mean.dtype == torch.float64 else torch.float32
    b, d = coefficients.to(dtype), variance.to(dtype)
    eye = torch.eye(mean.shape[1], dtype=dtype, device=mean.device)
    a = eye - transition_matrix(b, lags)
    chol = torch.linalg.solve_triangular(a, torch.diag_embed(d.transpose(1, 2).sqrt()), upper=False)
    return chol @ chol.transpose(-1, -2)


def sample_paths(mean, coefficients, variance, lags, samples=1, generator=None):
    """Ancestral paths [S,B,H,C]; no target is accepted or read."""
    _check(mean, coefficients, variance, lags)
    if type(samples) is not int or samples < 1:
        raise ValueError('samples must be a positive integer')
    dtype = torch.float64 if mean.dtype == torch.float64 else torch.float32
    m, b, d = (v.to(dtype) for v in (mean, coefficients, variance))
    noise = torch.randn((samples, *mean.shape), dtype=dtype, device=mean.device, generator=generator)
    residuals = []
    for h in range(mean.shape[1]):
        value = noise[:, :, h] * d[:, h].sqrt()
        for i, lag in enumerate(lags):
            if h >= lag:
                value = value + b[:, h, :, i] * residuals[h-lag]
        residuals.append(value)
    return m.unsqueeze(0) + torch.stack(residuals, dim=2)
