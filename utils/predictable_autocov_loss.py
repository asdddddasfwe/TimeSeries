"""Bias-corrected finite-window autocovariance loss, without model normalization edits.

The correction is a frozen estimate of E[C_noise(k)|history], not the raw
process lag covariance. Window centering and the biased /H denominator matter.
With exact correction and fixed history-dependent weights/scales, the squared
auxiliary risk targets the conditional mean's sample autocovariance in expectation.
This guarantee does not apply to inaccurate reference residual corrections.
"""
import torch

DEFAULT_LAGS = (0, 1, 2, 4, 8, 12, 24, 32, 48)


def sample_autocovariance(sequence, lags=DEFAULT_LAGS):
    """[B,H,C] -> [B,C,K], noncircular, centered per window, denominator H."""
    if sequence.ndim != 3 or not sequence.is_floating_point():
        raise ValueError('Expected floating [B,H,C]')
    horizon = sequence.shape[1]
    if not lags or len(set(lags)) != len(lags) or any(type(k) is not int or not 0 <= k < horizon for k in lags):
        raise ValueError('Lags must be unique integers in [0,H)')
    z = sequence.float() if sequence.dtype in (torch.float16, torch.bfloat16) else sequence
    z = z - z.mean(1, keepdim=True)
    return torch.stack([(z[:, :horizon-k] * z[:, k:]).sum(1) / horizon for k in lags], -1)


def expected_centered_autocovariance(covariance, lags=DEFAULT_LAGS):
    """[...,H,H] noise covariance -> [...,K] expectation after window centering.

    No white-noise or stationarity assumption; covariance may be nonstationary.
    """
    if covariance.ndim < 2 or covariance.shape[-1] != covariance.shape[-2]:
        raise ValueError('Expected square covariance matrices')
    horizon = covariance.shape[-1]
    if not lags or len(set(lags)) != len(lags) or any(type(k) is not int or not 0 <= k < horizon for k in lags):
        raise ValueError('Invalid covariance lags')
    s = covariance.float() if covariance.dtype in (torch.float16, torch.bfloat16) else covariance
    centered = s - s.mean(-1, keepdim=True) - s.mean(-2, keepdim=True) + s.mean((-1, -2), keepdim=True)
    return torch.stack([centered.diagonal(offset=k, dim1=-2, dim2=-1).sum(-1) / horizon for k in lags], -1)


def predictable_autocov_loss(prediction, target, noise_correction, channel_variance,
                              reliability=None, lags=DEFAULT_LAGS):
    """Return scalar squared structure loss and MSE; all supervision is detached.

    correction: [C,K] or [B,C,K], allowed negative at nonzero lags (and noisy
    corrected targets can be negative even at lag zero; never clamp them).
    channel_variance: positive [C], fitted from training data only and frozen.
    reliability: nonnegative [K], [C,K] or [B,C,K], determined before seeing the
    current target. Weights are NOT renormalized by surviving reliability mass.
    """
    if prediction.shape != target.shape or prediction.device != target.device:
        raise ValueError('Prediction and target must match')
    cp = sample_autocovariance(prediction, lags)
    cy = sample_autocovariance(target.detach(), lags)
    correction = torch.as_tensor(noise_correction, dtype=cp.dtype, device=cp.device).detach()
    if correction.shape not in (cp.shape, cp.shape[1:]):
        raise ValueError('Correction must have shape [C,K] or [B,C,K]')
    variance = torch.as_tensor(channel_variance, dtype=cp.dtype, device=cp.device).detach()
    if variance.shape != (cp.shape[1],) or not bool(torch.isfinite(variance).all() and (variance > 0).all()):
        raise ValueError('Positive finite frozen per-channel scale required')
    if not bool(torch.isfinite(correction).all()):
        raise ValueError('Nonfinite correction')
    weights = torch.ones_like(cp) if reliability is None else torch.as_tensor(reliability, dtype=cp.dtype, device=cp.device).detach()
    if weights.shape not in (cp.shape, cp.shape[1:], (len(lags),)) or not bool(torch.isfinite(weights).all() and (weights >= 0).all()):
        raise ValueError('Invalid reliability')
    target_covariance = cy - correction
    error = (cp - target_covariance) / variance[None, :, None]
    structure = (weights * error.square()).mean()
    mse = (prediction - target.detach()).square().mean()
    return dict(mse=mse, structure=structure)
