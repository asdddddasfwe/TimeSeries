"""Supervise existing MA1 output branches in dataset-standardized coordinates.

Targets use a fixed future-window MA, solely as training labels. The covariance
of the predicted seasonal curve is a restricted Gaussian parameterization,
not an identified conditional error covariance. No model parameter is added.
"""
import math
from unittest.mock import patch

import torch
from torch.nn import functional as F

from utils.prediction_autocov_nll import prediction_gaussian_nll


def decompose_target(target, kernel=25):
    """[B,H,C] -> seasonal, trend; centered odd MA with replicated endpoints."""
    if target.ndim != 3 or not target.is_floating_point():
        raise ValueError('Expected floating target [B,H,C]')
    if type(kernel) is not int or kernel < 1 or kernel % 2 != 1:
        raise ValueError('MA kernel must be a positive odd integer')
    if not bool(torch.isfinite(target).all()):
        raise ValueError('Nonfinite target')
    dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
    target = target.detach().to(dtype)
    padded = F.pad(target.transpose(1, 2), (kernel//2, kernel//2), mode='replicate')
    trend = F.avg_pool1d(padded, kernel, stride=1).transpose(1, 2)
    return target - trend, trend


def forward_branches(model, history):
    """One unmodified model forward; retain live eight-component gradients.

    Seasonal = sum(first four) * existing history stdev.
    Trend = sum(last four) * existing history stdev + existing history mean.
    Keep the original total forward/denorm bit-for-bit; S+T may differ by FP32
    rounding. Nested diagnostic capture remains supported; patches always exit.
    """
    norm = model.normalize
    if norm.affine or norm.subtract_last or model.features != 'M':
        raise ValueError('Branch supervision supports current non-affine, mean-based M model only')
    captured = []
    original = model.predict_components

    def capture(x):
        parts = original(x)
        captured.append(parts)
        return parts

    with patch.object(model, 'predict_components', new=capture):
        prediction = model(history)
    if len(captured) != 1 or captured[0].shape != (*prediction.shape, 8):
        raise RuntimeError('Expected exactly one forward with eight output components')
    parts = captured[0]
    seasonal, trend = parts[..., :4].sum(-1), parts[..., 4:].sum(-1)
    if not norm.non_norm:
        seasonal = seasonal * norm.stdev
        trend = trend * norm.stdev + norm.mean
    if not torch.allclose(seasonal + trend, prediction, atol=2e-5, rtol=1e-6):
        raise RuntimeError('Branch reconstruction differs from original denorm')
    return dict(prediction=prediction, seasonal=seasonal, trend=trend)


def branch_losses(outputs, target, arm, diagonal_floor=.01, kernel=25, seasonal_weight=1.):
    """mse: trend MSE + w*season MSE; nll: trend MSE + w*season NLL.

    The NLL uses the uncentered seasonal error. Only its covariance calculation
    centers the prediction. Total MSE is diagnostic, never silently added.
    This composite objective is not the joint likelihood of the original Y.
    """
    if arm not in ('mse', 'nll'):
        raise ValueError('Unknown branch objective')
    if not math.isfinite(seasonal_weight) or seasonal_weight <= 0:
        raise ValueError('Seasonal weight must be finite and positive')
    for key in ('prediction', 'seasonal', 'trend'):
        if outputs[key].shape != target.shape or outputs[key].device != target.device:
            raise ValueError('Branch and target shapes/devices must match')
        if not bool(torch.isfinite(outputs[key]).all()):
            raise ValueError('Nonfinite branch prediction')
    season_target, trend_target = decompose_target(target, kernel)
    season_error = outputs['seasonal'] - season_target
    trend_error = outputs['trend'] - trend_target
    season_mse, trend_mse = season_error.square().mean(), trend_error.square().mean()
    with torch.set_grad_enabled(torch.is_grad_enabled() and arm == 'nll'):
        nll, diagnostics = prediction_gaussian_nll(
            outputs['seasonal'], season_target, diagonal_floor, True)
    total = trend_mse + seasonal_weight * (season_mse if arm == 'mse' else nll)
    if not bool(torch.isfinite(total)):
        raise RuntimeError('Nonfinite branch objective')
    diagnostics.update(
        season_bias=float(season_error.detach().mean()),
        trend_bias=float(trend_error.detach().mean()),
        season_window_bias_rms=float(season_error.detach().mean(1).square().mean().sqrt()),
        trend_window_bias_rms=float(trend_error.detach().mean(1).square().mean().sqrt()),
        target_season_variance=float(season_target.var(1, unbiased=False).mean()),
        season_mse=float(season_mse.detach()), trend_mse=float(trend_mse.detach()))
    return dict(loss=total, mse=(outputs['prediction']-target.detach()).square().mean(),
                season_mse=season_mse, trend_mse=trend_mse, nll=nll.detach(),
                nll_diagnostics=diagnostics)
