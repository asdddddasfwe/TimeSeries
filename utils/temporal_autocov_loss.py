"""Amplitude-sensitive autocovariance regularizer for [batch, time, channel].

Design prototype only; no existing experiment imports this module.
Unlike ordinary ACF matching, both covariances use the target variance.
"""
import math

import torch
import torch.nn.functional as F


def target_scaled_autocov_loss(prediction, target,
                              lags=(1, 2, 4, 8, 12, 24, 48),
                              variance_floor=0.01, zero_lag_weight=0.25):
    """Return an auxiliary scalar, to be added to pure-network MSE.

    Inputs must share [B, H, C] and the dataset-standardized output space.
    Covariances use 1/H, independent temporal centering per B,C, no wrap.
    The target and its variance are detached. FP16/BF16 accumulate in FP32.
    variance_floor has squared-target units, NOT standard-deviation units.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError('Expected matching [batch, time, channel] tensors')
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError('Inputs must be floating point')
    if prediction.device != target.device or min(prediction.shape) < 1:
        raise ValueError('Inputs must be nonempty and on the same device')
    horizon = prediction.shape[1]
    lags = tuple(lags)
    if (not lags or any(type(k) is not int or not 0 < k < horizon for k in lags)
            or len(set(lags)) != len(lags)):
        raise ValueError('Lags must be unique integers in [1, horizon-1]')
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError('variance_floor must be finite and positive')
    if not math.isfinite(zero_lag_weight) or not 0 < zero_lag_weight < 1:
        raise ValueError('zero_lag_weight must be strictly between 0 and 1')
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        pred = prediction.to(dtype)
        truth = target.detach().to(dtype)
        pred = pred - pred.mean(dim=1, keepdim=True)
        truth = truth - truth.mean(dim=1, keepdim=True)
        target_variance = truth.square().mean(dim=1)
        denominator = target_variance + variance_floor
        def penalty(delta):
            return F.huber_loss(delta, torch.zeros_like(delta), delta=1.0,
                                reduction='mean')
        result = zero_lag_weight * penalty(
            (pred.square().mean(dim=1) - target_variance) / denominator)
        overlap_sum = sum(horizon - k for k in lags)
        for lag in lags:
            pred_cov = (pred[:, lag:] * pred[:, :-lag]).sum(dim=1) / horizon
            true_cov = (truth[:, lag:] * truth[:, :-lag]).sum(dim=1) / horizon
            weight = (1 - zero_lag_weight) * (horizon - lag) / overlap_sum
            result = result + weight * penalty((pred_cov - true_cov) / denominator)
        return result
