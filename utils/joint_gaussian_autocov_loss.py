"""Joint observed-history/forecast temporal covariance Gaussian KL.

Only the forecast receives gradients. The shared observed history supplies
cross-boundary lag pairs; centering is over the full concatenated sequence.
"""
import torch
from utils.gaussian_autocov_loss import gaussian_autocov_kl_loss

DEFAULT_OFFSETS = (0, 1, 2, 4, 8, 12, 24, 48, 96)


def joint_gaussian_autocov_kl_loss(history, prediction, target,
                                 offsets=DEFAULT_OFFSETS,
                                 variance_floor=0.01, jitter=0.01,
                                 reduction='mean'):
    """KL(N(0,S_[history,target]) || N(0,S_[history,prediction])) / D.

    history [B,L,C], prediction/target [B,H,C] share standardized units.
    For this study L=H=96, D=9. Matrices use biased lag covariance of the
    centered full 192-point sequences, with denominator 192 and common
    detached variance(history,target)+variance_floor normalization.
    Full-sequence centering couples means: the centered historical portions
    need not be identical even though the observed raw history is identical.
    This is joint lag-covariance matching, not a conditional Gaussian KL.
    """
    if history.ndim != 3 or prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError('Expected history [B,L,C] and matching forecasts [B,H,C]')
    if min(history.shape) < 1 or history.shape[0] != prediction.shape[0] or history.shape[2] != prediction.shape[2]:
        raise ValueError('History batch/channel dimensions must match forecast')
    if history.device != prediction.device or target.device != prediction.device:
        raise ValueError('All tensors must share a device')
    if not all(t.is_floating_point() for t in (history, prediction, target)):
        raise ValueError('Inputs must be floating point')
    dtype = torch.promote_types(torch.promote_types(history.dtype, prediction.dtype), target.dtype)
    fixed_history = history.detach().to(dtype)
    predicted_sequence = torch.cat((fixed_history, prediction.to(dtype)), dim=1)
    target_sequence = torch.cat((fixed_history, target.detach().to(dtype)), dim=1)
    return gaussian_autocov_kl_loss(predicted_sequence, target_sequence,
                                  offsets=offsets, variance_floor=variance_floor,
                                  jitter=jitter, reduction=reduction)
