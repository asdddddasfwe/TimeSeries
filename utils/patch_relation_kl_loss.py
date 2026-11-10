"""Rowwise KL(truth relations || prediction relations), excluding self edges.

Each B,H,C forecast is split into N disjoint length-L raw-value patches.
RBF logits are -sum(coordinate differences squared)/(2*sigma**2).
Rows compare the same temporal indices. No cross prediction/truth kernel,
centering, extra normalization, history, or cross-variable relationships.
"""
import math
import torch


def patch_log_probabilities(patches, bandwidth=1.):
    """Return B,C,N,N-1 log probabilities; ascending off-diagonal columns."""
    n = patches.shape[-2]
    logits = -.5 * ((patches.unsqueeze(-2) - patches.unsqueeze(-3)) / bandwidth).square().sum(-1)
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=patches.device)
    # Remove rather than multiply masked -inf entries: prevents 0*(-inf) NaNs.
    logits = logits[..., off_diagonal].reshape(*patches.shape[:-2], n, n - 1)
    return logits.log_softmax(-1)


def parts(prediction, target, bandwidth=1., length=4):
    if prediction.ndim != 3 or prediction.shape != target.shape or not all(prediction.shape):
        raise ValueError('Matching nonempty B,H,C tensors required')
    if isinstance(length, bool) or not isinstance(length, int) or length < 1 or prediction.shape[1] % length or prediction.shape[1] // length < 2:
        raise ValueError('Need at least two disjoint length-L patches')
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Finite positive bandwidth required')
    if prediction.device != target.device or not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError('Floating tensors on same device required')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite relation KL input')
    p = prediction.double().transpose(1, 2).unfold(-1, length, length)
    y = target.detach().double().transpose(1, 2).unfold(-1, length, length)
    lp = patch_log_probabilities(p, bandwidth)
    with torch.no_grad():
        lq = patch_log_probabilities(y, bandwidth)
        q = lq.exp()
    row_kl = (q * (lq - lp)).sum(-1)
    if not torch.isfinite(row_kl).all() or row_kl.min() < -1e-10:
        raise RuntimeError('Invalid relation KL')
    raw = row_kl.mean()
    value = row_kl.clamp_min(0).mean().to(prediction.dtype)
    diag = dict(relation_kl=float(value.detach()),
                prediction_relation_entropy=float(-(lp.detach().exp() * lp.detach()).sum(-1).mean()),
                target_relation_entropy=float(-(q * lq).sum(-1).mean()),
                relation_cross_entropy=float(-(q * lp.detach()).sum(-1).mean()),
                minimum_unclipped_kl=float(row_kl.detach().min()),
                unclipped_relation_kl=float(raw.detach()), patch_count=p.shape[-2])
    return value, diag
