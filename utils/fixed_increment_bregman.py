"""Parameter-free mult lag increment Bregman loss for [batch, time, channel].

Matches signed increments, not an estimated autocovariance or a Gaussian NLL.
All valid pairs are equally weighted. No extra normalization or learned metric.
"""
import math
import torch


LAGS = (1, 4, 12, 24)


def increments(sequence, lags=LAGS):
    if sequence.ndim != 3 or not sequence.is_floating_point():
        raise ValueError('Expected floating [batch, time, channel] tensor')
    if not lags or len(set(lags)) != len(lags):
        raise ValueError('Lags must be nonempty and unique')
    if any(isinstance(k, bool) or not isinstance(k, int) or not 0 < k < sequence.shape[1] for k in lags):
        raise ValueError('Lags must be positive integers below the horizon')
    # No circular wraparound, padding, or changes to temporal alignment.
    return torch.cat([sequence[:, k:] - sequence[:, :-k] for k in lags], dim=1)


def increment_bregman(prediction, target, beta=1., lags=LAGS):
    """D_psi(delta_target, delta_prediction), psi(s)=s^2/2+beta*logcosh(s)."""
    if prediction.shape != target.shape or prediction.device != target.device:
        raise ValueError('Prediction and target must have matching shapes/devices')
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    rp = increments(prediction.double(), lags)
    ry = increments(target.detach().double(), lags)
    delta = ry - rp
    quadratic = .5 * delta.square()
    nonlinear = (torch.logaddexp(ry, -ry) - torch.logaddexp(rp, -rp)
                 - torch.tanh(rp) * delta)
    value = (quadratic + beta * nonlinear).mean()
    if not torch.isfinite(value) or float(value.detach()) < -1e-10:
        raise RuntimeError('Invalid fixed increment Bregman divergence')
    diagnostics = dict(
        quadratic=float(quadratic.detach().mean()),
        logcosh_divergence=float(nonlinear.detach().mean()),
        prediction_increment_rms=float(rp.detach().square().mean().sqrt()),
        target_increment_rms=float(ry.square().mean().sqrt()),
        curvature_mean=float((1 + beta * (1 - torch.tanh(rp.detach()).square())).mean()),
        pair_count=rp.shape[1],
    )
    return value.to(prediction.dtype), diagnostics


def objective(prediction, target, history, arm, config):
    """Compatible trainer interface; history is deliberately unused."""
    if arm not in ('mse', 'quadratic', 'bregman'):
        raise ValueError('Unknown arm')
    weight = config['weight']
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('weight must be finite and nonnegative')
    beta = 0. if arm == 'quadratic' else config['beta']
    lags = tuple(config.get('lags', LAGS))
    if arm == 'mse':
        with torch.no_grad():
            relation, diagnostics = increment_bregman(prediction, target, beta, lags)
    else:
        relation, diagnostics = increment_bregman(prediction, target, beta, lags)
    mse = (prediction - target.detach()).square().mean()
    total = mse if arm == 'mse' else mse + weight * relation
    return dict(loss=total, mse=mse, relation=relation.detach(),
                relation_diagnostics=diagnostics)
