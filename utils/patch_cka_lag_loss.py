"""Centered kernel alignment and directed lagged patch RKHS covariance.

Inputs are B,H,C forecasts in dataset scaler units. Disjoint length-4 raw
patches keep coordinate and time order. No extra normalization or parameters.
"""
import math
import torch

LAGS = (1, 2)
NORM_FLOOR = 1e-12
METRIC_KEYS = ('cka', 'cka_alignment', 'cka_target_degenerate_fraction',
               'cka_prediction_degenerate_fraction', 'patch_lag') + tuple(
    f'lag{k}_{name}' for k in LAGS for name in
    ('distance', 'prediction_norm2', 'target_norm2', 'cross_inner_product'))


def center(matrix):
    return (matrix - matrix.mean(-1, keepdim=True)
            - matrix.mean(-2, keepdim=True)
            + matrix.mean((-1, -2), keepdim=True))


def gram(a, b, bandwidth):
    return torch.exp(-.5 * ((a.unsqueeze(-2) - b.unsqueeze(-3))
                           / bandwidth).square().sum(-1))


def parts(prediction, target, bandwidth=1., length=4, objective='cka'):
    """Return selected auxiliary and common diagnostics; target is detached.

    CKA uses biased centered full self-Grams, including the diagonal. A target
    Gram with Frobenius norm <=1e-12 carries no alignment information: its loss
    contribution is zero and its frequency is reported. Prediction norm uses
    the same floor. Constant predictions against a nonconstant target have loss
    one but can have zero RBF gradient; the point MSE remains essential.

    C_l = Phi(p[:-l])' J Phi(p[l:]) / (N-l). We compare the FULL operators,
    not just their norms. Their transpose represents reversed transitions;
    reversing both sequences preserves distance, so this is not a causal test.
    """
    if objective not in ('cka', 'patch_lag'):
        raise ValueError('Unknown patch objective')
    if prediction.ndim != 3 or prediction.shape != target.shape or not all(prediction.shape):
        raise ValueError('Expected matching nonempty B,H,C tensors')
    if (prediction.device != target.device or not prediction.is_floating_point()
            or not target.is_floating_point()):
        raise ValueError('Expected floating tensors on the same device')
    if (not isinstance(length, int) or isinstance(length, bool) or length < 1
            or prediction.shape[1] % length or prediction.shape[1] // length <= max(LAGS) + 1):
        raise ValueError('Need at least four complete patches')
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Invalid bandwidth')
    p = prediction.double().transpose(1, 2).unfold(-1, length, length)
    y = target.detach().double().transpose(1, 2).unfold(-1, length, length)
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite patch input')
    pp, yy, py = gram(p, p, bandwidth), gram(y, y, bandwidth), gram(p, y, bandwidth)
    a, b = center(pp), center(yy)
    an2, bn2 = a.square().sum((-1, -2)), b.square().sum((-1, -2))
    usable = bn2 > NORM_FLOOR ** 2
    alignment = (a * b).sum((-1, -2)) / (
        an2.clamp_min(NORM_FLOOR ** 2).sqrt() * bn2.clamp_min(NORM_FLOOR ** 2).sqrt())
    raw_cka = torch.where(usable, 1 - alignment, torch.zeros_like(alignment))
    if not torch.isfinite(raw_cka).all() or raw_cka.min() < -1e-10 or alignment.min() < -1e-10:
        raise RuntimeError('Invalid centered kernel alignment')
    cka = raw_cka.clamp_min(0).mean()
    diag = dict(cka=float(cka.detach()), cka_alignment=float(alignment.detach().mean()),
                cka_target_degenerate_fraction=float((~usable).double().mean()),
                cka_prediction_degenerate_fraction=float((an2 <= NORM_FLOOR ** 2).double().mean()),
                minimum_unclipped_cka=float(raw_cka.detach().min()))
    distances = []
    minimum = 0.
    for lag in LAGS:
        # Center each shortened pair set separately, AFTER slicing the Grams.
        left = (..., slice(None, -lag), slice(None, -lag))
        right = (..., slice(lag, None), slice(lag, None))
        pn = (center(pp[left]) * center(pp[right])).mean((-1, -2))
        yn = (center(yy[left]) * center(yy[right])).mean((-1, -2))
        cross = (center(py[left]) * center(py[right])).mean((-1, -2))
        raw = pn + yn - 2 * cross
        if not torch.isfinite(raw).all() or raw.min() < -1e-10:
            raise RuntimeError('Invalid lagged patch covariance distance')
        distance = raw.clamp_min(0).mean()
        distances.append(distance)
        minimum = min(minimum, float(raw.detach().min()))
        for name, value in [('distance', distance), ('prediction_norm2', pn.mean()),
                            ('target_norm2', yn.mean()), ('cross_inner_product', cross.mean())]:
            diag[f'lag{lag}_{name}'] = float(value.detach())
    patch_lag = torch.stack(distances).mean()
    diag.update(patch_lag=float(patch_lag.detach()), minimum_unclipped_distance=minimum)
    return (cka if objective == 'cka' else patch_lag).to(prediction.dtype), diag
