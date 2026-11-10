"""Convex-potential loss for nonlinear, history-conditioned temporal relations.

g(v) = b + a'v + beta'tanh(Wv), with fixed coordinate/adjacent-sum bases.
Only local Bayesian coefficients are estimated, with no auxiliary optimizer.
F(v, d) = hypot(1, d-g(v))-1 + C(theta)*||v||^2/2 is convex, where
C = 4/(3*sqrt(3))*sum_j |beta_j|*||W_j||^2. The score is D_F(y, p),
NOT the squared difference between nonlinear residuals.

The positive-weight finite posterior quadrature defines a convex potential in
its own right; it is not an exact posterior integral, NLL, or calibrated model.
MSE + weight*D has the conditional mean as its unique unrestricted optimum.
That statement does not guarantee convexity in prediction or training gains.

Inputs [batch, time, channel] remain in the existing StandardScaler units.
History and labels are detached; all prediction tangent derivatives are kept.
No changes to forecasting models, normalization, or existing experiment runners.
"""
from functools import lru_cache
import math

import torch


ARMS = ('mse', 'compensation', 'nonlinear')


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer')


def _sequence(value, order):
    _positive_integer(order, 'order')
    if (value.ndim != 3 or min(value.shape) < 1 or value.shape[1] <= order
            or value.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Expected nonempty FP32/FP64 [B,T,C] with T > order')
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite sequence')


def fixed_basis(order=4, device=None):
    """[2*q-1,q] fixed unit directions; zero offsets and no random draws."""
    _positive_integer(order, 'order')
    eye = torch.eye(order, dtype=torch.float64, device=device)
    return torch.cat((eye, (eye[:-1] + eye[1:]) / math.sqrt(2.)), dim=0)


def features(sequence, order=4):
    """Return d [B,C,T-q], v [B,C,T-q,q], phi [B,C,T-q,3*q].

    Coefficient order is intercept, linear parents newest first, tanh bases.
    """
    _sequence(sequence, order)
    z = sequence.double().transpose(1, 2)
    parents = torch.stack([z[..., order-j:z.shape[-1]-j]
                           for j in range(1, order+1)], dim=-1)
    nonlinear = torch.tanh(parents @ fixed_basis(order, z.device).T)
    design = torch.cat((torch.ones_like(parents[..., :1]), parents, nonlinear), -1)
    return z[..., order:], parents, design


@torch.no_grad()
def posterior(history, order=4, noise_variance=1., prior_variance=1.):
    """FP64 Gaussian regression posterior; history only; no inverse or jitter."""
    if any(not math.isfinite(v) or v <= 0 for v in (noise_variance, prior_variance)):
        raise ValueError('Variances must be finite and positive')
    d, _, design = features(history.detach(), order)
    dimension = design.shape[-1]
    eye = torch.eye(dimension, dtype=design.dtype, device=design.device)
    eye = eye.expand(*design.shape[:2], dimension, dimension)
    precision = design.transpose(-1, -2) @ design / noise_variance + eye / prior_variance
    factor = torch.linalg.cholesky(precision)
    rhs = (design.transpose(-1, -2) @ d.unsqueeze(-1)) / noise_variance
    mean = torch.cholesky_solve(rhs, factor).squeeze(-1)
    covariance = torch.cholesky_solve(eye, factor)
    covariance = (covariance + covariance.transpose(-1, -2)) * .5
    chol = torch.linalg.cholesky(covariance)
    if not all(torch.isfinite(v).all() for v in (mean, covariance, chol)):
        raise RuntimeError('Nonfinite nonlinear Bayesian posterior')
    return mean, covariance, chol


@lru_cache(maxsize=16)
def _normal_nodes(dimension, count, seed):
    _positive_integer(dimension, 'dimension')
    _positive_integer(count, 'count')
    if count < 2 or count % 2 or (count // 2) & (count // 2 - 1):
        raise ValueError('Node count must be twice a power of two')
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
    uniform = engine.draw(count // 2, dtype=torch.float64)
    nodes = math.sqrt(2.) * torch.erfinv(2 * uniform - 1)
    if not torch.isfinite(nodes).all():
        raise RuntimeError('Nonfinite quadrature nodes')
    return torch.cat((nodes, -nodes), dim=0)


def normal_nodes(dimension=12, count=64, seed=20261010):
    """A private fixed Sobol RNG; clone prevents callers corrupting the cache."""
    return _normal_nodes(dimension, count, seed).clone()


def _local_terms(pd, pv, pf, yd, yv, yf, coefficients, basis):
    """Unreduced [B,C,S,R] terms, with exact analytic prediction tangents."""
    order = pv.shape[-1]
    gp = torch.einsum('bcrd,bcsd->bcsr', pf, coefficients)
    gy = torch.einsum('bcrd,bcsd->bcsr', yf, coefficients)
    rp, ry = pd.unsqueeze(-2) - gp, yd.unsqueeze(-2) - gy
    sp = torch.hypot(torch.ones_like(rp), rp)
    sy = torch.hypot(torch.ones_like(ry), ry)
    # Stable scalar pseudo-Huber Bregman divergence via unit-vector distance.
    # Equals rho(ry)-rho(rp)-rho'(rp)*(ry-rp), avoiding large subtractions.
    scalar = .5 * sy * ((ry / sy - rp / sp).square()
                        + (sy.reciprocal() - sp.reciprocal()).square())
    delta = yv - pv
    beta = coefficients[..., 1+order:]
    projection = delta @ basis.T
    h_p, h_y = pf[..., 1+order:], yf[..., 1+order:]
    # The affine part has zero Taylor remainder. Keep h_p's derivative.
    remainder = h_y - h_p - (1 - h_p.square()) * projection
    tangent = -(rp / sp) * torch.einsum('bcrj,bcsj->bcsr', remainder, beta)
    curvature = (4. / (3. * math.sqrt(3.))) * (
        beta.abs() * basis.square().sum(-1)).sum(-1)
    compensation = .5 * curvature.unsqueeze(-1) * delta.square().sum(-1).unsqueeze(-2)
    return scalar, tangent, compensation, curvature


def score_from_coefficients(prediction, target, coefficients, order=4, chunk_size=16):
    """Average D_F(Y,p) over fixed equal-weight coefficients [B,C,S,3*q].

    This public primitive also supports offline checks of fixed coefficients.
    Coefficients are detached. Returned score retains FP64 for numerical audit.
    Only the total divergence is nonnegative; the tangent term can be negative.
    """
    _positive_integer(chunk_size, 'chunk_size')
    pd, pv, pf = features(prediction, order)
    if prediction.shape != target.shape or prediction.device != target.device:
        raise ValueError('Prediction and target must have matching shapes and devices')
    yd, yv, yf = features(target.detach(), order)
    if (coefficients.ndim != 4 or coefficients.shape[:2] != pd.shape[:2]
            or coefficients.shape[-1] != pf.shape[-1] or coefficients.shape[-2] < 1
            or coefficients.device != prediction.device):
        raise ValueError('Expected coefficients [B,C,S,3*order] on prediction device')
    coefficients = coefficients.detach().double()
    if not torch.isfinite(coefficients).all():
        raise ValueError('Nonfinite coefficients')
    basis = fixed_basis(order, prediction.device)
    parts = [pd.new_zeros(()) for _ in range(4)]
    minimum = float('inf')
    for sample in coefficients.split(chunk_size, dim=-2):
        scalar, tangent, compensation, curvature = _local_terms(
            pd, pv, pf, yd, yv, yf, sample, basis)
        divergence = scalar + tangent + compensation
        scale = scalar.abs() + tangent.abs() + compensation.abs()
        if (not torch.isfinite(divergence).all()
                or (divergence < -1e-10 * (1 + scale)).any()):
            raise RuntimeError('Invalid nonlinear relation divergence')
        minimum = min(minimum, float(divergence.detach().min()))
        fraction = sample.shape[-2] / coefficients.shape[-2]
        for i, term in enumerate((scalar, tangent, compensation, curvature)):
            parts[i] = parts[i] + term.mean() * fraction
    value = parts[0] + parts[1] + parts[2]
    return value, dict(scalar_bregman=parts[0], tangent_correction=parts[1],
                      curvature_compensation=parts[2], curvature_mean=parts[3],
                      minimum_local_divergence=minimum, relation_count=pd.shape[-1],
                      quadrature_nodes=coefficients.shape[-2])


def relation(prediction, target, post, order=4, nodes=64, chunk_size=16, seed=20261010):
    """Finite posterior expectation; separate compensation-only control available."""
    mean, covariance, chol = (v.detach().double() for v in post)
    expected = (prediction.shape[0], prediction.shape[2], 3 * order)
    if (mean.shape != expected or covariance.shape != (*expected, expected[-1])
            or chol.shape != covariance.shape
            or any(v.device != prediction.device for v in (mean, covariance, chol))):
        raise ValueError('Posterior dimensions/device do not match prediction')
    if not all(torch.isfinite(v).all() for v in (mean, covariance, chol)):
        raise ValueError('Nonfinite posterior')
    eps = normal_nodes(mean.shape[-1], nodes, seed).to(prediction.device)
    coefficients = mean.unsqueeze(-2) + torch.einsum('bcij,sj->bcsi', chol, eps)
    value, parts = score_from_coefficients(prediction, target, coefficients, order, chunk_size)
    parts.update(posterior_variance_mean=float(covariance.diagonal(dim1=-2, dim2=-1).mean()),
                 posterior_cholesky_min=float(chol.diagonal(dim1=-2, dim2=-1).min()))
    return value, parts


def objective(prediction, target, history, arm='nonlinear', config=None):
    """Existing trainer-compatible dict: loss/mse/relation/relation_diagnostics.

    Example: objective(p, y, x, 'nonlinear', {'weight': .1})['loss'].backward().
    'compensation' uses only the IDENTICAL posterior curvature penalty, so a
    later experiment can distinguish nonlinear relationships from reweighting.
    'mse' uses pure MSE, recording the same full score under no_grad.
    Weight .1 is a starting default, not an empirically selected optimum.
    """
    config = {} if config is None else config
    if arm not in ARMS:
        raise ValueError('Unknown nonlinear relation arm')
    weight = config.get('weight', .1)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Weight must be finite and nonnegative')
    order = config.get('order', 4)
    _sequence(prediction, order)
    _sequence(history, order)
    if history.shape[0] != prediction.shape[0] or history.shape[2] != prediction.shape[2]:
        raise ValueError('History batch/channel dimensions must match prediction')
    if history.device != prediction.device:
        raise ValueError('History and prediction devices must match')
    post = posterior(history, order, config.get('noise_variance', 1.), config.get('prior_variance', 1.))
    args = (prediction, target, post, order, config.get('nodes', 64),
            config.get('chunk_size', 16), config.get('seed', 20261010))
    if arm == 'mse' or weight == 0:
        with torch.no_grad():
            value, parts = relation(*args)
    else:
        value, parts = relation(*args)
    used = parts['curvature_compensation'] if arm == 'compensation' else value
    mse = (prediction - target.detach()).square().mean()
    loss = mse if arm == 'mse' or weight == 0 else mse + weight * used.to(prediction.dtype)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite nonlinear relation objective')
    diagnostics = {k: float(v.detach()) if torch.is_tensor(v) else v for k, v in parts.items()}
    diagnostics.update(weight=weight, applied_weight=0. if arm == 'mse' else weight,
                       used_relation=float(used.detach()), full_divergence=float(value.detach()))
    return dict(loss=loss, mse=mse, relation=value.detach().to(prediction.dtype),
                relation_diagnostics=diagnostics)
