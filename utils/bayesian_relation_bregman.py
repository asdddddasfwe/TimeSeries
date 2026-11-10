"""History-only Bayesian AR(4) relation score, not a predictive NLL.

Gaussian posterior is analytic. Quadratic posterior expectation is exact;
logcosh expectation uses fixed antithetic Sobol normal quadrature (64 nodes).
No learned auxiliary network, extra normalization, or future-label posterior.
"""
from functools import lru_cache
import math
import torch


ARMS = ('mse', 'mean', 'quadratic', 'bregman')


def features(sequence, order=4):
    """[B,T,C] -> destinations [B,C,T-q], parents+intercept [B,C,T-q,q+1]."""
    if sequence.ndim != 3 or sequence.shape[1] <= order or order < 1:
        raise ValueError('Invalid sequence shape or AR order')
    z = sequence.transpose(1, 2)
    h = z.shape[-1]
    parents = torch.stack([z[..., order-j:h-j] for j in range(1, order+1)], -1)
    return z[..., order:], torch.cat((parents, torch.ones_like(parents[..., :1])), -1)


@torch.no_grad()
def posterior(history, order=4, noise_variance=1., prior_variance=1.):
    """Zero prior mean; variance in existing StandardScaler units, no rescaling."""
    if any(not math.isfinite(v) or v <= 0 for v in (noise_variance, prior_variance)):
        raise ValueError('Prior and noise variances must be finite and positive')
    d, z = features(history.detach().double(), order)
    eye = torch.eye(order+1, dtype=z.dtype, device=z.device).expand(*z.shape[:2], -1, -1)
    precision = z.transpose(-1, -2) @ z / noise_variance + eye / prior_variance
    chol_precision = torch.linalg.cholesky(precision)
    rhs = (z.transpose(-1, -2) @ d.unsqueeze(-1)) / noise_variance
    mean = torch.cholesky_solve(rhs, chol_precision).squeeze(-1)
    covariance = torch.cholesky_solve(eye, chol_precision)
    covariance = (covariance + covariance.transpose(-1, -2)) * .5
    chol_covariance = torch.linalg.cholesky(covariance)
    if not all(torch.isfinite(v).all() for v in (mean, covariance, chol_covariance)):
        raise RuntimeError('Nonfinite Bayesian posterior')
    return mean, covariance, chol_covariance


@lru_cache(maxsize=16)
def normal_nodes(dimension=5, count=64):
    """Fixed local Sobol RNG, independent of model/dropout/global RNG state."""
    if count < 2 or count % 2 or (count//2) & (count//2-1):
        raise ValueError('Node count must be twice a power of two')
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=20211009)
    uniform = engine.draw(count//2, dtype=torch.double)
    nodes = math.sqrt(2.) * torch.erfinv(2*uniform-1)
    if not torch.isfinite(nodes).all():
        raise RuntimeError('Nonfinite fixed quadrature nodes')
    return torch.cat((nodes, -nodes), 0)


def relation(prediction, target, post, mode='bregman', order=4, nodes=64):
    if mode not in ('mean', 'quadratic', 'bregman'):
        raise ValueError('Unknown relation mode')
    if prediction.shape != target.shape or prediction.device != target.device:
        raise ValueError('Prediction and target must match')
    mean, cov, chol = (v.detach() for v in post)
    pd, pf = features(prediction.double(), order)
    yd, yf = features(target.detach().double(), order)
    ed, ef = pd-yd, pf-yf
    re = ed - (ef*mean.unsqueeze(-2)).sum(-1)
    mean_quadratic = .5*re.square().mean()
    uncertainty = .5*torch.einsum('bcrd,bcde,bcre->bcr', ef, cov, ef).mean()
    nonlinear = prediction.double().new_zeros(())
    if mode == 'bregman':
        eps = normal_nodes(order+1, nodes).to(device=prediction.device)
        theta = mean.unsqueeze(-2) + torch.einsum('bcij,sj->bcsi', chol, eps)
        # Chunk posterior samples. No random draws or detach in prediction path.
        for sample in theta.split(16, dim=-2):
            rp = pd.unsqueeze(-2) - torch.einsum('bcrd,bcsd->bcsr', pf, sample)
            ry = yd.unsqueeze(-2) - torch.einsum('bcrd,bcsd->bcsr', yf, sample)
            term = (torch.logaddexp(ry, -ry) - torch.logaddexp(rp, -rp)
                    - torch.tanh(rp)*(ry-rp))
            nonlinear = nonlinear + term.mean() * (sample.shape[-2]/nodes)
    value = mean_quadratic
    if mode != 'mean':
        value = value + uncertainty
    if mode == 'bregman':
        value = value + nonlinear
    if not torch.isfinite(value) or float(value.detach()) < -1e-10:
        raise RuntimeError('Invalid Bayesian relation divergence')
    diagnostics = dict(mean_quadratic=float(mean_quadratic.detach()),
        uncertainty_quadratic=float(uncertainty.detach()), logcosh_divergence=float(nonlinear.detach()),
        posterior_coefficient_abs_mean=float(mean[..., :order].abs().mean()),
        posterior_coefficient_variance_mean=float(cov.diagonal(dim1=-2,dim2=-1)[..., :order].mean()),
        posterior_cholesky_min=float(chol.diagonal(dim1=-2,dim2=-1).min()),
        relation_count=pd.shape[-1], quadrature_nodes=nodes if mode=='bregman' else 0)
    return value.to(prediction.dtype), diagnostics


def objective(prediction, target, history, arm, config):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    weight = config['weight']
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Invalid relation weight')
    order = config.get('order', 4)
    post = posterior(history, order, config.get('noise_variance', 1.), config.get('prior_variance', 1.))
    mode = 'bregman' if arm == 'mse' else arm
    if arm == 'mse':
        with torch.no_grad():
            value, diagnostics = relation(prediction, target, post, mode, order, config.get('nodes', 64))
    else:
        value, diagnostics = relation(prediction, target, post, mode, order, config.get('nodes', 64))
    mse = (prediction-target.detach()).square().mean()
    loss = mse if arm == 'mse' else mse + weight*value
    return dict(loss=loss, mse=mse, relation=value.detach(), relation_diagnostics=diagnostics)
