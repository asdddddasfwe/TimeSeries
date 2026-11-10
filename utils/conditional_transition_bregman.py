"""History-conditioned affine transitions and target-first Bregman loss.

No instance standard deviation, covariance inverse, or model normalization change.
The operator is fitted separately and frozen before forecasting training.
"""
import math
from pathlib import Path
import torch
from torch import nn


class TransitionOperator(nn.Module):
    """Separate per-channel histories; shared network, no cross-channel values.

    Coefficients have shape [B,C,R,q], intercepts [B,C,R]. R indexes valid
    (future parent endpoint, lag) pairs. Absolute horizon and channel identity
    are metadata; only the 96 historical values are inputs to coefficients.
    """
    def __init__(self, history=96, horizon=96, channels=7, q=4,
                 lags=(1, 4, 12, 24), hidden=32, coefficient_bound=2.):
        super().__init__()
        self.options = dict(history=history, horizon=horizon, channels=channels,
                            q=q, lags=list(lags), hidden=hidden,
                            coefficient_bound=coefficient_bound)
        self.q, self.coefficient_bound = q, coefficient_bound
        ends = [(t, k) for k in lags for t in range(q-1, horizon-k)]
        if not ends:
            raise ValueError('No valid transitions')
        self.register_buffer('parents', torch.tensor([[t-j for j in range(q)] for t,k in ends]))
        self.register_buffer('destinations', torch.tensor([t+k for t,k in ends]))
        self.register_buffer('queries', torch.tensor([[t/(horizon-1), k/horizon] for t,k in ends]))
        self.register_buffer('channel_codes', torch.eye(channels))
        self.encoder = nn.Sequential(nn.Linear(history, hidden), nn.Tanh(),
                                     nn.Linear(hidden, 16), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(16+2+channels, hidden), nn.Tanh(),
                                     nn.Linear(hidden, q+1))

    def forward(self, history):
        if history.shape[1:] != (self.options['history'], self.options['channels']):
            raise ValueError('Unexpected historical input shape')
        features = self.encoder(history.transpose(1, 2))
        b,c,_ = features.shape
        r = len(self.destinations)
        features = features[:,:,None,:].expand(b,c,r,-1)
        queries = self.queries[None,None].expand(b,c,-1,-1)
        codes = self.channel_codes[None,:,None,:].expand(b,-1,r,-1)
        raw = self.decoder(torch.cat((features,queries,codes),-1))
        # Fixed bound controls the operator norm, without data standardization.
        a = self.coefficient_bound/self.q * torch.tanh(raw[...,:self.q])
        return a, raw[...,self.q]

    def residual(self, sequence, coefficients):
        a,b = coefficients
        z = sequence.transpose(1,2)
        return z[...,self.destinations] - (a*z[...,self.parents]).sum(-1) - b


def logcosh(value):
    return torch.logaddexp(value, -value)-math.log(2.)


def bregman_transition(prediction, target, operator, coefficients, beta=1.):
    """D_psi(r_target,r_prediction); NOT weighted squared error W(p)e^2.

    Double precision prevents cancellation in the logcosh Bregman term;
    inputs/forecast parameters can remain FP32. Target and coefficients stop
    gradient; prediction residual and tanh(prediction residual) must not detach.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    a,b = [v.detach().double() for v in coefficients]
    rp = operator.residual(prediction.double(), (a,b))
    ry = operator.residual(target.detach().double(), (a,b))
    delta = ry-rp
    quadratic = .5*delta.square()
    nonlinear = logcosh(ry)-logcosh(rp)-torch.tanh(rp)*delta
    value = (quadratic+beta*nonlinear).mean()
    if not torch.isfinite(value) or float(value.detach()) < -1e-10:
        raise RuntimeError('Invalid transition Bregman divergence')
    diagnostics = dict(quadratic=float(quadratic.detach().mean()),
        logcosh_divergence=float(nonlinear.detach().mean()),
        prediction_residual_rms=float(rp.detach().square().mean().sqrt()),
        target_residual_rms=float(ry.square().mean().sqrt()),
        curvature_mean=float((1+beta*(1-torch.tanh(rp.detach()).square())).mean()),
        coefficient_abs_mean=float(a.abs().mean()),
        coefficient_l1_max=float(a.abs().sum(-1).max()))
    return value.to(prediction.dtype), diagnostics


_OPERATORS = {}


def load_operator(path, device):
    key = (str(Path(path).resolve()), str(device))
    if key not in _OPERATORS:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = TransitionOperator(**payload['options'])
        model.load_state_dict(payload['state_dict'], strict=True)
        model.to(device).eval().requires_grad_(False)
        _OPERATORS[key] = model
    return _OPERATORS[key]


def objective(prediction, target, history, arm, config):
    if arm not in ('mse','quadratic','bregman'):
        raise ValueError('Unknown arm')
    op = load_operator(config['operator_checkpoint'], prediction.device)
    with torch.no_grad():
        coefficients = op(history.detach())
    beta = 0. if arm == 'quadratic' else config['beta']
    if arm == 'mse':
        with torch.no_grad():
            relation, diagnostics = bregman_transition(prediction,target,op,coefficients,beta)
    else:
        relation, diagnostics = bregman_transition(prediction,target,op,coefficients,beta)
    mse = (prediction-target.detach()).square().mean()
    total = mse if arm == 'mse' else mse+config['weight']*relation
    return dict(loss=total, mse=mse, relation=relation.detach(),
                relation_diagnostics=diagnostics)
