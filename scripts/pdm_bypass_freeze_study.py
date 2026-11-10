"""Isolate raw-trend co-adaptation without changing forecast equations."""
import copy
import torch
from utils.timemix_decay_study import parameter_groups as old_groups
from scripts.train_timemix_pdm_trend_smooth_residual import initialize_pair as old_initialize
from models.OscillatorPDMReadoutSplit import Model as ReadoutSplit
from models.OscillatorPDMTrendSmoothResidual import Model as SmoothResidual

ARMS = ('seasonal_trend', 'smooth_residual', 'seasonal_trend_fixed', 'smooth_residual_fixed')
PREFIXES = ('raw_trend_mix.', 'raw_trend_heads.')

def targeted(name):
    return name.startswith(PREFIXES)

def configure(model, arm):
    if arm not in ARMS:
        raise ValueError('Unknown bypass-freeze arm')
    # Non-affine normalization and parameter-free raw decomposition ensure that
    # fixing these parameters fixes this function of a given observed history.
    if model.normalize.affine or list(model.raw_trend_decomposition.parameters()):
        raise ValueError('Raw bypass has unexpected trainable upstream state')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not (arm.endswith('_fixed') and targeted(name)))
    return model

def build_model(cfg, arm):
    cls = SmoothResidual if arm.startswith('smooth_residual') else ReadoutSplit
    model = cls(cfg) if cls is SmoothResidual else cls(cfg, variant='seasonal')
    return configure(model, arm)

def initialize_pair(cfg):
    old = old_initialize(cfg)
    models = {arm: old[arm] for arm in ARMS[:2]}
    for arm in ARMS[2:]:
        models[arm] = copy.deepcopy(models[arm.removesuffix('_fixed')])
    for arm, model in models.items():
        configure(model, arm)
        for name, value in models[ARMS[0]].state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    return models

def parameter_groups(model, arm):
    if arm not in ARMS:
        raise ValueError('Unknown bypass-freeze arm')
    groups = old_groups(model, 'adam_norm_bias')
    groups = [{**g, 'params': [p for p in g['params'] if p.requires_grad]} for g in groups]
    groups = [g for g in groups if g['params']]
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    return groups

def raw_state(model):
    return {n: p.detach().clone() for n,p in model.named_parameters() if targeted(n)}

def bypass_audit(model, initial, fixed):
    distance = 0.; grad_square = 0.; count = 0
    for name, p in model.named_parameters():
        if not targeted(name):
            continue
        if fixed and (p.requires_grad or p.grad is not None or not torch.equal(p.detach(), initial[name])):
            raise RuntimeError('Frozen raw bypass changed or received a gradient: ' + name)
        distance += float((p.detach().double()-initial[name].double()).square().sum())
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise RuntimeError('Nonfinite raw bypass gradient')
            grad_square += float(p.grad.detach().double().square().sum())
        count += p.numel()
    return dict(fixed=fixed, parameter_count=count, distance_from_initial=distance**.5,
                gradient_norm=grad_square**.5)

class PredictionGroupCapture:
    """Observe the existing component forward only; never invoke it twice."""
    def __init__(self, model):
        self.model = model
        self.original = model.predict_components
        self.summary = None
    def __enter__(self):
        def wrapped(x):
            parts = self.original(x)
            with torch.no_grad():
                # Components here omit history mean, as in saved validation captures.
                groups = torch.stack([parts[...,i:i+4].detach().double().sum(-1)
                                      for i in (0,4,8)], -1)
                groups = groups * self.model.normalize.stdev.detach().double()[...,None]
                energy = groups.square().mean((0,1))
                self.summary = dict(order=['recurrent','feature_trend','raw_trend'],
                    rms_per_channel=energy.sqrt().cpu().tolist(),
                    temporal_mean_per_channel=groups.mean((0,1)).cpu().tolist(),
                    temporal_mean_rms_per_channel=groups.mean(1).square().mean(0).sqrt().cpu().tolist())
            return parts
        self.model.predict_components = wrapped
        return self
    def __exit__(self, *args):
        # Restore class method dispatch, avoiding a permanent instance override.
        del self.model.predict_components
        self.model = None
        self.original = None
