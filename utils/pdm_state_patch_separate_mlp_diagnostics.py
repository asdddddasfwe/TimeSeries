"""Same-forward trend head energies and separate MSE/D gradients."""
import math
import torch


class FeatureStatistics:
    def __init__(self, model):
        self.totals, self.handles = {}, []
        for i, head in enumerate(model.feature_trend_heads):
            key = str(i)
            def hook(module, args, output, key=key):
                with torch.no_grad():
                    x, y = args[0].detach().double(), output.detach().double()
                    assert torch.isfinite(x).all() and torch.isfinite(y).all()
                    d = y.diff(dim=-1)
                    boundary = torch.arange(d.shape[-1], device=d.device)%4 == 3
                    row = self.totals.setdefault(key, dict(input_count=0, output_count=0,
                        input_square=0., prediction_square=0., prediction_sum=0.,
                        boundary_square=0., boundary_count=0, interior_square=0., interior_count=0))
                    row['input_count'] += x.numel(); row['output_count'] += y.numel()
                    row['input_square'] += float(x.square().sum())
                    row['prediction_square'] += float(y.square().sum())
                    row['prediction_sum'] += float(y.sum())
                    for label, mask in [('boundary', boundary), ('interior', ~boundary)]:
                        row[label+'_square'] += float(d[...,mask].square().sum())
                        row[label+'_count'] += d[...,mask].numel()
            self.handles.append(head.register_forward_hook(hook))

    def close(self):
        for handle in self.handles: handle.remove()

    def summary(self):
        return {k: dict(input_count=v['input_count'], output_count=v['output_count'],
            input_rms=math.sqrt(v['input_square']/v['input_count']),
            prediction_rms=math.sqrt(v['prediction_square']/v['output_count']),
            prediction_mean=v['prediction_sum']/v['output_count'],
            boundary_increment_rms=math.sqrt(v['boundary_square']/v['boundary_count']),
            interior_increment_rms=math.sqrt(v['interior_square']/v['interior_count'])) for k,v in self.totals.items()}


def gradient_probe(model, mse, distance):
    named = list(model.named_parameters())
    gm = torch.autograd.grad(mse, [p for _, p in named], retain_graph=True, allow_unused=True)
    gd = torch.autograd.grad(distance, [p for _, p in named], retain_graph=True, allow_unused=True)
    scopes = {'season_feature_mlp': lambda n: '.out_cross_layer.' in n, 'trend_feature_mlp': lambda n: '.trend_out_cross_layer.' in n, 'all': lambda n: True, 'trend_heads': lambda n: n.startswith('feature_trend_heads.'), 'trend_temporal': lambda n: n.startswith('feature_trend_heads.') and '.temporal.' in n, 'trend_readout': lambda n: n.startswith('feature_trend_heads.') and '.readout.' in n, 'readout_hidden': lambda n: '.readout.input_projection.' in n,
        'state_patch_heads': lambda n: '.readout.output_projection.' in n,
        'trend_condition': lambda n: '.trend_condition.' in n,
        'fine_trend_head': lambda n: n.startswith('feature_trend_heads.0.'),
        'future_projection': lambda n: '.temporal_projection.' in n,
        'history_attention': lambda n: '.blocks.' in n,
        'pdm': lambda n: n.startswith('pdm_blocks.')}
    for scale in range(4):
        scopes['trend_scale'+str(scale)] = lambda n, s=scale: n.startswith(f'feature_trend_heads.{s}.')
    result = {}
    for label, include in scopes.items():
        mm = dd = dot = 0.; count = 0
        for (name, _), a, b in zip(named, gm, gd):
            if not include(name):
                continue
            count += 1
            if a is not None:
                assert torch.isfinite(a).all()
                mm += float(a.detach().double().square().sum())
            if b is not None:
                assert torch.isfinite(b).all()
                dd += float(b.detach().double().square().sum())
            if a is not None and b is not None:
                dot += float((a.detach().double()*b.detach().double()).sum())
        result[label] = dict(parameter_tensors=count, mse_norm=math.sqrt(mm), d_norm=math.sqrt(dd),
            weighted_ratio=math.sqrt(dd)/max(math.sqrt(mm), 1e-30), cosine=dot/max(math.sqrt(mm*dd), 1e-30))
    return result
