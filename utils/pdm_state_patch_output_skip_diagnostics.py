"""Same-forward energies and gradients for independent mean-patch output skip."""
import math
import torch
from torch.nn import functional as F


class FeatureStatistics:
    def __init__(self, model):
        self.totals, self.pending, self.handles = {}, {}, []
        for i, encoder in enumerate(model.seasonals):
            key = str(i)
            def projection_hook(module, args, output, key=key, encoder=encoder):
                with torch.no_grad():
                    future = output.detach().transpose(1, 2)
                    future = future + encoder.future_positions.to(future)
                    n, h, d = future.shape
                    self.pending[key] = {'input': future.reshape(n, h // 4, 4, d).mean(2)}
            def activation_hook(module, args, output, key=key):
                self.pending[key]['hidden'] = output.detach()
            def readout_hook(module, args, output, key=key):
                with torch.no_grad():
                    cached = self.pending.pop(key)
                    x, r = cached['input'].double(), cached['hidden'].double()
                    active = len(args) == 2
                    if active:
                        assert torch.equal(cached['input'], args[1].detach())
                    z = r  # hidden features are never fused
                    w, bias = module.output_projection.weight.detach().double(), module.output_projection.bias.detach().double()
                    recurrent = F.linear(r, w, bias)
                    bypass = F.linear(x, module.skip_projection.weight.detach().double()) if active else torch.zeros_like(recurrent)
                    predicted = output.detach().double()
                    assert all(torch.isfinite(v).all() for v in (x, r, z, recurrent, bypass, predicted))
                    row = self.totals.setdefault(key, dict(count=0, output_count=0, input_square=0.,
                        recurrence_square=0., combined_square=0., inner_product=0., recurrent_prediction_square=0.,
                        bypass_prediction_square=0., prediction_square=0., prediction_inner_product=0., reconstruction_max_abs=0., skip_active=active, skip_weight_norm=0.))
                    row['count'] += x.numel()
                    row['output_count'] += predicted.numel()
                    for name, value in [('input_square', x.square().sum()), ('recurrence_square', r.square().sum()),
                        ('combined_square', z.square().sum()), ('inner_product', (x*r).sum()),
                        ('recurrent_prediction_square', recurrent.square().sum()),
                        ('bypass_prediction_square', bypass.square().sum()), ('prediction_square', predicted.square().sum()), ('prediction_inner_product',(recurrent*bypass).sum())]:
                        row[name] += float(value)
                    row['skip_weight_norm'] = float(module.skip_projection.weight.detach().double().norm()) if active else 0.
                    row['reconstruction_max_abs'] = max(row['reconstruction_max_abs'], float((recurrent+bypass-predicted).abs().max()))
            self.handles.extend([encoder.temporal_projection.register_forward_hook(projection_hook),
                encoder.decoder.readout.activation.register_forward_hook(activation_hook),
                encoder.decoder.readout.register_forward_hook(readout_hook)])

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.pending.clear()

    def summary(self):
        result = {}
        for key, row in self.totals.items():
            x, r, z = [math.sqrt(row[name]/row['count']) for name in ('input_square', 'recurrence_square', 'combined_square')]
            result[key] = dict(count=row['count'], output_count=row['output_count'], skip_active=row['skip_active'],
                input_rms=x, recurrence_rms=r, recurrent_hidden_rms=z, recurrence_to_input_rms=r/max(x, 1e-30),
                input_recurrence_cosine=row['inner_product']/max(math.sqrt(row['input_square']*row['recurrence_square']), 1e-30),
                recurrent_prediction_rms=math.sqrt(row['recurrent_prediction_square']/row['output_count']),
                bypass_prediction_rms=math.sqrt(row['bypass_prediction_square']/row['output_count']),
                prediction_rms=math.sqrt(row['prediction_square']/row['output_count']),
                reconstruction_max_abs=row['reconstruction_max_abs'], skip_weight_norm=row['skip_weight_norm'],
                projected_path_cosine=row['prediction_inner_product']/max(math.sqrt(row['recurrent_prediction_square']*row['bypass_prediction_square']),1e-30),
                bypass_to_recurrent_prediction_rms=math.sqrt(row['bypass_prediction_square']/max(row['recurrent_prediction_square'],1e-30)))
        return result


def gradient_probe(model, mse, distance):
    named = list(model.named_parameters())
    gm = torch.autograd.grad(mse, [p for _, p in named], retain_graph=True, allow_unused=True)
    gd = torch.autograd.grad(distance, [p for _, p in named], retain_graph=True, allow_unused=True)
    scopes = {'all': lambda n: True, 'output_skip': lambda n: '.skip_projection.' in n, 'readout_hidden': lambda n: '.readout.input_projection.' in n,
        'state_patch_heads': lambda n: '.readout.output_projection.' in n,
        'trend_condition': lambda n: '.trend_condition.' in n,
        'fine_trend_head': lambda n: n.startswith('feature_trend_heads.0.'),
        'future_projection': lambda n: '.temporal_projection.' in n,
        'history_attention': lambda n: '.blocks.' in n,
        'pdm': lambda n: n.startswith('pdm_blocks.')}
    for scale in range(4):
        scopes['output_skip_scale'+str(scale)] = lambda n, s=scale: n == f'seasonals.{s}.decoder.readout.skip_projection.weight'
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
