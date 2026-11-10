"""Same-forward trend transforms plus inherited twelve-component diagnostics."""
import math
import torch
from utils.pdm_readout_split_diagnostics import ReadoutSplitCapture


def prelinear_parameter_audit(model):
    result = {}
    for i, block in enumerate(model.pdm_blocks):
        for j, linear in enumerate(getattr(block.mixing_multi_scale_trend, 'pre_linears', [])):
            identity = torch.eye(linear.in_features, device=linear.weight.device)
            result[f'layer{i}_length{linear.in_features}'] = dict(
                identity_distance=float((linear.weight.detach()-identity).norm()),
                weight_norm=float(linear.weight.detach().norm()),
                bias_norm=float(linear.bias.detach().norm()),
                weight_gradient_norm=float(linear.weight.grad.norm()) if linear.weight.grad is not None else None,
                bias_gradient_norm=float(linear.bias.grad.norm()) if linear.bias.grad is not None else None)
    return result


class SelectedValidationCapture(ReadoutSplitCapture):
    def __init__(self, *args):
        super().__init__(*args)
        self.transforms = {}
        self.parameters = prelinear_parameter_audit(self.model)

    def __enter__(self):
        super().__enter__()
        for i, block in enumerate(self.model.pdm_blocks):
            for linear in getattr(block.mixing_multi_scale_trend, 'pre_linears', []):
                key = f'layer{i}_length{linear.in_features}'
                self.handles.append(linear.register_forward_hook(
                    lambda module, args, value, key=key: self.transformed(key, args[0], value)))
        return self

    def transformed(self, key, before, after):
        row = self.transforms.setdefault(key, dict(elements=0, diff_elements=0,
            before_square=0., after_square=0., change_square=0., before_diff_square=0., after_diff_square=0.))
        before, after = before.detach().double(), after.detach().double()
        if not torch.isfinite(after).all():
            raise RuntimeError('Nonfinite pre-fusion trend')
        row['elements'] += before.numel()
        row['change_square'] += float((after-before).square().sum())
        row['diff_elements'] += before[..., 2:].numel()
        for name, value in [('before', before), ('after', after)]:
            row[name+'_square'] += float(value.square().sum())
            difference = value[..., 2:] - 2*value[..., 1:-1] + value[..., :-2]
            row[name+'_diff_square'] += float(difference.square().sum())

    def report(self, prediction, truth):
        result = super().report(prediction, truth)
        result['prelinear_parameters'] = self.parameters
        result['prelinear_transforms'] = {k: {
            **{name+'_rms': math.sqrt(v[name+'_square']/v['elements']) for name in ['before','after','change']},
            **{name+'_second_difference_rms': math.sqrt(v[name+'_diff_square']/v['diff_elements']) for name in ['before','after']}
        } for k,v in self.transforms.items()}
        return result
