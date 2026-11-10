"""Experimental optimizer regrouping; model defaults remain unchanged."""
from torch import nn


def norm_bias_no_decay_groups(model, weight_decay):
    """Preserve model groups, exempt LayerNorm parameters and named biases.

    Existing dynamics exclusions remain intact. Gain offsets and attention
    residual gates are not biases and retain their original decay setting.
    """
    exempt = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if isinstance(module, nn.LayerNorm) or name == 'bias' or name.endswith('_bias'):
                exempt.add(id(parameter))
    groups = []
    for original in model.optimizer_parameter_groups(weight_decay):
        if original['weight_decay'] == 0:
            groups.append(dict(original))
            continue
        regular = [p for p in original['params'] if id(p) not in exempt]
        protected = [p for p in original['params'] if id(p) in exempt]
        if regular:
            groups.append({**original, 'params': regular})
        if protected:
            groups.append({**original, 'params': protected, 'weight_decay': 0.,
                           'group_name': original['group_name'] + '_norm_bias_no_decay'})
    actual = [id(p) for group in groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != {id(p) for p in model.parameters()}:
        raise ValueError('Regrouping must cover every model parameter exactly once')
    return groups
