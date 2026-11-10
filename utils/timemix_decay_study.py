"""Isolated decay-mechanism controls; no changes to production defaults."""
import torch
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups

ARMS = ("adam_norm_bias", "adamw_norm_bias", "adam_coarse_path")


def parameter_groups(model, arm, weight_decay=1e-4):
    if arm not in ARMS:
        raise ValueError("Unknown study arm: " + arm)
    groups = norm_bias_no_decay_groups(model, weight_decay)
    if arm == "adam_coarse_path":
        # Protect local coarse encoder/decoder and its incoming 24->12 map.
        # Shared gains and finer-scale encoders retain the reference policy.
        protected = {id(p) for module in (model.seasonals[3],
                     model.season_mix.down_sampling_layers[2]) for p in module.parameters()}
        revised = []
        for group in groups:
            for exempt in (False, True):
                params = [p for p in group["params"] if (id(p) in protected) == exempt]
                if params:
                    revised.append({**group, "params": params,
                                    "weight_decay": 0. if exempt else group["weight_decay"]})
        groups = revised
    ids = [id(p) for g in groups for p in g["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}:
        raise ValueError("Groups must cover every parameter exactly once")
    return groups


def build_optimizer(groups, arm, lr=1e-3):
    if arm not in ARMS:
        raise ValueError("Unknown study arm: " + arm)
    cls = torch.optim.AdamW if arm == "adamw_norm_bias" else torch.optim.Adam
    return cls(groups, lr=lr)


def sample_response_std(output, batch, channels):
    """RMS variation across windows at fixed variable/time; not position variation.

    Seasonal readout hooks return [B*C,H,1], with b0c0,b0c1,... order.
    Compute in double but retain the FP32 forward values. No future targets.
    """
    if output.shape[0] != batch * channels or batch < 2:
        raise ValueError("Need at least two windows and matching flattened channels")
    values = output.detach().double().reshape(batch, channels, -1)
    return float(values.var(dim=0, correction=0).mean().sqrt())
