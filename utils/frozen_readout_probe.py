"""Shared helpers retained for the best blend and latest context probe.
Extracted unchanged from historical diagnostic scripts during cleanup.
"""
import hashlib
import json
import numpy as np
import torch
from scipy.fft import dct
from torch.utils.data import Subset
from data_provider.data_loader import Dataset_ETT_minute
def collect(model, x):
    """Return prediction and [B,H,C,66] design in final standardized units."""
    hidden, hooks = {}, []
    for i in (0, 1):
        layer = model.seasonals[i].decoder.readout.output_projection
        hooks.append(layer.register_forward_pre_hook(
            lambda module, args, i=i: hidden.update({i: args[0].detach()})))
    try:
        pred = model(x)
    finally:
        for hook in hooks:
            hook.remove()
    norm = model.normalize
    assert not (norm.affine or norm.non_norm or norm.subtract_last)
    b, h, c = pred.shape
    designs = []
    for i in (0, 1):
        z = hidden[i].reshape(b, c, h, -1).transpose(1, 2)
        z = torch.cat([z, torch.ones_like(z[..., :1])], -1)
        designs.append(z.double() * norm.stdev.double().unsqueeze(-1))
    return pred, torch.cat(designs, -1)

def metrics(pred, truth):
    p, y = pred.astype(np.float64), truth.astype(np.float64)
    e = p - y
    amp_p = np.mean((p-p.mean(1, keepdims=True))**2, axis=(0,1))
    amp_y = np.mean((y-y.mean(1, keepdims=True))**2, axis=(0,1))
    return dict(mse=float(np.mean(e**2)), mae=float(np.mean(abs(e))),
                low4_mse=float(np.sum(dct(e,axis=1,norm='ortho')[:,:4]**2)/e.size),
                per_channel_mse=np.mean(e**2,axis=(0,1)).tolist(),
                per_channel_mae=np.mean(abs(e),axis=(0,1)).tolist(),
                amplitude_ratio=float(np.sqrt(amp_p.mean()/amp_y.mean())),
                per_channel_amplitude_ratio=np.sqrt(amp_p/amp_y).tolist())

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tensors(ds, ids):
    batch = [ds[int(i)] for i in ids]
    return (torch.tensor(np.stack([v[0] for v in batch]), dtype=torch.float32, device='cuda'),
            torch.tensor(np.stack([v[1][-96:] for v in batch]), dtype=torch.float32, device='cuda'))
