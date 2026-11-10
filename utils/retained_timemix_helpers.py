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
def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def dataset(cfg, flag, limit):
    if flag not in ('train', 'val'):
        raise ValueError('This study never opens the test split')
    data = Dataset_ETT_minute(cfg, cfg.root_path, flag=flag, size=[96, 0, 96],
                             features='M', data_path='ETTm1.csv', timeenc=1, freq='h')
    if limit:
        ids = np.unique(np.linspace(0, len(data)-1, min(limit, len(data)), dtype=int)).tolist()
        return Subset(data, ids)
    return data

def lowpass(x, modes=4):
    """Orthogonal DCT-II projection along forecast time, including the DC mode."""
    length = x.shape[1]
    k = np.arange(min(modes, length))
    basis = np.cos(np.pi / length * (np.arange(length)[:, None] + .5) * k)
    basis *= np.sqrt(2 / length)
    basis[:, 0] /= np.sqrt(2)
    return np.einsum('hk,nkc->nhc', basis, np.einsum('hk,nhc->nkc', basis, x))

def metrics(pred, truth):
    error = pred - truth
    mse = np.mean(error ** 2)
    pc, yc = pred - pred.mean(1, keepdims=True), truth - truth.mean(1, keepdims=True)
    corr = np.sum(pc * yc, axis=1) / np.maximum(np.sqrt(np.sum(pc**2, axis=1) * np.sum(yc**2, axis=1)), 1e-12)
    valid = (np.std(truth, axis=1) > 1e-5) & (np.std(pred, axis=1) > 1e-5)
    return dict(mse=float(mse), mae=float(np.mean(np.abs(error))),
        mean_bias=float(np.mean(error)),
        window_offset_mse=float(np.mean(error.mean(1)**2)),
        low_frequency_error_share={str(k):float(np.mean(lowpass(error,k)**2)/max(mse,1e-12)) for k in (2,4,8)},
        centered_amplitude_ratio=float(np.sqrt(np.mean(pc**2)/max(np.mean(yc**2),1e-12))),
        difference_rms_ratio=float(np.sqrt(np.mean(np.diff(pred,axis=1)**2)/max(np.mean(np.diff(truth,axis=1)**2),1e-12))),
        centered_correlation_mean=float(np.mean(corr[valid])) if valid.any() else None,
        horizon_mse=np.mean(error**2,axis=(0,2)).tolist())

WATCH = ('value_embedding.weight','temporal_projection.weight','decoder.drive_norm.weight',
         'decoder.drive_projection.weight','decoder.readout.input_projection.weight',
         'decoder.readout.output_projection.weight','decoder.future_gate.projection.weight')

def summarize_tensor(t):
    x=t.detach().double()
    return dict(rms=float(x.square().mean().sqrt()),std=float(x.std(correction=0)),max_abs=float(x.abs().max()))

def setup_hooks(model):
    captured={}
    handles=[]
    for i,e in enumerate(model.seasonals):
        handles.append(e.decoder.readout.register_forward_pre_hook(lambda m,a,i=i: captured.update({f'states_{i}':a[0]})))
        handles.append(e.decoder.readout.register_forward_hook(lambda m,a,v,i=i: captured.update({f'output_{i}':v})))
        handles.append(e.decoder.drive_norm.register_forward_hook(lambda m,a,v,i=i: captured.update({f'normalized_future_{i}':v})))
        handles.append(e.decoder.drive_projection.register_forward_pre_hook(lambda m,a,i=i: captured.update({f'drive_features_{i}':a[0]})))
        handles.append(e.decoder.drive_projection.register_forward_hook(lambda m,a,v,i=i: captured.update({f'drive_{i}':v})))
        handles.append(e.temporal_projection.register_forward_pre_hook(lambda m,a,i=i: captured.update({f'encoded_{i}':a[0]})))
    return captured,handles

def trace(model,captured,step,loss,wd_by_id):
    row=dict(step=step,loss=float(loss.detach()),scales=[])
    for i,e in enumerate(model.seasonals):
        item=dict(scale=i,activations={k.rsplit('_',1)[0]:summarize_tensor(v) for k,v in captured.items() if k.endswith('_'+str(i))},parameters={})
        for n,p in e.named_parameters():
            if n in WATCH:
                g=p.grad
                w=p.detach().double()
                entry=dict(norm=float(w.norm()),gradient_norm=None if g is None else float(g.double().norm()),decay_norm=float(w.norm()*wd_by_id[id(p)]),weight_decay=wd_by_id[id(p)])
                if g is not None:
                    entry['gradient_dot_weight']=float((g.double()*w).sum())
                item['parameters'][n]=entry
        row['scales'].append(item)
    return row
