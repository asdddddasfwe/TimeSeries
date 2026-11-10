"""Capture mechanism evidence during the already scheduled selected validation."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import models.OscillatorPDMScaleAllocation as forecast_module


class SelectedValidationCapture:
    def __init__(self, model, folder, arm, windows):
        self.model, self.folder, self.arm = model, Path(folder), arm
        self.windows, self.offset = windows, 0
        self.cache, self.heads, self.handles = [], {}, []
        self.statistics = {str(i):dict(elements=0, trend_square=0., input_square=0.,
                            curvature_elements=0, trend_second_difference_square=0.)
                           for i in range(len(model.pdm_blocks))}
        self.max_reconstruction_error = 0.

    def __enter__(self):
        self.components = np.lib.format.open_memmap(self.folder/(self.arm+'_validation_components.npy'),
                            mode='w+', dtype='float32', shape=(self.windows,96,self.model.enc_in,8))
        self.means = np.lib.format.open_memmap(self.folder/(self.arm+'_validation_history_mean.npy'),
                            mode='w+', dtype='float32', shape=(self.windows,1,self.model.enc_in))
        original = forecast_module.calibrated_seasonal_forecast
        def capture(*args, **kwargs):
            value = original(*args, **kwargs)
            self.cache.append(value.detach())
            return value
        # A Mock side_effect records every call argument, retaining GPU histories.
        # Install the function itself so no cross-batch call history is created.
        self.patch = patch.object(forecast_module,'calibrated_seasonal_forecast',new=capture)
        self.patch.start()
        self.handles.append(self.model.register_forward_pre_hook(self.before))
        for i, head in enumerate(self.model.raw_trend_heads):
            self.handles.append(head.register_forward_hook(
                lambda module,args,value,i=i:self.heads.__setitem__(i,value.detach().transpose(1,2))))
        for i, block in enumerate(self.model.pdm_blocks):
            self.handles.append(block.decompsition.register_forward_hook(
                lambda module,args,value,i=i:self.decomposition(i,args[0],value[1])))
        self.handles.append(self.model.register_forward_hook(self.after))
        return self

    def before(self, module, args):
        self.cache.clear(); self.heads.clear()

    def decomposition(self, layer, x, trend):
        if x.shape[1] != 12:
            return
        x, trend = x.detach().double(), trend.detach().double()
        curvature = trend[:,2:] - 2*trend[:,1:-1] + trend[:,:-2]
        s = self.statistics[str(layer)]
        s['elements'] += trend.numel()
        s['trend_square'] += float(trend.square().sum())
        s['input_square'] += float(x.square().sum())
        s['curvature_elements'] += curvature.numel()
        s['trend_second_difference_square'] += float(curvature.square().sum())

    def after(self, module, args, prediction):
        if len(self.cache)!=4 or len(self.heads)!=4:
            raise RuntimeError('Incomplete component capture')
        components = torch.stack(self.cache+[self.heads[i] for i in range(4)],-1)*module.normalize.stdev.unsqueeze(-1)
        mean = module.normalize.mean
        if not torch.isfinite(components).all():
            raise RuntimeError('Nonfinite selected-validation component')
        error = float((components.double().sum(-1)+mean.double()-prediction.double()).abs().max())
        self.max_reconstruction_error = max(self.max_reconstruction_error,error)
        if error>2e-5:
            raise RuntimeError('Branch reconstruction mismatch')
        end=self.offset+len(prediction)
        self.components[self.offset:end]=components.cpu().numpy()
        self.means[self.offset:end]=mean.cpu().numpy()
        self.offset=end
        # Only the current batch may own device tensors. Saved evidence is on CPU.
        self.cache.clear()
        self.heads.clear()

    def __exit__(self, typ, value, traceback):
        try:
            for handle in self.handles:
                handle.remove()
            self.patch.stop()
        finally:
            self.handles.clear()
            self.cache.clear()
            self.heads.clear()
            # The patch replacement closes over this object; break that cycle.
            self.patch = None
            self.model = None
            self.components.flush()
            self.means.flush()
        if typ is None and self.offset!=self.windows:
            raise RuntimeError('Incomplete selected-validation capture')

    def report(self, prediction, truth):
        p,y=prediction.astype('float64'),truth.astype('float64')
        z=np.asarray(self.components[...,3],dtype='float64')
        base=(p-y)**2;removed=(p-z-y)**2
        layers={}
        for key,s in self.statistics.items():
            layers[key]=dict(trend_rms=(s['trend_square']/s['elements'])**.5,
                            input_rms=(s['input_square']/s['elements'])**.5,
                            second_difference_rms=(s['trend_second_difference_square']/s['curvature_elements'])**.5)
        return dict(windows=self.offset,components=str(self.folder/(self.arm+'_validation_components.npy')),
            component_order=['pdm96','pdm48','pdm24','pdm12','trend96','trend48','trend24','trend12'],
            reconstruction_max_abs_error=self.max_reconstruction_error,coarse_decomposition=layers,
            intact_mse=float(base.mean()),frozen_remove_coarse_mse=float(removed.mean()),
            intact_per_channel_mse=base.mean(axis=(0,1)).tolist(),
            frozen_remove_coarse_per_channel_mse=removed.mean(axis=(0,1)).tolist(),
            coarse_mse_excess_per_channel=(base-removed).mean(axis=(0,1)).tolist(),
            coarse_mse_excess_quarters=[float((base-removed)[ix].mean()) for ix in np.array_split(np.arange(len(p)),min(4,len(p)))],
            caveat='Frozen deletion is conditional contribution, not retraining. Components in dataset-standardized units; history mean added once.')
