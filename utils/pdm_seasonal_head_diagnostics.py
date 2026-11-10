"""Bounded same-forward 8/12 component capture for the paired head study."""
from unittest.mock import patch
import numpy as np
import torch
from utils.pdm_scaled_ma_diagnostics import SelectedValidationCapture as BaseCapture

class SplitCapture(BaseCapture):
    def __enter__(self):
        self.components = np.lib.format.open_memmap(self.folder/(self.arm+'_validation_components.npy'),
            mode='w+', dtype='float32', shape=(self.windows,96,self.model.enc_in,12))
        self.means = np.lib.format.open_memmap(self.folder/(self.arm+'_validation_history_mean.npy'),
            mode='w+', dtype='float32', shape=(self.windows,1,self.model.enc_in))
        original = self.model.predict_components
        def capture(x):
            value = original(x)
            self.cache.append(value.detach())
            return value
        self.patch = patch.object(self.model, 'predict_components', new=capture)
        self.patch.start()
        self.handles.append(self.model.register_forward_pre_hook(self.before))
        for i, block in enumerate(self.model.pdm_blocks):
            self.handles.append(block.decompsition.register_forward_hook(
                lambda module,args,value,i=i:self.decomposition(i,args[0],value[1])))
        self.handles.append(self.model.register_forward_hook(self.after))
        return self

    def after(self, module, args, prediction):
        if len(self.cache) != 1:
            raise RuntimeError('Incomplete 12-component capture')
        components = self.cache[0] * module.normalize.stdev.unsqueeze(-1)
        mean = module.normalize.mean
        if not torch.isfinite(components).all():
            raise RuntimeError('Nonfinite selected-validation components')
        error = float((components.double().sum(-1)+mean.double()-prediction.double()).abs().max())
        self.max_reconstruction_error = max(self.max_reconstruction_error,error)
        if error > 2e-5:
            raise RuntimeError('Branch reconstruction mismatch')
        end = self.offset + len(prediction)
        self.components[self.offset:end] = components.cpu().numpy()
        self.means[self.offset:end] = mean.cpu().numpy()
        self.offset = end
        self.cache.clear()

    def report(self, prediction, truth):
        result = super().report(prediction, truth)
        result['component_order'] = [group+str(length) for group in
            ('season','feature_trend','raw_trend') for length in (96,48,24,12)]
        result['coarse_component_definition'] = 'season12 only; differs from merged PDM12'
        result['conditional_deletions'] = {}
        error = prediction.astype('float64') - truth.astype('float64')
        base = error**2
        for name, indices in {'feature_trend_all':[4,5,6,7], 'raw_trend_all':[8,9,10,11],
                              'season_all':[0,1,2,3], 'coarse_season_plus_feature_trend':[3,7]}.items():
            z = np.asarray(self.components[...,indices],dtype='float64').sum(-1)
            removed = (error-z)**2
            result['conditional_deletions'][name] = dict(mse=float(removed.mean()),
                mse_excess_per_channel=(base-removed).mean(axis=(0,1)).tolist())
        return result


def SelectedValidationCapture(model, folder, arm, windows):
    cls = SplitCapture if hasattr(model, 'feature_trend_heads') else BaseCapture
    return cls(model, folder, arm, windows)
