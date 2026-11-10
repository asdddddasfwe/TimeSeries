"""Capture original 12-component and new 8-component validation forwards."""
import math
from unittest.mock import patch

import numpy as np
import torch

from utils.pdm_seasonal_head_diagnostics import SplitCapture
from utils.pdm_readout_split_diagnostics import ReadoutSplitCapture


class SingleFFTCapture(SplitCapture):
    def __init__(self, *args):
        super().__init__(*args)
        self.terminal = {}
        self.terminal_reconstruction_error = 0.
        self.stream_statistics = {}
        self.split_calls = 0
        self.encoding_calls = 0

    def __enter__(self):
        self.components = np.lib.format.open_memmap(self.folder/(self.arm+'_validation_components.npy'),
            mode='w+', dtype='float32', shape=(self.windows,96,self.model.enc_in,8))
        self.means = np.lib.format.open_memmap(self.folder/(self.arm+'_validation_history_mean.npy'),
            mode='w+', dtype='float32', shape=(self.windows,1,self.model.enc_in))
        original = self.model.predict_components
        def capture(x):
            value = original(x)
            if value.shape[-1] != 8:
                raise RuntimeError('Expected eight components')
            self.cache.append(value.detach())
            return value
        self.patch = patch.object(self.model, 'predict_components', new=capture)
        self.patch.start()
        self.handles.append(self.model.register_forward_pre_hook(self.before))
        self.handles.append(self.model.initial_decomposition.register_forward_hook(self.initial_split))
        for encoder in self.model.seasonals:
            if len(encoder.blocks) != 1:
                raise RuntimeError('Unexpected FFT depth')
        for i, block in enumerate(self.model.pdm_blocks):
            self.handles.append(block.register_forward_hook(
                lambda module,args,output,i=i:self.streams(i,output)))
        self.handles.append(self.model.register_forward_hook(self.after))
        return self

    def initial_split(self, module, args, output):
        self.split_calls += 1
        ReadoutSplitCapture.capture_terminal(self,module,args,output)

    def streams(self, layer, output):
        for group, histories in zip(('season','trend'), output):
            for h in histories:
                x = h.detach().double()
                if not torch.isfinite(x).all():
                    raise RuntimeError('Nonfinite stream features')
                key = f'{layer}:{group}:{x.shape[1]}'
                row = self.stream_statistics.setdefault(key, dict(elements=0,square=0.,diff_elements=0,diff_square=0.))
                d = x[:,2:]-2*x[:,1:-1]+x[:,:-2]
                row['elements'] += x.numel(); row['square'] += float(x.square().sum())
                row['diff_elements'] += d.numel(); row['diff_square'] += float(d.square().sum())

    def report(self, prediction, truth):
        error = prediction.astype('float64')-truth.astype('float64')
        base = error**2
        conditional = {}
        for name, indices in {'recurrent_all':[0,1,2,3], 'feature_trend_all':[4,5,6,7],
                              'coarse_recurrent_plus_feature_trend':[3,7]}.items():
            z = np.asarray(self.components[...,indices],dtype='float64').sum(-1)
            removed = (error-z)**2
            conditional[name] = dict(mse=float(removed.mean()),
                mse_excess_per_channel=(base-removed).mean(axis=(0,1)).tolist())
        return dict(windows=self.offset, components=str(self.folder/(self.arm+'_validation_components.npy')),
            component_order=[group+str(length) for group in ('season_recurrent','feature_trend') for length in (96,48,24,12)],
            reconstruction_max_abs_error=self.max_reconstruction_error,
            initial_split_reconstruction_max_abs_error=self.terminal_reconstruction_error,
            initial_split_calls=self.split_calls,
            initial_decomposition={k:{**{name+'_rms':math.sqrt(v[name+'_square']/v['elements']) for name in ['h','s','t']},
                'trend_second_difference_rms':math.sqrt(v['curvature_square']/v['curvature_elements'])} for k,v in self.terminal.items()},
            stream_statistics={k:dict(rms=math.sqrt(v['square']/v['elements']),
                second_difference_rms=math.sqrt(v['diff_square']/v['diff_elements'])) for k,v in self.stream_statistics.items()},
            intact_mse=float(base.mean()),intact_per_channel_mse=base.mean((0,1)).tolist(),
            conditional_deletions=conditional,
            caveat='Saved-component deletion is conditional contribution, not retraining; shared weights and one joint loss; validation reused')


def SelectedValidationCapture(model, folder, arm, windows):
    cls = SingleFFTCapture if hasattr(model, 'initial_decomposition') else ReadoutSplitCapture
    return cls(model,folder,arm,windows)
