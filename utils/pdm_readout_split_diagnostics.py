"""Same-forward decoder-entrance reconstruction and bounded component capture."""
import math
import torch
from utils.pdm_seasonal_head_diagnostics import SplitCapture
from utils.pdm_scaled_ma_diagnostics import SelectedValidationCapture as BaselineCapture

class ReadoutSplitCapture(SplitCapture):
    def __init__(self, model, folder, arm, windows):
        super().__init__(model, folder, arm, windows)
        self.route = model.readout_split_variant
        self.terminal = {}
        self.terminal_reconstruction_error = 0.

    def __enter__(self):
        super().__enter__()
        self.handles.append(self.model.readout_decomposition.register_forward_hook(self.capture_terminal))
        return self

    def capture_terminal(self, module, args, output):
        h=args[0].detach().double();s,t=[v.detach().double() for v in output]
        error=float((h-s-t).abs().max())
        self.terminal_reconstruction_error=max(self.terminal_reconstruction_error,error)
        if error>2e-5:raise RuntimeError('Decoder entrance S+T differs from H')
        key=str(h.shape[1]);v=self.terminal.setdefault(key,dict(elements=0,h_square=0.,s_square=0.,t_square=0.,curvature_elements=0,curvature_square=0.))
        v['elements']+=h.numel()
        for name,x in [('h',h),('s',s),('t',t)]:v[name+'_square']+=float(x.square().sum())
        diff=t[:,2:]-2*t[:,1:-1]+t[:,:-2]
        v['curvature_elements']+=diff.numel();v['curvature_square']+=float(diff.square().sum())

    def report(self,prediction,truth):
        result=super().report(prediction,truth)
        prefix='full_recurrent' if self.route=='full' else 'season_recurrent'
        result['component_order']=[group+str(length) for group in (prefix,'feature_trend','raw_trend') for length in (96,48,24,12)]
        result['coarse_component_definition']=prefix+'12; report paired coarse recurrent+feature trend separately'
        result['conditional_deletions']['recurrent_all']=result['conditional_deletions'].pop('season_all')
        result['conditional_deletions']['coarse_recurrent_plus_feature_trend']=result['conditional_deletions'].pop('coarse_season_plus_feature_trend')
        result['terminal_reconstruction_max_abs_error']=self.terminal_reconstruction_error
        result['decoder_entrance']={k:{**{name+'_rms':math.sqrt(v[name+'_square']/v['elements']) for name in ['h','s','t']},
            'trend_second_difference_rms':math.sqrt(v['curvature_square']/v['curvature_elements'])} for k,v in self.terminal.items()}
        return result

def SelectedValidationCapture(model,folder,arm,windows):
    cls=ReadoutSplitCapture if hasattr(model,'readout_decomposition') else BaselineCapture
    return cls(model,folder,arm,windows)
