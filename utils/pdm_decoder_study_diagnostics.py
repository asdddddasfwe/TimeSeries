"""Record selected-validation FFT routing within the existing model forward."""
from unittest.mock import patch
import numpy as np
from utils.pdm_single_fft_diagnostics import SingleFFTCapture
from models.OscillatorPDMSeasonalFFT import select_seasonal_feature_periods


class SelectedValidationCapture(SingleFFTCapture):
    def __enter__(self):
        self.mixing_depth = len(self.model.pdm_blocks)
        self.direct = hasattr(self.model.seasonals[0], 'direct_readout')
        super().__enter__()
        self.routing = {}
        def select(season, top_k):
            periods, weights = select_seasonal_feature_periods(season, top_k)
            length = str(season.shape[1])
            row = self.routing.setdefault(length, dict(series=0, top1_counts={}, weighted_period_mass={}))
            p, w = periods.detach().cpu().numpy(), weights.detach().double().cpu().numpy()
            row['series'] += len(p)
            for period in np.unique(p):
                key = str(int(period))
                row['top1_counts'][key] = row['top1_counts'].get(key, 0) + int((p[:, 0] == period).sum())
                row['weighted_period_mass'][key] = row['weighted_period_mass'].get(key, 0.) + float(w[p == period].sum())
            return periods, weights
        target = 'models.OscillatorPDMSegmentPE'
        self.route_patch = patch(target + '.select_seasonal_feature_periods', new=select)
        self.route_patch.start()
        self.trend_shifts = {}
        for i, encoder in enumerate(self.model.seasonals):
            if self.direct:
                continue
            self.handles.append(encoder.decoder.trend_condition.register_forward_hook(
                lambda module,args,value,i=i:self.record_trend_shift(i,value)))
        return self

    def record_trend_shift(self, scale, value):
        x = value.detach().double()
        row = self.trend_shifts.setdefault(str(scale), dict(count=0, decay_square=0., frequency_square=0.))
        decay, frequency = x.chunk(2, -1)
        row['count'] += decay.numel()
        row['decay_square'] += float(decay.square().sum())
        row['frequency_square'] += float(frequency.square().sum())

    def __exit__(self, *args):
        self.route_patch.stop()
        return super().__exit__(*args)

    def report(self, prediction, truth):
        result = super().report(prediction, truth)
        result['fft_routing'] = self.routing
        result['history_position_added'] = False
        result['segment_position_added'] = True
        result['segment_position_definition'] = 'fixed sinusoidal cycle/phase on disjoint feature halves; valid tokens only; before axial block'
        result['future_position_added'] = True
        result['trend_conditioned_dynamics'] = not self.direct
        result['seasonal_decoder'] = 'direct_affine' if self.direct else 'trend_conditioned_oscillator'
        if self.direct:
            result['component_order'] = [v.replace('season_recurrent', 'season_direct') for v in result['component_order']]
            deletions = result['conditional_deletions']
            deletions['seasonal_direct_all'] = deletions.pop('recurrent_all')
            deletions['coarse_direct_plus_feature_trend'] = deletions.pop('coarse_recurrent_plus_feature_trend')
        result['mixing_depth'] = self.mixing_depth
        result['trend_logit_shift'] = {k:dict(decay_rms=(v['decay_square']/v['count'])**.5, frequency_rms=(v['frequency_square']/v['count'])**.5) for k,v in self.trend_shifts.items()}
        return result


# Keep old-control sampling exactly unchanged; use direct-head-specific hooks.
from utils.retained_timemix_helpers import setup_hooks as oscillator_hooks, trace as oscillator_trace, summarize_tensor

def setup_hooks(model):
    if not hasattr(model.seasonals[0], 'direct_readout'):
        return oscillator_hooks(model)
    captured, handles = {}, []
    for i, encoder in enumerate(model.seasonals):
        handles.append(encoder.direct_readout.register_forward_pre_hook(
            lambda module,args,i=i:captured.update({f'future_features_{i}':args[0]})))
        handles.append(encoder.direct_readout.register_forward_hook(
            lambda module,args,value,i=i:captured.update({f'output_{i}':value})))
        handles.append(encoder.temporal_projection.register_forward_pre_hook(
            lambda module,args,i=i:captured.update({f'encoded_{i}':args[0]})))
    return captured, handles

def trace(model, captured, step, loss, wd_by_id):
    if not hasattr(model.seasonals[0], 'direct_readout'):
        return oscillator_trace(model, captured, step, loss, wd_by_id)
    row=dict(step=step, loss=float(loss.detach()), scales=[])
    for i, encoder in enumerate(model.seasonals):
        item=dict(scale=i, activations={k.rsplit('_',1)[0]:summarize_tensor(v) for k,v in captured.items() if k.endswith('_'+str(i))}, parameters={})
        for name, parameter in encoder.named_parameters():
            if name not in ('value_embedding.weight', 'temporal_projection.weight', 'direct_readout.weight', 'direct_readout.bias'):
                continue
            weight, gradient = parameter.detach().double(), parameter.grad.double()
            item['parameters'][name]=dict(norm=float(weight.norm()), gradient_norm=float(gradient.norm()),
                decay_norm=float(weight.norm()*wd_by_id[id(parameter)]), weight_decay=wd_by_id[id(parameter)],
                gradient_dot_weight=float((gradient*weight).sum()))
        row['scales'].append(item)
    return row
