"""Record selected-validation FFT routing within the existing model forward."""
from unittest.mock import patch
import numpy as np
from utils.pdm_single_fft_diagnostics import SingleFFTCapture
from models.OscillatorPDMSeasonalFFT import select_seasonal_feature_periods


class SelectedValidationCapture(SingleFFTCapture):
    def __enter__(self):
        self.mixing_depth = len(self.model.pdm_blocks)
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
        result['trend_conditioned_dynamics'] = True
        result['mixing_depth'] = self.mixing_depth
        result['trend_logit_shift'] = {k:dict(decay_rms=(v['decay_square']/v['count'])**.5, frequency_rms=(v['frequency_square']/v['count'])**.5) for k,v in self.trend_shifts.items()}
        return result
