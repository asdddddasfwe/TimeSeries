"""Record selected-validation FFT routing within the existing model forward."""
from unittest.mock import patch
import numpy as np
from utils.pdm_single_fft_diagnostics import SingleFFTCapture
from models.OscillatorPDMSeasonalFFT import select_seasonal_feature_periods


class SelectedValidationCapture(SingleFFTCapture):
    def __enter__(self):
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
        target = ('models.OscillatorPDMSegmentPE' if self.arm == 'segment_pe' else 'models.OscillatorPDMSeasonalFFT')
        self.route_patch = patch(target + '.select_seasonal_feature_periods', new=select)
        self.route_patch.start()
        return self

    def __exit__(self, *args):
        self.route_patch.stop()
        return super().__exit__(*args)

    def report(self, prediction, truth):
        result = super().report(prediction, truth)
        result['fft_routing'] = self.routing
        result['history_position_added'] = False
        result['segment_position_added'] = self.arm == 'segment_pe'
        result['segment_position_definition'] = 'fixed sinusoidal cycle/phase on disjoint feature halves; valid tokens only; before axial block'
        result['future_position_added'] = True
        return result
