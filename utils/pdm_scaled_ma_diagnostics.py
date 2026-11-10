"""Capture every PDM scale during selected validation without extra forwards."""
import math
from utils.pdm_coarse_ma_diagnostics import SelectedValidationCapture as BaseCapture


class SelectedValidationCapture(BaseCapture):
    def __init__(self, model, folder, arm, windows):
        super().__init__(model, folder, arm, windows)
        self.all_scales = {}

    def decomposition(self, layer, x, trend):
        super().decomposition(layer, x, trend)
        key = f'{layer}:{x.shape[1]}'
        stats = self.all_scales.setdefault(key, dict(elements=0, input_square=0., trend_square=0., curvature_elements=0, curvature_square=0.))
        x, trend = x.detach().double(), trend.detach().double()
        curvature = trend[:, 2:] - 2*trend[:, 1:-1] + trend[:, :-2]
        stats['elements'] += trend.numel()
        stats['input_square'] += float(x.square().sum())
        stats['trend_square'] += float(trend.square().sum())
        stats['curvature_elements'] += curvature.numel()
        stats['curvature_square'] += float(curvature.square().sum())

    def report(self, prediction, truth):
        result = super().report(prediction, truth)
        result['all_scale_decomposition'] = {
            key: dict(input_rms=math.sqrt(s['input_square']/s['elements']),
                      trend_rms=math.sqrt(s['trend_square']/s['elements']),
                      second_difference_rms=math.sqrt(s['curvature_square']/s['curvature_elements']))
            for key, s in self.all_scales.items()
        }
        return result
