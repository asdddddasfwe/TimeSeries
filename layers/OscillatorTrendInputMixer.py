"""Give the existing FFT encoder full trend trajectories without changing routing."""
import torch
from torch import nn

from layers.LagARFormer import select_fft_periods
from layers.OscillatorMixer import OscillatorSeasonal


class TrendInputSeasonal(OscillatorSeasonal):
    def __init__(self, encoder):
        nn.Module.__init__(self)
        self.length, self.horizon = encoder.length, encoder.horizon
        self.period_top_k = encoder.period_top_k
        for name, module in encoder.named_children():
            self.add_module(name, module)
        for name, value in encoder.named_buffers(recurse=False):
            self.register_buffer(name, value.detach().clone())
        self.trend_embedding = nn.Linear(1, self.value_embedding.out_features, bias=False)
        nn.init.zeros_(self.trend_embedding.weight)

    def encode_history(self, season, trend=None):
        if trend is None or trend.shape != season.shape:
            raise ValueError("Trend-input encoder requires aligned [N,T] season and trend")
        # Geometry remains selected from the SAME seasonal input as the baseline.
        periods, weights = select_fft_periods(season, self.period_top_k)
        tokens = self.value_embedding(season[..., None])
        tokens = tokens + self.trend_embedding(trend[..., None]) + self.positions.to(tokens)
        return self.encode_periods(tokens, periods, weights)
