"""Trend-conditioned PDM with repeated residual MA before its single FFT."""
from numbers import Integral
from models.OscillatorPDMTrendCondition import Model as TrendCondition
from models.OscillatorPDMScaledMA import ScaleAwareDecomposition


class ResidualDecomposition(ScaleAwareDecomposition):
    """For [B*C,T,D], repeatedly remove MA(residual), accumulating the trend."""
    def __init__(self, base_kernel, base_length, scale_lengths, passes=3):
        if isinstance(passes, bool) or not isinstance(passes, Integral) or passes < 1:
            raise ValueError('passes must be a positive integer')
        super().__init__(base_kernel, base_length, scale_lengths)
        self.passes = passes

    def forward(self, x):
        # Keep the first application bitwise identical to the existing operator.
        season, trend = super().forward(x)
        for _ in range(1, self.passes):
            season, extracted = super().forward(season)
            trend = trend + extracted
        return season, trend


class Model(TrendCondition):
    def __init__(self, configs):
        super().__init__(configs)
        self.residual_decomposition_passes = getattr(configs, 'residual_decomposition_passes', 3)
        self.initial_decomposition = ResidualDecomposition(
            configs.moving_avg, configs.seq_len, self.lengths,
            self.residual_decomposition_passes)
