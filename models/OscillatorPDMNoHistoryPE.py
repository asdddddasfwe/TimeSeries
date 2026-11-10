"""Single seasonal FFT with history positional addition removed before MA.

Future decoder positions remain unchanged. Inherited fixed history-position
buffers are retained only for exact state compatibility, never read by forward.
"""
from models.OscillatorPDMSeasonalFFT import Model as SingleFFT
from utils.logger import get_logger


class Model(SingleFFT):
    def __init__(self, configs):
        super().__init__(configs)
        get_logger(__name__).info(
            "OscillatorPDMNoHistoryPE: no historical positional addition; "
            "MA splits only embedded observations; seasonal FFT unchanged; "
            "future decoder positional encoding retained")

    def encode_multiscale(self, raw_scales):
        normalized = [self.normalize(raw_scales[0], "norm")]
        normalized += [norm(x, "norm") for norm, x in zip(self.coarse_normalizers, raw_scales[1:])]
        batch, _, channels = normalized[0].shape
        seasons, trends = [], []
        for encoder, x in zip(self.seasonals, normalized):
            flat = x.transpose(1, 2).reshape(batch * channels, encoder.length)
            tokens = encoder.value_embedding(flat[..., None])
            season, trend = self.initial_decomposition(tokens)
            seasons.append(season)
            trends.append(trend)
        seasons = self.encode_seasonal_features(seasons)
        for pdm in self.pdm_blocks:
            seasons, trends = pdm(seasons, trends)
        return normalized, (seasons, trends)
