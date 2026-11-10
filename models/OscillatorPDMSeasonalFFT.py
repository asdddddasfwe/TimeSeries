"""One MA split of embedded features, followed by separate seasonal/trend paths.

Seasonal features receive ONE FFT periodic encoding before scale mixing. No intermediate merge,
re-decomposition, decoder split, or raw-trend output bypass.
"""
import torch
from torch import nn

from models.OscillatorFeaturePDMTimeMix import encode_one_fft_layer
from models.OscillatorPDMNoRawBypass import Model as NoRawBypass
from utils.logger import get_logger


def select_seasonal_feature_periods(season, top_k):
    """[B*C,T,D] seasonal features -> periods/weights [B*C,K].

    Average non-DC spectral magnitudes over features BEFORE ranking. Taking
    the feature mean before FFT could cancel opposite-sign feature channels.
    FFT only routes periodic encoding: no spectral filtering or inverse FFT.
    """
    if season.ndim != 3 or season.shape[1] < 2 or season.shape[2] < 1 or top_k < 1:
        raise ValueError("Expected seasonal features [N,T>=2,D>=1] and positive top_k")
    work = season.float() if season.dtype in (torch.float16, torch.bfloat16) else season
    amplitudes = torch.fft.rfft(work, dim=1).abs()[:, 1:, :].mean(-1)
    indices = torch.argsort(amplitudes, dim=-1, descending=True, stable=True)
    indices = indices[:, :min(top_k, amplitudes.shape[1])]
    weights = amplitudes.gather(1, indices).softmax(-1)
    return season.shape[1] // (indices + 1), weights


class SeparateScaleMixing(nn.Module):
    """Process already separated streams without another decomposition.

    Each layer's channel MLP is shared but evaluated independently on the two
    streams. This preserves parameters, not the former merged-MLP equation.
    """
    def __init__(self, original):
        super().__init__()
        if not original.channel_independence:
            raise ValueError("Seasonal FFT PDM requires channel-independent features")
        self.mixing_multi_scale_season = original.mixing_multi_scale_season
        self.mixing_multi_scale_trend = original.mixing_multi_scale_trend
        self.out_cross_layer = original.out_cross_layer

    def forward(self, seasons, trends):
        expected = len(self.mixing_multi_scale_season.down_sampling_layers) + 1
        if len(seasons) != expected or len(trends) != expected:
            raise ValueError("Expected one seasonal and trend history per scale")
        mixed_seasons = self.mixing_multi_scale_season([s.transpose(1, 2) for s in seasons])
        mixed_trends = self.mixing_multi_scale_trend([t.transpose(1, 2) for t in trends])
        season_out = [s + self.out_cross_layer(m) for s, m in zip(seasons, mixed_seasons)]
        trend_out = [t + self.out_cross_layer(m) for t, m in zip(trends, mixed_trends)]
        return season_out, trend_out


class Model(NoRawBypass):
    def __init__(self, configs):
        super().__init__(configs)
        if not self.pdm_blocks:
            raise ValueError("Seasonal FFT model requires at least one processing layer")
        # Parameter-free MA belongs only at the encoder entrance. Remove all
        # per-layer decompositions and the inherited final readout split.
        self.initial_decomposition = self.readout_decomposition
        del self.readout_decomposition
        self.pdm_blocks = nn.ModuleList([SeparateScaleMixing(block) for block in self.pdm_blocks])
        # Keep only the first periodic encoder; remove all later parameters,
        # not merely their forward calls. e_layers now controls mixing depth.
        for encoder in self.seasonals:
            encoder.blocks = nn.ModuleList([encoder.blocks[0]])
        get_logger(__name__).info(
            "OscillatorPDMSeasonalFFT overrides inherited routing descriptions: "
            "ONE moving-average split of embedded features per scale; seasonal "
            "features receive ONE periodic encoding before separate scale-mixing layers; "
            "shared channel MLP evaluated separately; no intermediate merge, "
            "no decoder decomposition, no raw bypass")

    def encode_multiscale(self, raw_scales):
        """Return normalized scales and (seasonal histories, trend histories)."""
        normalized = [self.normalize(raw_scales[0], "norm")]
        normalized += [norm(x, "norm") for norm, x in zip(self.coarse_normalizers, raw_scales[1:])]
        batch, _, channels = normalized[0].shape
        seasons, trends = [], []
        for encoder, x in zip(self.seasonals, normalized):
            flat = x.transpose(1, 2).reshape(batch * channels, encoder.length)
            tokens = encoder.value_embedding(flat[..., None])
            season, trend = self.initial_decomposition(tokens + encoder.positions.to(tokens))
            seasons.append(season)
            trends.append(trend)
        seasons = self.encode_seasonal_features(seasons)
        for pdm in self.pdm_blocks:
            seasons, trends = pdm(seasons, trends)
        return normalized, (seasons, trends)

    def encode_seasonal_features(self, seasons):
        """One periodic-encoding stage per scale, independent of mixing depth."""
        if len(seasons) != len(self.seasonals):
            raise ValueError("Expected one seasonal history per scale")
        return [
            encode_one_fft_layer(encoder, season,
                                 *select_seasonal_feature_periods(season, encoder.period_top_k), 0)
            for encoder, season in zip(self.seasonals, seasons)
        ]

    def decoder_histories(self, histories):
        """Use the two propagated streams directly; never decompose again."""
        return histories
