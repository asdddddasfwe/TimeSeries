"""MAFirst: seasonal and trend histories use independent sliding Transformers."""
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import (
    Model as SeasonOnly, SeasonalSlidingPatchEncoder)


class Model(SeasonOnly):
    def __init__(self, configs):
        super().__init__(configs)
        encoders = []
        for scale in range(len(self.seasonals)):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20211031 + scale)
                encoders.append(SeasonalSlidingPatchEncoder(
                    configs.d_model, configs.d_ff, configs.dropout,
                    heads=4, patch_length=4, stride=2))
        # Same architecture as season, independently initialized and stored.
        self.trend_patch_encoders = nn.ModuleList(encoders)

    def encode_multiscale(self, raw_scales):
        normalized = [self.normalize(raw_scales[0], 'norm')]
        normalized += [norm(x, 'norm') for norm, x in
                       zip(self.coarse_normalizers, raw_scales[1:])]
        batch, _, channels = normalized[0].shape
        seasons, trends = [], []
        for encoder, x in zip(self.seasonals, normalized):
            observed = x.transpose(1, 2).reshape(batch * channels, encoder.length, 1)
            seasonal, trend = self.initial_decomposition(observed)
            seasons.append(F.linear(seasonal, encoder.value_embedding.weight, None))
            trends.append(encoder.value_embedding(trend))
        seasons = self.encode_seasonal_features(seasons)
        # Only addition relative to the season-only arm, before original PDM.
        trends = [encoder(history) for encoder, history in
                  zip(self.trend_patch_encoders, trends)]
        for pdm in self.pdm_blocks:
            seasons, trends = pdm(seasons, trends)
        return normalized, (seasons, trends)
