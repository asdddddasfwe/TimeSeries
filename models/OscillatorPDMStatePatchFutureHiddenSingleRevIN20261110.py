"""ETTh1-H720 ablation with one RevIN before four-scale average pooling."""
from torch.nn import functional as F

from models.OscillatorPDMStatePatchSpectralTimevaryFutureHiddenV320261105 import (
    InheritedCombination,
)


class Model(InheritedCombination):
    def __init__(self, configs):
        if (configs.seq_len, configs.pred_len) != (96, 720):
            raise ValueError("Expected input 96 and horizon 720")
        super().__init__(configs)
        if self.normalize.affine or any(norm.affine for norm in self.coarse_normalizers):
            raise ValueError("This ablation requires the original affine=False configuration")
        del self.coarse_normalizers

    def downsample(self, history):
        return super().downsample(self.normalize(history, "norm"))

    def encode_multiscale(self, normalized):
        batch, _, channels = normalized[0].shape
        seasons, trends = [], []
        for encoder, history in zip(self.seasonals, normalized):
            observed = history.transpose(1, 2).reshape(batch * channels, encoder.length, 1)
            seasonal, trend = self.initial_decomposition(observed)
            seasons.append(F.linear(seasonal, encoder.value_embedding.weight, None))
            trends.append(encoder.value_embedding(trend))
        seasons = self.encode_seasonal_features(seasons)
        for pdm in self.pdm_blocks:
            seasons, trends = pdm(seasons, trends)
        return normalized, (seasons, trends)
