"""H720: one complete RevIN call before all pooling; one affine pair only."""
from torch.nn import functional as F
from models.MAFirstKernelH720AffineFollowup import Model as FrozenAffine

class Model(FrozenAffine):
    def __init__(self, configs):
        super().__init__(configs)
        # Remove the three unused normalizers and their six affine tensors.
        # All retained backbone/head initialization stays identical to scratch.
        del self.coarse_normalizers

    def downsample(self, x):
        if x.ndim != 3 or x.shape[1:] != (96, 7):
            raise ValueError('Expected observed history [B,96,7]')
        normalized = self.normalize(x, 'norm')
        scales = [normalized]
        for _ in range(3):
            normalized = F.avg_pool1d(normalized.transpose(1, 2), 2, 2).transpose(1, 2)
            scales.append(normalized)
        return scales

    def encode_multiscale(self, normalized_scales):
        # Already fully normalized including the sole gamma/beta, before pooling.
        assert len(normalized_scales) == 4
        batch, _, channels = normalized_scales[0].shape
        seasons, trends = [], []
        for encoder, x in zip(self.seasonals, normalized_scales):
            observed = x.transpose(1, 2).reshape(batch * channels, encoder.length, 1)
            seasonal, trend = self.initial_decomposition(observed)
            seasons.append(F.linear(seasonal, encoder.value_embedding.weight, None))
            trends.append(encoder.value_embedding(trend))
        seasons = self.encode_seasonal_features(seasons)
        for pdm in self.pdm_blocks:
            seasons, trends = pdm(seasons, trends)
        return normalized_scales, (seasons, trends)
