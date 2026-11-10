"""H720: normalize the original observed history before any scale pooling.

Preserve all264 states and all four affine modules. One detached original96
mean/stdev is shared; each existing affine is applied before its own pooling.
No coarse-scale variance normalization occurs after downsampling.
"""
from torch.nn import functional as F
from models.MAFirstKernelH720AffineFollowup import Model as FrozenAffine

class Model(FrozenAffine):
    def downsample(self, x):
        if x.ndim != 3 or x.shape[1:] != (96, 7):
            raise ValueError('Expected observed original history [B,96,7]')
        norms = [self.normalize, *self.coarse_normalizers]
        assert len(norms) == 4
        assert all(n.affine and not n.non_norm and not n.subtract_last for n in norms)
        # Compute shared statistics once, solely from original observed history.
        self.normalize._get_statistics(x)
        mean, stdev = self.normalize.mean, self.normalize.stdev
        standardized = (x - mean) / stdev
        scales = []
        for level, norm in enumerate(norms):
            norm.mean, norm.stdev = mean, stdev
            value = standardized * norm.affine_weight + norm.affine_bias
            for _ in range(level):
                value = F.avg_pool1d(value.transpose(1, 2), 2, 2).transpose(1, 2)
            scales.append(value)
        return scales

    def encode_multiscale(self, normalized_scales):
        # downsample() already normalized BEFORE pooling; never normalize again.
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
