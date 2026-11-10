"""Disconnected MA1 with TimeMixer-style PDM fusion and decoder re-splitting.
No trend-to-dynamics logit conditioning. No extra FFT or trainable parameters.
"""
import copy
from torch import nn
from models.OscillatorPDMStatePatchDisconnected import Model as Disconnected
from utils.logger import get_logger


class FusedScaleMixing(nn.Module):
    """[N,T,D] streams -> merged channel MLP/residual -> MA decoder streams.

    F = (S + T) + MLP(MS_season(S) + MS_trend(T)).
    Return (F - MA(F), MA(F)) so the unchanged two decoder types can consume it.
    This is a TimeMixer-inspired hybrid, not the full TimeMixer predictor.
    """
    def __init__(self, original, decomposition):
        super().__init__()
        self.mixing_multi_scale_season = original.mixing_multi_scale_season
        self.mixing_multi_scale_trend = original.mixing_multi_scale_trend
        self.out_cross_layer = original.out_cross_layer
        self.decomposition = copy.deepcopy(decomposition)

    def forward(self, seasons, trends):
        expected = len(self.mixing_multi_scale_season.down_sampling_layers) + 1
        if len(seasons) != expected or len(trends) != expected:
            raise ValueError('Expected seasonal/trend histories at all scales')
        ms = self.mixing_multi_scale_season([s.transpose(1, 2) for s in seasons])
        mt = self.mixing_multi_scale_trend([t.transpose(1, 2) for t in trends])
        outputs = [self.decomposition((s + t) + self.out_cross_layer(a + b))
                   for s, t, a, b in zip(seasons, trends, ms, mt)]
        return [v[0] for v in outputs], [v[1] for v in outputs]


class Model(Disconnected):
    def __init__(self, configs):
        super().__init__(configs)
        self.pdm_blocks = nn.ModuleList([
            FusedScaleMixing(block, self.initial_decomposition) for block in self.pdm_blocks])
        get_logger(__name__).info(
            'PDM fusion override: one entrance MA/seasonal FFT, each PDM merges '
            'season/trend before channel MLP plus S+T residual, then scale-aware '
            'MA re-splits for unchanged seasonal A24 and linear trend decoders; '
            'no explicit trend-condition logits. Not full TimeMixer.')
