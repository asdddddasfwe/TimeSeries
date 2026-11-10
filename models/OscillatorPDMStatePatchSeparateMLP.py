"""Old mean-patch model with independently learned seasonal/trend PDM MLPs.

Keep the original ``out_cross_layer`` for the seasonal path and clone it for
the trend path. Identical initial functions; no new activation or forward call.
"""
import copy
from torch import nn
from models.OscillatorPDMStatePatch import Model as Original


class SeparateMLPScaleMixing(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.mixing_multi_scale_season = source.mixing_multi_scale_season
        self.mixing_multi_scale_trend = source.mixing_multi_scale_trend
        self.out_cross_layer = source.out_cross_layer
        self.trend_out_cross_layer = copy.deepcopy(source.out_cross_layer)

    def forward(self, seasons, trends):
        expected = len(self.mixing_multi_scale_season.down_sampling_layers) + 1
        if len(seasons) != expected or len(trends) != expected:
            raise ValueError('Expected one seasonal/trend history per scale')
        mixed_s = self.mixing_multi_scale_season([s.transpose(1, 2) for s in seasons])
        mixed_t = self.mixing_multi_scale_trend([t.transpose(1, 2) for t in trends])
        return ([s + self.out_cross_layer(m) for s, m in zip(seasons, mixed_s)],
                [t + self.trend_out_cross_layer(m) for t, m in zip(trends, mixed_t)])


class Model(Original):
    def __init__(self, configs):
        super().__init__(configs)
        self.pdm_blocks = nn.ModuleList([SeparateMLPScaleMixing(b) for b in self.pdm_blocks])

    def clone_trend_mlps(self):
        """Call after loading regenerated common scratch weights."""
        for block in self.pdm_blocks:
            block.trend_out_cross_layer.load_state_dict(block.out_cross_layer.state_dict())
