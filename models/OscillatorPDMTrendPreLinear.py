"""Identity-initialized temporal trend maps before each PDM trend mixer."""
import torch
from torch import nn

from models.OscillatorPDMReadoutSplit import Model as ReadoutSplit
from models.TimeMixer import MultiScaleTrendMixing
from utils.logger import get_logger


class PreLinearTrendMixing(MultiScaleTrendMixing):
    """Transform each [B*C,D,T_i] trend along time, then mix coarse to fine.

    Reuse the original upsampling modules and state-dict paths. Each PDM layer
    owns separate temporal maps; weights are shared over variables/features.
    """

    def __init__(self, original, lengths):
        nn.Module.__init__(self)
        self.lengths = tuple(lengths)
        self.up_sampling_layers = original.up_sampling_layers
        # Identity maps need no random initialization; preserve the caller RNG
        # so constructing this model does not perturb paired training streams.
        with torch.random.fork_rng(devices=[]):
            self.pre_linears = nn.ModuleList([
                nn.Linear(length, length) for length in self.lengths
            ])
        for linear in self.pre_linears:
            nn.init.eye_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, trend_list):
        if len(trend_list) != len(self.lengths):
            raise ValueError("Expected one trend tensor per configured scale")
        for trend, length in zip(trend_list, self.lengths):
            if trend.ndim != 3 or trend.shape[-1] != length:
                raise ValueError("Expected trend [B*C,D,T_i] in configured scale order")
        transformed = [linear(trend) for linear, trend
                       in zip(self.pre_linears, trend_list)]
        return super().forward(transformed)


class Model(ReadoutSplit):
    """Seasonal-residual recurrent heads with pre-fusion trend linear maps."""

    def __init__(self, configs):
        super().__init__(configs, variant="seasonal")
        for block in self.pdm_blocks:
            block.mixing_multi_scale_trend = PreLinearTrendMixing(
                block.mixing_multi_scale_trend, self.lengths)
        self.register_buffer("trend_prelinear_version", torch.tensor(1, dtype=torch.long))
        get_logger(__name__).info(
            "OscillatorPDMTrendPreLinear overrides inherited unchanged-PDM description: "
            "each PDM layer applies independent identity-initialized temporal "
            "Linear(T_i,T_i) maps before coarse-to-fine feature-trend mixing; "
            "lengths=%s, no added activation; seasonal-residual recurrence, "
            "feature-trend heads and raw MA25 bypass retained", self.lengths)
