"""Smooth only the learned temporal correction before PDM trend fusion."""
import torch
from torch import nn
from layers.Autoformer_EncDec import moving_avg
from models.OscillatorPDMScaledMA import scaled_odd_kernel
from models.OscillatorPDMTrendPreLinear import Model as PreLinearModel
from models.TimeMixer import MultiScaleTrendMixing
from utils.logger import get_logger


class SmoothTrendCorrection(nn.Module):
    """[B*C,D,L] -> T + MA(WT+b-T), with replicated endpoints."""
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.average = moving_avg(kernel, stride=1)

    def forward(self, trend, mapped):
        correction = mapped - trend
        return trend + self.average(correction.transpose(1, 2)).transpose(1, 2)


class SmoothResidualTrendMixing(MultiScaleTrendMixing):
    def __init__(self, original, base_kernel, base_length):
        nn.Module.__init__(self)
        self.lengths = original.lengths
        self.up_sampling_layers = original.up_sampling_layers
        self.pre_linears = original.pre_linears
        self.correction_filters = nn.ModuleList([
            SmoothTrendCorrection(scaled_odd_kernel(base_kernel, base_length, length))
            for length in self.lengths])

    def forward(self, trend_list):
        if len(trend_list) != len(self.lengths):
            raise ValueError('Expected one trend tensor per configured scale')
        transformed = []
        for trend, length, linear, smoother in zip(
                trend_list, self.lengths, self.pre_linears, self.correction_filters):
            if trend.ndim != 3 or trend.shape[-1] != length:
                raise ValueError('Expected trend [B*C,D,T_i] in configured scale order')
            transformed.append(smoother(trend, linear(trend)))
        return super().forward(transformed)


class Model(PreLinearModel):
    def __init__(self, configs):
        super().__init__(configs)
        for block in self.pdm_blocks:
            block.mixing_multi_scale_trend = SmoothResidualTrendMixing(
                block.mixing_multi_scale_trend, configs.moving_avg, configs.seq_len)
        self.register_buffer('trend_smooth_residual_version', torch.tensor(1, dtype=torch.long))
        get_logger(__name__).info(
            'OscillatorPDMTrendSmoothResidual overrides free prelinear mapping: '
            'T+MA(WT+b-T) before each PDM trend mixer; automatic kernels=%s; '
            'same W=I/b=0 initialization, parameters and decay as free prelinear; '
            'no gate or amplitude bound; moving average is not a strict spectral projector',
            [f.kernel for f in self.pdm_blocks[0].mixing_multi_scale_trend.correction_filters])
