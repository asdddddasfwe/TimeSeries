"""Layerwise TimeMixer PDM on multiscale FFT features, with oscillator decoding.

No raw-history season/trend split and no seasonal residual renormalization in
feature mode. Per-scale history Normalize precedes embedding, as in TimeMixer.
The FFT encoder and full four-scale rank16 recurrence are retained; PDM replaces
one-shot seasonal mixing and the separate raw-trend prediction bypass.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from models.OscillatorLowRankTimeMix import Model as LowRank
from models.TimeMixer import PastDecomposableMixing
from layers.StandardNorm import Normalize
from layers.LagARFormer import select_fft_periods
from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorTimeMix import complex_oscillator_scan


def encode_one_fft_layer(encoder, tokens, periods, weights, layer_index):
    """Apply exactly one existing axial layer to each FFT image and fuse it."""
    if tokens.dtype in (torch.float16, torch.bfloat16):
        tokens = tokens.float()
    groups = {}
    for series, row in enumerate(periods.detach().cpu().tolist()):
        for period in set(row):
            groups.setdefault(period, []).append(series)
    block = encoder.blocks[layer_index]
    projections = block.projection_parameters()
    fused = torch.zeros_like(tokens)
    for period in sorted(groups):
        selected = torch.tensor(groups[period], dtype=torch.long, device=tokens.device)
        group = tokens[selected]
        n, length, width = group.shape
        image = F.pad(group, (0, 0, 0, (-length) % period)).reshape(n, -1, period, width).permute(0, 3, 1, 2).contiguous()
        image = block(image, projections)
        encoded = image.permute(0, 2, 3, 1).reshape(n, -1, width)[:, :length]
        merged = (weights[selected] * (periods[selected] == period)).sum(-1).to(tokens)
        fused = fused.index_add(0, selected, encoded * merged[:, None, None])
    return fused


class Model(LowRank):
    def __init__(self, configs, enabled=None):
        super().__init__(configs)
        self.feature_decomposition = bool(getattr(configs, 'feature_decomposition', True) if enabled is None else enabled)
        if not self.feature_decomposition:
            return
        if bool(getattr(configs, 'seasonal_normalize', False)):
            raise ValueError('Feature PDM must not apply seasonal residual normalization')
        cfg = copy.copy(configs)
        cfg.down_sampling_layers, cfg.down_sampling_window = 3, 2
        cfg.channel_independence, cfg.decomp_method = 1, 'moving_avg'
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(cfg) for _ in range(cfg.e_layers)])
        for block in self.pdm_blocks:
            # These members are unused in the upstream PDM forward. Remove
            # dormant LayerNorm parameters; PDM itself performs no LayerNorm.
            del block.layer_norm
            del block.dropout
        self.coarse_normalizers = nn.ModuleList([Normalize(self.enc_in, affine=False) for _ in range(3)])
        del self.decomposition
        del self.season_mix
        del self.trend_mix
        del self.trend_heads
        for encoder in self.seasonals:
            del encoder.trend_embedding
        self.register_buffer('feature_pdm_version', torch.tensor(1, dtype=torch.long))

    def initialize_from_control(self, control):
        if self.feature_decomposition:
            raise ValueError('Initialize common weights through the paired experiment initializer')
        return super().initialize_from_control(control)

    def encode_multiscale(self, raw_scales):
        """Normalize each observed-history scale, embed, then FFT -> PDM per layer."""
        normalized = [self.normalize(raw_scales[0], 'norm')]
        normalized += [normalizer(x, 'norm') for normalizer, x in zip(self.coarse_normalizers, raw_scales[1:])]
        batch, _, channels = normalized[0].shape
        histories, routing = [], []
        for encoder, x in zip(self.seasonals, normalized):
            flat = x.transpose(1, 2).reshape(batch * channels, encoder.length)
            routing.append(select_fft_periods(flat, encoder.period_top_k))
            tokens = encoder.value_embedding(flat[..., None])
            histories.append(tokens + encoder.positions.to(tokens))
        for layer_index, pdm in enumerate(self.pdm_blocks):
            histories = [encode_one_fft_layer(e, h, *route, layer_index)
                         for e, h, route in zip(self.seasonals, histories, routing)]
            histories = pdm(histories)
        return normalized, histories

    def forecast(self, x_enc, zero_acf=False, intervention='normal'):
        if not self.feature_decomposition:
            return super().forecast(x_enc, zero_acf, intervention)
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected observed history [B,seq_len,enc_in]')
        if zero_acf or intervention != 'normal':
            raise ValueError('Feature PDM study supports the normal forecasting path only')
        scales, histories = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        outputs = [calibrated_seasonal_forecast(e, x, alpha, beta, encoded_history=h,
                   scan_function=complex_oscillator_scan) for e, x, h in zip(self.seasonals, scales, histories)]
        prediction = torch.stack(outputs, -1).sum(-1)
        prediction = self.normalize(prediction, 'denorm')
        return prediction[..., -1:] if self.features == 'MS' and self.c_out == 1 else prediction
