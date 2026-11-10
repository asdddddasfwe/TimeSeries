"""FFT season + sliding-patch trend -> history oscillators -> PDM -> FMM.

Recurrence encodes observed features at each native history resolution. It is
not the old A24 future decoder: only FMM maps history to the forecast horizon.
This module is independent of the frozen experiment models.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from layers.LagARFormer import LagAxialBlock, sinusoidal_positions
from layers.OscillatorMixer import OscillatorDecoder
from layers.OscillatorTimeMix import complex_oscillator_scan
from layers.StandardNorm import Normalize
from models.OscillatorPDMScaledMA import ScaleAwareDecomposition
from models.OscillatorPDMSegmentPE import encode_positioned_segments
from models.OscillatorPDMSeasonalFFT import select_seasonal_feature_periods
from models.TimeMixer import MultiScaleSeasonMixing, MultiScaleTrendMixing


class SeasonalFFTEncoder(nn.Module):
    """Reuse the existing FFT routing, cycle/phase PE and axial attention."""

    def __init__(self, width, d_ff, dropout, top_k):
        super().__init__()
        self.period_top_k = top_k
        self.blocks = nn.ModuleList([LagAxialBlock(width, d_ff, dropout)])

    def forward(self, history):
        return encode_positioned_segments(
            self, history, *select_seasonal_feature_periods(history, self.period_top_k))


class SlidingPatchTrendEncoder(nn.Module):
    """[N,T,D] -> overlapping patch tokens -> attention -> [N,T,D].

Right endpoint replication covers incomplete final patches. Learned patch
readouts are overlap-averaged, cropped, then added to the input and normalized.
Attention is within one observed window/variable, never across batch entries.
"""

    def __init__(self, width, d_ff, dropout, heads=4, patch_length=4, stride=2):
        super().__init__()
        if patch_length < 2 or not 1 <= stride <= patch_length:
            raise ValueError('Require patch_length >= 2 and 1 <= stride <= patch_length')
        if heads < 1 or width % heads:
            raise ValueError('Trend attention heads must divide d_model')
        self.width, self.patch_length, self.stride = width, patch_length, stride
        self.patch_embedding = nn.Linear(patch_length * width, width)
        self.attention = nn.TransformerEncoderLayer(
            width, heads, d_ff, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.patch_readout = nn.Linear(width, patch_length * width)
        self.output_norm = nn.LayerNorm(width)

    def patchify(self, history):
        if history.ndim != 3 or history.shape[-1] != self.width or history.shape[1] < 1:
            raise ValueError('Expected nonempty [N,T,D] trend history')
        length = history.shape[1]
        padded_length = max(length, self.patch_length)
        padded_length += (-(padded_length - self.patch_length)) % self.stride
        padded = F.pad(history.transpose(1, 2), (0, padded_length - length), mode='replicate')
        patches = padded.unfold(-1, self.patch_length, self.stride).permute(0, 2, 3, 1)
        return patches, padded_length

    def overlap_average(self, patches, padded_length, length):
        n, count, size, width = patches.shape
        columns = patches.permute(0, 3, 2, 1).reshape(n, width * size, count)
        summed = F.fold(columns, (1, padded_length), (1, size), stride=(1, self.stride))
        counts = F.fold(patches.new_ones(1, size, count), (1, padded_length),
                        (1, size), stride=(1, self.stride))
        return (summed / counts).squeeze(2).transpose(1, 2)[:, :length]

    def forward(self, history):
        patches, padded_length = self.patchify(history)
        n, count, size, width = patches.shape
        tokens = self.patch_embedding(patches.reshape(n, count, size * width))
        positions = sinusoidal_positions(count, width).to(tokens)
        encoded = self.attention(tokens + positions)
        decoded = self.patch_readout(encoded).reshape(n, count, size, width)
        restored = self.overlap_average(decoded, padded_length, history.shape[1])
        return self.output_norm(history + restored)


class HistoryOscillator(nn.Module):
    """Encode T observed tokens by T stable complex transitions, zero start.

Context conditions A on this branch's mean/last observed features. No trend
conditioning of the seasonal A, future drive, future PE or A**4 is present.
State 2*modes -> width -> width reads one feature vector per history position.
The window encoder is not causal: attention/context may use the whole history.
"""

    def __init__(self, length, width, modes, dropout):
        super().__init__()
        source = OscillatorDecoder(length, length, width, modes, dropout)
        self.length, self.width, self.modes = length, width, modes
        for name in ('context_norm', 'condition', 'log_decay', 'frequency_logits',
                     'drive_norm', 'drive_projection', 'drive_dropout'):
            setattr(self, name, getattr(source, name))
        self.readout = nn.Sequential(nn.Linear(2 * modes, width), nn.GELU(), nn.Linear(width, width))

    def scan_inputs(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.length, self.width):
            raise ValueError('History oscillator requires its native [N,T,D] shape')
        context = self.context_norm(torch.cat((history.mean(1), history[:, -1]), -1))
        dtype = torch.float64 if history.dtype == torch.float64 else torch.float32
        decay, frequency = self.condition(context).to(dtype).chunk(2, -1)
        radius = torch.exp(-F.softplus(self.log_decay.to(dtype) + decay))
        angle = math.pi * torch.sigmoid(self.frequency_logits.to(dtype) + frequency)
        transition = torch.stack((radius * angle.cos(), radius * angle.sin()), -1)
        drive = self.drive_projection(self.drive_dropout(F.gelu(self.drive_norm(history))))
        drive = drive.reshape(history.shape[0], self.length, self.modes, 2)
        return transition[:, None].expand(-1, self.length, -1, -1), drive, transition.new_zeros(
            history.shape[0], self.modes, 2)

    def forward(self, history):
        states = complex_oscillator_scan(*self.scan_inputs(history))
        return self.readout(states.flatten(2).to(self.readout[0].weight.dtype))


class EncodedPDM(nn.Module):
    """TimeMixer directional maps and merged channel-MLP residual, no re-split."""

    def __init__(self, configs):
        super().__init__()
        self.season_mix = MultiScaleSeasonMixing(configs)
        self.trend_mix = MultiScaleTrendMixing(configs)
        self.channel_mlp = nn.Sequential(nn.Linear(configs.d_model, configs.d_ff),
                                         nn.GELU(), nn.Linear(configs.d_ff, configs.d_model))

    def forward(self, seasons, trends):
        ms = self.season_mix([s.transpose(1, 2) for s in seasons])
        mt = self.trend_mix([t.transpose(1, 2) for t in trends])
        return [s + t + self.channel_mlp(a + b) for s, t, a, b in zip(seasons, trends, ms, mt)]


class Model(nn.Module):
    """Four scales, independent branch encoders, one PDM, TimeMixer-style FMM."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = configs.seq_len, configs.pred_len, configs.enc_in
        self.features, self.c_out = getattr(configs, 'features', 'M'), configs.c_out
        self.task_name = configs.task_name
        if self.task_name not in ('long_term_forecast', 'short_term_forecast'):
            raise ValueError('Only forecasting tasks are supported')
        if self.seq_len < 16 or self.seq_len % 8 or min(self.pred_len, self.enc_in) < 1:
            raise ValueError('Require seq_len >=16 divisible by8 and positive horizon/channels')
        if self.features not in ('M', 'MS', 'S') or (self.features == 'S' and self.enc_in != 1):
            raise ValueError('Invalid feature mode')
        if self.c_out != self.enc_in and not (self.features == 'MS' and self.c_out == 1):
            raise ValueError('c_out must equal enc_in, except MS permits c_out=1')
        for name, expected in (('e_layers', 1), ('down_sampling_layers', 3),
                               ('down_sampling_window', 2), ('down_sampling_method', 'avg'),
                               ('channel_independence', 1), ('use_norm', 1),
                               ('decomp_method', 'moving_avg')):
            if getattr(configs, name, expected) != expected:
                raise ValueError(f'This architecture requires {name}={expected}')
        width, d_ff, dropout = configs.d_model, configs.d_ff, configs.dropout
        modes, top_k = getattr(configs, 'ssm_modes', 8), getattr(configs, 'top_k', 5)
        if min(width, d_ff, modes, top_k) < 1 or width < 2 or not 0 <= dropout < 1:
            raise ValueError('Invalid feature, oscillator or attention configuration')
        self.dynamics_weight_decay = getattr(configs, 'ssm_dynamics_weight_decay', 0.)
        if not math.isfinite(self.dynamics_weight_decay) or self.dynamics_weight_decay < 0:
            raise ValueError('Dynamics weight decay must be finite and nonnegative')
        self.lengths = [self.seq_len // stride for stride in (1, 2, 4, 8)]
        self.normalizers = nn.ModuleList([Normalize(self.enc_in, affine=False) for _ in self.lengths])
        self.embeddings = nn.ModuleList([nn.Linear(1, width) for _ in self.lengths])
        self.decomposition = ScaleAwareDecomposition(configs.moving_avg, self.seq_len, self.lengths)
        self.season_encoders = nn.ModuleList([
            SeasonalFFTEncoder(width, d_ff, dropout, top_k) for _ in self.lengths])
        self.trend_encoders = nn.ModuleList([
            SlidingPatchTrendEncoder(width, d_ff, dropout,
                                     getattr(configs, 'trend_patch_heads', 4),
                                     getattr(configs, 'trend_patch_length', 4),
                                     getattr(configs, 'trend_patch_stride', 2)) for _ in self.lengths])
        self.season_oscillators = nn.ModuleList([
            HistoryOscillator(t, width, modes, dropout) for t in self.lengths])
        self.trend_oscillators = nn.ModuleList([
            HistoryOscillator(t, width, modes, dropout) for t in self.lengths])
        cfg = copy.copy(configs)
        cfg.down_sampling_layers, cfg.down_sampling_window = 3, 2
        self.pdm = EncodedPDM(cfg)
        self.predict_layers = nn.ModuleList([nn.Linear(t, self.pred_len) for t in self.lengths])
        self.projection_layer = nn.Linear(width, 1)

    @staticmethod
    def downsample(x):
        scales = [x]
        for _ in range(3):
            x = F.avg_pool1d(x.transpose(1, 2), 2, 2).transpose(1, 2)
            scales.append(x)
        return scales

    def encode_branches(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected observed history [B,seq_len,enc_in]')
        seasons, trends = [], []
        for i, raw in enumerate(self.downsample(x)):
            scale = self.normalizers[i](raw, 'norm')
            flat = scale.transpose(1, 2).reshape(-1, self.lengths[i], 1)
            season, trend = self.decomposition(self.embeddings[i](flat))
            seasons.append(self.season_oscillators[i](self.season_encoders[i](season)))
            trends.append(self.trend_oscillators[i](self.trend_encoders[i](trend)))
        return seasons, trends

    def predict_components(self, x):
        """Four normalized FMM predictions [B,H,C,4], after season/trend merge."""
        merged = self.pdm(*self.encode_branches(x))
        outputs = []
        for head, features in zip(self.predict_layers, merged):
            future = head(features.transpose(1, 2)).transpose(1, 2)
            prediction = self.projection_layer(future)
            outputs.append(prediction.reshape(x.shape[0], self.enc_in, self.pred_len).transpose(1, 2))
        return torch.stack(outputs, -1)

    def forecast(self, x):
        prediction = self.normalizers[0](self.predict_components(x).sum(-1), 'denorm')
        return prediction[..., -1:] if self.features == 'MS' and self.c_out == 1 else prediction

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Forecast-time observations and calendar tensors are intentionally unused.
        return self.forecast(x_enc)

    def optimizer_parameter_groups(self, weight_decay):
        protected = set()
        for encoder in [*self.season_oscillators, *self.trend_oscillators]:
            protected.update((id(encoder.log_decay), id(encoder.frequency_logits)))
            protected.update(id(p) for p in encoder.condition.parameters())
        return [dict(params=[p for p in self.parameters() if id(p) not in protected],
                     weight_decay=weight_decay, group_name='regular'),
                dict(params=[p for p in self.parameters() if id(p) in protected],
                     weight_decay=self.dynamics_weight_decay, group_name='ssm_dynamics')]
