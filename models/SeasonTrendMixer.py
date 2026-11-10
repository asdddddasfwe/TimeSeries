"""Seasonal period-patch attention plus four-scale DLinear-style trend processing."""
import torch
from torch import nn
from torch.nn import functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.StandardNorm import Normalize
from layers.SeasonTrendMixer import PeriodPatchEncoder, TemporalScaleMixing
from utils.logger import get_logger


def temporal_linear(length, output_length, identity=False):
    layer = nn.Linear(length, output_length)
    with torch.no_grad():
        if identity:
            nn.init.eye_(layer.weight)
        else:
            nn.init.constant_(layer.weight, 1.0 / length)
        nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    """Forecast [B, T, C] independently per channel and sum four scale forecasts."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name not in ("long_term_forecast", "short_term_forecast"):
            raise ValueError("SeasonTrendMixer only supports forecasting tasks")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.features = getattr(configs, "features", "M")
        self.c_out = getattr(configs, "c_out", self.enc_in)
        if self.seq_len % 8 or self.seq_len < 16:
            raise ValueError("seq_len must be divisible by 8, with seq_len / 8 >= 2")
        if self.pred_len < 1 or self.enc_in < 1:
            raise ValueError("pred_len and enc_in must be positive")
        if self.features not in ("M", "S", "MS"):
            raise ValueError("features must be M, S, or MS")
        if self.features == "S" and self.enc_in != 1:
            raise ValueError("S forecasting requires enc_in=1")
        if self.c_out != self.enc_in and not (self.features == "MS" and self.c_out == 1):
            raise ValueError("c_out must equal enc_in, except MS allows c_out=1")
        for name, expected in (("down_sampling_layers", 3), ("down_sampling_window", 2),
                               ("down_sampling_method", "avg"), ("channel_independence", 1),
                               ("use_norm", 1), ("decomp_method", "moving_avg")):
            if getattr(configs, name, expected) != expected:
                raise ValueError(f"SeasonTrendMixer requires {name}={expected}")
        d_model = getattr(configs, "d_model", 32)
        n_heads = getattr(configs, "n_heads", 4)
        d_ff = getattr(configs, "d_ff", 64)
        depth = getattr(configs, "e_layers", 2)
        top_k = getattr(configs, "top_k", 3)
        dropout = getattr(configs, "dropout", 0.2)
        moving_avg = getattr(configs, "moving_avg", 25)
        if min(d_model, n_heads, d_ff, depth, top_k) < 1 or d_model % n_heads:
            raise ValueError("positive dimensions required; d_model must be divisible by n_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if moving_avg < 1 or moving_avg % 2 == 0:
            raise ValueError("moving_avg must be a positive odd number")
        self.lengths = [self.seq_len // (2 ** i) for i in range(4)]
        self.normalize = Normalize(self.enc_in, affine=False)
        self.decomposition = series_decomp(moving_avg)
        self.season_encoders = nn.ModuleList([
            PeriodPatchEncoder(length, d_model, n_heads, d_ff, depth, dropout, top_k)
            for length in self.lengths[:2]
        ])
        self.season_linears = nn.ModuleList([
            temporal_linear(length, length, identity=True) for length in self.lengths[2:]
        ])
        self.trend_linears = nn.ModuleList([
            temporal_linear(length, length, identity=True) for length in self.lengths
        ])
        self.season_mixing = TemporalScaleMixing(self.lengths, fine_to_coarse=True)
        self.trend_mixing = TemporalScaleMixing(self.lengths, fine_to_coarse=False)
        self.season_heads = nn.ModuleList([
            temporal_linear(length, self.pred_len) for length in self.lengths
        ])
        self.trend_heads = nn.ModuleList([
            temporal_linear(length, self.pred_len) for length in self.lengths
        ])
        get_logger(__name__).info(
            "SeasonTrendMixer: scales=%s; one RevIN and one original-scale moving average "
            "(%d); six identity temporal preprocessors (season=%s, trend=%s); "
            "FFT per sample/channel on fine seasonal scales, top_k=%d, "
            "period=floor(local_length/frequency_index), DC excluded, duplicate weights merged; "
            "encoding=period_patch_conv; %d Pre-LN Transformer layers; one period per token; "
            "patch-local residual Conv1d(1,D,k=3)-GELU-Conv1d(D,1,k=1); "
            "shared Linear(T,D)/Linear(D,T); sinusoidal position added to every patch token; "
            "right-zero-pad patches, no masks, decode/crop then period-weighted fusion; "
            "season fine-to-coarse / trend coarse-to-fine Linear-GELU-Linear mixing; "
            "eight uniform temporal heads; sum seasonal+trend and SUM four scales; "
            "one inverse RevIN; channel-independent",
            self.lengths, moving_avg, self.lengths[2:], self.lengths, top_k, depth,
        )

    @staticmethod
    def downsample(x):
        scales = [x]
        for _ in range(3):
            x = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            scales.append(x)
        return scales

    def forecast(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"Expected input [B, {self.seq_len}, {self.enc_in}]")
        seasonal, trend = self.decomposition(self.normalize(x_enc, "norm"))
        season_scales, trend_scales = self.downsample(seasonal), self.downsample(trend)
        seasons = [encoder(value) for encoder, value in
                   zip(self.season_encoders, season_scales[:2])]
        seasons.extend(layer(value.transpose(1, 2)).transpose(1, 2)
                       for layer, value in zip(self.season_linears, season_scales[2:]))
        trends = [layer(value.transpose(1, 2)).transpose(1, 2)
                  for layer, value in zip(self.trend_linears, trend_scales)]
        seasons = self.season_mixing(seasons)
        trends = self.trend_mixing(trends)
        predictions = [
            (season_head(season.transpose(1, 2))
             + trend_head(trend.transpose(1, 2))).transpose(1, 2)
            for season_head, trend_head, season, trend in
            zip(self.season_heads, self.trend_heads, seasons, trends)
        ]
        output = self.normalize(torch.stack(predictions, dim=0).sum(dim=0), "denorm")
        if self.features == "MS" and self.c_out == 1:
            output = output[:, :, -1:]
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Calendar marks, decoder values and masks are intentionally unused.
        return self.forecast(x_enc)
