"""Four-resolution decomposition, FFT axial encoding and lag-message forecasts."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.StandardNorm import Normalize
from layers.LagARFormer import LagMessageSeasonal
from utils.logger import get_logger


class Model(nn.Module):
    """Four native-rate forecasts -> original-rate linear fusion -> inverse RevIN."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len, self.pred_len, self.enc_in = configs.seq_len, configs.pred_len, configs.enc_in
        self.relation_weight_decay = getattr(configs, "lag_relation_weight_decay", -1.0)
        if (not math.isfinite(self.relation_weight_decay) or
                (self.relation_weight_decay < 0 and self.relation_weight_decay != -1)):
            raise ValueError("lag_relation_weight_decay must be -1 (inherit) or nonnegative")
        self.features = getattr(configs, "features", "M")
        self.c_out = getattr(configs, "c_out", self.enc_in)
        if self.task_name not in ("long_term_forecast", "short_term_forecast"):
            raise ValueError("LagARFormer only supports forecasting tasks")
        if self.seq_len < 16 or self.seq_len % 8 or self.pred_len < 1 or self.enc_in < 1:
            raise ValueError("Require seq_len divisible by 8 and >=16; positive pred_len/enc_in")
        if self.features not in ("M", "S", "MS"):
            raise ValueError("features must be M, S, or MS")
        if self.features == "S" and self.enc_in != 1:
            raise ValueError("S requires enc_in=1")
        if self.c_out != self.enc_in and not (self.features == "MS" and self.c_out == 1):
            raise ValueError("c_out must equal enc_in, except MS permits c_out=1")
        for name, expected in (("down_sampling_layers", 3), ("down_sampling_window", 2),
                               ("down_sampling_method", "avg"), ("channel_independence", 1),
                               ("use_norm", 1), ("decomp_method", "moving_avg")):
            if getattr(configs, name, expected) != expected:
                raise ValueError(f"LagARFormer requires {name}={expected}")
        moving_avg = getattr(configs, "moving_avg", 25)
        width = getattr(configs, "d_model", 32)
        d_ff, depth = getattr(configs, "d_ff", 64), getattr(configs, "e_layers", 2)
        dropout = getattr(configs, "dropout", 0.2)
        if min(width, d_ff, depth) < 1:
            raise ValueError("d_model, d_ff and e_layers must be positive")
        if moving_avg < 1 or moving_avg % 2 == 0 or not 0 <= dropout < 1:
            raise ValueError("Require positive odd moving_avg and dropout in [0,1)")
        for name in ("ar_max_lag", "ar_rho", "ar_forcing_rank", "ar_forcing_scale"):
            if hasattr(configs, name):
                raise ValueError(f"{name} was removed; use lag_max/lag_layers for message propagation")
        lag_limit = getattr(configs, "lag_max", 0)
        if not 0 <= lag_limit < self.seq_len:
            raise ValueError("lag_max must be 0 (all local lags) or in [1, seq_len-1]")
        message_depth = getattr(configs, "lag_layers", 2)
        chunk_size = getattr(configs, "lag_chunk_size", 8)
        self.strides = [1, 2, 4, 8]
        self.lengths = [self.seq_len // stride for stride in self.strides]
        self.horizons = [(self.pred_len + stride - 1) // stride for stride in self.strides]
        self.lag_limits = [length - 1 if lag_limit == 0 else min(lag_limit, length - 1)
                           for length in self.lengths]
        self.normalize = Normalize(self.enc_in, affine=False)
        self.decomposition = series_decomp(moving_avg)
        self.seasonals = nn.ModuleList([
            LagMessageSeasonal(length, horizon, width, d_ff, depth, dropout,
                               getattr(configs, "top_k", 5), local_lag, message_depth, chunk_size)
            for length, horizon, local_lag in zip(self.lengths, self.horizons, self.lag_limits)
        ])
        self.trend_heads = nn.ModuleList([
            nn.Linear(length, horizon) for length, horizon in zip(self.lengths, self.horizons)
        ])
        for length, head in zip(self.lengths, self.trend_heads):
            nn.init.constant_(head.weight, 1.0 / length)
            nn.init.zeros_(head.bias)
        # Unconstrained learned scale readout, initialized to an arithmetic mean.
        self.scale_fusion = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.scale_fusion.weight, 0.25)
        self.register_buffer("multiresolution_version", torch.tensor(5, dtype=torch.long))
        get_logger(__name__).info(
            "LagARFormer v5: multiresolution=%s, strides=%s, native horizons=%s; "
            "one RevIN and original-resolution moving-average decomposition (%d); "
            "season/trend separately avg-pool x2 three times; channel-independent; "
            "four independent seasonal encoders and four Linear(T_i,H_i) trend heads; "
            "encoding=fft_timemixerpp_row_column, periods=floor(T_i/bin); "
            "FFT top_k=%d per sample/channel/scale, unmasked right-zero-padding; "
            "%d parallel row/column groups, d_ff=Q/K width, n_heads unused; "
            "decoder=parallel_lag_messages, %d layers, local lag limits=%s; "
            "local lag tau equals stride*tau original time steps; "
            "Linear(T_i,H_i)+local future positions; chunk size=%d; "
            "native-rate future message sources, then repeat_interleave(stride) and crop H; "
            "sum season+trend per scale; learned Linear(4,1) fusion initialized to MEAN; "
            "one inverse RevIN; no AR/DCT decoder or additional lag-selection gate",
            self.lengths, self.strides, self.horizons, moving_avg,
            getattr(configs, "top_k", 5), depth, message_depth, self.lag_limits, chunk_size,
        )

    def optimizer_parameter_groups(self, weight_decay):
        """Override only ACF/lag encoders; source, target and all other weights inherit."""
        if self.relation_weight_decay == -1:
            return [{"params": list(self.parameters()), "weight_decay": weight_decay,
                     "group_name": "all"}]
        relation_ids = {
            id(parameter)
            for encoder in self.seasonals
            for module in [encoder.lag_embedding] + [
                block.lag_projection for block in encoder.message_blocks]
            for parameter in module.parameters()
        }
        regular, relation = [], []
        for parameter in self.parameters():
            (relation if id(parameter) in relation_ids else regular).append(parameter)
        return [
            {"params": regular, "weight_decay": weight_decay, "group_name": "regular"},
            {"params": relation, "weight_decay": self.relation_weight_decay,
             "group_name": "lag_relation"},
        ]

    @staticmethod
    def downsample(x):
        """Aligned non-overlapping averages on the history; no future data needed."""
        scales = [x]
        for _ in range(3):
            x = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            scales.append(x)
        return scales

    def restore_resolution(self, prediction, stride):
        """Local future bin j maps to original steps [j*stride, (j+1)*stride)."""
        return prediction.repeat_interleave(stride, dim=1)[:, :self.pred_len]

    def prepare_trend_scales(self, trend):
        """Original trend pyramid; experimental subclasses may mix before prediction."""
        return self.downsample(trend)

    def predict_scales(self, season, trend, zero_acf=False):
        """Normalized inputs -> [B,H,C,4] predictions on the original time grid."""
        season_scales, trend_scales = self.downsample(season), self.prepare_trend_scales(trend)
        predictions = []
        for stride, encoder, head, seasonal, baseline in zip(
                self.strides, self.seasonals, self.trend_heads, season_scales, trend_scales):
            native = encoder(seasonal, zero_acf=zero_acf) + head(baseline.transpose(1, 2)).transpose(1, 2)
            predictions.append(self.restore_resolution(native, stride))
        return torch.stack(predictions, dim=-1)

    def forecast(self, x_enc, zero_acf=False):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"Expected [B,{self.seq_len},{self.enc_in}]")
        season, trend = self.decomposition(self.normalize(x_enc, "norm"))
        predictions = self.predict_scales(season, trend, zero_acf=zero_acf)
        output = self.normalize(self.scale_fusion(predictions).squeeze(-1), "denorm")
        return output[:, :, -1:] if self.features == "MS" and self.c_out == 1 else output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc)
