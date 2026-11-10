"""Trend-input oscillators with TimeMixer directional mixing and summed heads."""
import copy

import torch
from torch import nn

from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorTimeMix import FullResolutionGatedDecoder, forecast_positions, complex_oscillator_scan
from models.OscillatorTrendInputMixer import Model as TrendInputModel
from models.TimeMixer import MultiScaleSeasonMixing, MultiScaleTrendMixing
from utils.logger import get_logger


class Model(TrendInputModel):
    """Preserve the oscillator backbone; restore TimeMixer's two fusion stages."""

    def __init__(self, configs):
        super().__init__(configs)
        cfg = copy.copy(configs)
        cfg.down_sampling_layers, cfg.down_sampling_window = 3, 2
        # Direct reuse, including GELU and cumulative additive residuals.
        self.season_mix = MultiScaleSeasonMixing(cfg)
        self.trend_mix = MultiScaleTrendMixing(cfg)
        self.horizons = [self.pred_len] * 4
        for stride, length, encoder in zip(self.strides, self.lengths, self.seasonals):
            encoder.horizon = self.pred_len
            encoder.temporal_projection = nn.Linear(length, self.pred_len)
            encoder.future_positions = forecast_positions(
                length, self.pred_len, encoder.value_embedding.out_features, stride)
            encoder.decoder = FullResolutionGatedDecoder(encoder.decoder, self.pred_len, stride)
        self.trend_heads = nn.ModuleList([nn.Linear(t, self.pred_len) for t in self.lengths])
        for length, head in zip(self.lengths, self.trend_heads):
            nn.init.constant_(head.weight, 1. / length)
            nn.init.zeros_(head.bias)
        del self.scale_fusion
        self.register_buffer("timemixer_fusion_version", torch.tensor(1, dtype=torch.long))
        get_logger(__name__).info(
            "OscillatorTimeMix v1 OVERRIDES inherited native-horizon/fusion logs: "
            "seasonal FFT encoded features mix fine->coarse; raw trend histories "
            "mix coarse->fine; BOTH reuse TimeMixer Linear/GELU/Linear residual maps; "
            "trend input to FFT encoders remains the unmixed aligned history; "
            "all four heads predict full H=%d directly, NO repeat upsampling; "
            "sum season+trend per scale, then SUM four scales, NO learned scale weights; "
            "coarse historical transitions converted to original-step future transitions; "
            "native-complex affine scan (same equations and checkpoint keys); "
            "one inverse RevIN; old model checkpoints require separate old models",
            self.pred_len)

    def load_base_checkpoint(self, state):
        raise ValueError("Different fusion and head shapes: train OscillatorTimeMix from scratch; "
                         "restore only its own checkpoint with strict=True")

    def prepare_trend_scales(self, trend, scales=None):
        scales = self.downsample(trend) if scales is None else scales
        return self.trend_mix([x.transpose(1, 2) for x in scales])

    def predict_scales(self, season, trend, zero_acf=False, intervention="normal", history_stats=None):
        if zero_acf or history_stats is not None:
            raise ValueError("OscillatorTimeMix has no ACF/statistics adapter")
        alpha, beta = self.branch_gains()
        if intervention == "uncalibrated":
            alpha, beta = torch.ones_like(alpha), torch.ones_like(beta)
            intervention = "normal"
        trend_mode = intervention if intervention in ("zero_trend_input", "reverse_trend_input") else "normal"
        if trend_mode != "normal":
            intervention = "normal"
        batch, _, channels = season.shape
        seasons, trends = self.downsample(season), self.downsample(trend)
        encoded = []
        for encoder, s, t in zip(self.seasonals, seasons, trends):
            context = torch.zeros_like(t) if trend_mode == "zero_trend_input" else t
            if trend_mode == "reverse_trend_input":
                context = context.flip(1)
            history = encoder.encode_history(
                s.transpose(1, 2).reshape(batch * channels, encoder.length),
                context.transpose(1, 2).reshape(batch * channels, encoder.length))
            encoded.append(history.transpose(1, 2))  # [B*C,D,T_i]
        histories = self.season_mix(encoded)  # [B*C,T_i,D], fine -> coarse
        trends = self.prepare_trend_scales(trend, scales=trends)  # [B,T_i,C], coarse -> fine
        predictions = []
        for encoder, head, s, t, history in zip(self.seasonals, self.trend_heads, seasons, trends, histories):
            seasonal = calibrated_seasonal_forecast(
                encoder, s, alpha, beta, intervention, encoded_history=history, scan_function=complex_oscillator_scan)
            predictions.append(seasonal + head(t.transpose(1, 2)).transpose(1, 2))
        return torch.stack(predictions, dim=-1)  # [B,H,C,4], all on original grid

    def forecast(self, x_enc, zero_acf=False, intervention="normal"):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"Expected [B,{self.seq_len},{self.enc_in}]")
        season, trend = self.decomposition(self.normalize(x_enc, "norm"))
        predictions = self.predict_scales(season, trend, zero_acf, intervention)
        output = self.normalize(predictions.sum(-1), "denorm")
        return output[..., -1:] if self.features == "MS" and self.c_out == 1 else output
