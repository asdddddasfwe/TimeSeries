"""Four-resolution FFT encoding with parallel oscillatory state-space decoding."""
import copy
import math

import torch

from layers.OscillatorMixer import OscillatorSeasonal
from models.LagARFormer import Model as FourScaleModel
from utils.logger import get_logger


class Model(FourScaleModel):
    def __init__(self, configs):
        modes = getattr(configs, "ssm_modes", 8)
        dynamics_decay = getattr(configs, "ssm_dynamics_weight_decay", 0.)
        if modes < 1 or not math.isfinite(dynamics_decay) or dynamics_decay < 0:
            raise ValueError("ssm_modes must be positive; ssm_dynamics_weight_decay finite/nonnegative")
        # Reuse existing construction/validation and history initialization.
        # Legacy lag options do not configure this decoder.
        cfg = copy.copy(configs)
        cfg.lag_max, cfg.lag_layers, cfg.lag_chunk_size = 0, 1, 1
        cfg.lag_relation_weight_decay = -1.
        super().__init__(cfg)
        self.dynamics_weight_decay = dynamics_decay
        for index, encoder in enumerate(self.seasonals):
            self.seasonals[index] = OscillatorSeasonal(encoder, modes, getattr(configs, "dropout", .2))
        self.register_buffer("oscillator_model_version", torch.tensor(1, dtype=torch.long))
        get_logger(__name__).info(
            "OscillatorMixer v1 replaces ALL legacy lag-message decoders above: "
            "four resolutions=%s; native horizons=%s; unchanged FFT row/column history "
            "encoding, separate linear trend heads and learned scale fusion; "
            "%d two-dimensional oscillators per scale; history-conditioned decay/frequency "
            "shared over each horizon; initial state and parallel driving sequence from history; "
            "same-layer recurrence s[h]=A*s[h-1]+b[h] via work-efficient tree scan "
            "(O(H) work, O(log H) dependency depth for fixed state size); "
            "radius=exp(-softplus(rate)), frequency=pi*sigmoid(logit); "
            "ALL seasonal outputs read states, no F0 prediction bypass; "
            "one RevIN/inverse; calendar/future observations ignored; "
            "dynamics weight_decay=%g; no ACF tokens, lag gate or triangular solve",
            self.lengths, self.horizons, modes, dynamics_decay)

    def optimizer_parameter_groups(self, weight_decay):
        protected = set()
        for encoder in self.seasonals:
            decoder = encoder.decoder
            protected.update((id(decoder.log_decay), id(decoder.frequency_logits)))
            protected.update(id(p) for p in decoder.condition.parameters())
        return [
            {"params": [p for p in self.parameters() if id(p) not in protected],
             "weight_decay": weight_decay, "group_name": "regular"},
            {"params": [p for p in self.parameters() if id(p) in protected],
             "weight_decay": self.dynamics_weight_decay, "group_name": "ssm_dynamics"},
        ]

    def predict_scales(self, season, trend, zero_acf=False, intervention="normal"):
        if zero_acf:
            raise ValueError("OscillatorMixer has no ACF-token decoder")
        predictions = []
        for stride, encoder, head, seasonal, baseline in zip(
                self.strides, self.seasonals, self.trend_heads,
                self.downsample(season), self.prepare_trend_scales(trend)):
            native = encoder(seasonal, intervention=intervention)
            native = native + head(baseline.transpose(1, 2)).transpose(1, 2)
            predictions.append(self.restore_resolution(native, stride))
        return torch.stack(predictions, dim=-1)

    def forecast(self, x_enc, zero_acf=False, intervention="normal"):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"Expected [B,{self.seq_len},{self.enc_in}]")
        season, trend = self.decomposition(self.normalize(x_enc, "norm"))
        predictions = self.predict_scales(season, trend, zero_acf, intervention)
        output = self.normalize(self.scale_fusion(predictions).squeeze(-1), "denorm")
        return output[..., -1:] if self.features == "MS" and self.c_out == 1 else output
