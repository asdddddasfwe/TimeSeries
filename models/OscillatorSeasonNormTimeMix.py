"""History-only reversible normalization of the seasonal branch.

The trend branch and original whole-input RevIN remain unchanged. No new
learned parameters or state-dict entries. This targets location/scale drift,
not time-varying ACF or a guarantee of stationarity.
"""
import math
import torch
from models.OscillatorLowRankTimeMix import Model as LowRank
from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorTimeMix import complex_oscillator_scan


def normalize_season_history(season, floor=0.1):
    """[B,L,C] -> normalized history, detached [B,1,C] mean and scale.

    Statistics use observed history only. Variance is population variance.
    A fixed std floor prevents amplification of nearly constant histories.
    """
    if season.ndim != 3 or not math.isfinite(floor) or floor <= 0:
        raise ValueError("Expected history [B,L,C] and positive finite std floor")
    mean = season.mean(1, keepdim=True).detach()
    scale = season.var(1, keepdim=True, unbiased=False).sqrt().clamp_min(floor).detach()
    return (season - mean) / scale, mean, scale


class Model(LowRank):
    def __init__(self, configs, enabled=None):
        super().__init__(configs)
        self.seasonal_normalize = bool(getattr(configs, 'seasonal_normalize', True) if enabled is None else enabled)
        self.seasonal_std_floor = float(getattr(configs, 'seasonal_std_floor', .1))
        if not math.isfinite(self.seasonal_std_floor) or self.seasonal_std_floor <= 0:
            raise ValueError("Invalid seasonal std floor")

    def predict_scales(self, season, trend, zero_acf=False, intervention="normal", history_stats=None):
        if not self.seasonal_normalize:
            return super().predict_scales(season, trend, zero_acf, intervention, history_stats)
        season, location, scale = normalize_season_history(season, self.seasonal_std_floor)
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
            # Restore seasonal units; distribute the single location term
            # over heads so their sum adds the historical mean exactly once.
            restored = scale * seasonal + location / len(self.seasonals)
            predictions.append(restored + head(t.transpose(1, 2)).transpose(1, 2))
        return torch.stack(predictions, dim=-1)  # [B,H,C,4], all on original grid

