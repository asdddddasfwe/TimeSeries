"""Only the finest future drive loses temporal accumulation; history memory stays."""
import torch
from models.OscillatorTimeMix import Model as TimeMix
from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorTimeMix import complex_oscillator_scan
from utils.logger import get_logger


def history_response_plus_local_drive(transition, drive, initial):
    """Return prod(A[0:h])*z0 + b[h], with no propagation of earlier b.

    Operates before the nonlinear readout. A and the historical state remain
    trainable; coarse-scale future scans and all historical scans are unchanged.
    """
    return complex_oscillator_scan(transition, torch.zeros_like(drive), initial) + drive


class Model(TimeMix):
    """Independent experiment model; identical parameters to OscillatorTimeMix.

    Load checkpoints with this model name: tensor shapes intentionally match the
    control, but the finest-scale forecast equation differs.
    """
    def __init__(self, configs):
        super().__init__(configs)
        get_logger(__name__).info(
            "OscillatorLocalDriveTimeMix OVERRIDE: finest seasonal future state "
            "is A^h*z0+b_h; retain full history scan, coarse future recurrence, "
            "nonlinear readout and TimeMixer mixing/sum. No new parameters.")

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
        for scale, (encoder, head, s, t, history) in enumerate(zip(self.seasonals, self.trend_heads, seasons, trends, histories)):
            seasonal = calibrated_seasonal_forecast(
                encoder, s, alpha, beta, intervention, encoded_history=history, scan_function=(history_response_plus_local_drive if scale == 0 else complex_oscillator_scan))
            predictions.append(seasonal + head(t.transpose(1, 2)).transpose(1, 2))
        return torch.stack(predictions, dim=-1)  # [B,H,C,4], all on original grid

