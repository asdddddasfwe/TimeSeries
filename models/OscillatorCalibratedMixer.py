"""Minimal variable-specific history/drive calibration of OscillatorHistoryMixer."""
import torch
from torch import nn

from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from models.OscillatorHistoryMixer import Model as HistoryModel
from utils.logger import get_logger


class Model(HistoryModel):
    def __init__(self, configs):
        super().__init__(configs)
        # Effective gains start at one. Ordinary weight decay on zero-centered
        # offsets shrinks toward the original model, not toward zero output.
        self.history_gain_offset = nn.Parameter(torch.zeros(self.enc_in))
        self.drive_gain_offset = nn.Parameter(torch.zeros(self.enc_in))
        self.register_buffer("oscillator_calibration_version", torch.tensor(1, dtype=torch.long))
        get_logger(__name__).info(
            "OscillatorCalibratedMixer v1: %d added scalar parameters (2 per input variable); "
            "alpha=1+history_gain_offset and beta=1+drive_gain_offset start at 1; "
            "gains shared over four scales and forecast positions; no cross-variable mixing; "
            "alpha scales historical final state, beta scales FUTURE drive only; "
            "trend unchanged, calibration before scale fusion and single inverse RevIN; "
            "unbounded gains, ordinary weight decay on zero-centered offsets; "
            "same two parallel scans, no auxiliary loss", 2 * self.enc_in)

    def branch_gains(self):
        """Effective alpha/beta [C]; variable order follows dataset feature order."""
        return 1 + self.history_gain_offset, 1 + self.drive_gain_offset

    def predict_scales(self, season, trend, zero_acf=False, intervention="normal", history_stats=None):
        if zero_acf:
            raise ValueError("OscillatorCalibratedMixer has no ACF-token decoder")
        alpha, beta = self.branch_gains()
        if intervention == "uncalibrated":
            alpha, beta = torch.ones_like(alpha), torch.ones_like(beta)
            intervention = "normal"
        predictions = []
        for stride, encoder, head, seasonal, baseline in zip(
                self.strides, self.seasonals, self.trend_heads,
                self.downsample(season), self.prepare_trend_scales(trend)):
            native = calibrated_seasonal_forecast(encoder, seasonal, alpha, beta, intervention, history_stats)
            native = native + head(baseline.transpose(1, 2)).transpose(1, 2)
            predictions.append(self.restore_resolution(native, stride))
        return torch.stack(predictions, dim=-1)
