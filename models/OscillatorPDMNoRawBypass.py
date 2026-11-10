"""Seasonal readout split with the raw-input trend bypass removed.

This changes only the output paths of OscillatorPDMReadoutSplit(seasonal).
Full-history FFT routing and the two PDM blocks remain unchanged.
"""
import torch

from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorTimeMix import complex_oscillator_scan
from models.OscillatorPDMReadoutSplit import Model as ReadoutSplit
from utils.logger import get_logger


class Model(ReadoutSplit):
    def __init__(self, configs):
        super().__init__(configs, variant="seasonal")
        # Construct first to preserve the control's initialization RNG and all
        # retained weights. Delete modules, rather than freeze or zero outputs:
        # raw-bypass parameters must not enter state_dict or any optimizer.
        del self.raw_trend_decomposition
        del self.raw_trend_mix
        del self.raw_trend_heads
        self.raw_trend_bypass = False
        get_logger(__name__).info(
            "OscillatorPDMNoRawBypass overrides inherited routing descriptions: "
            "raw MA bypass removed; recurrent4 + feature_trend4 only; "
            "full-history FFT/PDM backbone unchanged")

    def predict_components(self, x_enc):
        """Return normalized [B,pred_len,C,8]: recurrent4, feature_trend4."""
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected observed history [B,seq_len,enc_in]")
        scales, histories = self.encode_multiscale(self.downsample(x_enc))
        recurrent, trends = self.decoder_histories(histories)
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        outputs = [
            calibrated_seasonal_forecast(
                encoder, scale, alpha, beta, encoded_history=history,
                scan_function=complex_oscillator_scan)
            for encoder, scale, history in zip(self.seasonals, scales, recurrent)
        ]
        feature = [
            head(trend).reshape(batch, channels, self.pred_len).transpose(1, 2)
            for head, trend in zip(self.feature_trend_heads, trends)
        ]
        return torch.stack(outputs + feature, dim=-1)

    def forecast(self, x_enc, zero_acf=False, intervention="normal"):
        if zero_acf or intervention != "normal":
            raise ValueError("No-raw-bypass model supports normal forecasting only")
        parts = self.predict_components(x_enc)
        prediction = parts[..., :4].sum(-1) + parts[..., 4:].sum(-1)
        prediction = self.normalize(prediction, "denorm")
        return prediction[..., -1:] if self.features == "MS" and self.c_out == 1 else prediction
