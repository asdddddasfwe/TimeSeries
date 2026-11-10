"""Linear trend baseline plus a nonlinear branch observing season AND trend."""
import torch

from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorTrendInputMixer import TrendInputSeasonal
from models.OscillatorGatedMLPMixer import Model as GatedMLPModel
from utils.logger import get_logger


class Model(GatedMLPModel):
    def __init__(self, configs):
        super().__init__(configs)
        for i, encoder in enumerate(self.seasonals):
            self.seasonals[i] = TrendInputSeasonal(encoder)
        self.register_buffer("trend_input_version", torch.tensor(1, dtype=torch.long))
        get_logger(__name__).info(
            "OscillatorTrendInputMixer: four zero-initialized bias-free trend "
            "embeddings added BEFORE FFT image encoding, %d new parameters; "
            "shared RevIN coordinates; FFT periods still selected from season only; "
            "linear trend heads, future gates, dynamics and readouts unchanged; "
            "no future observations; complex branch is a nonlinear correction, "
            "not a supervised seasonal component", sum(
                e.trend_embedding.weight.numel() for e in self.seasonals))

    def load_base_checkpoint(self, state):
        own = self.state_dict()
        extra = {k for k in own if '.trend_embedding.' in k or k == 'trend_input_version'}
        if set(state) != set(own) - extra:
            raise ValueError("Expected an exact OscillatorGatedMLPMixer checkpoint")
        for key in extra:
            if '.trend_embedding.' in key:
                own[key] = torch.zeros_like(own[key])
        own.update(state)
        return self.load_state_dict(own, strict=True)

    def predict_scales(self, season, trend, zero_acf=False, intervention="normal", history_stats=None):
        if zero_acf or history_stats is not None:
            raise ValueError("Trend-input model has no ACF/statistics adapter")
        alpha, beta = self.branch_gains()
        if intervention == "uncalibrated":
            alpha, beta = torch.ones_like(alpha), torch.ones_like(beta)
            intervention = "normal"
        trend_mode = intervention if intervention in ("zero_trend_input", "reverse_trend_input") else "normal"
        if trend_mode != "normal":
            intervention = "normal"
        predictions = []
        for stride, encoder, head, s, t in zip(
                self.strides, self.seasonals, self.trend_heads,
                self.downsample(season), self.prepare_trend_scales(trend)):
            context = torch.zeros_like(t) if trend_mode == "zero_trend_input" else t
            if trend_mode == "reverse_trend_input":
                context = context.flip(1)
            native = calibrated_seasonal_forecast(encoder, s, alpha, beta,
                                                  intervention, history_trend=context)
            native = native + head(t.transpose(1, 2)).transpose(1, 2)
            predictions.append(self.restore_resolution(native, stride))
        return torch.stack(predictions, dim=-1)
