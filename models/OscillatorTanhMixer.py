"""OscillatorCalibratedMixer with a pointwise Tanh before state readout."""
from layers.OscillatorTanhMixer import TanhReadout
from models.OscillatorCalibratedMixer import Model as CalibratedModel
from utils.logger import get_logger


class Model(CalibratedModel):
    def __init__(self, configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            encoder.decoder.readout = TanhReadout(encoder.decoder.readout)
        get_logger(__name__).info(
            "OscillatorTanhMixer v1 overrides LINEAR readout above: "
            "calibrated states -> elementwise Tanh -> original bias-free Linear(2J,1); "
            "same parameter count and two affine parallel scans per scale; "
            "activation is OUTSIDE recurrence and does not feed back into states; "
            "alpha/beta scale state contributions before Tanh, not additive final predictions; "
            "all four seasonal readouts use Tanh; trends/fusion/inverse RevIN unchanged")
