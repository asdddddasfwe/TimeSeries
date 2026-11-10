"""Future-gated oscillator with Linear -> GELU -> Linear state readout."""
from layers.OscillatorGatedMLPMixer import GELUMLPReadout
from models.OscillatorGatedMixer import Model as GatedModel
from utils.logger import get_logger


class Model(GatedModel):
    def __init__(self, configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            decoder = encoder.decoder
            decoder.readout = GELUMLPReadout(2 * decoder.modes, hidden_features=32)
        get_logger(__name__).info(
            "OscillatorGatedMLPMixer v1 REPLACES Tanh readout above: "
            "calibrated scanned states -> Linear(2J,32) -> GELU -> Linear(32,1); "
            "four independent readouts, shared across positions and variables; "
            "standard Linear weight initialization and zero biases; no Tanh, "
            "extra readout dropout, residual bypass or output activation; "
            "future gates and history/future affine scans unchanged; all readout "
            "parameters use regular weight decay; trend/fusion/inverse RevIN unchanged")
