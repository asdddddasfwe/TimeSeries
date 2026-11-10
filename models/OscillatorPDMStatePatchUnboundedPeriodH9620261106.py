"""H96 extension of the frozen unbounded period angle model; no H720 learned weights."""
from models.OscillatorPDMStatePatchUnboundedPeriodAngle20261105 import InheritedCombination

class Model(InheritedCombination):
    def __init__(self, configs):
        if (configs.seq_len, configs.pred_len) != (96, 96) or configs.combo_direct or not (configs.combo_spectral and configs.combo_timevary):
            raise ValueError("Expected input96 H96 spectral and bounded rotation")
        super().__init__(configs)
