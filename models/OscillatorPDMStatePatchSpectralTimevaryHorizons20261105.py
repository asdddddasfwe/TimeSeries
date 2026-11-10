"""Horizon extension of the frozen MAFirst spectral and bounded rotation combination."""
from models.OscillatorPDMStatePatchBestCombination20261104 import InheritedCombination


class Model(InheritedCombination):
    def __init__(self, configs):
        if (configs.seq_len != 96 or configs.pred_len not in (96, 192, 336, 720)
                or not (configs.combo_spectral and configs.combo_timevary)
                or configs.combo_direct):
            raise ValueError('Expected ETTm1 input 96, a supported horizon, and spectral plus bounded rotation')
        super().__init__(configs)
