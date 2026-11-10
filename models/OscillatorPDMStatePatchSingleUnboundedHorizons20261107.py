"""Historical-hidden-conditioned unbounded physical periods for ETTm1/ETTm2.

The inherited forward is unchanged: the angle head sees history mean and last,
while projected future hidden remains the existing additive state drive only.
"""
from models.OscillatorPDMStatePatchUnboundedPeriodAngle20261105 import InheritedCombination

class Model(InheritedCombination):
    def __init__(self, configs):
        if configs.seq_len != 96 or configs.pred_len not in (96,192,336,720):
            raise ValueError('Expected input96 and forecast96/192/336/720')
        if configs.combo_direct or not (configs.combo_spectral and configs.combo_timevary):
            raise ValueError('Expected inherited spectral/history-conditioned dynamic rotation')
        super().__init__(configs)
