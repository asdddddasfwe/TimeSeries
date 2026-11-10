"""MA-first 0.318: independent embeddings and no learned cross-branch edge.

SeparateLinear already separates scalar embeddings; the 0.318 base already
separates PDM channel MLPs, scale maps and prediction heads. Remove the sole
remaining trend->season dynamics edge, retaining the original FFT and heads.
Shared observed-data normalization/MA and the total forecast loss are fixed
preprocessing/joint supervision, not learned exchanges between branches.
"""
from models.OscillatorPDMStatePatchMAFirstSeparateLinear import Model as SeparateLinear
from models.OscillatorPDMStatePatchDisconnected import DisconnectedDecoder


class Model(SeparateLinear):
    def __init__(self,configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            encoder.decoder=DisconnectedDecoder(encoder.decoder)

    def optimizer_parameter_groups(self,weight_decay):
        from models.OscillatorPDMSegmentPE import Model as SegmentPE
        return SegmentPE.optimizer_parameter_groups(self,weight_decay)
