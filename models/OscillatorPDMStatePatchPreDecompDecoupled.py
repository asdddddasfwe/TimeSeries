"""Independent FFT/sliding patch codecs and PDM paths; no cross dynamics."""
import copy
from torch import nn
from models.OscillatorPDMStatePatchPreDecomp import Model as PreDecomp
from models.OscillatorPDMStatePatchDisconnected import DisconnectedDecoder


class Model(PreDecomp):
    def __init__(self,configs):
        super().__init__(configs)
        self.trend_patch_codecs = copy.deepcopy(self.shared_patch_codecs)
        self.season_patch_codecs = self.shared_patch_codecs
        del self.shared_patch_codecs
        for codec in self.season_patch_codecs:
            codec.patch_projection.register_parameter('bias',None)
            codec.patch_readout.register_parameter('bias',None)
        for encoder in self.seasonals: encoder.decoder = DisconnectedDecoder(encoder.decoder)

    def codecs(self,index):
        return self.season_patch_codecs[index],self.trend_patch_codecs[index]

    def optimizer_parameter_groups(self,weight_decay):
        from models.OscillatorPDMSegmentPE import Model as SegmentPE
        return SegmentPE.optimizer_parameter_groups(self,weight_decay)
