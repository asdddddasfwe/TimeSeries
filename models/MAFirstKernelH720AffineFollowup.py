"""New H720 affine-only follow-up; preserve frozen kernel/parallel implementation."""
import copy
from torch import nn
from layers.StandardNorm import Normalize
from models.MAFirstKernelTransferAblation import Model as FrozenKernelParallel
from models.MAFirstKernelTransferAblation import kernel_supervision, parallel_future_states

class Model(FrozenKernelParallel):
    def __init__(self, configs):
        # The frozen parent only constructs the OFF arm. Replace solely its
        # four Normalize modules; all original state names stay unchanged.
        parent_configs = copy.copy(configs)
        parent_configs.study_affine = False
        parent_configs.study_arm = 'kernel'
        parent_configs.study_parallel = True
        parent_configs.study_feedback_enabled = True
        super().__init__(parent_configs)
        if configs.pred_len != 720 or configs.enc_in != 7:
            raise ValueError('This follow-up is fixed to H720, seven variables')
        self.normalize = Normalize(self.enc_in, affine=True)
        self.coarse_normalizers = nn.ModuleList(
            [Normalize(self.enc_in, affine=True) for _ in range(3)])
        self.affine_enabled = True
