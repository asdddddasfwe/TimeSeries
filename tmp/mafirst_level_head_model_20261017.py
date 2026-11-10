"""History statistics to additive smooth level correction; all backbone weights train."""
import torch
from torch import nn
from models.OscillatorPDMStatePatchMAFirst import Model as Original

class Model(Original):
    def __init__(self, configs):
        super().__init__(configs)
        # torch.zeros consumes no RNG. All original initialization draws remain intact.
        self.level_head_weight = nn.Parameter(torch.zeros(self.enc_in, 2, 4))
        self.level_head_bias = nn.Parameter(torch.zeros(self.enc_in, 4))

    def level_correction(self, x):
        # Statistics of observed history only, in dataset-standardized units.
        features = torch.stack((x.mean(1), x.std(1, unbiased=False)), dim=-1).detach()
        coefficients = torch.einsum('bcf,cfk->bck', features, self.level_head_weight) + self.level_head_bias
        t = torch.linspace(0, 3, self.pred_len, device=x.device, dtype=x.dtype)
        basis = (1 - (t[:, None] - torch.arange(4, device=x.device, dtype=x.dtype)).abs()).clamp_min(0)
        return torch.einsum('hk,bck->bhc', basis, coefficients)

    def forecast(self, x_enc, zero_acf=False, intervention='normal'):
        prediction = super().forecast(x_enc, zero_acf, intervention)
        correction = self.level_correction(x_enc)
        if self.features == 'MS' and self.c_out == 1:
            correction = correction[..., -1:]
        return prediction + correction
