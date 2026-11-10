"""Single-factor intervention: MA3 only on 12-point PDM features."""
from torch import nn
from layers.Autoformer_EncDec import series_decomp
from models.OscillatorPDMScaleAllocation import Model as ScaleModel


class ScaleSelectiveDecomposition(nn.Module):
    """Keep the original MA25 object on every length except the declared 12."""
    def __init__(self, original, coarse_kernel):
        super().__init__()
        self.original = original
        self.coarse = series_decomp(coarse_kernel)
        self.coarse_kernel = coarse_kernel

    def forward(self, x):
        return self.coarse(x) if x.shape[1] == 12 else self.original(x)


class Model(ScaleModel):
    def __init__(self, configs, variant=None):
        super().__init__(configs, variant='bypass')
        self.coarse_ma_variant = variant or getattr(configs, 'coarse_ma_variant', 'ma25')
        if self.coarse_ma_variant not in ('ma25', 'coarse_ma3'):
            raise ValueError('Unknown coarse MA intervention')
        if configs.seq_len != 96 or configs.moving_avg != 25 or self.lengths != [96, 48, 24, 12]:
            raise ValueError('This intervention requires 96/48/24/12 histories and base MA25')
        if self.coarse_ma_variant == 'coarse_ma3':
            for block in self.pdm_blocks:
                block.decompsition = ScaleSelectiveDecomposition(block.decompsition, 3)
