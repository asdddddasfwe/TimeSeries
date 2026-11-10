"""Scale-aware PDM moving averages derived from the observed-history lengths."""
from numbers import Integral

from torch import nn

from layers.Autoformer_EncDec import series_decomp
from models.OscillatorPDMScaleAllocation import Model as ScaleModel


def scaled_odd_kernel(base_kernel, base_length, scale_length):
    """Nearest positive odd integer to base_kernel * scale_length / base_length.

    Exact ties round upward. Integer arithmetic avoids floating-point rounding
    differences. The minimum is one, even after repeated downsampling.
    """
    for name, value in (
        ("base_kernel", base_kernel),
        ("base_length", base_length),
        ("scale_length", scale_length),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if base_kernel % 2 != 1:
        raise ValueError("base_kernel must be odd for same-length endpoint padding")
    if scale_length > base_length:
        raise ValueError("scale_length must not exceed base_length")
    return 2 * ((base_kernel * scale_length) // (2 * base_length)) + 1


class ScaleAwareDecomposition(nn.Module):
    """Decompose [B*C,T,D] using a kernel derived from its actual time length."""

    def __init__(self, base_kernel, base_length, scale_lengths):
        super().__init__()
        lengths = tuple(scale_lengths)
        if not lengths or len(set(lengths)) != len(lengths):
            raise ValueError("Expected nonempty, distinct scale lengths")
        self.base_kernel = base_kernel
        self.base_length = base_length
        self.kernels = {
            length: scaled_odd_kernel(base_kernel, base_length, length)
            for length in lengths
        }
        self.decompositions = nn.ModuleDict({
            str(length): series_decomp(kernel)
            for length, kernel in self.kernels.items()
        })

    def forward(self, x):
        if x.ndim != 3 or str(x.shape[1]) not in self.decompositions:
            raise ValueError("Expected [B*C,T,D] with a configured history length")
        return self.decompositions[str(x.shape[1])](x)


class Model(ScaleModel):
    """Keep the full PDM/readout/bypass; scale only feature MA windows."""

    def __init__(self, configs):
        # Validate before constructing layers or consuming initialization RNG.
        scaled_odd_kernel(configs.moving_avg, configs.seq_len, configs.seq_len)
        super().__init__(configs, variant="bypass")
        self.ma_kernels = {
            length: scaled_odd_kernel(configs.moving_avg, configs.seq_len, length)
            for length in self.lengths
        }
        for block in self.pdm_blocks:
            block.decompsition = ScaleAwareDecomposition(
                configs.moving_avg, configs.seq_len, self.lengths
            )
