"""Matched MAFirst period ablation with an unchanged initial cosine basis."""
import math

import torch
from torch import nn
from torch.nn.utils import parametrize

from models.OscillatorPDMStatePatchSpectralTimevaryFutureHiddenV320261105 import InheritedCombination


class LearnedCosinePeriods(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.offsets = nn.Parameter(torch.zeros(4))
        indices = torch.arange(steps, dtype=torch.float32)[:, None] + 0.5
        harmonics = torch.arange(1, 5, dtype=torch.float32)
        self.register_buffer("phase", math.pi * indices * harmonics / steps, persistent=False)

    def forward(self, original):
        phase = self.phase
        # Center on the stored reference to retain its exact values at zero offsets
        # even when CPU and CUDA cos have different last-bit rounding.
        return original + ((phase * (-self.offsets).exp()).cos() - phase.cos())


class Model(InheritedCombination):
    def __init__(self, configs):
        if (configs.seq_len, configs.pred_len) != (96, 96):
            raise ValueError("This ablation uses input 96 and forecast 96")
        if configs.combo_direct or not configs.combo_spectral or not configs.combo_timevary:
            raise ValueError("Expected the inherited future-hidden frequency head")
        super().__init__(configs)
        parametrize.register_parametrization(
            self, "position_frequency_basis", LearnedCosinePeriods(self.pred_len // 4)
        )

    @property
    def learnable_period_offsets(self):
        return self.parametrizations.position_frequency_basis[0].offsets
