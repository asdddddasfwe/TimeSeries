"""MAFirst: replace only seasonal FFT/axial encoding with sliding patch attention."""
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from models.OscillatorPDMStatePatchMAFirst import Model as Original
from models.OscillatorDualEncoderTimeMixer import SlidingPatchTrendEncoder


class SeasonalSlidingPatchEncoder(SlidingPatchTrendEncoder):
    """Observed [N,T,D] -> patch4/stride2 -> Transformer -> native [N,T,D].

    Tokens retain chronological order. Fixed 1-D patch PE; one pre-norm
    4-head Transformer with FF64/GELU and configured dropout. Linear patch
    decoding and overlap averaging restore T, then residual + LayerNorm.
    No FFT routing, spectral token, future observation or new forecast head.
    """
    def forward(self, history):
        with sdpa_kernel(SDPBackend.MATH):
            return super().forward(history)


class Model(Original):
    def __init__(self, configs):
        super().__init__(configs)
        # Remove obsolete axial parameters rather than leave unused weights.
        for encoder in self.seasonals:
            del encoder.blocks
        encoders = []
        for scale in range(len(self.seasonals)):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20211027 + scale)
                encoders.append(SeasonalSlidingPatchEncoder(
                    configs.d_model, configs.d_ff, configs.dropout,
                    heads=4, patch_length=4, stride=2))
        self.season_patch_encoders = nn.ModuleList(encoders)

    def encode_seasonal_features(self, seasons):
        if len(seasons) != len(self.season_patch_encoders):
            raise ValueError("Expected four seasonal histories")
        return [encoder(history) for encoder, history
                in zip(self.season_patch_encoders, seasons)]
