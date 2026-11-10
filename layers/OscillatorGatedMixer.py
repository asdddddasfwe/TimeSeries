"""History-conditioned gates on generated future features, outside the scan."""
import torch
from torch import nn
from torch.nn import functional as F

from layers.OscillatorHistoryMixer import HistoryInitializedDecoder


class FutureFeatureGate(nn.Module):
    """Return [N,H,D] gains from normalized future features and past context.

    Each series is independent. Parameters are shared across forecast positions
    and variables, while the gains depend on their features. Zero initialization
    gives exactly one via 2*sigmoid(0), preserving the ungated initial forecast.
    """

    def __init__(self, width):
        super().__init__()
        self.context_norm = nn.LayerNorm(2 * width, elementwise_affine=False)
        self.projection = nn.Linear(3 * width, width)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, normalized_future, history):
        context = self.context_norm(torch.cat((history.mean(1), history[:, -1]), dim=-1))
        context = context[:, None].expand(-1, normalized_future.shape[1], -1)
        return 2 * torch.sigmoid(self.projection(torch.cat((normalized_future, context), dim=-1)))


class GatedHistoryDecoder(HistoryInitializedDecoder):
    """Only future drive features are gated; historical state scanning is shared."""

    def __init__(self, decoder):
        super().__init__(decoder)
        self.future_gate = FutureFeatureGate(self.drive_projection.in_features)
        self.register_buffer("future_feature_gate_version", torch.tensor(1, dtype=torch.long))

    def future_drive(self, future_features, history=None):
        if history is None:
            raise ValueError("Future feature gating requires encoded historical context")
        normalized = self.drive_norm(future_features)
        features = F.gelu(normalized) * self.future_gate(normalized, history)
        return self.drive_projection(self.drive_dropout(features))
