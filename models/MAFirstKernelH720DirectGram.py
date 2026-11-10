"""Direct dense similarity matrix head; preserve the full single RevIN model."""
import torch
from torch import nn
from models.MAFirstKernelH720SingleRevIN import Model as FrozenSingleRevIN


def direct_similarity(logits):
    """Symmetric [0,1] similarity with unit diagonal; no PSD constraint."""
    symmetric = (logits + logits.transpose(-1, -2)) * 0.5
    similarity = torch.sigmoid(symmetric)
    diagonal = torch.eye(logits.shape[-1], device=logits.device, dtype=torch.bool)
    return torch.where(diagonal, torch.ones_like(similarity), similarity)


class DirectFutureGram(nn.Module):
    def __init__(self, length, patches, width):
        super().__init__()
        self.temporal = nn.Linear(length, patches, bias=False)
        # Keep the historical scale information previously used by the RBF head.
        # Each future row directly predicts all patch similarities, not coordinates.
        self.rows = nn.Sequential(nn.Linear(2 * width + 1, 16), nn.GELU(), nn.Linear(16, patches, bias=False))

    def forward(self, seasonal, trend, raw_scale):
        joint = torch.cat((seasonal, trend), dim=-1)
        future = self.temporal(joint.transpose(1, 2)).transpose(1, 2)
        history_scale = raw_scale.expand(-1, future.shape[1], 1)
        logits = self.rows(torch.cat((future, history_scale), dim=-1))
        return direct_similarity(logits)


class Model(FrozenSingleRevIN):
    def __init__(self, configs):
        super().__init__(configs)
        self.kernel_head = DirectFutureGram(self.seq_len, self.pred_len // 4, configs.d_model)
        self.direct_matrix_head = True
