"""Frozen trained MAFirst control with a trainable post-PDM correction only."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import torch
from torch import nn
from models.OscillatorPDMStatePatchMAFirst import Model as Control
from models.OscillatorPDMStatePatchMAFirstTrendPatchTransformer import PatchTransformerTrendHead

CONTROL_SHA256 = "752a3a72a5cf17c6a643adde5e8a5c76af6334009f4e387bc6e0a5eacc026f22"

class CorrectionHead(PatchTransformerTrendHead):
    def __init__(self, original, scale):
        super().__init__(original, scale)
        # Original affine heads remain exclusively in the frozen control.
        del self.temporal
        del self.readout

    def forward(self, history):
        return self.nonlinear_readout(self.encode_history(history).flatten(1))

class CorrectionHeads(nn.ModuleList):
    def optimizer_parameter_groups(self, weight_decay):
        return [dict(params=list(self.parameters()), weight_decay=weight_decay,
            group_name="post_pdm_correction")]

class Model(nn.Module):
    """Correction units are converted using the control's instance stdev."""
    def __init__(self, configs):
        super().__init__()
        self.base = Control(configs)
        self.corrections = CorrectionHeads([
            CorrectionHead(head, scale)
            for scale, head in enumerate(self.base.feature_trend_heads)
        ])
        self.pred_len = configs.pred_len
        self.base.requires_grad_(False)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def frozen_features(self, x):
        histories = []
        handles = [
            head.register_forward_pre_hook(lambda module, args: histories.append(args[0]))
            for head in self.base.feature_trend_heads
        ]
        try:
            with torch.no_grad():
                parts = self.base.predict_components(x)
                base_prediction = self.base.normalize(
                    parts[..., :4].sum(-1) + parts[..., 4:].sum(-1), "denorm")
                scale = self.base.normalize.stdev.detach().clone()
                mean = self.base.normalize.mean.detach().clone()
        finally:
            for handle in handles:
                handle.remove()
        assert len(histories) == 4
        return base_prediction, [h.detach() for h in histories], parts, scale, mean

    def forward(self, x, return_details=False):
        base, histories, parts, scale, mean = self.frozen_features(x)
        batch, _, channels = x.shape
        corrections = torch.stack([
            head(h).reshape(batch, channels, self.pred_len).transpose(1, 2)
            for head, h in zip(self.corrections, histories)
        ], -1)
        physical_corrections = corrections * scale.unsqueeze(-1)
        prediction = base + physical_corrections.sum(-1)
        if return_details:
            return prediction, dict(base_prediction=base, corrections=physical_corrections,
                components=parts * scale.unsqueeze(-1), history_mean=mean)
        return prediction

def load_trained_control(root, logdir):
    root = Path(root)
    folder = root / "tmp/mafirst_memory_factorial_20261017_full_h720_control_seed2021"
    checkpoint = folder / "best.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CONTROL_SHA256
    cfg = json.loads((folder / "model_config.json").read_text())
    cfg.update(log_dir=str(logdir), tensorboard=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20211014)
        model = Model(SimpleNamespace(**cfg))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.base.load_state_dict(state, strict=True)
    assert len(state) == 236
    assert all(torch.equal(state[k], model.base.state_dict()[k]) for k in state)
    assert sum(p.numel() for p in model.base.parameters()) == 320450
    assert sum(p.numel() for p in model.corrections.parameters()) == 414272
    assert len(list(model.corrections.parameters())) == 72
    return model, cfg
