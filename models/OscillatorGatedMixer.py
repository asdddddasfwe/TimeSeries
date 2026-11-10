"""OscillatorTanhMixer with history-conditioned future feature generation gates."""
from layers.OscillatorGatedMixer import GatedHistoryDecoder
from models.OscillatorTanhMixer import Model as TanhModel
from utils.logger import get_logger


class Model(TanhModel):
    def __init__(self, configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            encoder.decoder = GatedHistoryDecoder(encoder.decoder)
        count = sum(p.numel() for encoder in self.seasonals
                    for p in encoder.decoder.future_gate.parameters())
        get_logger(__name__).info(
            "OscillatorGatedMixer v1: %d added parameters; four independent future "
            "feature gates g=2*sigmoid(Linear(concat(LN(future),LN(history_mean,last)))); "
            "zero-initialized logits give g=1; per-window/variable/position/feature gains; "
            "drive=Linear(Dropout(g*GELU(LN(future)))); no normalization after gating; "
            "historical drive unchanged; same two affine scans and post-scan Tanh; "
            "all gates use observed history only; ordinary weight decay", count)
