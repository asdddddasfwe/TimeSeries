"""Controlled history-state initialization variant of OscillatorMixer."""
import torch

from layers.OscillatorHistoryMixer import HistoryInitializedDecoder
from models.OscillatorMixer import Model as OscillatorModel
from utils.logger import get_logger


class Model(OscillatorModel):
    def __init__(self, configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            encoder.decoder = HistoryInitializedDecoder(encoder.decoder)
        self.register_buffer("oscillator_history_model_version", torch.tensor(1, dtype=torch.long))
        get_logger(__name__).info(
            "OscillatorHistoryMixer v1 overrides summary initialization above: "
            "each seasonal branch scans encoded historical features from zero; "
            "the historical final state initializes the future scan without detach; "
            "history/future SHARE transition, drive LayerNorm/GELU/dropout/projection; "
            "no summary initial-state head, no auxiliary loss; unchanged four scales, "
            "FFT row/column encoding, trend heads, future features and fusion; "
            "bidirectional history encoding is not causal historical one-step prediction; "
            "two parallel scans, O(T+H) work for fixed state size")
