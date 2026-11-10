"""Historical future-hidden-conditioned dynamic frequency, extended to all ETT horizons.

Only the public horizon guard changes. The frozen V3 inherited model computes its
future features from the observed input; no future targets enter the forward pass.
"""
from models.OscillatorPDMStatePatchSpectralTimevaryFutureHiddenV320261105 import InheritedCombination


def Model(configs):
    if configs.seq_len != 96 or configs.pred_len not in (96, 192, 336, 720):
        raise ValueError('Expected input96 and forecast96/192/336/720')
    if configs.combo_direct or not configs.combo_spectral or not configs.combo_timevary:
        raise ValueError('This experiment combines spectral, future hidden, and dynamic frequency')
    return InheritedCombination(configs)
