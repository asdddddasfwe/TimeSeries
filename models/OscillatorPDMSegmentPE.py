"""Position-aware FFT segments, after the sole seasonal/trend decomposition.

History PE is absent before MA and period selection. Fixed two-axis positions
are added only to valid seasonal image tokens before the axial block.
"""
import torch
from torch.nn import functional as F

from layers.LagARFormer import sinusoidal_positions
from models.OscillatorPDMNoHistoryPE import Model as NoHistoryPE
from models.OscillatorPDMSeasonalFFT import select_seasonal_feature_periods
from utils.logger import get_logger


def segment_positions(length, period, width, reference):
    """[1,D,ceil(T/P),P] fixed PE; first channels mark cycles, rest phase.

    Coordinates start at zero within this observed window. Padded positions
    receive zero PE, preserving the existing zero-padding convention. Padding
    remains unmasked in attention, as in the parent encoder.
    """
    if length < 2 or not 1 <= period <= length or width < 2:
        raise ValueError("Require length >= 2, 1 <= period <= length, width >= 2")
    cycles = (length + period - 1) // period
    cycle_width = width // 2
    phase_width = width - cycle_width
    cycle = sinusoidal_positions(cycles, cycle_width).to(reference)
    phase = sinusoidal_positions(period, phase_width).to(reference)
    grid = torch.cat((cycle[:, None, :].expand(-1, period, -1),
                      phase[None, :, :].expand(cycles, -1, -1)), dim=-1)
    valid = torch.arange(cycles * period, device=reference.device).reshape(cycles, period) < length
    grid = grid * valid[..., None]
    return grid.permute(2, 0, 1).unsqueeze(0).contiguous()


def encode_positioned_segments(encoder, tokens, periods, weights):
    """Group FFT routes, reshape, add 2-D PE once, encode, restore and fuse.

    Periods and fusion weights must be computed from seasonal features BEFORE
    this call. No FFT is applied to the position-augmented features.
    """
    if tokens.dtype in (torch.float16, torch.bfloat16):
        tokens = tokens.float()
    groups = {}
    for series, row in enumerate(periods.detach().cpu().tolist()):
        for period in set(row):
            groups.setdefault(period, []).append(series)
    block = encoder.blocks[0]
    projections = block.projection_parameters()
    fused = torch.zeros_like(tokens)
    for period in sorted(groups):
        selected = torch.tensor(groups[period], dtype=torch.long, device=tokens.device)
        group = tokens[selected]
        count, length, width = group.shape
        image = F.pad(group, (0, 0, 0, (-length) % period))
        image = image.reshape(count, -1, period, width).permute(0, 3, 1, 2).contiguous()
        image = image + segment_positions(length, period, width, image)
        image = block(image, projections)
        encoded = image.permute(0, 2, 3, 1).reshape(count, -1, width)[:, :length]
        merged = (weights[selected] * (periods[selected] == period)).sum(-1).to(tokens)
        fused = fused.index_add(0, selected, encoded * merged[:, None, None])
    return fused


class Model(NoHistoryPE):
    def __init__(self, configs):
        super().__init__(configs)
        if self.seasonals[0].value_embedding.out_features < 2:
            raise ValueError("Segment positional encoding requires d_model >= 2")
        get_logger(__name__).info(
            "OscillatorPDMSegmentPE: no PE before MA or FFT period selection; "
            "fixed cycle/phase PE after seasonal segment reshape, before axial "
            "attention; future decoder positions retained")

    def encode_seasonal_features(self, seasons):
        if len(seasons) != len(self.seasonals):
            raise ValueError("Expected one seasonal history per scale")
        return [encode_positioned_segments(
                    encoder, season,
                    *select_seasonal_feature_periods(season, encoder.period_top_k))
                for encoder, season in zip(self.seasonals, seasons)]
