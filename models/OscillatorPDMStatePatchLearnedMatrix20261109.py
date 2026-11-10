"""ETTm1-H720 ablation with freely learned future state matrices."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.OscillatorPDMStatePatchBestCombination20261104 import InheritedCombination
from models.OscillatorPDMStatePatchBestSpectralContinuousRNG20261103 import (
    Model as SpectralModel,
    spectral_dynamics,
)


def fixed_matrix_scan(matrix, drive, initial):
    """Compute z[t] = A z[t-1] + u[t] with a shared A per mode.

    matrix: [modes, 2, 2], drive: [batch * channels, steps, modes, 2],
    initial: [batch * channels, modes, 2]. Each doubling combines
    adjacent affine blocks using their common transition power.
    """
    def apply(power, states):
        return torch.einsum("mij,btmj->btmi", power, states)

    first = drive[:, :1] + apply(matrix, initial[:, None])
    states = torch.cat((first, drive[:, 1:]), dim=1)
    power = matrix
    span = 1
    while span < drive.shape[1]:
        tail = states[:, span:] + apply(power, states[:, :-span])
        states = torch.cat((states[:, :span], tail), dim=1)
        span *= 2
        if span < drive.shape[1]:
            power = power @ power
    return states


class Model(InheritedCombination):
    def __init__(self, configs):
        if (configs.seq_len, configs.pred_len) != (96, 720):
            raise ValueError("Expected history 96 and forecast horizon 720")
        if not (configs.combo_spectral and configs.combo_timevary):
            raise ValueError("Use the reference spectral/time-varying construction config")
        super().__init__(configs)
        # Preserve the reference constructor's RNG sequence before dropping
        # the unused conditioning heads.
        del self.position_frequency_heads
        del self.position_frequency_basis
        self.timevary_enabled = False
        matrices = []
        for encoder in self.seasonals:
            decoder = encoder.decoder
            with torch.no_grad():
                radius = torch.exp(-F.softplus(decoder.log_decay) / decoder.stride)
                angle = math.pi * decoder.frequency_logits.sigmoid() / decoder.stride
                real, imag = radius * angle.cos(), radius * angle.sin()
                first = torch.stack((real, -imag), dim=-1)
                second = torch.stack((imag, real), dim=-1)
                matrices.append(torch.stack((first, second), dim=-2))
        self.future_transition = nn.Parameter(torch.stack(matrices))

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected [batch, 96, channels]")
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        spectrum = self.spectral_encoder(SpectralModel.spectrum_features(self, scales[0]))
        history_gain, drive_gain = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent, diagnostics = [], []
        for scale, (encoder, history, trend) in enumerate(zip(self.seasonals, seasons, trends)):
            decoder = encoder.decoder
            future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
            future = future + encoder.future_positions.to(future)
            n, horizon, width = future.shape
            steps = horizon // 4
            future = future.reshape(n, steps, 4, width).mean(dim=2)
            shift = self.spectral_conditions[scale](spectrum)
            _, initial, _, _ = spectral_dynamics(decoder, history, trend, shift)
            drive = decoder.future_drive(future, history).reshape(n, steps, decoder.modes, 2)
            initial = initial * history_gain.repeat(batch)[:, None, None]
            drive = drive * drive_gain.repeat(batch)[:, None, None, None]
            matrix = self.future_transition[scale].to(initial)
            states = fixed_matrix_scan(matrix, drive.to(initial), initial)
            output = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
            recurrent.append(output.reshape(batch, channels, horizon).transpose(1, 2))
            diagnostics.append(dict(
                context_error=0.0,
                trend_delta_energy=0.0,
                is_season_tensor=True,
                history_length=history.shape[1],
                spectral_before_history_scan=True,
                spectral_shift_max_abs=shift.detach().abs().max(),
                learned_future_matrix=True,
            ))
        feature = [
            head(trend).reshape(batch, channels, self.pred_len).transpose(1, 2)
            for head, trend in zip(self.feature_trend_heads, trends)
        ]
        self.last_gate_context_diagnostics = diagnostics
        return torch.stack(recurrent + feature, dim=-1)

