"""MAFirst combinations for ETTm1 with 96 observed and 720 forecast points."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from layers.OscillatorTimeMix import complex_oscillator_scan
from models.OscillatorPDMStatePatchBestAllDirectParametersDistDFRNG20261103 import (
    Model as DirectModel,
    parameter_context,
)
from models.OscillatorPDMStatePatchBestSpectralContinuousRNG20261103 import (
    Model as SpectralModel,
    spectral_dynamics,
)
from models.OscillatorFixedTimeAngle20261105 import position_transitions as fixed_position_transitions

def position_transitions(*args):
    return fixed_position_transitions(*args,phase_bounded=False)
from models.OscillatorPDMStatePatchMAFirstSeasonSliding import Model as BaselineModel


def direct_history(decoder, history, trend, shift):
    if shift is None:
        return decoder.dynamics(history, trend)[1]
    context = parameter_context(decoder, history, trend)
    dtype = torch.float64 if context.dtype == torch.float64 else torch.float32
    decay, angle = (decoder.history_parameter_head(context).to(dtype) + shift.to(dtype)).chunk(2, -1)
    radius = (-F.softplus(decay)).exp()
    angle = math.pi * angle.sigmoid()
    transition = torch.stack((radius * angle.cos(), radius * angle.sin()), -1)
    return decoder.encode_history_states(history, transition)[:, -1]


def direct_sequence(decoder, history, trend, head, basis, correction):
    context = parameter_context(decoder, history, trend)
    steps = len(basis)
    inputs = torch.cat((context[:, None].expand(-1, steps, -1),
                        basis.to(context)[None].expand(len(context), -1, -1)), -1)
    dtype = torch.float64 if context.dtype == torch.float64 else torch.float32
    decay, angle = head(inputs).to(dtype).chunk(2, -1)
    radius = (-F.softplus(decay) / decoder.stride).exp()
    upper = math.pi / decoder.stride
    angle = upper * angle.sigmoid()
    delta = None
    if correction is not None:
        seasonal_context = decoder.context_norm(torch.cat((history.mean(1), history[:, -1]), -1))
        coefficients = correction(seasonal_context).reshape(len(history), decoder.modes, 4)
        raw = torch.einsum('nmk,tk->ntm', coefficients.to(dtype), basis.to(angle))
        delta = .25 * torch.minimum(angle, upper - angle) * raw.tanh()
        angle = angle + delta
    return torch.stack((radius * angle.cos(), radius * angle.sin()), -1), delta


class Combination:
    def setup_combination(self, configs):
        self.spectral_enabled = configs.combo_spectral
        self.timevary_enabled = configs.combo_timevary
        if self.spectral_enabled:
            self.spectral_encoder = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.LayerNorm(32))
            self.spectral_conditions = nn.ModuleList(nn.Linear(32, 16) for _ in self.seasonals)
            for head in self.spectral_conditions:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        if self.timevary_enabled:
            self.position_frequency_heads = nn.ModuleList(
                nn.Linear(2 * e.decoder.drive_projection.in_features, 4 * e.decoder.modes, bias=False)
                for e in self.seasonals
            )
            for head in self.position_frequency_heads:
                nn.init.zeros_(head.weight)
            if not self.direct:
                steps = self.pred_len // 4
                basis = torch.cos(math.pi * (torch.arange(steps, dtype=torch.float32)[:, None] + .5)
                                  * torch.arange(1, 5, dtype=torch.float32)[None, :] / steps)
                self.register_buffer('position_frequency_basis', basis)
        self.gate_context_policy = 'season'

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected [batch, 96, channels]')
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        spectrum = None
        if self.spectral_enabled:
            spectrum = self.spectral_encoder(SpectralModel.spectrum_features(self, scales[0]))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent, diagnostics = [], []
        for i, (encoder, history, trend) in enumerate(zip(self.seasonals, seasons, trends)):
            decoder = encoder.decoder
            future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
            future = future + encoder.future_positions.to(future)
            n, horizon, width = future.shape
            steps = horizon // 4
            future = future.reshape(n, steps, 4, width).mean(2)
            shift = None if spectrum is None else self.spectral_conditions[i](spectrum)
            correction = self.position_frequency_heads[i] if self.timevary_enabled else None
            diag = dict(context_error=0., trend_delta_energy=0., is_season_tensor=True,
                        history_length=history.shape[1], direct_generation=self.direct,
                        spectral_before_history_scan=shift is not None,
                        time_varying_rotation=correction is not None)
            if self.direct:
                initial = direct_history(decoder, history, trend, shift)
                sequence, delta = direct_sequence(
                    decoder, history, trend, self.direct_future_parameter_heads[i],
                    self.direct_parameter_position_basis, correction,
                )
                if delta is not None:
                    diag['frequency_delta_max_abs'] = delta.detach().abs().max()
            else:
                if shift is None:
                    transition, initial, _, angle = decoder.dynamics(history, trend)
                else:
                    transition, initial, _, angle = spectral_dynamics(decoder, history, trend, shift)
                if correction is None:
                    sequence = transition[:, None].expand(-1, steps, -1, -1)
                else:
                    sequence, rotation_diag = position_transitions(
                        decoder, history, transition, angle, correction, self.position_frequency_basis,
                    )
                    diag['frequency_delta_max_abs'] = rotation_diag['frequency_delta_max_abs']
            if shift is not None:
                diag['spectral_shift_max_abs'] = shift.detach().abs().max()
            drive = decoder.future_drive(future, history).reshape(n, steps, decoder.modes, 2)
            initial = initial * alpha.repeat(batch)[:, None, None]
            drive = drive * beta.repeat(batch)[:, None, None, None]
            states = complex_oscillator_scan(sequence, drive, initial)
            outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
            recurrent.append(outputs.reshape(batch, channels, horizon).transpose(1, 2))
            diagnostics.append(diag)
        feature = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, t in zip(self.feature_trend_heads, trends)]
        self.last_gate_context_diagnostics = diagnostics
        return torch.stack(recurrent + feature, -1)


class InheritedCombination(Combination, BaselineModel):
    direct = False

    def __init__(self, configs):
        super().__init__(configs)
        self.setup_combination(configs)


class DirectCombination(Combination, DirectModel):
    direct = True

    def __init__(self, configs):
        super().__init__(configs)
        self.setup_combination(configs)


def Model(configs):
    if (configs.seq_len, configs.pred_len) != (96, 720):
        raise ValueError('This experiment fixes input 96 and horizon 720')
    cls = DirectCombination if configs.combo_direct else InheritedCombination
    return cls(configs)

