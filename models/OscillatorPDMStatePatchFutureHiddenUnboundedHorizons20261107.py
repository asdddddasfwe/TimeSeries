"""Future-hidden-conditioned dynamic rotation with two unbounded learned periods.

The only new trainable values are two global offsets shared by the four scales.
P=[96,384]*exp(eta); physical block centers t=4*(k+.5), no clamp or phase cap.
The inherited cosine buffer remains solely to load the shared untrained reference;
the forward correction uses the learned physical-time sine/cosine basis below.
"""
import math
import torch
from torch import nn
from models.OscillatorPDMStatePatchSpectralTimevaryFutureHiddenV320261105 import (
    InheritedCombination, SpectralModel, spectral_dynamics, complex_oscillator_scan,
    future_rotation_delta, direct_history, direct_sequence,
)
from models.OscillatorUnboundedPeriodAngle20261105 import learned_periods, features


class Model(InheritedCombination):
    def __init__(self, configs):
        if configs.seq_len != 96 or configs.pred_len not in (96,192,336,720):
            raise ValueError('Expected input96 and forecast96/192/336/720')
        if configs.combo_direct or not configs.combo_spectral or not configs.combo_timevary:
            raise ValueError('Expected inherited spectral/future-hidden/dynamic-frequency combination')
        super().__init__(configs)
        self.learnable_period_offsets = nn.Parameter(torch.zeros(2))

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected [batch, 96, channels]')
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        spectrum = None
        if self.spectral_enabled:
            spectrum = self.spectral_encoder(SpectralModel.spectrum_features(self, scales[0]))
        alpha, beta = self.branch_gains()
        periods = learned_periods(self.learnable_period_offsets)
        t = 4 * (torch.arange(self.pred_len // 4, device=x_enc.device, dtype=x_enc.dtype) + .5)
        learned_basis = features(t, periods)
        batch, _, channels = x_enc.shape
        recurrent, diagnostics = [], []
        for i, (encoder, history, trend) in enumerate(zip(self.seasonals, seasons, trends)):
            decoder = encoder.decoder
            future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
            future = future + encoder.future_positions.to(future)
            n, horizon, width = future.shape
            steps = horizon // 4
            future_blocks = future.reshape(n, steps, 4, width)
            future = future_blocks.mean(2)
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
                    upper = math.pi / decoder.stride
                    delta = future_rotation_delta(
                        decoder, future_blocks, angle[:, None], upper,
                        learned_basis, correction,
                    )
                    co, si = delta.cos(), delta.sin()
                    real, imag = transition.unbind(-1)
                    sequence = torch.stack((real[:, None] * co - imag[:, None] * si,
                                            real[:, None] * si + imag[:, None] * co), -1)
                    diag['frequency_delta_max_abs'] = delta.detach().abs().max()
                    diag['future_hidden_conditioning'] = True
                    diag['learnable_unbounded_periods'] = True
                    diag['period_1'] = periods[0].detach()
                    diag['period_2'] = periods[1].detach()
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

