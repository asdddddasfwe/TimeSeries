"""Four pointwise drives folded into one MAFirst state update per forecast block."""
import torch

from layers.OscillatorTimeMix import complex_oscillator_scan
from models.OscillatorPDMStatePatchBestCombination20261104 import InheritedCombination
from models.OscillatorPDMStatePatchBestSpectralContinuousRNG20261103 import (
    Model as SpectralModel,
    spectral_dynamics,
)
from models.OscillatorPDMStatePatchBestTimeVaryingRotation import position_transitions


def fold_four_steps(point_transition, point_drives):
    """[N, blocks, modes, 2] and [N, blocks, 4, modes, 2] -> block transition/drive."""
    step = torch.view_as_complex(point_transition.contiguous())
    drives = torch.view_as_complex(point_drives.to(point_transition.dtype).contiguous())
    first, second, third, fourth = drives.unbind(dim=2)
    step_squared = step * step
    block_drive = ((step * first + second) * step + third) * step + fourth
    return torch.view_as_real(step_squared * step_squared), torch.view_as_real(block_drive)


class Model(InheritedCombination):
    def __init__(self, configs):
        super().__init__(configs)
        if (configs.seq_len, configs.pred_len) != (96, 720) or configs.combo_direct:
            raise ValueError('Expected MAFirst S gate with 96 observed and 720 forecast points')

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
            n, horizon, _ = future.shape
            steps = horizon // 4
            if spectrum is None:
                transition, initial, _, angle = decoder.dynamics(history, trend)
            else:
                shift = self.spectral_conditions[i](spectrum)
                transition, initial, _, angle = spectral_dynamics(decoder, history, trend, shift)
            diag = dict(context_error=0., trend_delta_energy=0., is_season_tensor=True,
                        history_length=history.shape[1], pointwise_drives_folded=True,
                        mean4_drive=False, spectral_before_history_scan=spectrum is not None,
                        time_varying_rotation=self.timevary_enabled)
            if self.timevary_enabled:
                sequence, rotation_diag = position_transitions(
                    decoder, history, transition, angle, self.position_frequency_heads[i],
                    self.position_frequency_basis,
                )
                diag['frequency_delta_max_abs'] = rotation_diag['frequency_delta_max_abs']
            else:
                sequence = transition[:, None].expand(-1, steps, -1, -1)
            if spectrum is not None:
                diag['spectral_shift_max_abs'] = shift.detach().abs().max()
            point_drives = decoder.future_drive(future, history).reshape(n, steps, 4, decoder.modes, 2)
            point_drives = point_drives * beta.repeat(batch)[:, None, None, None, None]
            block_transition, block_drive = fold_four_steps(sequence, point_drives)
            initial = initial * alpha.repeat(batch)[:, None, None]
            states = complex_oscillator_scan(block_transition, block_drive, initial)
            outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
            recurrent.append(outputs.reshape(batch, channels, horizon).transpose(1, 2))
            diagnostics.append(diag)
        feature = [head(trend).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, trend in zip(self.feature_trend_heads, trends)]
        self.last_gate_context_diagnostics = diagnostics
        return torch.stack(recurrent + feature, dim=-1)
