"""Generate 24 patch features directly, then one original-A state per four points.

No 96-position future feature tensor or runtime patch aggregation. At scratch
initialization only, fold the old linear output rows and fixed positional
buffers into their four-row averages. This is algebraically equivalent to the
old mean driver in real arithmetic, not bitwise equivalent in floating point.
"""
import torch
from torch import nn

from layers.OscillatorTimeMix import complex_oscillator_scan
from models.OscillatorPDMStatePatch import Model as MeanPoolModel, convert_pointwise_state


def fold_mean_driver_state(state):
    """Convert a mean-model scratch state; only final projection rows and PE shrink."""
    result = {key: value.clone() for key, value in state.items()}
    for scale in range(4):
        prefix = f'seasonals.{scale}.'
        keys = [key for key in state if key.startswith(prefix + 'temporal_projection.')
                and key.endswith('.weight')]
        final = sorted(keys)[-1]
        for key in (final, final.replace('.weight', '.bias'), prefix + 'future_positions'):
            value = state[key]
            if value.shape[0] % 4:
                raise ValueError('Expected four consecutive output rows per patch')
            result[key] = value.reshape(value.shape[0] // 4, 4, *value.shape[1:]).mean(1)
    return result


def state_patch_forecast(encoder, x, history_gain, drive_gain, history, trend):
    batch, _, channels = x.shape
    # [BC,L,D] -> [BC,24,D], with no intermediate 96-position future features.
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    n, patches, _ = future.shape
    decoder = encoder.decoder
    transition, initial, _, _ = decoder.dynamics(history, trend)
    drive = decoder.future_drive(future, history).reshape(n, patches, decoder.modes, 2)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    drive = drive * drive_gain.repeat(batch)[:, None, None, None]
    states = complex_oscillator_scan(transition[:, None].expand(-1, patches, -1, -1),
                                    drive, initial)
    outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    if outputs.shape != (n, patches, 4):
        raise RuntimeError('Expected four values per state')
    return outputs.reshape(batch, channels, 4 * patches).transpose(1, 2)


class Model(MeanPoolModel):
    def __init__(self, configs):
        super().__init__(configs)
        initial = fold_mean_driver_state(self.state_dict())
        for encoder in self.seasonals:
            projection = encoder.temporal_projection
            old = projection[-1] if isinstance(projection, nn.Sequential) else projection
            with torch.random.fork_rng(devices=[]):
                direct = nn.Linear(old.in_features, self.pred_len // 4)
            if isinstance(projection, nn.Sequential):
                projection[-1] = direct
            else:
                encoder.temporal_projection = direct
            encoder.future_positions = encoder.future_positions[::4].clone()
        self.load_state_dict(initial, strict=True)

    def load_pointwise_initial_state(self, state):
        self.load_state_dict(fold_mean_driver_state(convert_pointwise_state(state)), strict=True)

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected history B,seq_len,enc_in')
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent = [state_patch_forecast(encoder, x, alpha, beta, season, trend)
                     for encoder, x, season, trend in zip(self.seasonals, scales, seasons, trends)]
        feature = [head(trend).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, trend in zip(self.feature_trend_heads, trends)]
        return torch.stack(recurrent + feature, -1)
