"""Ordered patch drivers with one recurrent state reading out four values.

Each seasonal scale concatenates four consecutive future features (including
their positional encodings) in time-major order, then projects 4D -> D. The
projection starts at [I/4, I/4, I/4, I/4], but all entries learn independently.
History scanning, original A, 24 future recurrences and the four-value readout
are inherited from StatePatch. No future observations or output feedback.
"""
import torch
from torch import nn

from layers.OscillatorTimeMix import complex_oscillator_scan
from models.OscillatorPDMStatePatch import (
    Model as MeanPoolModel,
    convert_pointwise_state,
)


def reset_patch_projection(projection):
    """Initialize a time-major concatenation projection to arithmetic mean."""
    width = projection.out_features
    with torch.no_grad():
        identity = torch.eye(width, device=projection.weight.device,
                             dtype=projection.weight.dtype)
        projection.weight.copy_(identity.repeat(1, 4) / 4)
        projection.bias.zero_()


def project_patch_drivers(encoder, future):
    """[BC, H, D] -> [BC, H/4, D], with no mixing between patches."""
    n, horizon, width = future.shape
    if horizon % 4:
        raise ValueError('Forecast horizon must be divisible by 4')
    patches = future.reshape(n, horizon // 4, 4 * width)
    return encoder.patch_driver_projection(patches)


def state_patch_forecast(encoder, x, history_gain, drive_gain, history, trend):
    batch, _, channels = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    n, horizon, _ = future.shape
    future = project_patch_drivers(encoder, future)
    decoder = encoder.decoder
    transition, initial, _, _ = decoder.dynamics(history, trend)
    drive = decoder.future_drive(future, history).reshape(n, horizon // 4,
                                                        decoder.modes, 2)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    drive = drive * drive_gain.repeat(batch)[:, None, None, None]
    states = complex_oscillator_scan(
        transition[:, None].expand(-1, horizon // 4, -1, -1), drive, initial)
    outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    if outputs.shape != (n, horizon // 4, 4):
        raise RuntimeError('Expected four values per state')
    return outputs.reshape(batch, channels, horizon).transpose(1, 2)


class Model(MeanPoolModel):
    def __init__(self, configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            width = encoder.future_positions.shape[-1]
            # The deterministic extra initialization must not advance RNG.
            with torch.random.fork_rng(devices=[]):
                projection = nn.Linear(4 * width, width)
            reset_patch_projection(projection)
            encoder.patch_driver_projection = projection

    def load_mean_pool_state(self, state):
        """Load an old StatePatch state strictly and reset only the new layers."""
        projection_state = {
            key: value for key, value in self.state_dict().items()
            if '.patch_driver_projection.' in key
        }
        if set(state) & set(projection_state):
            raise ValueError('Expected a mean-pool checkpoint without projections')
        self.load_state_dict(dict(projection_state, **state), strict=True)
        for encoder in self.seasonals:
            reset_patch_projection(encoder.patch_driver_projection)

    def load_pointwise_initial_state(self, state):
        self.load_mean_pool_state(convert_pointwise_state(state))

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected history B,seq_len,enc_in')
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent = [
            state_patch_forecast(encoder, x, alpha, beta, season, trend)
            for encoder, x, season, trend in zip(self.seasonals, scales,
                                                seasons, trends)
        ]
        feature = [
            head(trend).reshape(batch, channels, self.pred_len).transpose(1, 2)
            for head, trend in zip(self.feature_trend_heads, trends)
        ]
        return torch.stack(recurrent + feature, -1)
