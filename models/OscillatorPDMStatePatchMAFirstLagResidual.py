"""Future hidden-state shortcut from k-lag, with the audited original history.

For k < lag: h[k] = A[k] h[k-1] + u[k]. From k == lag onward:
h[k] = mix * A[k] h[k-1] + (1-mix) * h[k-lag] + u[k].
h[0] is the original encoded history endpoint. No invented negative-time
states, output feedback, extra parameters, or changes to the historical scan.
"""
import math
import torch

from layers.OscillatorTimeMix import complex_oscillator_scan as base_scan
from models.OscillatorPDMStatePatchMAFirst import Model as Original


def lagged_scan(transition, drive, initial, lag=8, mix=0.5):
    """Block affine scans of length lag; gradients retain all delayed paths.

    Inputs are A,u [N,K,J,2], h0 [N,J,2], as in the original complex scan.
    The first lag-1 states use the original recurrence; each following block
    can scan in parallel because its delayed inputs belong to earlier blocks.
    """
    if isinstance(lag, bool) or int(lag) != lag or int(lag) < 1:
        raise ValueError('lag must be a positive integer')
    lag, mix = int(lag), float(mix)
    if not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError('mix must be finite and in [0, 1]')
    if transition.shape != drive.shape or drive.ndim != 4 or drive.shape[-1] != 2 or drive.shape[1] < 1:
        raise ValueError('Expected transition/drive [N,K>=1,J,2]')
    if initial.shape != (drive.shape[0], drive.shape[2], 2):
        raise ValueError('Expected initial [N,J,2]')
    horizon = drive.shape[1]
    if mix == 1.0 or lag > horizon:
        return base_scan(transition, drive, initial)
    blocks = []
    previous = initial
    prefix = min(lag - 1, horizon)
    if prefix:
        states = base_scan(transition[:, :prefix], drive[:, :prefix], initial)
        blocks.append(states)
        previous = states[:, -1]
        delayed = torch.cat((initial[:, None], states), 1)
    else:
        delayed = initial[:, None]
    for begin in range(prefix, horizon, lag):
        end = min(begin + lag, horizon)
        adjusted_drive = drive[:, begin:end] + (1.0 - mix) * delayed[:, :end-begin]
        states = base_scan(mix * transition[:, begin:end], adjusted_drive, previous)
        blocks.append(states)
        previous, delayed = states[:, -1], states
    return torch.cat(blocks, 1)


def state_patch_forecast(encoder, x, history_gain, drive_gain, history, trend, lag, mix):
    batch, _, channels = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    n, horizon, width = future.shape
    if horizon % 4:
        raise ValueError('Forecast horizon must be divisible by 4')
    future = future.reshape(n, horizon // 4, 4, width).mean(2)
    decoder = encoder.decoder
    transition, initial, _, _ = decoder.dynamics(history, trend)
    drive = decoder.future_drive(future, history).reshape(n, horizon // 4, decoder.modes, 2)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    drive = drive * drive_gain.repeat(batch)[:, None, None, None]
    states = lagged_scan(transition[:, None].expand_as(drive), drive, initial, lag, mix)
    output = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    return output.reshape(batch, channels, horizon).transpose(1, 2)


class Model(Original):
    def __init__(self, configs):
        super().__init__(configs)
        self.future_residual_lag = int(getattr(configs, 'future_residual_lag', 8))
        self.future_residual_mix = float(getattr(configs, 'future_residual_mix', 0.5))

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected history [B,seq_len,enc_in]')
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent = [state_patch_forecast(e, x, alpha, beta, s, t,
                                         self.future_residual_lag, self.future_residual_mix)
                     for e, x, s, t in zip(self.seasonals, scales, seasons, trends)]
        feature = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, t in zip(self.feature_trend_heads, trends)]
        return torch.stack(recurrent + feature, -1)
