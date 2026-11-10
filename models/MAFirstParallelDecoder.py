"""Parallel future patches with the original MAFirst parameter/state budget.

History encoding and history state scanning remain recurrent. Only future
s[k] = A*s[k-1] + u[k] becomes s[k] = A*h_history + u[k], independently in k.
No extra parameters, positional buffers, output feedback or target statistics.
"""
import torch
from models.OscillatorPDMStatePatchMAFirst import Model as Original


def parallel_future_states(transition, drive, initial):
    """Inputs A[N,J,2], u[N,K,J,2], h[N,J,2]; output [N,K,J,2]."""
    if (drive.ndim != 4 or drive.shape[-1] != 2
            or transition.shape != initial.shape
            or initial.shape != (drive.shape[0], drive.shape[2], 2)):
        raise ValueError("Expected A/h[N,J,2] and u[N,K,J,2]")
    dtype = torch.promote_types(drive.dtype, initial.dtype)
    dtype = torch.promote_types(dtype, transition.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    a = torch.view_as_complex(transition.to(dtype).contiguous())
    h = torch.view_as_complex(initial.to(dtype).contiguous())
    anchor = torch.view_as_real(a * h)
    return drive.to(dtype) + anchor[:, None]


def parallel_state_patch_forecast(encoder, x, history_gain, drive_gain, history, trend):
    batch, _, channels = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    n, horizon, width = future.shape
    if horizon % 4:
        raise ValueError("Forecast horizon must be divisible by four")
    future = future.reshape(n, horizon // 4, 4, width).mean(2)
    decoder = encoder.decoder
    transition, initial, _, _ = decoder.dynamics(history, trend)
    drive = decoder.future_drive(future, history).reshape(n, horizon // 4, decoder.modes, 2)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    drive = drive * drive_gain.repeat(batch)[:, None, None, None]
    states = parallel_future_states(transition, drive, initial)
    outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    if outputs.shape != (n, horizon // 4, 4):
        raise RuntimeError("Expected four values per future patch")
    return outputs.reshape(batch, channels, horizon).transpose(1, 2)


class Model(Original):
    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected history B,seq_len,enc_in")
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        seasonal = [parallel_state_patch_forecast(e, x, alpha, beta, s, t)
                    for e, x, s, t in zip(self.seasonals, scales, seasons, trends)]
        feature = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, t in zip(self.feature_trend_heads, trends)]
        return torch.stack(seasonal + feature, -1)
