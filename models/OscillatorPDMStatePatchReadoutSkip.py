"""Fixed patch-feature skip into the old mean-patch decoder's hidden readout.

Keep the original model and state_dict exactly. At each of 24 future steps,
u is the mean of four projected D32 features, including original future PE.
The new readout is W_out(GELU(W_in h + b_in) + u) + b_out.
This bypasses the state16 path, not mean pooling; no new scalar prediction head.
"""
import torch
from torch import nn

from layers.OscillatorTimeMix import complex_oscillator_scan
from models.OscillatorPDMStatePatch import Model as BaseModel


class PatchSkipReadout(nn.Module):
    """Preserve readout parameters/buffers and add a parameter-free D-wide skip."""

    def __init__(self, source):
        super().__init__()
        for name, module in source.named_children():
            self.add_module(name, module)
        for name, parameter in source.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, buffer in source.named_buffers(recurse=False):
            self.register_buffer(name, buffer)
        self.in_features = source.in_features
        self.out_features = source.out_features

    @property
    def weight(self):
        return self.input_projection.weight

    def forward(self, states, patch_features):
        hidden = self.activation(self.input_projection(states))
        if hidden.shape != patch_features.shape:
            raise ValueError('Expected matching [B*C, H/4, D] readout and patch features')
        return self.output_projection(hidden + patch_features)


def skip_patch_forecast(encoder, x, history_gain, drive_gain, history, trend):
    batch, _, channels = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    n, horizon, width = future.shape
    if horizon % 4:
        raise ValueError('Forecast horizon must be divisible by four')
    patch_features = future.reshape(n, horizon // 4, 4, width).mean(2)
    decoder = encoder.decoder
    transition, initial, _, _ = decoder.dynamics(history, trend)
    drive = decoder.future_drive(patch_features, history).reshape(n, horizon // 4, decoder.modes, 2)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    drive = drive * drive_gain.repeat(batch)[:, None, None, None]
    states = complex_oscillator_scan(transition[:, None].expand(-1, horizon // 4, -1, -1), drive, initial)
    outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype), patch_features)
    if outputs.shape != (n, horizon // 4, 4):
        raise RuntimeError('Expected four values per recurrent patch state')
    return outputs.reshape(batch, channels, horizon).transpose(1, 2)


class Model(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            head = encoder.decoder.readout
            if head.input_projection.out_features != encoder.value_embedding.out_features:
                raise ValueError('Patch width must match the readout hidden width')
            encoder.decoder.readout = PatchSkipReadout(head)

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected history [B, seq_len, enc_in]')
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        seasonal = [skip_patch_forecast(e, x, alpha, beta, s, t)
                    for e, x, s, t in zip(self.seasonals, scales, seasons, trends)]
        feature = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, t in zip(self.feature_trend_heads, trends)]
        return torch.stack(seasonal + feature, -1)
