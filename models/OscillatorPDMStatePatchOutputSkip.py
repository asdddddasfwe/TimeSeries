"""Independent output skip for the old mean-patch forecasting model.
Each seasonal head emits Wout GELU(Win h+bin)+bout + Wskip u.
u remains the original mean4 future feature including PE. Wskip has no bias,
starts at zero, and does not share storage with Wout. No hidden feature addition.
"""
import torch
from torch import nn

from layers.OscillatorTimeMix import complex_oscillator_scan
from models.OscillatorPDMStatePatch import Model as BaseModel


class PatchOutputSkipReadout(nn.Module):
    """Preserve original readout and add an independent zero-initialized D-to-4 output path."""

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
        # Avoid shifting scratch RNG while registering a separately learned head.
        with torch.random.fork_rng(devices=[]):
            self.skip_projection = nn.Linear(source.input_projection.out_features,
                                             source.output_projection.out_features, bias=False)
        nn.init.zeros_(self.skip_projection.weight)

    @property
    def weight(self):
        return self.input_projection.weight

    def forward(self, states, patch_features):
        hidden = self.activation(self.input_projection(states))
        if hidden.shape != patch_features.shape:
            raise ValueError('Expected matching [B*C, H/4, D] readout and patch features')
        return self.output_projection(hidden) + self.skip_projection(patch_features)


def output_skip_patch_forecast(encoder, x, history_gain, drive_gain, history, trend):
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
            encoder.decoder.readout = PatchOutputSkipReadout(head)

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected history [B, seq_len, enc_in]')
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        seasonal = [output_skip_patch_forecast(e, x, alpha, beta, s, t)
                    for e, x, s, t in zip(self.seasonals, scales, seasons, trends)]
        feature = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, t in zip(self.feature_trend_heads, trends)]
        return torch.stack(seasonal + feature, -1)
