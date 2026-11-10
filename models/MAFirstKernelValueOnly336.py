"""Kernel feedback retained; only future scan may become parallel. No time-kernel change inside the model."""
import torch
from torch import nn
from layers.StandardNorm import Normalize
from models.OscillatorPDMStatePatchMAFirst import Model as Original
from models.MAFirstParallelDecoder import Model as Parallel, parallel_future_states
from models.OscillatorPDMStatePatch import complex_oscillator_scan


def value_gram(coordinates):
    """RBF sigma=1 on four-point raw dataset-standardized patch coordinates."""
    distance = (coordinates.unsqueeze(-2) - coordinates.unsqueeze(-3)).square().sum(-1)
    return torch.exp(-0.5 * distance)


def kernel_supervision(predicted, target):
    # Targets only supervise the head; they never become forecast inputs.
    batch, horizon, channels = target.shape
    patches = target.detach().transpose(1, 2).reshape(batch * channels, horizon // 4, 4)
    gold = value_gram(patches)
    n = gold.shape[-1]
    mask = ~torch.eye(n, dtype=torch.bool, device=gold.device)
    error = (predicted - gold).square()[..., mask]
    return error.mean(), gold


class FutureValueGram(nn.Module):
    def __init__(self, length, patches, width):
        super().__init__()
        self.temporal = nn.Linear(length, patches, bias=False)
        self.coordinates = nn.Sequential(nn.Linear(2 * width, 16), nn.GELU(), nn.Linear(16, 4, bias=False))

    def forward(self, seasonal, trend, raw_scale):
        joint = torch.cat((seasonal, trend), -1)
        future = self.temporal(joint.transpose(1, 2)).transpose(1, 2)
        coordinates = self.coordinates(future) * raw_scale
        return value_gram(coordinates)


def recurrent_future_states(transition, drive, initial):
    return complex_oscillator_scan(transition[:, None].expand(-1, drive.shape[1], -1, -1), drive, initial)


class Model(Original):
    def __init__(self, configs):
        super().__init__(configs)
        self.study_arm = configs.study_arm
        assert self.study_arm == "kernel" and not configs.study_affine
        self.study_parallel = bool(configs.study_parallel)
        self.affine_enabled = bool(configs.study_affine)
        if self.affine_enabled:
            self.normalize = Normalize(self.enc_in, affine=True)
            self.coarse_normalizers = nn.ModuleList([Normalize(self.enc_in, affine=True) for _ in range(3)])
        if self.study_arm == 'kernel':
            patches = self.pred_len // 4
            self.kernel_head = FutureValueGram(self.seq_len, patches, configs.d_model)
            self.kernel_season_feedback = nn.ModuleList([nn.Linear(patches, configs.d_model) for _ in range(4)])
            self.kernel_trend_feedback = nn.ModuleList([nn.Linear(patches, configs.d_model) for _ in range(4)])
            # Initial forecasts equal the chosen recurrent base, not a random extra head.
            for layer in [*self.kernel_season_feedback, *self.kernel_trend_feedback]:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif self.study_arm not in ('affine', 'parallel'):
            raise ValueError('Unknown preregistered study arm')

    def inverse_coordinates(self):
        norm = self.normalize
        scale = norm.stdev
        offset = norm.mean
        if norm.affine:
            scale = scale / (norm.affine_weight + norm.eps * norm.eps)
            offset = offset - norm.affine_bias * scale
        return scale, offset

    def predict_components(self, x_enc):
        if self.study_arm == 'parallel':
            return Parallel.predict_components(self, x_enc)
        if self.study_arm == 'affine':
            return super().predict_components(x_enc)
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        batch, _, channels = x_enc.shape
        scale, _ = self.inverse_coordinates()
        raw_scale = scale.transpose(1, 2).reshape(batch * channels, 1, 1)
        gram = self.kernel_head(seasons[0], trends[0], raw_scale)
        self.predicted_gram = gram
        relation = gram / gram.sum(-1, keepdim=True)
        alpha, beta = self.branch_gains()
        seasonal, trend_outputs = [], []
        for i, (encoder, head, x, s, t) in enumerate(zip(self.seasonals, self.feature_trend_heads, scales, seasons, trends)):
            future = encoder.temporal_projection(s.transpose(1, 2)).transpose(1, 2)
            future = future + encoder.future_positions.to(future)
            n, horizon, width = future.shape
            future = future.reshape(n, horizon // 4, 4, width).mean(2)
            future = future + self.kernel_season_feedback[i](relation)
            decoder = encoder.decoder
            transition, initial, _, _ = decoder.dynamics(s, t)
            drive = decoder.future_drive(future, s).reshape(n, horizon // 4, decoder.modes, 2)
            initial = initial * alpha.repeat(batch)[:, None, None]
            drive = drive * beta.repeat(batch)[:, None, None, None]
            states = (parallel_future_states if self.study_parallel else recurrent_future_states)(transition, drive, initial)
            values = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
            seasonal.append(values.reshape(batch, channels, horizon).transpose(1, 2))
            future_t = head.temporal(t.transpose(1, 2)).transpose(1, 2)
            future_t = future_t + self.kernel_trend_feedback[i](relation).repeat_interleave(4, dim=1)
            values_t = head.readout(future_t).squeeze(-1)
            trend_outputs.append(values_t.reshape(batch, channels, horizon).transpose(1, 2))
        return torch.stack(seasonal + trend_outputs, -1)
