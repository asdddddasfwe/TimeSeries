"""Condition seasonal oscillator dynamics on the existing final trend stream."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from layers.OscillatorTimeMix import FullResolutionGatedDecoder, complex_oscillator_scan
from models.OscillatorPDMSegmentPE import Model as SegmentPE


class TrendConditionedDecoder(FullResolutionGatedDecoder):
    """Add zero-initialized trend corrections to decay/frequency logits only.

    Summary [N,2D] is mean and last token of same-scale final trend features.
    History/future drives and future gate still use seasonal features only.
    """
    def __init__(self, decoder):
        nn.Module.__init__(self)
        for name in ('length', 'horizon', 'modes', 'stride'):
            setattr(self, name, getattr(decoder, name))
        for name, module in decoder.named_children():
            self.add_module(name, module)
        for name, parameter in decoder.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, value in decoder.named_buffers(recurse=False):
            self.register_buffer(name, value)
        width = self.drive_projection.in_features
        self.trend_context_norm = nn.LayerNorm(2 * width, elementwise_affine=False)
        with torch.random.fork_rng(devices=[]):
            self.trend_condition = nn.Linear(2 * width, 2 * self.modes, bias=False)
        nn.init.zeros_(self.trend_condition.weight)

    def dynamics(self, history, trend_history=None):
        if trend_history is None or trend_history.shape != history.shape:
            raise ValueError('Expected same-scale seasonal/trend histories [N,T,D]')
        context = self.context_norm(torch.cat((history.mean(1), history[:, -1]), -1))
        trend = self.trend_context_norm(torch.cat((trend_history.mean(1), trend_history[:, -1]), -1))
        dtype = torch.float64 if context.dtype == torch.float64 else torch.float32
        shifts = self.condition(context).to(dtype) + self.trend_condition(trend).to(dtype)
        decay_shift, frequency_shift = shifts.chunk(2, -1)
        rate = F.softplus(self.log_decay.to(dtype) + decay_shift)
        radius = torch.exp(-rate)
        frequency = math.pi * torch.sigmoid(self.frequency_logits.to(dtype) + frequency_shift)
        transition = torch.stack((radius * frequency.cos(), radius * frequency.sin()), -1)
        initial = self.encode_history_states(history, transition)[:, -1]
        if self.stride != 1:
            radius = radius.pow(1. / self.stride)
            frequency = frequency / self.stride
            transition = torch.stack((radius * frequency.cos(), radius * frequency.sin()), -1)
        return transition, initial, radius, frequency


def conditioned_forecast(encoder, x, history_gain, drive_gain, history, trend):
    """Same calibrated scan/readout, with trend passed only to dynamics."""
    batch, _, channels = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    decoder = encoder.decoder
    transition, initial, _, _ = decoder.dynamics(history, trend)
    drive = decoder.future_drive(future, history)
    drive = drive.reshape(batch * channels, encoder.horizon, decoder.modes, 2)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    drive = drive * drive_gain.repeat(batch)[:, None, None, None]
    states = complex_oscillator_scan(transition[:, None].expand(-1, encoder.horizon, -1, -1), drive, initial)
    output = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype)).squeeze(-1)
    return output.reshape(batch, channels, encoder.horizon).transpose(1, 2)


class Model(SegmentPE):
    def __init__(self, configs):
        super().__init__(configs)
        for encoder in self.seasonals:
            encoder.decoder = TrendConditionedDecoder(encoder.decoder)

    def optimizer_parameter_groups(self, weight_decay):
        # New logit corrections follow the existing dynamics decay exemption.
        protected = {id(e.decoder.trend_condition.weight) for e in self.seasonals}
        groups = []
        for group in super().optimizer_parameter_groups(weight_decay):
            regular = [p for p in group['params'] if id(p) not in protected]
            added = [p for p in group['params'] if id(p) in protected]
            if regular:
                groups.append({**group, 'params': regular})
            if added:
                groups.append({**group, 'params': added, 'weight_decay': self.dynamics_weight_decay,
                               'group_name': 'trend_condition_dynamics'})
        return groups

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected history [B,seq_len,enc_in]')
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent = [conditioned_forecast(e, x, alpha, beta, s, t)
                     for e, x, s, t in zip(self.seasonals, scales, seasons, trends)]
        feature = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, t in zip(self.feature_trend_heads, trends)]
        return torch.stack(recurrent + feature, -1)
