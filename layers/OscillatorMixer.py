"""History-conditioned damped oscillators with a work-efficient affine scan."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from layers.LagARFormer import LagMessageSeasonal


def complex_product(left, right):
    """Complex multiplication using [..., real/imag] pairs, without complex dtype."""
    real = left[..., 0] * right[..., 0] - left[..., 1] * right[..., 1]
    imag = left[..., 0] * right[..., 1] + left[..., 1] * right[..., 0]
    return torch.stack((real, imag), dim=-1)


def _affine_prefix(a, b):
    """Inclusive prefixes of s -> a*s+b along dimension 1.

    Pair reduction and interleaved reconstruction visit O(H) elements over
    O(log H) recursive levels. No loop over future time steps or HxH matrix.
    Functionally constructed tensors preserve autograd; no in-place updates.
    """
    length = a.shape[1]
    if length == 1:
        return a, b
    pairs = length // 2
    left_a, right_a = a[:, :2 * pairs:2], a[:, 1::2]
    left_b, right_b = b[:, :2 * pairs:2], b[:, 1::2]
    odd_a, odd_b = _affine_prefix(
        complex_product(right_a, left_a),
        right_b + complex_product(right_a, left_b),
    )
    tail_a, tail_b = a[:, 2::2], b[:, 2::2]
    count = tail_a.shape[1]
    even_a = complex_product(tail_a, odd_a[:, :count])
    even_b = tail_b + complex_product(tail_a, odd_b[:, :count])
    interleaved_a = torch.stack((odd_a[:, :count], even_a), dim=2).flatten(1, 2)
    interleaved_b = torch.stack((odd_b[:, :count], even_b), dim=2).flatten(1, 2)
    return (torch.cat((a[:, :1], interleaved_a, odd_a[:, count:]), dim=1),
            torch.cat((b[:, :1], interleaved_b, odd_b[:, count:]), dim=1))


def parallel_oscillator_scan(transition, drive, initial):
    """Exact affine recurrence in real arithmetic, modulo floating-point order.

    transition/drive: [N,H,J,2]; initial: [N,J,2].
    Returns every same-layer state [N,H,J,2]. Half/bfloat16 recurrences are
    accumulated in float32, including rotation multiplication and prefix sums.
    """
    if (transition.ndim != 4 or transition.shape != drive.shape or
            transition.shape[-1] != 2 or transition.shape[1] < 1 or
            initial.shape != (drive.shape[0], drive.shape[2], 2)):
        raise ValueError("Expected transition/drive[N,H>=1,J,2] and initial[N,J,2]")
    if transition.device != drive.device or initial.device != drive.device:
        raise ValueError("All scan inputs must share a device")
    work_dtype = torch.promote_types(torch.promote_types(transition.dtype, drive.dtype), initial.dtype)
    if work_dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32
    a, b, start = transition.to(work_dtype), drive.to(work_dtype), initial.to(work_dtype)
    prefix_a, prefix_b = _affine_prefix(a, b)
    return prefix_b + complex_product(prefix_a, start[:, None])


class OscillatorDecoder(nn.Module):
    """Input-conditioned frequencies/decay; all future values read scanned states."""

    def __init__(self, length, horizon, width, modes=8, dropout=.2):
        super().__init__()
        if min(length, horizon, width, modes) < 1 or not 0 <= dropout < 1:
            raise ValueError("Require positive decoder dimensions and dropout in [0,1)")
        self.length, self.horizon, self.modes = length, horizon, modes
        self.context_norm = nn.LayerNorm(2 * width)
        self.condition = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                       nn.Linear(width, 2 * modes))
        nn.init.normal_(self.condition[-1].weight, std=.01)
        nn.init.zeros_(self.condition[-1].bias)
        # Spread initial memory times and frequencies; all remain trainable.
        memory = torch.logspace(math.log10(4.), math.log10(max(8, length, horizon)), modes)
        rate = memory.reciprocal()
        self.log_decay = nn.Parameter(torch.log(torch.expm1(rate)))
        periods = torch.logspace(math.log10(2.2), math.log10(max(4, 2 * length)), modes)
        frequency_fraction = 2. / periods
        self.frequency_logits = nn.Parameter(torch.logit(frequency_fraction))
        self.initial_state = nn.Linear(2 * width, 2 * modes)
        self.drive_norm = nn.LayerNorm(width)
        self.drive_projection = nn.Linear(width, 2 * modes, bias=False)
        self.drive_dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(2 * modes, 1, bias=False)
        nn.init.normal_(self.initial_state.weight, std=.02)
        nn.init.zeros_(self.initial_state.bias)
        nn.init.normal_(self.drive_projection.weight, std=.02)

    def dynamics(self, history):
        """One transition per sample/channel/scale, shared over this horizon."""
        context = self.context_norm(torch.cat((history.mean(1), history[:, -1]), dim=-1))
        conditioning = self.condition(context)
        # Keep the pole parameterization and state scan in at least float32.
        dtype = torch.float64 if context.dtype == torch.float64 else torch.float32
        decay_shift, frequency_shift = conditioning.to(dtype).chunk(2, dim=-1)
        rate = F.softplus(self.log_decay.to(dtype) + decay_shift)
        radius = torch.exp(-rate)
        frequency = math.pi * torch.sigmoid(self.frequency_logits.to(dtype) + frequency_shift)
        transition = torch.stack((radius * frequency.cos(), radius * frequency.sin()), dim=-1)
        initial = self.initial_state(context).to(dtype).reshape(-1, self.modes, 2)
        return transition, initial, radius, frequency

    def future_drive(self, future_features, history=None):
        """Map generated future features [N,H,D] to drive [N,H,2J]."""
        return self.drive_projection(self.drive_dropout(F.gelu(self.drive_norm(future_features))))

    def states(self, history, future_features, intervention="normal"):
        if (history.ndim != 3 or future_features.ndim != 3 or
                history.shape[1] != self.length or future_features.shape[1] != self.horizon or
                history.shape[0] != future_features.shape[0] or
                history.shape[2] != future_features.shape[2]):
            raise ValueError("Encoded history/future shapes differ from decoder configuration")
        transition, initial, _, _ = self.dynamics(history)
        drive = self.future_drive(future_features, history)
        drive = drive.reshape(history.shape[0], self.horizon, self.modes, 2)
        transition = transition[:, None].expand(-1, self.horizon, -1, -1)
        if intervention == "no_memory":
            transition = torch.zeros_like(transition)
        elif intervention == "zero_initial":
            initial = torch.zeros_like(initial)
        elif intervention != "normal":
            raise ValueError("Unknown oscillator intervention")
        return parallel_oscillator_scan(transition, drive, initial)

    def forward(self, history, future_features, intervention="normal"):
        states = self.states(history, future_features, intervention)
        # Float32 scan is retained until the usual autocast-aware output linear.
        return self.readout(states.flatten(2).to(self.readout.weight.dtype)).squeeze(-1)


class OscillatorSeasonal(LagMessageSeasonal):
    """Reuse only the existing FFT history encoder and future feature projection."""

    def __init__(self, encoder, modes, dropout):
        nn.Module.__init__(self)
        self.length, self.horizon = encoder.length, encoder.horizon
        self.period_top_k = encoder.period_top_k
        self.value_embedding = encoder.value_embedding
        self.blocks = encoder.blocks
        self.temporal_projection = encoder.temporal_projection
        self.register_buffer("positions", encoder.positions.detach().clone())
        self.register_buffer("future_positions", encoder.future_positions.detach().clone())
        self.register_buffer("oscillator_encoder_version", torch.tensor(1, dtype=torch.long))
        self.decoder = OscillatorDecoder(self.length, self.horizon,
                                         self.value_embedding.out_features, modes, dropout)

    def encode_periods(self, time_tokens, periods, weights):
        # Axial residual gates / LayerNorm may promote AMP attention results to
        # float32. Accumulate period groups in float32 as well: index_add needs
        # exactly matching source/destination dtypes.
        if time_tokens.dtype in (torch.float16, torch.bfloat16):
            time_tokens = time_tokens.float()
        return super().encode_periods(time_tokens, periods, weights)

    def forecast_from_history(self, x, history, zero_acf=False, intervention="normal"):
        if zero_acf:
            raise ValueError("OscillatorMixer has no ACF-token decoder; use no_memory intervention")
        if history.shape[:2] != x.shape:
            raise ValueError("Encoded history must preserve the seasonal time grid")
        future = self.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
        future = future + self.future_positions.to(future)
        return self.decoder(history, future, intervention)

    def forward(self, x, zero_acf=False, intervention="normal"):
        batch, length, channels = x.shape
        if length != self.length:
            raise ValueError("Seasonal input length differs from configured length")
        flat = x.transpose(1, 2).reshape(batch * channels, length)
        history = self.encode_history(flat)
        output = self.forecast_from_history(flat, history, zero_acf, intervention)
        return output.reshape(batch, channels, self.horizon).transpose(1, 2)
