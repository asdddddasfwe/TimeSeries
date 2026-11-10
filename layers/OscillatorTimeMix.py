"""Keep coarse historical scans and use original-step dynamics for full forecasts."""
import math

import torch
from torch.nn import functional as F

from layers.OscillatorGatedMixer import GatedHistoryDecoder


def forecast_positions(length, horizon, width, stride):
    """Future positions in local history units, after the last historical token."""
    positions = length - 1 + torch.arange(1, horizon + 1, dtype=torch.float32) / stride
    rates = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.) / width))
    encoding = torch.zeros(horizon, width)
    encoding[:, 0::2] = torch.sin(positions[:, None] * rates)
    encoding[:, 1::2] = torch.cos(positions[:, None] * rates[:width // 2])
    return encoding


def _complex_prefix(a, b):
    """Same balanced affine tree as the reference, with native complex products."""
    length = a.shape[1]
    if length == 1:
        return a, b
    pairs = length // 2
    la, ra = a[:, :2 * pairs:2], a[:, 1::2]
    lb, rb = b[:, :2 * pairs:2], b[:, 1::2]
    odd_a, odd_b = _complex_prefix(ra * la, rb + ra * lb)
    tail_a, tail_b = a[:, 2::2], b[:, 2::2]
    count = tail_a.shape[1]
    even_a = tail_a * odd_a[:, :count]
    even_b = tail_b + tail_a * odd_b[:, :count]
    inter_a = torch.stack((odd_a[:, :count], even_a), 2).flatten(1, 2)
    inter_b = torch.stack((odd_b[:, :count], even_b), 2).flatten(1, 2)
    return (torch.cat((a[:, :1], inter_a, odd_a[:, count:]), 1),
            torch.cat((b[:, :1], inter_b, odd_b[:, count:]), 1))


class _AffineScan(torch.autograd.Function):
    """Analytic reverse scan avoids autograd through the forward tree's slices.

    For s_t=a_t*s_(t-1)+b_t, the complex adjoint is
    g_t=upstream_t+conj(a_(t+1))*g_(t+1). Both directions use the same
    balanced tree; there is no time-step Python loop or gradient truncation.
    """

    @staticmethod
    def forward(ctx, a, b, start):
        pa, pb = _complex_prefix(a, b)
        states = pb + pa * start[:, None]
        ctx.save_for_backward(a, states, start)
        return states

    @staticmethod
    def backward(ctx, grad_output):
        a, states, start = ctx.saved_tensors
        reverse_a = torch.cat((torch.zeros_like(a[:, :1]), a[:, 1:].flip(1).conj()), 1)
        _, reverse_grad = _complex_prefix(reverse_a, grad_output.flip(1))
        grad = reverse_grad.flip(1)
        previous = torch.cat((start[:, None], states[:, :-1]), 1)
        return grad * previous.conj(), grad, grad[:, 0] * a[:, 0].conj()


def complex_oscillator_scan(transition, drive, initial):
    """Exact affine scan in real arithmetic; float32/64 accumulation, no division.

    Inputs/outputs retain the public real/imag pair representation. Internally
    native complex multiplication replaces six scalar operations and their
    select/stack backward nodes. Supports zero and time-varying transitions.
    """
    if (transition.ndim != 4 or transition.shape != drive.shape or
            transition.shape[-1] != 2 or transition.shape[1] < 1 or
            initial.shape != (drive.shape[0], drive.shape[2], 2)):
        raise ValueError("Expected transition/drive[N,H>=1,J,2] and initial[N,J,2]")
    if transition.device != drive.device or initial.device != drive.device:
        raise ValueError("All scan inputs must share a device")
    dtype = torch.promote_types(torch.promote_types(transition.dtype, drive.dtype), initial.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    a = torch.view_as_complex(transition.to(dtype).contiguous())
    b = torch.view_as_complex(drive.to(dtype).contiguous())
    start = torch.view_as_complex(initial.to(dtype).contiguous())
    return torch.view_as_real(_AffineScan.apply(a, b, start))


class FullResolutionGatedDecoder(GatedHistoryDecoder):
    """A_history covers stride samples; A_future is its principal stride-th root.

    The inherited dynamics first scans coarse historical features. Only the
    transition returned for future scanning is converted. Future drive remains
    learned at each original-grid position; no future values are used.
    """

    def __init__(self, decoder, horizon, stride):
        super().__init__(decoder)
        self.future_gate = decoder.future_gate
        self.horizon, self.stride = horizon, stride

    def encode_history_states(self, history, transition):
        drive = self.drive_projection(self.drive_dropout(F.gelu(self.drive_norm(history))))
        drive = drive.reshape(history.shape[0], self.length, self.modes, 2)
        initial = transition.new_zeros(history.shape[0], self.modes, 2)
        return complex_oscillator_scan(
            transition[:, None].expand(-1, self.length, -1, -1), drive, initial)

    def dynamics(self, history):
        transition, initial, radius, frequency = super().dynamics(history)
        if self.stride == 1:
            return transition, initial, radius, frequency
        radius = radius.pow(1. / self.stride)
        frequency = frequency / self.stride
        transition = torch.stack((radius * frequency.cos(), radius * frequency.sin()), dim=-1)
        return transition, initial, radius, frequency
