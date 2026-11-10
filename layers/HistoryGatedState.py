"""State-dependent history retrieval and gated, coupled recurrent updates.

Only observed, already fused seasonal/trend features enter this decoder.
There is no horizon-length learned projection, teacher forcing, oscillator
transition, or independent future-drive sequence. One real state emits a patch.
"""
import math

import torch
from torch import nn


def time_features(times, width):
    """Encode original-sample coordinates: times [L] -> [L, width]."""
    rates = torch.exp(
        torch.arange(0, width, 2, device=times.device, dtype=times.dtype)
        * (-math.log(10000.0) / width)
    )
    angles = times[:, None] * rates
    result = times.new_zeros(times.numel(), width)
    result[:, 0::2] = angles.sin()
    result[:, 1::2] = angles[:, :width // 2].cos()
    return result


class HistoryGatedStateDecoder(nn.Module):
    """Shared GRU cell for history initialization and future patch decoding.

    For each variable/scale, memory is [N,T,D] and state is [N,2*modes].
    Future query depends on the CURRENT state, so the history context and
    recurrent gates cannot be precomputed independently of the trajectory.
    Time/duration are learned conditioning features, not an ODE discretization.
    """

    def __init__(self, length, horizon, width, state_size, stride, readout,
                 dropout=0.0, patch_size=4):
        super().__init__()
        if horizon < patch_size or horizon % patch_size:
            raise ValueError("Horizon must be a positive multiple of patch_size")
        if min(length, width, state_size, stride) <= 0:
            raise ValueError("Length, width, state_size and stride must be positive")
        self.length = length
        self.horizon = horizon
        self.width = width
        self.state_size = state_size
        self.stride = stride
        self.patch_size = patch_size
        self.memory_projection = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
        )
        self.key_projection = nn.Linear(width, width, bias=False)
        self.value_projection = nn.Linear(width, width, bias=False)
        self.state_query = nn.Linear(state_size, width, bias=False)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(2 * width + 1),
            nn.Linear(2 * width + 1, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Dense hidden-to-hidden matrices allow all real state coordinates to mix.
        self.cell = nn.GRUCell(width, state_size)
        self.readout = readout

    def prepare_memory(self, season, trend):
        if (season.ndim != 3 or trend.shape != season.shape
                or season.shape[1:] != (self.length, self.width)):
            raise ValueError("Expected matching seasonal/trend [N,length,width]")
        return self.memory_projection(torch.cat((season, trend), dim=-1))

    def _cell_input(self, context, position, duration):
        position = position.expand(context.shape[0], -1)
        dt = context.new_full((context.shape[0], 1), math.log1p(duration))
        return self.input_projection(torch.cat((context, position, dt), dim=-1))

    def encode_history(self, memory, history_gain=None):
        # The upstream encoder is bidirectional over observed history. These
        # are latent updates, NOT causal one-step predictions of observations.
        times = torch.arange(
            1 - self.length, 1, device=memory.device, dtype=memory.dtype
        ) * self.stride
        positions = time_features(times, self.width)
        state = memory.new_zeros(memory.shape[0], self.state_size)
        for index in range(self.length):
            state = self.cell(
                self._cell_input(memory[:, index], positions[index], self.stride),
                state,
            )
        if history_gain is not None:
            state = state * history_gain.reshape(-1, 1)
        return state

    def decode(self, memory, initial, input_gain=None, steps=None,
               return_details=False):
        if (memory.ndim != 3 or memory.shape[1:] != (self.length, self.width)
                or initial.shape != (memory.shape[0], self.state_size)):
            raise ValueError("Memory/initial state shape mismatch")
        count = self.horizon // self.patch_size if steps is None else steps
        if not isinstance(count, int) or not 1 <= count <= self.horizon // self.patch_size:
            raise ValueError("Invalid number of future patches")
        keys = self.key_projection(memory)
        values = self.value_projection(memory)
        times = torch.arange(
            1, count + 1, device=memory.device, dtype=memory.dtype
        ) * self.patch_size
        positions = time_features(times, self.width)
        state = initial
        states, attention = [], []
        for index in range(count):
            query = self.state_query(state) + positions[index]
            scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
            weights = torch.softmax(scores / math.sqrt(self.width), dim=-1)
            context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
            cell_input = self._cell_input(context, positions[index], self.patch_size)
            if input_gain is not None:
                cell_input = cell_input * input_gain.reshape(-1, 1)
            state = self.cell(cell_input, state)
            states.append(state)
            if return_details:
                attention.append(weights)
        trajectory = torch.stack(states, dim=1)
        # Same per-scale 16 -> 32 -> GELU -> 4 readout as the control.
        prediction = self.readout(trajectory)
        if prediction.shape != (memory.shape[0], count, self.patch_size):
            raise RuntimeError("Readout must emit exactly one four-point patch")
        prediction = prediction.reshape(memory.shape[0], count * self.patch_size)
        if return_details:
            return prediction, dict(
                initial=initial, states=trajectory,
                attention=torch.stack(attention, dim=1),
            )
        return prediction

    def forward(self, season, trend, history_gain=None, input_gain=None,
                return_details=False):
        memory = self.prepare_memory(season, trend)
        initial = self.encode_history(memory, history_gain)
        return self.decode(memory, initial, input_gain, return_details=return_details)
