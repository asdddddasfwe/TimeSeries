"""FFT axial encoding and finite-depth parallel lag message propagation."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from layers.pypots_timemixerpp.layers import RowAttention, ColAttention


def autocorrelation(x, max_lag):
    """Biased ACF r[l] = sum(z[t] z[t-l]) / sum(z[t]^2), excluding lag zero.

    x: [N, T]. Each row is centered independently. Missing lags in a shorter
    window have statistic zero (their reliability is supplied separately).
    """
    if x.ndim != 2 or max_lag < 1:
        raise ValueError("ACF expects [N, T] and max_lag >= 1")
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    centered = work - work.mean(-1, keepdim=True)
    energy = centered.square().sum(-1).clamp_min(1e-8)
    # Zero padding makes this linear correlation, with no wrap at the boundary.
    length = work.shape[-1]
    fft_length = 1 << (2 * length - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=fft_length, dim=-1)
    correlation = torch.fft.irfft(spectrum * spectrum.conj(), n=fft_length, dim=-1)
    available = min(max_lag, length - 1)
    values = correlation[:, 1:available + 1] / energy[:, None]
    return F.pad(values, (0, max_lag - available))


def lag_statistics(x, max_lag):
    """All lags 1..max_lag in temporal order, with seven historical features.

    No ACF top-k selection or ACF-based weight normalization. Features are
    lag/T, full/recent/early ACF, recent-minus-early, and two pair fractions.
    """
    length = x.shape[-1]
    if not 1 <= max_lag < length:
        raise ValueError("Require 1 <= max_lag < sequence length")
    full = autocorrelation(x, max_lag)
    half = length // 2
    early = autocorrelation(x[:, :half], max_lag)
    recent = autocorrelation(x[:, -half:], max_lag)
    lags = torch.arange(1, max_lag + 1, device=x.device)
    lag_fraction = (lags / length).to(full).expand_as(full)
    pair_fraction = ((length - lags) / length).to(full).expand_as(full)
    half_pairs = ((half - lags).clamp_min(0) / half).to(full).expand_as(full)
    features = torch.stack((lag_fraction, full, recent, early, recent - early,
                            pair_fraction, half_pairs), dim=-1)
    return lags[None].expand(x.shape[0], -1), features


def select_fft_periods(x, top_k):
    """Per-series non-DC rFFT peaks -> floor(T/bin) and amplitude-softmax weights.

    x [N,T] is the original-resolution seasonal history, before embedding.
    These periods control image geometry; ACF-selected AR lags do not.
    """
    if x.ndim != 2 or x.shape[-1] < 2 or top_k < 1:
        raise ValueError("FFT period selection requires [N,T>=2] and positive top_k")
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    amplitudes = torch.fft.rfft(work, dim=-1).abs()[:, 1:]
    indices = torch.argsort(amplitudes, dim=-1, descending=True, stable=True)
    indices = indices[:, :min(top_k, amplitudes.shape[-1])]
    weights = amplitudes.gather(-1, indices).softmax(-1)
    return x.shape[-1] // (indices + 1), weights


def sinusoidal_positions(length, width):
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    rates = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(positions * rates)
    encoding[:, 1::2] = torch.cos(positions * rates[:width // 2])
    return encoding


class LagAxialBlock(nn.Module):
    """TimeMixer++ row/column attention on the same [N,D,cycles,period] image.

    Row attends within each FFT period; column attends across cycles
    at the same phase. Preserve the upstream zero-initialized residual gates.
    This single-scale adapter sums attention *updates* once, then normalizes.
    """

    def __init__(self, width, q_k_dim, dropout):
        super().__init__()
        self.row = RowAttention(width, q_k_dim)
        self.column = ColAttention(width, q_k_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def projection_parameters(self):
        # Keep the original parameter names/checkpoints; execute the six pointwise
        # projections as one matrix multiply instead of six small convolutions.
        projections = [
            layer for branch in (self.row, self.column)
            for layer in (branch.query_conv, branch.key_conv, branch.value_conv)
        ]
        weight = torch.cat([layer.weight.flatten(1) for layer in projections])
        bias = torch.cat([layer.bias for layer in projections])
        return weight, bias, [layer.out_channels for layer in projections]

    def forward(self, image, projections=None):
        if projections is None:
            projections = self.projection_parameters()
        weight, bias, sizes = projections
        tokens = image.permute(0, 2, 3, 1)
        projected = F.linear(tokens, weight, bias)
        rq, rk, rv, cq, ck, cv = projected.split(sizes, dim=-1)

        def attend(query, key, value):
            # Upstream TimeMixer++ uses unscaled logits and no attention dropout.
            # Pad V for fused CUDA kernels when its width differs from Q/K.
            width = value.shape[-1]
            if width < query.shape[-1]:
                value = F.pad(value, (0, query.shape[-1] - width))
            return F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, scale=1.0
            )[..., :width]

        row = attend(rq, rk, rv)
        column = attend(cq.transpose(1, 2), ck.transpose(1, 2),
                        cv.transpose(1, 2)).transpose(1, 2)
        # Both axes read the same input, with exactly one external residual.
        update = self.row.gamma * row + self.column.gamma * column
        mixed = image + self.dropout(update.permute(0, 3, 1, 2))
        return self.norm(mixed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class LagMessageBlock(nn.Module):
    """One synchronous update of all H future feature vectors.

    source[h,tau] = history[T+h-tau] for h-tau<0, else previous_future[h-tau].
    Every message in a layer reads the same immutable previous-layer snapshot.
    Chunking spans lags only; H is always computed in parallel.
    """

    def __init__(self, length, horizon, width, d_ff, max_lag, chunk_size, dropout):
        super().__init__()
        if not 1 <= max_lag < length or chunk_size < 1:
            raise ValueError("Require 1 <= max_lag < length and positive chunk_size")
        self.length, self.horizon, self.max_lag = length, horizon, max_lag
        self.chunk_size = chunk_size
        self.state_norm = nn.LayerNorm(width)
        self.target_projection = nn.Linear(width, width)
        self.source_projection = nn.Linear(width, width, bias=False)
        self.lag_projection = nn.Linear(width, width, bias=False)
        self.message_norm = nn.LayerNorm(width)
        self.update = nn.Sequential(
            nn.Linear(2 * width, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, width),
        )
        # Small residual initialization is not an amplitude bound.
        nn.init.normal_(self.update[-1].weight, std=0.01)
        nn.init.zeros_(self.update[-1].bias)
        indices = length + torch.arange(horizon)[:, None] - torch.arange(1, max_lag + 1)[None]
        self.register_buffer("source_indices", indices, persistent=False)

    @staticmethod
    def message_sum(target, source, lag, indices):
        """Factorized nonlinear messages; no softmax/tanh coefficient bound."""
        previous = source[:, indices, :]
        return F.gelu(target[:, :, None, :] + previous + lag[:, None, :, :]).sum(2)

    def aggregate(self, history, future, lag_tokens):
        if history.shape[1] != self.length or future.shape[1] != self.horizon:
            raise ValueError("History/future lengths differ from configured message block")
        if lag_tokens.shape[1] != self.max_lag:
            raise ValueError("Message block requires every lag in ascending order")
        normalized_future = self.state_norm(future)
        target = self.target_projection(normalized_future)
        # The concatenation is fixed for the complete layer: no in-place rollout.
        source = self.source_projection(
            torch.cat((self.state_norm(history), normalized_future), dim=1)
        )
        lag = self.lag_projection(lag_tokens)
        aggregated = torch.zeros_like(target)
        for start in range(0, self.max_lag, self.chunk_size):
            stop = min(start + self.chunk_size, self.max_lag)
            arguments = (target, source, lag[:, start:stop], self.source_indices[:, start:stop])
            if self.training and torch.is_grad_enabled():
                # Recompute chunk activations in backward instead of storing [N,H,L,D].
                part = checkpoint(
                    self.message_sum, *arguments, use_reentrant=False,
                    preserve_rng_state=False,  # message_sum has no random operations.
                )
            else:
                part = self.message_sum(*arguments)
            aggregated = aggregated + part
        # Feature scaling, not a bound on learned relation coefficients or output.
        return normalized_future, aggregated / math.sqrt(self.max_lag)

    def forward(self, history, future, lag_tokens):
        current, messages = self.aggregate(history, future, lag_tokens)
        delta = self.update(torch.cat((current, self.message_norm(messages)), dim=-1))
        return future + delta


class LagMessageSeasonal(nn.Module):
    """FFT period encoding -> full future features -> finite lag propagation."""

    def __init__(self, length, horizon, width, d_ff, depth, dropout,
                 top_k, max_lag, message_depth, chunk_size):
        super().__init__()
        if not 1 <= max_lag < length or min(top_k, message_depth, chunk_size) < 1:
            raise ValueError("Require valid lag range and positive top_k/message depth/chunk size")
        self.length, self.horizon = length, horizon
        self.max_lag = max_lag
        self.period_top_k = min(top_k, length // 2)
        self.value_embedding = nn.Linear(1, width)
        self.lag_embedding = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width))
        self.register_buffer("positions", sinusoidal_positions(length, width))
        self.register_buffer("future_positions", sinusoidal_positions(length + horizon, width)[length:])
        self.register_buffer("lag_message_encoder_version", torch.tensor(4, dtype=torch.long))
        self.blocks = nn.ModuleList([LagAxialBlock(width, d_ff, dropout) for _ in range(depth)])
        self.temporal_projection = nn.Linear(length, horizon)
        self.message_blocks = nn.ModuleList([
            LagMessageBlock(length, horizon, width, d_ff, max_lag, chunk_size, dropout)
            for _ in range(message_depth)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, 1)

    def encode_image(self, time_tokens, period, projections=None):
        """[N,T,D] -> [N,D,ceil(T/P),P] -> axial encoding -> restore/crop.

        P comes from FFT, not from an ACF lag. Right-zero padding is unmasked.
        """
        count, length, width = time_tokens.shape
        padded = F.pad(time_tokens, (0, 0, 0, (-length) % period))
        image = padded.reshape(count, -1, period, width).permute(0, 3, 1, 2).contiguous()
        for index, block in enumerate(self.blocks):
            image = block(image) if projections is None else block(image, projections[index])
        return image.permute(0, 2, 3, 1).reshape(count, -1, width)[:, :length]

    def encode_periods(self, time_tokens, periods, weights):
        """Encode each distinct FFT width once per series and fuse in time order.

        Identical floor(T/bin) values merge their weights. Grouping by width
        shares computation, never statistics/attention across samples/channels.
        """
        # One small device-to-host transfer rather than a CUDA nonzero sync for
        # every distinct period. Only integer routing leaves the device.
        groups = {}
        for series, row in enumerate(periods.detach().cpu().tolist()):
            for period in set(row):
                groups.setdefault(period, []).append(series)
        ordered = sorted(groups)
        # Pack once per forward, sharing this autograd node across period groups.
        projections = [block.projection_parameters() for block in self.blocks]
        packed = torch.tensor(
            [series for period in ordered for series in groups[period]],
            dtype=torch.long, device=time_tokens.device,
        )
        fused = torch.zeros_like(time_tokens)
        offset = 0
        for period in ordered:
            count = len(groups[period])
            selected = packed[offset:offset + count]
            offset += count
            matches = periods[selected] == period
            merged_weight = (weights[selected] * matches).sum(-1).to(time_tokens)
            encoded = self.encode_image(time_tokens[selected], period, projections)
            fused = fused.index_add(0, selected, encoded * merged_weight[:, None, None])
        return fused

    def encode_history(self, x):
        """Keep [N,T,D] features; no temporal averaging into a single context."""
        periods, period_weights = select_fft_periods(x, self.period_top_k)
        time_tokens = self.value_embedding(x[..., None])
        time_tokens = time_tokens + self.positions.to(time_tokens)
        return self.encode_periods(time_tokens, periods, period_weights)

    def forecast_features(self, x, zero_acf=False):
        history = self.encode_history(x)
        return self.forecast_from_history(x, history, zero_acf=zero_acf)

    def forecast_from_history(self, x, history, zero_acf=False):
        """Decode [N,T,D] history; lag statistics always use raw seasonal x [N,T]."""
        if history.shape[:2] != x.shape:
            raise ValueError("Encoded history must preserve the seasonal time grid")
        _, statistics = lag_statistics(x, self.max_lag)
        if zero_acf:
            # Diagnostic intervention only: preserve lag position and pair fractions.
            statistics = statistics.clone()
            statistics[..., 1:5] = 0
        lag_tokens = self.lag_embedding(statistics.to(x.dtype))
        future = self.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
        future = future + self.future_positions.to(future)
        for block in self.message_blocks:
            future = block(history, future, lag_tokens)
        return future

    def forward(self, x, zero_acf=False):
        batch, length, channels = x.shape
        if length != self.length:
            raise ValueError("Seasonal input length differs from configured seq_len")
        flat = x.transpose(1, 2).reshape(batch * channels, length)
        future = self.forecast_features(flat, zero_acf=zero_acf)
        output = self.output_head(self.final_norm(future)).squeeze(-1)
        return output.reshape(batch, channels, self.horizon).transpose(1, 2)
