"""Channel-independent seasonal encoders and temporal scale mixing."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def select_periods(x, top_k):
    """Return [N, K] period widths/weights for independent [N, T] series.

    DC is excluded. Width = floor(T / positive rFFT bin index).
    Stable sorting makes ties (including constant inputs) deterministic.
    """
    fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    amplitudes = torch.fft.rfft(fft_input, dim=-1).abs()[:, 1:]
    k = min(top_k, amplitudes.shape[-1])
    indices = torch.argsort(amplitudes, dim=-1, descending=True, stable=True)[:, :k]
    weights = amplitudes.gather(-1, indices).softmax(dim=-1)
    periods = x.shape[-1] // (indices + 1)
    return periods, weights.to(x.dtype)


class PeriodPatchEncoder(nn.Module):
    """One full FFT period per token, with shared variable-length patch projections.

    Each patch first gets a residual Conv1d-GELU-Conv1d local encoder.
    Its period-P features are right-zero-padded to maximum width T.
    Linear(T, D) encodes it; Linear(D, T) decodes it, retaining only the first
    P outputs. All periods/variables share these registered parameters.
    """

    def __init__(self, length, d_model, n_heads, d_ff, depth, dropout, top_k):
        super().__init__()
        self.length = length
        self.top_k = top_k
        # Shared across patches, periods and variables within this scale.
        # Kernel 3 models local neighbors; kernel 1 projects back to one channel.
        self.local_encoder = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, 1, kernel_size=1),
        )
        self.patch_embedding = nn.Linear(length, d_model)
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(length, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[:d_model // 2])
        # Position denotes patch order, not individual time points within a patch.
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])
        self.patch_decoder = nn.Linear(d_model, length)

    def encode_period(self, sequence, period):
        """[N,T] -> [N,ceil(T/P),D] tokens -> reconstructed [N,T]."""
        n, length = sequence.shape
        padded_length = math.ceil(length / period) * period
        padded = F.pad(sequence, (0, padded_length - length))
        patches = padded.reshape(n, padded_length // period, period)
        # Separate patches are separate batch entries: convolution cannot cross
        # patch boundaries or variables. Convolve before width padding to T.
        local_input = patches.reshape(-1, 1, period)
        local_features = self.local_encoder(local_input).reshape_as(patches)
        patches = patches + local_features
        # Width padding preserves the order of the locally encoded time points.
        patches = F.pad(patches, (0, self.length - period))
        tokens = self.patch_embedding(patches)
        tokens = tokens + self.position_encoding[:, :tokens.shape[1]].to(tokens.dtype)
        # No padding mask: the final partial patch remains a regular token.
        # A single full-window patch is valid, though it has no cross-patch context.
        for block in self.blocks:
            tokens = block(tokens)
        decoded = self.patch_decoder(tokens)[..., :period]
        return decoded.reshape(n, padded_length)[:, :length]

    def forward(self, x):
        # Each sample/channel is a separate sequence; attention is across patches only.
        batch, length, channels = x.shape
        flat = x.transpose(1, 2).reshape(batch * channels, length)
        periods, weights = select_periods(flat, self.top_k)
        mixed = torch.zeros_like(flat)
        # Coalesce duplicate widths and batch together the corresponding series.
        for period in torch.unique(periods).tolist():
            period_weights = (weights * (periods == period)).sum(dim=-1)
            members = torch.nonzero((periods == period).any(dim=-1), as_tuple=True)[0]
            decoded = self.encode_period(flat.index_select(0, members), period)
            weighted = decoded * period_weights.index_select(0, members)[:, None]
            mixed = mixed.index_add(0, members, weighted)
        return mixed.reshape(batch, channels, length).transpose(1, 2)


class TemporalScaleMixing(nn.Module):
    """TimeMixer Linear -> GELU -> Linear mappings with additive scale residuals."""

    def __init__(self, lengths, fine_to_coarse):
        super().__init__()
        self.order = list(range(len(lengths)))
        if not fine_to_coarse:
            self.order.reverse()
        self.mappings = nn.ModuleList([
            nn.Sequential(nn.Linear(lengths[source], lengths[target]), nn.GELU(),
                          nn.Linear(lengths[target], lengths[target]))
            for source, target in zip(self.order[:-1], self.order[1:])
        ])

    def forward(self, scales):
        outputs = list(scales)
        previous = scales[self.order[0]].transpose(1, 2)
        for mapping, index in zip(self.mappings, self.order[1:]):
            previous = scales[index].transpose(1, 2) + mapping(previous)
            outputs[index] = previous.transpose(1, 2)
        return outputs
