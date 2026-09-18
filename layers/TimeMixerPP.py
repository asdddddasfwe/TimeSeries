"""TimeMixer++ blocks independently implemented from arXiv:2410.16032v5.

Image layout: [batch, embedding, phase within period, period index].
See docs/TimeMixerPP.md for equation mapping and paper ambiguities.
"""

import torch
from torch import nn
from torch.nn import functional as F


def select_periods(coarsest, full_length, top_k):
    """Eq. (5): non-DC batch-mean top bins with differentiable sample weights.

    Float32 FFT supports AMP and odd lengths. The period numerator is original T,
    as printed in the paper, not the length of the downsampled FFT input.
    """
    amplitudes = torch.fft.rfft(coarsest.float(), dim=1).abs().mean(dim=-1)
    count = min(top_k, amplitudes.shape[1] - 1)
    if count < 1:
        raise ValueError("TimeMixerPP requires at least two coarsest-scale time points.")
    indices = amplitudes[:, 1:].mean(0).argsort(descending=True, stable=True)[:count] + 1
    periods = [(full_length + frequency - 1) // frequency
               for frequency in indices.detach().cpu().tolist()]
    return periods, amplitudes.index_select(1, indices).softmax(dim=-1)


def to_time_image(x, period):
    """Eq. (6): [B,T,D] -> [B,D,P,ceil(T/P)], preserving chronological order."""
    batch, length, channels = x.shape
    columns = (length + period - 1) // period
    x = F.pad(x, (0, 0, 0, columns * period - length))
    return x.reshape(batch, columns, period, channels).permute(0, 3, 2, 1).contiguous()


def from_time_image(image, length):
    """Eq. (10): restore chronological order and remove right padding."""
    batch, channels, period, columns = image.shape
    return image.permute(0, 3, 2, 1).reshape(batch, columns * period, channels)[:, :length]


class ChannelAttention(nn.Module):
    """Eq. (2): variates are tokens; coarsest time points supply features."""

    def __init__(self, length, dropout):
        super().__init__()
        self.qkv = nn.Linear(length, 3 * length)
        self.dropout = dropout

    def forward(self, x):
        q, k, v = self.qkv(x.transpose(1, 2)).chunk(3, dim=-1)
        mixed = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            dropout_p=self.dropout if self.training else 0.0)
        return mixed.squeeze(1).transpose(1, 2)


class AxialAttention(nn.Module):
    """Convolutional Q/K/V shared across scales and periods; one attention axis."""

    def __init__(self, d_model, n_heads, axis, dropout):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("TimeMixerPP d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.axis = axis
        self.dropout = dropout
        self.qkv = nn.Conv2d(d_model, 3 * d_model, 3, padding=1)
        self.projection = nn.Conv2d(d_model, d_model, 1)

    def forward(self, image, valid):
        batch, channels, period, columns = image.shape
        if self.axis == "season":
            other, length = columns, period
            order, restore = (0, 3, 2, 1), (0, 3, 2, 1)
        else:
            other, length = period, columns
            order, restore = (0, 2, 3, 1), (0, 3, 1, 2)

        def tokens(tensor):
            tensor = tensor.permute(order).reshape(
                batch * other, length, self.n_heads, channels // self.n_heads)
            return tensor.transpose(1, 2)

        q, k, v = [tokens(part) for part in self.qkv(image).chunk(3, dim=1)]
        key_mask = valid.permute(order).reshape(batch * other, length)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=key_mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(batch, other, length, channels)
        attended = attended.permute(restore).contiguous()
        return self.projection(attended).masked_fill(~valid, 0)


class TimeImageDecomposition(nn.Module):
    """Eq. (7), Fig. 7: within-period seasons and across-period trends."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.season = AxialAttention(d_model, n_heads, "season", dropout)
        self.trend = AxialAttention(d_model, n_heads, "trend", dropout)

    def forward(self, image, valid):
        return self.season(image, valid), self.trend(image, valid)


def _match_columns(image, columns):
    # Ceil(T/P) need not divide by two. Never resize the phase axis.
    image = image[..., :columns]
    return F.pad(image, (0, max(0, columns - image.shape[-1])))


class MultiScaleImageMixing(nn.Module):
    """Eqs. (8)-(9): bottom-up seasons, top-down trends, net stride two in F."""

    def __init__(self, n_scales, d_model, d_ff):
        super().__init__()
        self.season_down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_ff, 3, stride=(1, 2), padding=1),
                nn.GELU(), nn.Conv2d(d_ff, d_model, 3, padding=1),
            ) for _ in range(n_scales - 1)
        ])
        self.trend_up = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(d_model, d_ff, (3, 4),
                                   stride=(1, 2), padding=(1, 1)),
                nn.GELU(), nn.ConvTranspose2d(d_ff, d_model, 3, padding=1),
            ) for _ in range(n_scales - 1)
        ])

    def forward(self, seasons, trends, masks):
        seasons, trends = list(seasons), list(trends)
        for scale in range(1, len(seasons)):
            residual = self.season_down[scale - 1](seasons[scale - 1])
            residual = _match_columns(residual, seasons[scale].shape[-1])
            seasons[scale] = (seasons[scale] + residual).masked_fill(~masks[scale], 0)
        for scale in range(len(trends) - 2, -1, -1):
            residual = self.trend_up[scale](trends[scale + 1])
            residual = _match_columns(residual, trends[scale].shape[-1])
            trends[scale] = (trends[scale] + residual).masked_fill(~masks[scale], 0)
        return [season + trend for season, trend in zip(seasons, trends)]


class MixerBlock(nn.Module):
    """MRTI -> TID -> MCM -> MRM, with residual LayerNorm (Eqs. 4-11)."""

    def __init__(self, n_scales, d_model, d_ff, n_heads, top_k, dropout):
        super().__init__()
        self.top_k = top_k
        self.decomposition = TimeImageDecomposition(d_model, n_heads, dropout)
        self.mixing = MultiScaleImageMixing(n_scales, d_model, d_ff)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_scales)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, scales, valid_masks):
        periods, weights = select_periods(scales[-1], scales[0].shape[1], self.top_k)
        mixed = [torch.zeros_like(x) for x in scales]
        for resolution, period in enumerate(periods):
            images = [to_time_image(x, period) for x in scales]
            masks = [to_time_image(valid, period) for valid in valid_masks]
            decomposed = [self.decomposition(image, mask)
                          for image, mask in zip(images, masks)]
            seasons, trends = zip(*decomposed)
            outputs = self.mixing(seasons, trends, masks)
            for scale, image in enumerate(outputs):
                sequence = from_time_image(image, scales[scale].shape[1])
                weight = weights[:, resolution, None, None].to(sequence.dtype)
                mixed[scale] = mixed[scale] + weight * sequence
        return [
            norm(original + self.dropout(update)).masked_fill(~valid, 0)
            for norm, original, update, valid
            in zip(self.norms, scales, mixed, valid_masks)
        ]
