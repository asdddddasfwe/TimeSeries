# Vendored from PyPOTS commit 53b3eac34be9491ac3f28e65ee1993436e9318af. See UPSTREAM.json and LICENSE.
""" """

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn

from .layers import MixerBlock
from .revin import RevIN
from .attention import ScaledDotProductAttention, MultiHeadAttention
from .embedding import DataEmbedding


class BackboneTimeMixerPP(nn.Module):
    """Retained forecast path: independent channels, average pooling and mean ensemble."""

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        dropout: float,
        top_k: int,
        n_kernels: int,
        downsampling_layers: int,
        downsampling_window: int,
        use_norm: bool = False,
        freq="h",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.downsampling_layers = downsampling_layers
        self.downsampling_window = downsampling_window
        self.use_norm = use_norm

        # Preserve module names and construction order for the best checkpoint.
        self.enc_embedding = DataEmbedding(1, d_model, "fixed", freq, dropout, with_pos=False)
        if self.use_norm:
            self.revin_layers = nn.ModuleList([RevIN(n_features) for _ in range(downsampling_layers + 1)])
        self.encoder_model = nn.ModuleList([
            MixerBlock(n_steps, n_pred_steps, top_k, d_model, d_ffn, n_kernels,
                       downsampling_window)
            for _ in range(n_layers)
        ])

        d_time_model = n_steps // (downsampling_window**downsampling_layers)
        d_kv = d_time_model // n_heads
        full_attn_operator = ScaledDotProductAttention(d_kv**0.5, dropout)
        self.channel_mixing_attention = MultiHeadAttention(
            full_attn_operator, d_time_model, n_heads, d_kv, d_kv,
        )
        self.down_pool = nn.AvgPool1d(downsampling_window)
        self.predict_layers = nn.ModuleList([
            nn.Linear(n_steps // (downsampling_window**i), n_pred_steps)
            for i in range(downsampling_layers + 1)
        ])
        self.projection_layer = nn.Linear(d_model, 1, bias=True)

    def _multi_scale_inputs(self, x):
        scales = [x]
        current = x.permute(0, 2, 1)
        for _ in range(self.downsampling_layers):
            current = self.down_pool(current)
            scales.append(current.permute(0, 2, 1))
        return scales

    def forecast(self, x_enc):
        batch, _, channels = x_enc.size()
        x_list = []
        for i, x in enumerate(self._multi_scale_inputs(x_enc)):
            length = x.size(1)
            x = self.revin_layers[i](x, mode="norm") if self.use_norm else x
            x = x.permute(0, 2, 1).contiguous().reshape(batch * channels, length, 1)
            x_list.append(x)

        # Channel attention operates only on the coarsest scale.
        _, length, width = x_list[-1].size()
        coarse = x_list[-1].reshape(batch, channels, length * width)
        mixed, _ = self.channel_mixing_attention(coarse, coarse, coarse, None)
        x_list[-1] = mixed.reshape(batch * channels, length, width) + x_list[-1]

        enc_out_list = [self.enc_embedding(x, None) for x in x_list]
        for block in self.encoder_model:
            enc_out_list = block(enc_out_list)

        predictions = []
        for head, enc_out in zip(self.predict_layers, enc_out_list):
            output = head(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            output = self.projection_layer(output)
            output = output.reshape(batch, channels, -1).permute(0, 2, 1).contiguous()
            predictions.append(output)

        # All heads contribute; invert the finest-scale normalization exactly once.
        output = torch.stack(predictions, dim=0).mean(dim=0)
        return self.revin_layers[0](output, mode="denorm") if self.use_norm else output
