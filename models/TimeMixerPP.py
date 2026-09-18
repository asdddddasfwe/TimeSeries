"""TimeMixer++: A General Time Series Pattern Machine (ICLR 2025).

Independent paper-based implementation: https://arxiv.org/abs/2410.16032
Supports TSLib's five supervised tasks; no pretrained weights are bundled.
See docs/TimeMixerPP.md for architecture choices and reproduction limits.
"""

import torch
from torch import nn
from torch.nn import functional as F

from layers.Embed import TemporalEmbedding, TimeFeatureEmbedding
from layers.TimeMixerPP import ChannelAttention, MixerBlock
from utils.logger import get_logger


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        supported = {"long_term_forecast", "short_term_forecast", "imputation",
                     "anomaly_detection", "classification"}
        if self.task_name not in supported:
            raise ValueError(
                f"TimeMixerPP does not support task {self.task_name!r}. "
                "Zero/few-shot reproduction requires a separate pretraining and checkpoint pipeline."
            )
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.classification = self.task_name == "classification"
        self.forecasting = self.task_name in {"long_term_forecast", "short_term_forecast"}
        self.c_out = configs.c_out if not self.classification else self.enc_in
        self.use_norm = bool(getattr(configs, "use_norm", 1)) and not self.classification
        n_down = getattr(configs, "down_sampling_layers", 3)
        if n_down < 0 or self.seq_len < 2 or self.enc_in < 1 or self.c_out < 1:
            raise ValueError("TimeMixerPP needs seq_len >= 2, positive channels and nonnegative scales.")
        self.lengths = [self.seq_len // (2 ** scale) for scale in range(n_down + 1)]
        if self.lengths[-1] < 2:
            raise ValueError("Too many down_sampling_layers: coarsest sequence must have length >= 2.")
        if n_down and getattr(configs, "down_sampling_window", 2) != 2:
            raise ValueError("TimeMixerPP uses stride-two convolutions: set --down_sampling_window 2.")
        if n_down and getattr(configs, "down_sampling_method", None) not in {None, "conv"}:
            raise ValueError("TimeMixerPP requires learned downsampling: set --down_sampling_method conv.")
        if configs.e_layers < 1 or configs.top_k < 1:
            raise ValueError("TimeMixerPP e_layers and top_k must be positive.")
        if configs.n_heads < 1 or configs.d_model < 1 or configs.d_ff < 1:
            raise ValueError("TimeMixerPP n_heads, d_model and d_ff must be positive.")
        if configs.d_model % configs.n_heads:
            raise ValueError("TimeMixerPP d_model must be divisible by n_heads.")
        if self.forecasting and self.pred_len < 1:
            raise ValueError("Forecasting requires pred_len > 0.")
        self.target_only = (self.forecasting and self.c_out == 1 and self.enc_in > 1
                            and getattr(configs, "features", "M") == "MS")
        if self.c_out != self.enc_in and not self.target_only:
            raise ValueError("Use c_out=enc_in, or features=MS and c_out=1 for the last target channel.")

        # Eq. (1): registered convolutions; crop odd lengths to floor(T/2).
        self.downsample = nn.ModuleList([
            nn.Conv1d(self.enc_in, self.enc_in, 3, stride=2, padding=1)
            for _ in range(n_down)
        ])
        self.channel_attention = ChannelAttention(self.lengths[-1], configs.dropout)
        self.value_embedding = nn.Linear(self.enc_in, configs.d_model)
        self.time_embedding = None
        if self.forecasting or self.task_name == "imputation":
            if configs.embed == "timeF":
                self.time_embedding = TimeFeatureEmbedding(configs.d_model, freq=configs.freq)
            else:
                self.time_embedding = TemporalEmbedding(
                    configs.d_model, embed_type=configs.embed, freq=configs.freq)
        self.dropout = nn.Dropout(configs.dropout)
        self.blocks = nn.ModuleList([
            MixerBlock(len(self.lengths), configs.d_model, configs.d_ff,
                       configs.n_heads, configs.top_k, configs.dropout)
            for _ in range(configs.e_layers)
        ])
        if self.classification:
            if configs.num_class < 1:
                raise ValueError("Classification requires a positive num_class.")
            self.heads = nn.ModuleList([
                nn.Linear(length * configs.d_model, configs.num_class)
                for length in self.lengths
            ])
        else:
            output_length = self.pred_len if self.forecasting else self.seq_len
            self.time_heads = nn.ModuleList([
                nn.Linear(length, output_length) for length in self.lengths
            ])
            self.channel_heads = nn.ModuleList([
                nn.Linear(configs.d_model, self.c_out) for _ in self.lengths
            ])
        get_logger(__name__).info(
            "TimeMixerPP: scales=%s, blocks=%d, top_k=%d, channel mixing enabled",
            self.lengths, configs.e_layers, configs.top_k,
        )

    def _normalize(self, x, observed=None):
        # TSLib adaptation: per-instance stats, observed-only for imputation.
        if observed is None:
            observed = torch.ones_like(x, dtype=torch.bool)
        if observed.shape != x.shape:
            raise ValueError("Imputation mask must match x_enc [B,T,C].")
        observed = observed.bool()
        clean = torch.where(observed, x, 0)
        if not self.use_norm:
            return clean, None, None
        count = observed.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (clean.sum(dim=1, keepdim=True) / count).detach()
        centered = torch.where(observed, clean - mean, 0)
        std = (centered.square().sum(dim=1, keepdim=True) / count + 1e-5).sqrt().detach()
        return centered / std, mean, std

    def _encode(self, x, markers, valid):
        scales, masks = [x], [valid]
        marker_scales = [markers]
        for layer, length in zip(self.downsample, self.lengths[1:]):
            x = layer(x.transpose(1, 2))[..., :length].transpose(1, 2)
            valid = F.max_pool1d(valid.transpose(1, 2).float(), 2, 2).transpose(1, 2).bool()
            x = x.masked_fill(~valid, 0)
            scales.append(x)
            masks.append(valid)
            if markers is not None:
                markers = markers[:, ::2, :][:, :length]
            marker_scales.append(markers)
        scales[-1] = self.channel_attention(scales[-1]).masked_fill(~masks[-1], 0)
        embedded = []
        for scale, mark, keep in zip(scales, marker_scales, masks):
            value = self.value_embedding(scale)
            if mark is not None and self.time_embedding is not None:
                value = value + self.time_embedding(mark)
            embedded.append(self.dropout(value).masked_fill(~keep, 0))
        for block in self.blocks:
            embedded = block(embedded, masks)
        return embedded, masks

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Encoder-only: no decoder values or future markers are consumed.
        if x_enc.ndim != 3 or x_enc.shape[-1] != self.enc_in:
            raise ValueError(f"x_enc must have shape [B,T,{self.enc_in}].")
        if self.classification:
            if x_enc.shape[1] > self.seq_len:
                raise ValueError("Classification input exceeds configured seq_len.")
            valid = (torch.ones(x_enc.shape[:2], device=x_enc.device, dtype=torch.bool)
                     if x_mark_enc is None else x_mark_enc.bool())
            if valid.shape != x_enc.shape[:2]:
                raise ValueError("Classification padding mask must have shape [B,T].")
            pad = self.seq_len - x_enc.shape[1]
            x = F.pad(x_enc.masked_fill(~valid.unsqueeze(-1), 0), (0, 0, 0, pad))
            valid = F.pad(valid, (0, pad)).unsqueeze(-1)
            embedded, masks = self._encode(x, None, valid)
            outputs = [
                head(self.dropout(F.gelu(value)).masked_fill(~keep, 0).flatten(1))
                for head, value, keep in zip(self.heads, embedded, masks)
            ]
        else:
            if x_enc.shape[1] != self.seq_len:
                raise ValueError(f"Expected seq_len={self.seq_len}, received {x_enc.shape[1]}.")
            if self.task_name == "imputation" and mask is None:
                raise ValueError("Imputation requires an observed-value mask (1=observed, 0=missing).")
            observed = mask if self.task_name == "imputation" else None
            x, mean, std = self._normalize(x_enc, observed)
            valid = torch.ones_like(x[..., :1], dtype=torch.bool)
            markers = x_mark_enc if self.time_embedding is not None else None
            if markers is not None and markers.shape[:2] != x.shape[:2]:
                raise ValueError("Time markers must have shape [B,seq_len,time_features].")
            embedded, _ = self._encode(x, markers, valid)
            outputs = [
                channel_head(time_head(value.transpose(1, 2)).transpose(1, 2))
                for time_head, channel_head, value
                in zip(self.time_heads, self.channel_heads, embedded)
            ]
        # Eq. (3): equal-weight ensemble of scale-specific prediction heads.
        output = torch.stack(outputs, dim=0).mean(dim=0)
        if not self.classification and self.use_norm:
            if self.target_only:
                mean, std = mean[..., -1:], std[..., -1:]
            output = output * std + mean
        return output
