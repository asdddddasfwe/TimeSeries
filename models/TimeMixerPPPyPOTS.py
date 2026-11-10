"""TSLib forecasting adapter for a pinned PyPOTS TimeMixer++ implementation.

Source: WenjieDu/PyPOTS @ 53b3eac34be9491ac3f28e65ee1993436e9318af.
See docs/TimeMixerPPPyPOTS.md for the retained best configuration and local fixes.
"""

from torch import nn

from layers.pypots_timemixerpp.backbone import BackboneTimeMixerPP
from utils.logger import get_logger


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("TimeMixerPPPyPOTS currently supports forecasting tasks only.")
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        n_down = getattr(configs, "down_sampling_layers", 3)
        window = getattr(configs, "down_sampling_window", 2)
        if window < 2 or n_down < 1:
            raise ValueError("TimeMixerPPPyPOTS requires down_sampling_window >= 2 and down_sampling_layers >= 1.")
        divisor = window ** n_down
        if self.seq_len % divisor or self.seq_len // divisor < 2:
            raise ValueError("TimeMixerPPPyPOTS requires seq_len divisible by window**layers and coarsest length >= 2.")
        obsolete = [name for name in vars(configs)
                    if name.startswith("pypots_")]
        if obsolete:
            raise ValueError("PyPOTS ablation options were removed; omit: " + ", ".join(sorted(obsolete)))
        if getattr(configs, "channel_independence", 1) != 1:
            raise ValueError("TimeMixerPPPyPOTS retains channel_independence=1 only.")
        if (getattr(configs, "down_sampling_method", None) or "avg") != "avg":
            raise ValueError("TimeMixerPPPyPOTS retains down_sampling_method=avg only.")
        if self.enc_in < 1 or self.pred_len < 1:
            raise ValueError("TimeMixerPPPyPOTS requires enc_in and pred_len > 0.")
        if configs.e_layers < 1 or configs.top_k < 1 or configs.num_kernels < 1:
            raise ValueError("TimeMixerPPPyPOTS requires e_layers, top_k and num_kernels > 0.")
        if configs.d_model < 2 or configs.d_model % 2 or configs.d_ff < 1:
            raise ValueError("TimeMixerPPPyPOTS requires a positive even d_model and positive d_ff.")
        if not 1 <= configs.n_heads <= self.seq_len // divisor:
            raise ValueError("TimeMixerPPPyPOTS n_heads must be between 1 and the coarsest length.")
        self.target_only = configs.c_out == 1 and self.enc_in > 1 and configs.features == "MS"
        if configs.c_out != self.enc_in and not self.target_only:
            raise ValueError("Use c_out=enc_in, or features=MS with c_out=1.")

        self.backbone = BackboneTimeMixerPP(
            n_steps=self.seq_len,
            n_features=self.enc_in,
            n_pred_steps=self.pred_len,
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            d_ffn=configs.d_ff,
            n_heads=configs.n_heads,
            dropout=configs.dropout,
            top_k=configs.top_k,
            n_kernels=configs.num_kernels,
            downsampling_layers=n_down,
            downsampling_window=window,
            use_norm=bool(getattr(configs, "use_norm", 1)),
            freq=getattr(configs, "freq", "h"),
        )
        get_logger(__name__).info(
            "PyPOTS TimeMixerPP: downsampling=avg, CI=1, periods=coarsest, no padding mask, "
            "mean scale ensemble, within-period season / across-period trend; "
            "no calendar inputs; scales=%s, num_kernels=%d, use_norm=%s",
            [self.seq_len // window**i for i in range(n_down + 1)],
            configs.num_kernels, self.backbone.use_norm,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"Expected x_enc [B,{self.seq_len},{self.enc_in}].")
        # This follows the public PyPOTS forecast wrapper, which calls forecast(X, None).
        output = self.backbone.forecast(x_enc)
        return output[..., -1:] if self.target_only else output
