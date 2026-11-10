"""MAFirst with history-retrieving gated recurrence instead of oscillators.

Preserves input normalization, MAFirst/FFT/PDM, four independent scale
readouts, all feature-trend heads, channel gains and the eight-path sum.
Replaces oscillator decoders AND removes unused horizon temporal projections.
"""
import torch

from layers.HistoryGatedState import HistoryGatedStateDecoder
from models.OscillatorPDMStatePatchMAFirst import Model as MAFirst


class Model(MAFirst):
    def __init__(self, configs):
        if configs.pred_len < 16 or configs.pred_len % 4:
            raise ValueError("MAFirst requires horizon >=16 and divisible by four")
        super().__init__(configs)
        for encoder in self.seasonals:
            original = encoder.decoder
            width = encoder.value_embedding.out_features
            decoder = HistoryGatedStateDecoder(
                length=encoder.length, horizon=self.pred_len, width=width,
                state_size=2 * original.modes, stride=original.stride,
                readout=original.readout, dropout=configs.dropout,
            )
            encoder.decoder = decoder
            # Do not leave inactive learned parameters in the optimizer.
            del encoder.temporal_projection
            del encoder.future_positions
        self.register_buffer("history_gated_state_version", torch.tensor(1))

    def optimizer_parameter_groups(self, weight_decay):
        # Old oscillator-specific decay exemptions no longer apply.
        # The shared training helper further exempts norms and bias tensors.
        cell_biases = [
            bias for encoder in self.seasonals
            for bias in (encoder.decoder.cell.bias_ih, encoder.decoder.cell.bias_hh)
        ]
        protected = {id(p) for p in cell_biases}
        return [
            dict(params=[p for p in self.parameters() if id(p) not in protected],
                 weight_decay=weight_decay, group_name="regular"),
            # The old shared helper recognizes "bias", not GRU's "bias_ih/hh".
            dict(params=cell_biases, weight_decay=0.0, group_name="gru_bias_no_decay"),
        ]

    def load_reference_initialization(self, reference_state):
        """Copy retained scratch tensors exactly; new core remains freshly initialized.

        Accept only the original MAFirst architecture. Removed oscillator and
        horizon-projection tensors are reported, never silently adapted.
        This does not make the changed parameter count an equal-capacity ablation.
        """
        current = self.state_dict()
        common = current.keys() & reference_state.keys()
        removed = sorted(reference_state.keys() - current.keys())
        added = sorted(current.keys() - reference_state.keys())
        for name in removed:
            if not (name.startswith("seasonals.") and (
                    ".decoder." in name or ".temporal_projection." in name
                    or name.endswith(".future_positions"))):
                raise ValueError("Unexpected removed reference tensor: " + name)
        for name in added:
            if not (name == "history_gated_state_version"
                    or (name.startswith("seasonals.") and ".decoder." in name)):
                raise ValueError("Unexpected new model tensor: " + name)
        for name in common:
            if current[name].shape != reference_state[name].shape:
                raise ValueError("Reference tensor shape mismatch: " + name)
        current.update({name: reference_state[name] for name in common})
        self.load_state_dict(current, strict=True)
        return dict(copied=sorted(common), removed=removed, fresh=added)

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected observed history [B,seq_len,enc_in]")
        _, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent = []
        for encoder, season, trend in zip(self.seasonals, seasons, trends):
            output = encoder.decoder(
                season, trend, history_gain=alpha.repeat(batch),
                input_gain=beta.repeat(batch),
            )
            recurrent.append(output.reshape(batch, channels, self.pred_len).transpose(1, 2))
        feature = [
            head(trend).reshape(batch, channels, self.pred_len).transpose(1, 2)
            for head, trend in zip(self.feature_trend_heads, trends)
        ]
        return torch.stack(recurrent + feature, dim=-1)
