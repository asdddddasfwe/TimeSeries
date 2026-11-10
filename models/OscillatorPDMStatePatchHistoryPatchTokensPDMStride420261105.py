"""Use compact historical patch tokens for the oscillator's history scan."""
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from layers.OscillatorTimeMix import complex_oscillator_scan
from models.OscillatorPDMStatePatchBestCombination20261104 import InheritedCombination
from models.OscillatorPDMStatePatchBestSpectralContinuousRNG20261103 import (
    Model as SpectralModel,
    spectral_dynamics,
)
from models.OscillatorPDMStatePatchBestTimeVaryingRotation import position_transitions
from models.OscillatorDualEncoderTimeMixer import sinusoidal_positions


def encode_patch_views(encoder, history):
    patches, padded_length = encoder.patchify(history)
    batch, count, size, width = patches.shape
    embedded = encoder.patch_embedding(patches.reshape(batch, count, size * width))
    positions = sinusoidal_positions(count, width).to(embedded)
    with sdpa_kernel(SDPBackend.MATH):
        tokens = encoder.attention(embedded + positions)
    decoded = encoder.patch_readout(tokens).reshape(batch, count, size, width)
    restored = encoder.overlap_average(decoded, padded_length, history.shape[1])
    return tokens, encoder.output_norm(history + restored)


class Model(InheritedCombination):
    def __init__(self, configs):
        super().__init__(configs)
        if (self.seq_len, self.pred_len) != (96, 720) or not (
            self.spectral_enabled and self.timevary_enabled and not configs.combo_direct
        ):
            raise ValueError("Expected the original spectral + time-varying H720 decoder")
        for encoder, patch_encoder in zip(self.seasonals, self.season_patch_encoders):
            if (patch_encoder.patch_length, patch_encoder.stride) != (4, 2):
                raise ValueError("Expected the original four-point sliding encoder")
            patch_encoder.stride = 4
            if encoder.decoder.length % 4:
                raise ValueError("History must contain whole nonoverlapping patches")
            encoder.decoder.length //= 4

    def encode_with_patch_tokens(self, raw_scales):
        normalized = [self.normalize(raw_scales[0], "norm")]
        normalized += [norm(x, "norm") for norm, x in zip(self.coarse_normalizers, raw_scales[1:])]
        batch, _, channels = normalized[0].shape
        seasons, trends, tokens = [], [], []
        for encoder, patch_encoder, scale in zip(self.seasonals, self.season_patch_encoders, normalized):
            observed = scale.transpose(1, 2).reshape(batch * channels, encoder.length, 1)
            seasonal, trend = self.initial_decomposition(observed)
            seasonal = F.linear(seasonal, encoder.value_embedding.weight, None)
            patch_tokens, restored = encode_patch_views(patch_encoder, seasonal)
            tokens.append(patch_tokens)
            seasons.append(restored)
            trends.append(encoder.value_embedding(trend))
        before_pdm = seasons
        for pdm in self.pdm_blocks:
            seasons, trends = pdm(seasons, trends)

        # PDM keeps its original point-level projections. Return its seasonal
        # residual to the matching nonoverlapping tokens before history dynamics.
        mixed_tokens = [
            patch_tokens + (after - before).reshape(*patch_tokens.shape[:2], 4, patch_tokens.shape[-1]).mean(2)
            for patch_tokens, before, after in zip(tokens, before_pdm, seasons)
        ]
        return normalized, seasons, trends, mixed_tokens

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected [batch, 96, channels]")
        scales, seasons, trends, patch_tokens = self.encode_with_patch_tokens(self.downsample(x_enc))
        spectrum = self.spectral_encoder(SpectralModel.spectrum_features(self, scales[0]))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent, diagnostics = [], []
        for i, (encoder, history, trend, tokens) in enumerate(
            zip(self.seasonals, seasons, trends, patch_tokens)
        ):
            decoder = encoder.decoder
            future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
            future = future + encoder.future_positions.to(future)
            n, horizon, width = future.shape
            steps = horizon // 4
            future = future.reshape(n, steps, 4, width).mean(2)

            # Dynamics sees PDM-mixed patch tokens. The original decoder's fixed-length
            # projection, future gate and angle head use the restored view.
            patch_trend = trend.reshape(n, tokens.shape[1], 4, width).mean(2)
            shift = self.spectral_conditions[i](spectrum)
            transition, initial, _, angle = spectral_dynamics(decoder, tokens, patch_trend, shift)
            sequence, rotation_diag = position_transitions(
                decoder, history, transition, angle, self.position_frequency_heads[i],
                self.position_frequency_basis,
            )
            drive = decoder.future_drive(future, history).reshape(n, steps, decoder.modes, 2)
            initial = initial * alpha.repeat(batch)[:, None, None]
            drive = drive * beta.repeat(batch)[:, None, None, None]
            states = complex_oscillator_scan(sequence, drive, initial)
            outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
            recurrent.append(outputs.reshape(batch, channels, horizon).transpose(1, 2))
            diagnostics.append(dict(
                context_error=0., trend_delta_energy=0., is_season_tensor=True,
                history_length=tokens.shape[1], decoder_context_length=history.shape[1],
                spectral_before_history_scan=True, time_varying_rotation=True,
                frequency_delta_max_abs=rotation_diag["frequency_delta_max_abs"],
                spectral_shift_max_abs=shift.detach().abs().max(),
            ))
        feature = [head(trend).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, trend in zip(self.feature_trend_heads, trends)]
        self.last_gate_context_diagnostics = diagnostics
        return torch.stack(recurrent + feature, -1)
