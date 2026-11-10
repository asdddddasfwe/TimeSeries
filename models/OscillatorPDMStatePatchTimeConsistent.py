"""MAFirst with native-history and four-point-future time intervals aligned.

The original historical scan is preserved. DriveProjection outputs retain the
units of an integrated contribution over one native history interval. Future
point drives are converted to one-point units, then composed exactly across a
four-point block. No correction head, new parameter, or state buffer is added.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import torch
from models.OscillatorPDMStatePatchMAFirst import Model as Original
from layers.OscillatorTimeMix import complex_oscillator_scan

INITIAL_SHA256 = "6a572882d3161da56009437cad7a6251682f2688492e68dc4bd05d3cfc2c8086"


def _complex(pair):
    return torch.view_as_complex(pair.contiguous())


def native_interval_sum(unit_transition, stride):
    """Sum I+A+...+A**(stride-1); no cancellation-prone (1-A) division."""
    if stride not in (1, 2, 4, 8):
        raise ValueError("Expected one of the four original history strides")
    a = _complex(unit_transition)
    total = torch.ones_like(a)
    power = torch.ones_like(a)
    for _ in range(1, stride):
        power = power * a
        total = total + power
    return total


def compose_future_patch(unit_transition, native_point_drives, history_stride):
    """[N,J,2], [N,H,J,2] -> A4 [N,J,2], B4 [N,H/4,J,2].

    If a native-history interval has constant forcing b1 per original point,
    its integrated contribution is b_native = sum(A1**j, j=0..stride-1)*b1.
    Keep that existing native unit and invert only this bounded polynomial.
    Historical phase lies in (0, pi), so the valid model domain avoids the
    roots at a complete 2*pi native rotation. Final block state is at its
    right edge; the unchanged MLP reads the four points of that block.
    """
    if native_point_drives.ndim != 4 or native_point_drives.shape[1] % 4:
        raise ValueError("Expected four-point-divisible future drive [N,H,J,2]")
    a = _complex(unit_transition)
    b = _complex(native_point_drives)
    b = b / native_interval_sum(unit_transition, history_stride)[:, None, :]
    b = b.reshape(b.shape[0], -1, 4, b.shape[-1])
    a2 = a * a
    a3 = a2 * a
    weights = torch.stack((a3, a2, a, torch.ones_like(a)), dim=1)
    integrated = (b * weights[:, None, :, :]).sum(2)
    return torch.view_as_real(a2 * a2), torch.view_as_real(integrated)


def time_consistent_forecast(encoder, x, history_gain, drive_gain, history, trend):
    batch, _, channels = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    n, horizon, _ = future.shape
    decoder = encoder.decoder
    # This inherited method preserves A_history, its historical scan, and h0.
    # It returns the transition converted to one ORIGINAL sampling interval.
    a1, initial, _, _ = decoder.dynamics(history, trend)
    point_drives = decoder.future_drive(future, history).reshape(
        n, horizon, decoder.modes, 2)
    a4, block_drives = compose_future_patch(a1, point_drives, decoder.stride)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    block_drives = block_drives * drive_gain.repeat(batch)[:, None, None, None]
    states = complex_oscillator_scan(
        a4[:, None].expand(-1, horizon // 4, -1, -1), block_drives, initial)
    outputs = decoder.readout(states.flatten(2).to(decoder.readout.weight.dtype))
    if outputs.shape != (n, horizon // 4, 4):
        raise RuntimeError("Expected one four-value block per right-edge state")
    return outputs.reshape(batch, channels, horizon).transpose(1, 2)


class Model(Original):
    """Only future transition and drive discretization change."""

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected history [B,seq_len,enc_in]")
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent = [time_consistent_forecast(e, x, alpha, beta, s, t)
                     for e, x, s, t in zip(self.seasonals, scales, seasons, trends)]
        feature = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, t in zip(self.feature_trend_heads, trends)]
        return torch.stack(recurrent + feature, -1)


def load_original_scratch(root, logdir):
    """Use the audited control INITIAL state, never its trained best state."""
    folder = Path(root) / "tmp/mafirst_memory_factorial_20261017_full_h720_control_seed2021"
    checkpoint = folder / "initial.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != INITIAL_SHA256:
        raise ValueError("Original scratch initialization SHA mismatch")
    cfg = json.loads((folder / "model_config.json").read_text())
    cfg.update(model="OscillatorPDMStatePatchTimeConsistent",
               log_dir=str(logdir), tensorboard=False,
               factor_slow_memory=False, factor_readout_no_decay=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20211014)
        model = Model(SimpleNamespace(**cfg))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    assert len(state) == 236 and len(list(model.parameters())) == 204
    assert sum(p.numel() for p in model.parameters()) == 320450
    assert all(torch.equal(state[k], model.state_dict()[k]) for k in state)
    return model, cfg
