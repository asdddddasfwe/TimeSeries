"""Same MA9 backbone; isolate which existing state each four-point readout row reads."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.OscillatorPDMStatePatchMAKernel9 import Model as SameKernel9
from layers.OscillatorTimeMix import complex_oscillator_scan


def constant_drive_substates(transition, radius, frequency, block_drive, initial, endpoints):
    """Return four states per block; final state reuses the original tensor exactly.

    transition [N,J,2] and block_drive/endpoints [N,K,J,2] use the original
    block clock. Four constant-force substeps compose the SAME block map:
    a=A_patch**(1/4), b=B_patch/(1+a+a**2+a**3).
    This does not change the original-sample interpretation of the block clock.
    """
    a = torch.polar(radius.pow(0.25), frequency * 0.25)
    total = 1 + a + a.square() + a.square() * a
    drive = torch.view_as_complex(block_drive.contiguous()) / total[:, None]
    previous = torch.cat((initial[:, None], endpoints[:, :-1]), dim=1)
    state = torch.view_as_complex(previous.contiguous())
    values = []
    for _ in range(3):
        state = a[:, None] * state + drive
        values.append(torch.view_as_real(state))
    reconstructed = torch.view_as_real(a[:, None] * state + drive)
    # The last coordinate is the original scan output, not a recomputed copy.
    values.append(endpoints)
    return values, reconstructed


def read_phase(readout, states, phase, recompute):
    """Use an existing output row; readout has Linear/GELU/Linear, no dropout."""
    def function(x):
        return readout(x)[..., phase]
    if recompute and torch.is_grad_enabled():
        return checkpoint(function, states, use_reentrant=False, preserve_rng_state=False)
    return function(states)


def forecast(encoder, x, history_gain, drive_gain, history, trend, policy):
    batch, _, channels = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    n, horizon, width = future.shape
    pooled = future.reshape(n, horizon // 4, 4, width).mean(2)
    decoder = encoder.decoder
    transition, initial, radius, frequency = decoder.dynamics(history, trend)
    drive = decoder.future_drive(pooled, history).reshape(n, horizon // 4, decoder.modes, 2)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    drive = drive * drive_gain.repeat(batch)[:, None, None, None]
    endpoints = complex_oscillator_scan(
        transition[:, None].expand(-1, horizon // 4, -1, -1), drive, initial)
    if policy == "endpoint":
        outputs = decoder.readout(endpoints.flatten(2).to(decoder.readout.weight.dtype))
        substate_reconstruction = endpoints
    else:
        substates, substate_reconstruction = constant_drive_substates(
            transition, radius, frequency, drive, initial, endpoints)
        outputs = torch.stack([read_phase(
            decoder.readout, state.flatten(2).to(decoder.readout.weight.dtype), phase,
            recompute=True) for phase, state in enumerate(substates)], dim=-1)
    with torch.no_grad():
        p = outputs.detach().reshape(batch, channels, horizon // 4, 4).double()
        end = endpoints.detach().reshape(batch, channels, horizon // 4, decoder.modes, 2).double()
        error = (substate_reconstruction.detach() - endpoints.detach()).abs()
        diag = {
            "block_output_contrast_energy": (p - p.mean(-1, keepdim=True)).square().mean((0, 2, 3)),
            "block_state_square_energy": end.square().mean((0, 2, 3, 4)),
            "substate_reconstruction_max_abs": error.max(),
            "substate_reconstruction_relative": error.max() / endpoints.detach().abs().max().clamp_min(1),
            "selected_endpoint_bitwise_exact": True,
        }
    return outputs.reshape(batch, channels, horizon).transpose(1, 2), diag


class Model(SameKernel9):
    """No new parameter/state/buffer; keep MA9, clock, pooled drive and eight branches."""
    def __init__(self, configs):
        policy = getattr(configs, "state_readout_policy", None)
        if policy not in ("endpoint", "substate") or getattr(configs, "ma_kernel_policy", None) != "point9":
            raise ValueError("Require MA point9 and endpoint/substate readout policy")
        super().__init__(configs)
        self.state_readout_policy = policy

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected history [B,seq_len,enc_in]")
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent, diagnostics = [], []
        for encoder, x, season, trend in zip(self.seasonals, scales, seasons, trends):
            values, diag = forecast(encoder, x, alpha, beta, season, trend, self.state_readout_policy)
            recurrent.append(values)
            diagnostics.append(diag)
        feature = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                   for head, t in zip(self.feature_trend_heads, trends)]
        self.last_state_readout_diagnostics = diagnostics
        return torch.stack(recurrent + feature, -1)
