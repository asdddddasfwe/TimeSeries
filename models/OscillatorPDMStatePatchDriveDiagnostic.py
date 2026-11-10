"""MA9 drive-entry diagnostic: fixed block clock and matched block dropout."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.OscillatorPDMStatePatchReadoutDiagnostic import Model as ReadoutModel, read_phase
from layers.OscillatorTimeMix import complex_oscillator_scan


def phase_drives(decoder, features, history, policy):
    """[N,K,4,D] -> [N,K,4,J,2]; one dropout mask per block in both arms.

    Only the input to the existing nonlinear drive network differs: repeated
    mean vs four position features. Features are generated, not observed future.
    Operation shapes, parameters, random calls and recomputation match.
    """
    if policy not in ("mean", "phase"):
        raise ValueError("Require mean or phase drive policy")
    n, blocks, phases, width = features.shape
    assert phases == 4
    if policy == "mean":
        features = features.mean(2, keepdim=True).expand(-1, -1, 4, -1)
    mask = decoder.drive_dropout(torch.ones_like(features[:, :, 0]))

    def project(z, context, dropout_mask):
        normalized = decoder.drive_norm(z.reshape(n, blocks * 4, width))
        gated = F.gelu(normalized) * decoder.future_gate(normalized, context)
        gated = gated.reshape(n, blocks, 4, width) * dropout_mask[:, :, None]
        return decoder.drive_projection(gated).reshape(n, blocks, 4, decoder.modes, 2)

    if torch.is_grad_enabled():
        return checkpoint(project, features, history, mask, use_reentrant=False,
                          preserve_rng_state=False)
    return project(features, history, mask)


def driven_substates(transition, radius, frequency, projected, initial):
    """Four substeps compose one original A, never A**4.

    a=radius**.25*exp(i*frequency/4), S=1+a+a**2+a**3, b_q=g_q/S.
    Equal g_q recover the original constant-drive block map in real arithmetic.
    FP32 associativity is audited rather than called bitwise equality.
    """
    a = torch.polar(radius.pow(.25), frequency * .25)
    total = 1 + a + a.square() + a.square() * a
    point = torch.view_as_complex(projected.contiguous()) / total[:, None, None]
    effective = sum(a[:, None].pow(3-q) * point[:, :, q] for q in range(4))
    endpoints = complex_oscillator_scan(
        transition[:, None].expand(-1, projected.shape[1], -1, -1),
        torch.view_as_real(effective), initial)
    previous = torch.cat((initial[:, None], endpoints[:, :-1]), dim=1)
    state = torch.view_as_complex(previous.contiguous())
    substates = []
    for q in range(3):
        state = a[:, None] * state + point[:, :, q]
        substates.append(torch.view_as_real(state))
    reconstructed = torch.view_as_real(a[:, None] * state + point[:, :, 3])
    substates.append(endpoints)
    return substates, reconstructed, endpoints, torch.view_as_real(effective)


def forecast(encoder, x, history_gain, drive_gain, history, trend, policy):
    batch, _, channels = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    n, horizon, width = future.shape
    features = future.reshape(n, horizon // 4, 4, width)
    decoder = encoder.decoder
    transition, initial, radius, frequency = decoder.dynamics(history, trend)
    projected = phase_drives(decoder, features, history, policy)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    projected = projected * drive_gain.repeat(batch)[:, None, None, None, None]
    states, reconstructed, endpoints, effective = driven_substates(
        transition, radius, frequency, projected, initial)
    outputs = torch.stack([read_phase(decoder.readout,
        state.flatten(2).to(decoder.readout.weight.dtype), q, recompute=True)
        for q, state in enumerate(states)], dim=-1)
    with torch.no_grad():
        p = outputs.detach().reshape(batch, channels, horizon//4, 4).double()
        end = endpoints.detach().reshape(batch, channels, horizon//4, decoder.modes, 2).double()
        d = projected.detach().reshape(batch, channels, horizon//4, 4, decoder.modes, 2).double()
        fe = features.detach().reshape(batch, channels, horizon//4, 4, width).double()
        error = (reconstructed.detach()-endpoints.detach()).abs()
        diag = dict(block_output_contrast_energy=(p-p.mean(-1, keepdim=True)).square().mean((0,2,3)),
                    block_state_square_energy=end.square().mean((0,2,3,4)),
                    substate_reconstruction_max_abs=error.max(),
                    substate_reconstruction_relative=error.max()/endpoints.detach().abs().max().clamp_min(1),
                    selected_endpoint_bitwise_exact=states[-1] is endpoints,
                    projected_drive_phase_contrast_energy=(d-d.mean(3,keepdim=True)).square().mean((0,2,3,4,5)),
                    future_feature_phase_contrast_energy=(fe-fe.mean(3,keepdim=True)).square().mean((0,2,3,4)),
                    effective_block_drive_square_energy=effective.detach().double().square().mean(),
                    shared_block_dropout_mask=True, original_block_clock=True)
    return outputs.reshape(batch, channels, horizon).transpose(1,2), diag


class Model(ReadoutModel):
    """Existing weights, MA9, encoders, trends and eight branches only."""
    def __init__(self, configs):
        policy = getattr(configs, "state_drive_policy", None)
        if policy not in ("mean", "phase") or getattr(configs, "state_readout_policy", None) != "substate":
            raise ValueError("Require mean/phase drive and substate readout in both arms")
        super().__init__(configs)
        self.state_drive_policy = policy

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected history [B,seq_len,enc_in]")
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent, diagnostics = [], []
        for encoder, x, season, trend in zip(self.seasonals, scales, seasons, trends):
            values, diag = forecast(encoder, x, alpha, beta, season, trend, self.state_drive_policy)
            recurrent.append(values)
            diagnostics.append(diag)
        feature = [head(t).reshape(batch, channels, self.pred_len).transpose(1,2)
                   for head,t in zip(self.feature_trend_heads,trends)]
        self.last_state_readout_diagnostics = diagnostics
        return torch.stack(recurrent+feature,-1)
