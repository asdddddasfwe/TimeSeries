"""Only future-gate context changes S->S+T; no new parameters or bypass."""
import torch
from models.OscillatorPDMStatePatchDriveDiagnostic import Model as Original,phase_drives,driven_substates
from models.OscillatorPDMStatePatchReadoutDiagnostic import read_phase

def gate_context(season,trend,policy):
    if policy not in ("season","sum") or season.shape!=trend.shape:
        raise ValueError("Require matched features and season/sum context policy")
    return season if policy=="season" else season+trend

def forecast(encoder, x, history_gain, drive_gain, history, trend, policy, context_policy):
    batch, _, channels = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    n, horizon, width = future.shape
    features = future.reshape(n, horizon // 4, 4, width)
    decoder = encoder.decoder
    transition, initial, radius, frequency = decoder.dynamics(history, trend)
    context = gate_context(history, trend, context_policy)
    projected = phase_drives(decoder, features, context, policy)
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
    with torch.no_grad():
        expected = history if context_policy == 'season' else history + trend
        delta = (context.detach()-history.detach()).reshape(batch,channels,history.shape[1],width).double()
        diag.update(gate_context_policy=context_policy, gate_context_matches_expected_max_abs=(context.detach()-expected.detach()).abs().max(), gate_context_delta_mse_by_variable=delta.square().mean((0,2,3)), gate_context_is_season_tensor=context is history, no_new_state_or_RNG=True)
    return outputs.reshape(batch, channels, horizon).transpose(1,2), diag


class Model(Original):
    def __init__(self,configs):
        policy=getattr(configs,"gate_context_policy",None)
        if policy not in ("season","sum") or getattr(configs,"state_drive_policy",None)!="mean" or getattr(configs,"state_readout_policy",None)!="substate":
            raise ValueError("Require season/sum context, mean drive and substate readout")
        super().__init__(configs)
        self.gate_context_policy=policy

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("Expected history [B,seq_len,enc_in]")
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        recurrent, diagnostics = [], []
        for encoder, x, season, trend in zip(self.seasonals, scales, seasons, trends):
            values, diag = forecast(encoder, x, alpha, beta, season, trend, self.state_drive_policy, self.gate_context_policy)
            recurrent.append(values)
            diagnostics.append(diag)
        feature = [head(t).reshape(batch, channels, self.pred_len).transpose(1,2)
                   for head,t in zip(self.feature_trend_heads,trends)]
        self.last_state_readout_diagnostics = diagnostics
        return torch.stack(recurrent+feature,-1)
