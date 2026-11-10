"""Trend-conditioned oscillator with history-aware nonlinear future updates.

The encoder, normalization, historical scan, candidates, gains, and trend heads
are inherited unchanged. Only future seasonal state updates get a new residual.
History context is final seasonal-feature mean/last, not attention over all tokens.
"""
import torch
from models.OscillatorPDMTrendCondition import Model as TrendCondition
from layers.NonlinearOscillatorScan import (
    HistoryStateCorrection, parallel_nonlinear_scan, sequential_nonlinear_scan,
)


def prepare_recurrence(encoder, x, history_gain, drive_gain, history, trend):
    """Generate all candidate states once using the original gated drive map.

    Returns A[N,H,J,2], candidate[N,H,J,2], initial[N,J,2], context[N,2D].
    N=B*C; channels remain independent. Every refinement reuses these tensors,
    so dropout in candidate generation is sampled only once per model forward.
    """
    batch, _, channels = x.shape
    future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)
    future = future + encoder.future_positions.to(future)
    decoder = encoder.decoder
    transition, initial, _, _ = decoder.dynamics(history, trend)
    candidate = decoder.future_drive(future, history)
    candidate = candidate.reshape(batch * channels, encoder.horizon, decoder.modes, 2)
    initial = initial * history_gain.repeat(batch)[:, None, None]
    candidate = candidate * drive_gain.repeat(batch)[:, None, None, None]
    context = torch.cat((history.mean(1), history[:, -1]), -1)
    transition = transition[:, None].expand(-1, encoder.horizon, -1, -1)
    return transition, candidate, initial, context


class Model(TrendCondition):
    """Config: nonlinear_solver='parallel'/'sequential', iterations=4, hidden=32.

    Full configuration names are nonlinear_solver, nonlinear_iterations,
    nonlinear_hidden. Fixed iteration count defines the train/eval computation;
    changing it for evaluation may change predictions and is not automatic.
    """
    def __init__(self, configs):
        solver = getattr(configs, 'nonlinear_solver', 'parallel')
        iterations = getattr(configs, 'nonlinear_iterations', 4)
        hidden = getattr(configs, 'nonlinear_hidden', 32)
        if solver not in ('parallel', 'sequential'):
            raise ValueError('Unknown nonlinear solver')
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError('nonlinear_iterations must be a positive integer')
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError('nonlinear_hidden must be a positive integer')
        super().__init__(configs)
        self.nonlinear_solver = solver
        self.nonlinear_iterations = iterations
        self.nonlinear_hidden = hidden
        # Preserve the baseline RNG stream, including subsequent training dropout.
        with torch.random.fork_rng(devices=[]):
            for encoder in self.seasonals:
                decoder = encoder.decoder
                decoder.state_correction = HistoryStateCorrection(
                    decoder.modes, 2 * decoder.drive_projection.in_features, hidden)

    def predict_components(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError('Expected history [B,seq_len,enc_in]')
        scales, (seasons, trends) = self.encode_multiscale(self.downsample(x_enc))
        alpha, beta = self.branch_gains()
        batch, _, channels = x_enc.shape
        seasonal_outputs = []
        for encoder, x, season, trend in zip(self.seasonals, scales, seasons, trends):
            inputs = prepare_recurrence(encoder, x, alpha, beta, season, trend)
            correction = encoder.decoder.state_correction
            if self.nonlinear_solver == 'parallel':
                states = parallel_nonlinear_scan(*inputs, correction, self.nonlinear_iterations)
            else:
                states = sequential_nonlinear_scan(*inputs, correction)
            readout = encoder.decoder.readout
            values = readout(states.flatten(2).to(readout.weight.dtype)).squeeze(-1)
            seasonal_outputs.append(values.reshape(batch, channels, encoder.horizon).transpose(1, 2))
        trend_outputs = [head(t).reshape(batch, channels, self.pred_len).transpose(1, 2)
                         for head, t in zip(self.feature_trend_heads, trends)]
        return torch.stack(seasonal_outputs + trend_outputs, -1)
