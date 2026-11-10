"""Nonlinear residual recurrence with sequential and parallel-iteration solvers.

z[h] = A[h] z[h-1] + candidate[h] + correction(z[h-1], candidate[h], history)
A is represented by complex real/imag pairs. No future observations are inputs.
The parallel solver unrolls a fixed number of refinements with full autograd.
It is an approximation unless the recurrence residual is sufficiently small.
"""
import torch
from torch import nn
from layers.OscillatorMixer import complex_product
from layers.OscillatorTimeMix import complex_oscillator_scan


class HistoryStateCorrection(nn.Module):
    """[N,H,J,2] state/candidate plus repeated [N,C] historical summary.

    Zero output initialization preserves the affine baseline function. Only the
    output layer gets a nonzero gradient initially; inner layers learn once its
    weights move. There is no normalization, dropout, or detached state here.
    """
    def __init__(self, modes, context_width, hidden_width=32):
        super().__init__()
        if min(modes, context_width, hidden_width) < 1:
            raise ValueError('Correction dimensions must be positive')
        self.modes, self.context_width = modes, context_width
        self.input = nn.Linear(4 * modes + context_width, hidden_width)
        self.output = nn.Linear(hidden_width, 2 * modes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, previous, candidate, history):
        if (previous.ndim != 4 or previous.shape != candidate.shape or
                previous.shape[-2:] != (self.modes, 2) or
                history.shape != (previous.shape[0], self.context_width)):
            raise ValueError('Expected matching [N,H,J,2] and history[N,C]')
        context = history[:, None].expand(-1, previous.shape[1], -1)
        joined = torch.cat((previous.flatten(2), candidate.flatten(2), context), -1)
        return self.output(torch.tanh(self.input(joined))).reshape_as(previous)


def _validate(transition, candidate, initial, history):
    if (transition.ndim != 4 or transition.shape != candidate.shape or
            transition.shape[-1] != 2 or transition.shape[1] < 1 or
            initial.shape != (candidate.shape[0], candidate.shape[2], 2) or
            history.ndim != 2 or history.shape[0] != candidate.shape[0]):
        raise ValueError('Expected A/candidate[N,H>=1,J,2], initial[N,J,2], history[N,C]')
    tensors = (transition, candidate, initial, history)
    if any(t.device != candidate.device or t.dtype != candidate.dtype for t in tensors):
        raise ValueError('All recurrence inputs must share device and dtype')
    if candidate.dtype not in (torch.float32, torch.float64):
        raise ValueError('Nonlinear scan currently supports float32/float64 only')


def sequential_nonlinear_scan(transition, candidate, initial, history, correction):
    """Direct time-step reference, with gradients through every state."""
    _validate(transition, candidate, initial, history)
    current, values = initial, []
    for h in range(candidate.shape[1]):
        delta = correction(current[:, None], candidate[:, h:h+1], history)[:, 0]
        current = complex_product(transition[:, h], current) + candidate[:, h] + delta
        values.append(current)
    return torch.stack(values, 1)


def parallel_nonlinear_scan(transition, candidate, initial, history, correction, iterations=4):
    """Batch nonlinear corrections over H, then affine-scan the current round.

    The affine warm start costs one scan. Each refinement adds one batched MLP
    and one scan. Rounds are sequential, positions within each round parallel.
    There is no detach, tolerance-based branching, or silent serial fallback.
    For this finite causal recurrence, H rounds recover the serial solution in
    exact arithmetic; fewer rounds need not do so. FP rounding still differs.
    """
    _validate(transition, candidate, initial, history)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError('iterations must be a nonnegative integer')
    states = complex_oscillator_scan(transition, candidate, initial)
    for _ in range(iterations):
        previous = torch.cat((initial[:, None], states[:, :-1]), 1)
        delta = correction(previous, candidate, history)
        states = complex_oscillator_scan(transition, candidate + delta, initial)
    return states


def recurrence_residual(states, transition, candidate, initial, history, correction):
    """Return differentiable [N,H,J,2] equation residual, not solver error bounds."""
    _validate(transition, candidate, initial, history)
    if states.shape != candidate.shape:
        raise ValueError('States must have the same shape as candidate')
    previous = torch.cat((initial[:, None], states[:, :-1]), 1)
    expected = complex_product(transition, previous) + candidate
    return states - expected - correction(previous, candidate, history)
