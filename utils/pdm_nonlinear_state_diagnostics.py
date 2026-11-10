"""Same-forward solver diagnostics; no extra encoder/model predictions or fitting."""
from unittest.mock import patch
import torch
from layers.NonlinearOscillatorScan import parallel_nonlinear_scan, sequential_nonlinear_scan, recurrence_residual
from utils.pdm_decoder_study_diagnostics import setup_hooks as base_hooks, trace as base_trace
from utils.pdm_decoder_study_diagnostics import SelectedValidationCapture as BaseCapture


class SolverCapture:
    """Observe the fixed solver; sequential reference only on a bounded prefix.

    Reference comparisons reuse this forward's A/u/z0/c. They never replace the
    prediction or change the objective, gradients, RNG, or iteration count.
    """
    def __init__(self, model, reference_windows=0):
        self.model = model
        self.reference_series = reference_windows * model.enc_in
        self.rows = {}
        self.calls = 0
        self.active = hasattr(model, 'nonlinear_iterations')

    def __enter__(self):
        if self.active:
            self.patcher = patch('models.OscillatorPDMNonlinearState.parallel_nonlinear_scan', self.scan)
            self.patcher.start()
        return self

    def remove(self):
        if self.active:
            self.patcher.stop()
            self.active = False

    def __exit__(self, *args):
        self.remove()

    def scan(self, transition, candidate, initial, history, correction, iterations):
        states = parallel_nonlinear_scan(transition, candidate, initial, history, correction, iterations)
        scale = str(self.calls % len(self.model.seasonals))
        self.calls += 1
        row = self.rows.setdefault(scale, dict(elements=0, state_square=0., residual_square=0.,
            residual_max=0., correction_square=0., candidate_square=0., reference_series=0,
            reference_elements=0, reference_error_square=0., reference_error_max=0.))
        with torch.no_grad():
            a, u, z0, c, z = [v.detach() for v in (transition, candidate, initial, history, states)]
            residual = recurrence_residual(z, a, u, z0, c, correction)
            previous = torch.cat((z0[:, None], z[:, :-1]), 1)
            delta = correction(previous, u, c)
            for value in (z, residual, delta):
                if not torch.isfinite(value).all():
                    raise RuntimeError('Nonfinite nonlinear solver state/residual/correction')
            row['elements'] += z.numel()
            for key, value in [('state', z), ('residual', residual), ('correction', delta), ('candidate', u)]:
                row[key+'_square'] += float(value.double().square().sum())
            row['residual_max'] = max(row['residual_max'], float(residual.abs().max()))
            n = min(len(z), max(0, self.reference_series-row['reference_series']))
            if n:
                reference = sequential_nonlinear_scan(a[:n], u[:n], z0[:n], c[:n], correction)
                error = z[:n]-reference
                if not torch.isfinite(error).all():
                    raise RuntimeError('Nonfinite nonlinear sequential-reference difference')
                row['reference_series'] += n
                row['reference_elements'] += error.numel()
                row['reference_error_square'] += float(error.double().square().sum())
                row['reference_error_max'] = max(row['reference_error_max'], float(error.abs().max()))
        return states

    def report(self):
        result = {}
        for scale, row in self.rows.items():
            v = dict(row)
            for key in ('state', 'residual', 'correction', 'candidate'):
                v[key+'_rms'] = (row[key+'_square']/row['elements'])**.5
            v['residual_relative_l2'] = (row['residual_square']/max(row['state_square'],1e-30))**.5
            v['reference_error_rms'] = (row['reference_error_square']/max(row['reference_elements'],1))**.5
            result[scale] = v
        return result


def setup_hooks(model):
    captured, handles = base_hooks(model)
    observer = SolverCapture(model).__enter__()
    captured['solver_diagnostics'] = observer
    handles.append(observer)
    return captured, handles


def trace(model, captured, step, loss, wd_by_id):
    row = base_trace(model, {k:v for k,v in captured.items() if torch.is_tensor(v)}, step, loss, wd_by_id)
    row['solver_diagnostics'] = captured['solver_diagnostics'].report()
    row['nonlinear_parameters'] = {name:dict(norm=float(p.detach().double().norm()),
        gradient_norm=float(p.grad.double().norm()), weight_decay=wd_by_id[id(p)])
        for name,p in model.named_parameters() if '.state_correction.' in name}
    return row


class SelectedValidationCapture(BaseCapture):
    def __enter__(self):
        super().__enter__()
        self.solver_config = dict(nonlinear_solver=getattr(self.model,'nonlinear_solver',None),
            nonlinear_iterations=getattr(self.model,'nonlinear_iterations',0),
            nonlinear_hidden=getattr(self.model,'nonlinear_hidden',0))
        self.solver = SolverCapture(self.model, reference_windows=16).__enter__()
        return self

    def __exit__(self, *args):
        self.solver.remove()
        return super().__exit__(*args)

    def report(self, prediction, truth):
        result = super().report(prediction, truth)
        result.update(self.solver_config)
        result['solver_diagnostics'] = self.solver.report()
        nonlinear = self.solver_config['nonlinear_solver'] is not None
        result['sequential_reference_windows'] = 16 if nonlinear else 0
        result['seasonal_decoder'] = ('nonlinear_history_corrected_oscillator' if nonlinear
                                      else 'trend_conditioned_oscillator')
        return result
