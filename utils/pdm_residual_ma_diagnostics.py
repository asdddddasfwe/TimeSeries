"""Same-forward decomposition metadata, retaining all original diagnostics."""
from utils.pdm_decoder_study_diagnostics import setup_hooks, trace
from utils.pdm_decoder_study_diagnostics import SelectedValidationCapture as BaseCapture


class SelectedValidationCapture(BaseCapture):
    def __enter__(self):
        self.decomposition_passes = getattr(self.model, 'residual_decomposition_passes', 1)
        return super().__enter__()

    def report(self, prediction, truth):
        result = super().report(prediction, truth)
        result['residual_decomposition_passes'] = self.decomposition_passes
        result['decomposition_definition'] = 'S0=X; Sn=Sn-1-MA(Sn-1); Tn=sum extracted MA; same kernel each pass'
        return result
