"""Framework inference adapter for independently fitted neural/linear forecasts."""
import torch
from layers.LinearForecastBlend import LinearForecastBlend
from models.OscillatorTimeMix import Model as NeuralModel


class Model(LinearForecastBlend):
    """Load an assembled checkpoint; use the two-stage experiment to fit components.

    Ordinary end-to-end run.py training is intentionally unsupported: it would
    change the separately fitted estimator and invalidate its validation result.
    The checkpoint format equals LinearForecastBlend.state_dict().
    """
    def __init__(self, configs):
        if getattr(configs, 'is_training', 0):
            raise ValueError('OscillatorLinearBlend uses separately fitted components; load an assembled checkpoint with is_training=0')
        neural=NeuralModel(configs)
        coefficients=torch.zeros(configs.enc_in,configs.seq_len+1,configs.pred_len)
        super().__init__(neural,coefficients,neural_weight=.5)
        self._components_loaded=False
        for parameter in self.neural.parameters(): parameter.requires_grad_(False)
        self.eval()

    def train(self,mode=True):
        if mode:
            raise ValueError('This fixed blend is inference-only; train its components separately')
        return super().train(False)

    def load_state_dict(self,state_dict,strict=True,assign=False):
        self._components_loaded=False
        if not strict:
            raise ValueError('Both fitted components require strict checkpoint loading')
        result=super().load_state_dict(state_dict,strict=True,assign=assign)
        if not torch.isfinite(self.linear_coefficients).all() or not 0<=float(self.neural_weight)<=1:
            raise ValueError('Invalid fitted coefficients or blend weight')
        self._components_loaded=True
        return result

    def forward(self,x_enc,*args,**kwargs):
        if not self._components_loaded:
            raise RuntimeError('Load the assembled fitted checkpoint before forecasting')
        return super().forward(x_enc,*args,**kwargs)
