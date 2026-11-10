"""Fixed blend of a frozen neural forecast and a train-fitted temporal map."""
import torch
from torch import nn


class LinearForecastBlend(nn.Module):
    """One output per variable, no cross-variable observations or future inputs.

    Coefficients [C,T+1,H] are ridge temporal weights plus normalized bias.
    They must be fitted on training data outside this inference wrapper.
    The neural model keeps its own multi-scale fusion unchanged.
    """
    def __init__(self, neural, coefficients, neural_weight=0.5):
        super().__init__()
        weights=torch.as_tensor(coefficients,dtype=torch.float32)
        if weights.ndim!=3 or weights.shape[1]<2 or not torch.isfinite(weights).all():
            raise ValueError("Expected finite [C,T+1,H] coefficients")
        if not 0.<=neural_weight<=1.:
            raise ValueError("Blend weight must be in [0,1]")
        self.neural=neural
        self.register_buffer('linear_coefficients',weights.clone())
        self.register_buffer('neural_weight',torch.tensor(float(neural_weight)))

    def linear_forecast(self,x):
        weights=self.linear_coefficients.to(dtype=x.dtype)
        if x.ndim!=3 or x.shape[1]!=weights.shape[1]-1 or x.shape[2]!=weights.shape[0]:
            raise ValueError("Input must match fitted history length and variables")
        mean=x.mean(1,keepdim=True)
        std=torch.sqrt(x.var(1,keepdim=True,unbiased=False)+1e-5)
        z=torch.cat((x-mean,std),dim=1)
        output=mean+torch.einsum('btc,cth->bhc',z,weights)
        if getattr(self.neural,'features',None)=='MS' and self.neural.c_out==1:
            output=output[...,-1:]
        return output

    def forward(self,x_enc,*args,**kwargs):
        neural=self.neural(x_enc,*args,**kwargs)
        linear=self.linear_forecast(x_enc)
        if neural.shape!=linear.shape:
            raise ValueError("Neural and linear forecast shapes must match")
        return self.neural_weight*neural+(1-self.neural_weight)*linear
