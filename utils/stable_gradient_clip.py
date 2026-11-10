"""FP64 norm accumulation prevents overflow while clipping finite FP32 grads."""
import math
import torch


def clip_finite_gradients(parameters, max_norm=1.0):
    if not math.isfinite(max_norm) or max_norm<=0:raise ValueError('Positive finite max_norm required')
    parameters=list(parameters)
    if not parameters:raise ValueError('Expected parameters')
    squares=[]
    for parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Missing or nonfinite gradient; cannot repair by clipping')
        squares.append(parameter.grad.detach().double().square().sum())
    norm=torch.stack(squares).sum().sqrt()
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite FP64 gradient norm')
    factor=min(1.,max_norm/(float(norm)+1e-12))
    with torch.no_grad():
        for parameter in parameters:parameter.grad.mul_(factor)
    return float(norm),factor
