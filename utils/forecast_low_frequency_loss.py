"""Differentiable MA25 loss on total forecasts in dataset-scaled units."""
import torch
from functools import lru_cache


@lru_cache(maxsize=16)
def _average_matrix(length,kernel,device,dtype):
    offsets=torch.arange(kernel)-kernel//2
    indices=(torch.arange(length)[:,None]+offsets).clamp(0,length-1)
    matrix=torch.nn.functional.one_hot(indices,num_classes=length).double().sum(1)/kernel
    return matrix.to(device=device,dtype=dtype)


def moving_average_forecast(x, kernel=25):
    if x.ndim!=3 or kernel<1 or kernel%2!=1:
        raise ValueError('Expected [B,H,C] and positive odd kernel')
    # Fixed matrix implements replicated endpoints; backward is matrix transpose
    # rather than scatter into repeated endpoints. No trainable weights or RNG.
    matrix=_average_matrix(x.shape[1],kernel,str(x.device),x.dtype)
    return torch.matmul(matrix,x)


def low_frequency_mse(prediction,target):
    return (moving_average_forecast(prediction)-moving_average_forecast(target.detach())).square().mean()
