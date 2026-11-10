"""Shared helpers extracted from retired experiment entry points."""
import numpy as np


def solve_hierarchical(gram,rhs,ridge,sharing):
    """Mean_c [MSE_c + ridge*||W_c[:-1]||^2 + sharing*||W_c-W_mean||^2]."""
    channels,width,_=gram.shape
    penalty=np.eye(width);penalty[-1,-1]=0.
    blocks=np.zeros((channels,width,channels,width))
    for c in range(channels):
        blocks[c,:,c,:]=gram[c]+ridge*penalty+sharing*np.eye(width)
        for other in range(channels):blocks[c,:,other,:]-=sharing/channels*np.eye(width)
    matrix=blocks.reshape(channels*width,channels*width)
    weights=np.linalg.solve(matrix,rhs.reshape(channels*width,-1)).reshape(rhs.shape)
    gradient=np.einsum('cij,cjh->cih',gram+ridge*penalty[None],weights)-rhs+sharing*(weights-weights.mean(0,keepdims=True))
    if np.max(np.abs(gradient))>1e-7:raise AssertionError('Normal-equation residual too large')
    return weights
