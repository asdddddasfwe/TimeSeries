"""Deterministic time-kernel mass correction anchored at prediction length96."""
import math

def time_kernel_mass(patch_count):
    if not isinstance(patch_count,int) or patch_count<1:
        raise ValueError('patch_count must be a positive integer')
    return (patch_count+2*sum((patch_count-d)*math.exp(-d*d/8) for d in range(1,patch_count)))/(patch_count*patch_count)

def mmd_weight(horizon):
    if not isinstance(horizon,int) or horizon<4 or horizon%4:
        raise ValueError('horizon must be a positive multiple of4')
    return time_kernel_mass(24)/time_kernel_mass(horizon//4)
