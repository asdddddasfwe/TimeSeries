"""Independent NumPy indexing implementation; saved arrays only, no models."""
import numpy as np


def numpy_increment_parts(prediction, truth, bandwidth, length=4):
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError('Matching B,H,C arrays required')
    if length < 1 or prediction.shape[1] <= length or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Invalid patch length or bandwidth')
    total_q = total_k = count = 0
    for start in range(0, len(prediction), 64):
        p = np.asarray(prediction[start:start+64], dtype=np.float64)
        y = np.asarray(truth[start:start+64], dtype=np.float64)
        if not (np.isfinite(p).all() and np.isfinite(y).all()):
            raise ValueError('Nonfinite prediction or truth')
        error = p-y
        n = p.shape[1]-length
        q = np.zeros((len(p), n, p.shape[2]), dtype=np.float64)
        for offset in range(length):
            q += (error[:, offset+1:offset+1+n]-error[:, offset:offset+n])**2
        q /= 2*length*bandwidth**2
        total_q += q.sum()
        total_k += (-np.expm1(-q)).sum()
        count += q.size
    if count == 0:
        raise ValueError('Empty predictions')
    return dict(increment_quadratic=float(total_q/count), increment_kernel=float(total_k/count))
