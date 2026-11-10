"""Independent NumPy row distributions and KL, no torch or training helpers."""
import numpy as np


def numpy_relation_parts(prediction, truth, bandwidth=1., length=4):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.ndim != 3 or p.shape != y.shape or not all(p.shape) or length < 1 or p.shape[1] % length or p.shape[1] // length < 2:
        raise ValueError('Matching B,H,C forecasts required')
    if not np.isfinite(bandwidth) or bandwidth <= 0 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid input')
    n = p.shape[1] // length
    totals = np.zeros(4)
    count = 0
    indices = np.arange(n)[:, None] * length + np.arange(length)[None, :]
    for start in range(0, len(p), 32):
        a = p[start:start+32, indices, :].transpose(0, 3, 1, 2)
        b = y[start:start+32, indices, :].transpose(0, 3, 1, 2)
        for i in range(n):
            others = [j for j in range(n) if j != i]
            def distribution(v):
                distance = np.zeros((*v.shape[:2], n-1))
                for k in range(length):
                    distance += ((v[:, :, i, k, None] - v[:, :, others, k]) / bandwidth) ** 2
                z = -.5 * distance
                z -= z.max(-1, keepdims=True)
                log_prob = z - np.log(np.exp(z).sum(-1, keepdims=True))
                return log_prob, np.exp(log_prob)
            lp, pp = distribution(a)
            lq, qq = distribution(b)
            kl = (qq * (lq-lp)).sum(-1)
            if not np.isfinite(kl).all() or kl.min() < -1e-10:
                raise RuntimeError('Invalid independent KL')
            totals += [kl.clip(0).sum(), -(pp*lp).sum(), -(qq*lq).sum(), -(qq*lp).sum()]
            count += kl.size
    return dict(zip(('relation_kl', 'prediction_relation_entropy', 'target_relation_entropy', 'relation_cross_entropy'), map(float, totals/count)))
