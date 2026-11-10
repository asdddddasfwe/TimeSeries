"""Independent NumPy alignment and lagged operator-distance audit, no torch."""
import numpy as np


def numpy_relation_parts(prediction, truth, bandwidth=1., length=4):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if (p.ndim != 3 or p.shape != y.shape or not all(p.shape) or length < 1
            or p.shape[1] % length or p.shape[1] // length < 4):
        raise ValueError('Matching B,H,C with at least four complete patches required')
    if not np.isfinite(bandwidth) or bandwidth <= 0 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid input')
    n = p.shape[1] // length
    total, count = {}, 0
    # Matrix multiplication by J deliberately differs from training's mean algebra.
    centering = {m: np.eye(m) - np.ones((m, m)) / m for m in (n, n-1, n-2)}
    for start in range(0, len(p), 32):
        a = p[start:start+32].transpose(0, 2, 1).reshape(-1, n, length)
        b = y[start:start+32].transpose(0, 2, 1).reshape(-1, n, length)
        def kernel(x, z):
            squared = np.zeros((len(x), n, n))
            for d in range(length):
                squared += (x[:, :, d, None] - z[:, None, :, d]) ** 2
            return np.exp(-squared / (2 * bandwidth**2))
        pp, yy, py = kernel(a, a), kernel(b, b), kernel(a, b)
        j = centering[n]
        pc, yc = j @ pp @ j, j @ yy @ j
        pn, yn = np.linalg.norm(pc, axis=(1, 2)), np.linalg.norm(yc, axis=(1, 2))
        alignment = np.einsum('bij,bij->b', pc, yc) / (np.maximum(pn, 1e-12) * np.maximum(yn, 1e-12))
        raw = np.where(yn > 1e-12, 1-alignment, 0.)
        if not np.isfinite(raw).all() or raw.min() < -1e-10:
            raise RuntimeError('Invalid independent CKA')
        values = dict(cka=raw.clip(0), cka_alignment=alignment,
                      cka_target_degenerate_fraction=(yn <= 1e-12).astype(float),
                      cka_prediction_degenerate_fraction=(pn <= 1e-12).astype(float))
        distances = []
        for lag in (1, 2):
            m, j = n-lag, centering[n-lag]
            def inner(g):
                left, right = j @ g[:, :m, :m] @ j, j @ g[:, lag:, lag:] @ j
                return np.trace(left @ right.transpose(0, 2, 1), axis1=1, axis2=2) / m**2
            pn, yn, cross = inner(pp), inner(yy), inner(py)
            raw = pn+yn-2*cross
            if not np.isfinite(raw).all() or raw.min() < -1e-10:
                raise RuntimeError('Invalid independent lag distance')
            distances.append(raw.clip(0))
            for name, value in [('distance', raw.clip(0)), ('prediction_norm2', pn),
                                ('target_norm2', yn), ('cross_inner_product', cross)]:
                values[f'lag{lag}_{name}'] = value
        values['patch_lag'] = (distances[0]+distances[1])/2
        for key, value in values.items():
            total[key] = total.get(key, 0.) + float(value.sum())
        count += len(a)
    return {k: v/count for k, v in total.items()}
