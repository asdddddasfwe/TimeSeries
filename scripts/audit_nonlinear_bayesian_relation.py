"""Independent NumPy reference for the nonlinear relation loss (no model).

Call numpy_score(prediction, target, history, nodes) with saved positive-weight
normal quadrature nodes. No torch/loss import, data fitting beyond the specified
history-only conjugate posterior, or prediction forward is performed here.
"""
import math
import numpy as np


def numpy_features(sequence, order=4):
    z = np.asarray(sequence, dtype=np.float64).transpose(0, 2, 1)
    # Independently build the fixed coordinate and adjacent-sum directions.
    basis = np.zeros((2 * order - 1, order))
    for j in range(order):
        basis[j, j] = 1.
    for j in range(order - 1):
        basis[order + j, j:j+2] = 1. / math.sqrt(2.)
    parents = np.stack([z[..., t-order:t][..., ::-1]
                        for t in range(order, z.shape[-1])], axis=-2)
    nonlinear = np.tanh(np.einsum('bcrq,jq->bcrj', parents, basis))
    design = np.concatenate((np.ones_like(parents[..., :1]), parents, nonlinear), -1)
    return z[..., order:], parents, design, basis


def numpy_score(prediction, target, history, nodes, order=4,
                noise_variance=1., prior_variance=1.):
    d, _, design, basis = numpy_features(history, order)
    precision = np.swapaxes(design, -1, -2) @ design / noise_variance
    precision += np.eye(design.shape[-1]) / prior_variance
    covariance = np.linalg.solve(precision, np.broadcast_to(np.eye(design.shape[-1]), precision.shape))
    mean = np.linalg.solve(precision, (np.swapaxes(design, -1, -2) @ d[..., None]) / noise_variance)[..., 0]
    covariance = .5 * (covariance + np.swapaxes(covariance, -1, -2))
    theta = mean[..., None, :] + np.einsum('bcij,sj->bcsi', np.linalg.cholesky(covariance), nodes)
    pd, pv, pf, _ = numpy_features(prediction, order)
    yd, yv, yf, _ = numpy_features(target, order)
    score = compensation = tangent = scalar = 0.
    for sample in range(theta.shape[-2]):
        coef = theta[..., sample, :]
        beta = coef[..., 1+order:]
        rp = pd - (pf * coef[..., None, :]).sum(-1)
        ry = yd - (yf * coef[..., None, :]).sum(-1)
        gradient_g = coef[..., None, 1:1+order] + np.einsum(
            'bcj,bcrj,jq->bcrq', beta, 1 - pf[..., 1+order:] ** 2, basis)
        curvature = (4. / (3. * math.sqrt(3.))) * (np.abs(beta) * (basis**2).sum(-1)).sum(-1)
        dv = yv - pv
        comp = .5 * curvature[..., None] * (dv**2).sum(-1)
        # Direct potential difference minus analytic tangent; unlike torch's
        # unit-vector scalar divergence / tanh remainder decomposition.
        rho_difference = np.hypot(1., ry) - np.hypot(1., rp)
        local = rho_difference - (rp / np.hypot(1., rp)) * (
            yd - pd - (gradient_g * dv).sum(-1)) + comp
        scalar_part = rho_difference - (rp / np.hypot(1., rp)) * (ry-rp)
        score += local.mean() / theta.shape[-2]
        compensation += comp.mean() / theta.shape[-2]
        scalar += scalar_part.mean() / theta.shape[-2]
        tangent += (local - comp - scalar_part).mean() / theta.shape[-2]
    return dict(mean=mean, covariance=covariance, divergence=float(score),
                curvature_compensation=float(compensation), scalar_bregman=float(scalar),
                tangent_correction=float(tangent))
