"""Independent conditional-Gaussian checks for the corrected multi-point NLL."""
import math
import unittest

import numpy as np
import torch

from utils.label_covariance_nll import label_gaussian_nll


def sequential_reference(error, covariance):
    total = 0.
    for i in range(len(error)):
        if i:
            beta = np.linalg.solve(covariance[:i, :i], covariance[:i, i])
            variance = covariance[i, i] - covariance[i, :i] @ beta
            innovation = error[i] - beta @ error[:i]
        else:
            variance, innovation = covariance[0, 0], error[0]
        total += .5 * (innovation**2 / variance + np.log(variance) + np.log(2*np.pi))
    return total / len(error)


class LabelCovarianceTests(unittest.TestCase):
    def test_three_correlated_points(self):
        cov = np.full((3, 3), .5) + .5 * np.eye(3)
        beta = np.linalg.solve(cov[:2, :2], cov[:2, 2])
        np.testing.assert_allclose(beta, [1/3, 1/3])
        self.assertAlmostEqual(cov[2, 2] - cov[2, :2] @ beta, 2/3)
        pred = torch.tensor([[[.2], [.5], [-.8]]], dtype=torch.double)
        y = torch.zeros_like(pred)
        loss, _ = label_gaussian_nll(pred, y, torch.tensor(cov)[None])
        self.assertAlmostEqual(float(loss.detach()), sequential_reference(-pred.numpy().ravel(), cov), places=12)
        e = -pred.numpy().ravel()
        wrong = .5 * (e[0]**2 + np.log(2*np.pi))
        for i in (1, 2):
            var = 1 - i * .25
            wrong += .5 * ((e[i] - .5*e[:i].sum())**2/var + np.log(var) + np.log(2*np.pi))
        self.assertGreater(abs(float(loss.detach())-wrong/3), .001)

    def test_numpy_joint_and_conditional_and_gradient(self):
        rng = np.random.default_rng(12)
        a = rng.normal(size=(2, 5, 5))
        cov = a @ a.transpose(0, 2, 1) + .2*np.eye(5)
        p = torch.tensor(rng.normal(size=(3, 5, 2)), requires_grad=True)
        y = torch.tensor(rng.normal(size=(3, 5, 2)), requires_grad=True)
        sigma = torch.tensor(cov, requires_grad=True)
        loss, _ = label_gaussian_nll(p, y, sigma)
        err = y.detach().numpy() - p.detach().numpy()
        joint, conditional, gradients = [], [], np.zeros_like(err)
        for b in range(3):
            for c in range(2):
                e = err[b, :, c]
                sol = np.linalg.solve(cov[c], e)
                joint.append((e @ sol + np.linalg.slogdet(cov[c])[1] + 5*np.log(2*np.pi))/10)
                conditional.append(sequential_reference(e, cov[c]))
                gradients[b, :, c] = -sol / 30
        self.assertAlmostEqual(float(loss.detach()), np.mean(joint), places=12)
        self.assertAlmostEqual(float(loss.detach()), np.mean(conditional), places=12)
        loss.backward()
        np.testing.assert_allclose(p.grad.numpy(), gradients, rtol=1e-12, atol=1e-12)
        self.assertIsNone(y.grad)
        self.assertIsNone(sigma.grad)
        self.assertTrue(torch.autograd.gradcheck(lambda x: label_gaussian_nll(x, y, sigma)[0], (p,)))

    def test_identity_and_invalid_covariance(self):
        p = torch.ones(2, 4, 1, dtype=torch.double)
        y = torch.zeros_like(p)
        identity = torch.eye(4, dtype=torch.double)[None]
        loss, _ = label_gaussian_nll(p, y, identity)
        self.assertAlmostEqual(float(loss.detach()), .5+.5*math.log(2*math.pi))
        with self.assertRaises(torch.linalg.LinAlgError):
            label_gaussian_nll(p, y, -identity)
        asymmetric = identity.clone()
        asymmetric[0, 0, 1] = .5
        with self.assertRaises(ValueError):
            label_gaussian_nll(p, y, asymmetric)


if __name__ == '__main__':
    unittest.main()
