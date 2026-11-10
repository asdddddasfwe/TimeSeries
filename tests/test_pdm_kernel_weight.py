"""Weight and gradient instrumentation checks for the unchanged kernel loss."""
import unittest
from types import SimpleNamespace
import torch
from scripts import train_timemix_pdm_kernel_weight as study
from scripts import train_timemix_pdm_kernel_autocovariance as original


class KernelWeightTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def check_objective(self, device):
        p = torch.randn(2, 96, 3, device=device, requires_grad=True)
        y = torch.randn_like(p, requires_grad=True)
        history = torch.randn_like(p, requires_grad=True)
        config = study.loss_configuration(SimpleNamespace())
        old = original.objective(p, y, history, 'kernel', original.loss_configuration(None))
        gm = torch.autograd.grad(old['mse'], p, retain_graph=True)[0]
        gd = torch.autograd.grad(old['loss'] - old['mse'], p, retain_graph=True)[0]
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state() if device == 'cuda' else None
        for arm, weight in study.WEIGHTS.items():
            result = study.objective(p, y, history, arm, config)
            self.assertTrue(torch.equal(result['relation'], old['relation']))
            self.assertTrue(torch.equal(result['loss'], result['mse'] + weight * result['relation']))
            gp, gy, gh = torch.autograd.grad(result['loss'], (p, y, history), retain_graph=True, allow_unused=True)
            torch.testing.assert_close(gp, gm + weight * gd, atol=2e-8, rtol=3e-5)
            self.assertIsNone(gy)
            self.assertIsNone(gh)
            self.assertTrue(torch.isfinite(gp).all())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        if cuda_rng is not None:
            self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))

    def test_cpu_objective(self):
        self.check_objective('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_objective(self):
        self.check_objective('cuda')

    def test_gradient_probe_does_not_change_backward_or_rng(self):
        model = torch.nn.Linear(3, 3)
        x, y = torch.randn(2, 96, 3), torch.randn(2, 96, 3)
        config = study.loss_configuration(None)
        for arm, weight in study.WEIGHTS.items():
            model.zero_grad(set_to_none=True)
            parts = study.objective(model(x), y, x, arm, config)
            expected = torch.autograd.grad(parts['loss'], tuple(model.parameters()), retain_graph=True)
            rng = torch.get_rng_state().clone()
            d = study.gradient_comparison(model, parts, weight)
            self.assertGreater(d['all']['weighted_kernel_norm'], 0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            parts['loss'].backward()
            for p, grad in zip(model.parameters(), expected):
                self.assertTrue(torch.equal(p.grad, grad))

    def test_joint_stale_and_selection(self):
        stale = 0
        for improvement, expected in [(True, 0), (False, 1), (False, 2), (True, 0), (False, 1)]:
            stale = study.joint_stale_count(stale, improvement)
            self.assertEqual(stale, expected)
        self.assertEqual(study.selection_score({'neural_mse': .4, 'relation': 0}), .4)
        self.assertEqual(study.learning_rate('lambda5', 0, 30), .001)
        self.assertEqual(study.learning_rate('lambda10', 30, 30), 0)


if __name__ == '__main__':
    unittest.main()
