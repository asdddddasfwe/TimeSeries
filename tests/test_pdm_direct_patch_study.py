"""Paired scratch initialization, inherited objective and harmless diagnostics."""
import unittest
import torch

from test_pdm_direct_patch import cfg
from scripts import train_timemix_pdm_direct_patch as study
from scripts import train_timemix_pdm_state_patch as mean_study


class DirectStudyTests(unittest.TestCase):
    def check_pair(self, device):
        torch.set_num_threads(2)
        study.seed_all(2021)
        models = study.initialize_pair(cfg())
        study.seed_all(2021)
        reference = mean_study.initialize_pair(cfg())['mse'].to(device)
        models = {k: v.to(device) for k, v in models.items()}
        for key, value in models['mse'].state_dict().items():
            self.assertTrue(torch.equal(value, models['patch_covariance'].state_dict()[key]), key)
        x = torch.randn(2, 96, 7, device=device)
        target = torch.randn_like(x, requires_grad=True)
        for training in (False, True):
            outputs, rngs = [], []
            for model in [*models.values(), reference]:
                model.train(training)
                study.seed_all(123)
                outputs.append(model(x))
                rngs.append(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state())
            self.assertTrue(torch.equal(outputs[0], outputs[1]))
            self.assertTrue(all(torch.equal(rngs[0], s) for s in rngs))
            torch.testing.assert_close(outputs[0], outputs[2], rtol=2e-5, atol=2e-6)
        model = models['patch_covariance']
        result = study.objective(model(x), target, x, 'patch_covariance', study.loss_configuration(None))
        parameters = tuple(model.parameters())
        before = torch.autograd.grad(result['loss'], parameters, retain_graph=True)
        diag = study.gradient_comparison(model, result, 1.)
        self.assertEqual(diag['direct_temporal_projection']['parameter_tensors'], 10)
        self.assertGreater(diag['direct_temporal_projection']['mse_norm'], 0.)
        self.assertGreater(diag['direct_temporal_projection']['auxiliary_norm'], 0.)
        self.assertTrue(all(p.grad is None for p in parameters))
        result['loss'].backward()
        self.assertIsNone(target.grad)
        for parameter, expected in zip(parameters, before):
            self.assertTrue(torch.equal(parameter.grad, expected))
            self.assertTrue(torch.isfinite(expected).all())
        prediction = models['mse'](x)
        pure = study.objective(prediction, target, x, 'mse', study.loss_configuration(None))
        self.assertTrue(torch.equal(pure['loss'], (prediction-target.detach()).square().mean()))
        stale = 0
        for improved, expected in [(True, 0), (False, 1), (True, 0), (False, 1),
                                   (False, 2), (False, 3), (False, 4), (False, 5)]:
            stale = study.joint_stale_count(stale, improved)
            self.assertEqual(stale, expected)

    def test_pair_cpu(self):
        self.check_pair('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_pair_cuda(self):
        self.check_pair('cuda')


if __name__ == '__main__':
    unittest.main()
