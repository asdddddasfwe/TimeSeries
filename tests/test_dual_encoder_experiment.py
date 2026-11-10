"""Selection and same-forward gradient-probe contracts of the new runner."""
import copy
import unittest
from pathlib import Path

import torch

from test_pdm_direct_patch import cfg
from models.OscillatorDualEncoderTimeMixer import Model
from scripts.run_timemix_dual_encoder_experiment import gradient_probe, learning_rate, verify_policy
from utils.patch_covariance_loss import parts


class ExperimentTests(unittest.TestCase):
    def test_probe_preserves_rng_and_backward(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            torch.manual_seed(2021)
            model = Model(cfg()).train()
            reference = copy.deepcopy(model)
            x, y = torch.randn(2, 96, 7), torch.randn(2, 96, 7)
            for instance, probe in [(model, True), (reference, False)]:
                torch.manual_seed(42)
                prediction = instance(x)
                mse = (prediction-y).square().mean()
                distance, _ = parts(prediction, y)
                if probe:
                    rng = torch.get_rng_state().clone()
                    values = gradient_probe(instance, mse, distance)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertTrue(all(p.grad is None for p in instance.parameters()))
                    self.assertGreater(values['trend_attention']['mse_norm'], 0.)
                (mse+distance).backward()
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)

    def test_strict_validation_selection_and_update_count(self):
        curves = [dict(epoch=i, learning_rate=learning_rate(i), train_loss=.5,
                       train_mse=.4, train_d=.1, val={'mse': v})
                  for i, v in enumerate([.4, .3, .3], 1)]
        last = dict(curves=curves, best=.3, best_epoch=2, stale=1, epoch=3, updates=6,
                    optimizer={'state': {0: {'step': torch.tensor(6)}}})
        self.assertEqual(verify_policy(Path('.'), last, 33)['selected_epoch'], 2)
        last['best_epoch'] = 3
        with self.assertRaises(AssertionError):
            verify_policy(Path('.'), last, 33)
        self.assertEqual(learning_rate(1), .001)
        self.assertEqual(learning_rate(31), 0.)


if __name__ == '__main__':
    unittest.main()
