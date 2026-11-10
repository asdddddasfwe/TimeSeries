"""Paired stopping, parameter matching and diagnostics isolated from training."""
import copy
import unittest

import torch

from test_pdm_state_patch import cfg
from scripts.run_timemix_pdm_state_patch_output_skip_experiment import (
    initialize_pair, initialization_audit, gradient_probe, verify_policy, FeatureStatistics)
from scripts.run_timemix_dual_encoder_experiment import learning_rate
from utils.patch_covariance_loss import parts


class OutputSkipExperimentTests(unittest.TestCase):
    def test_same_forward_diagnostics_do_not_change_backward(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            torch.manual_seed(2021)
            models = initialize_pair(cfg())
            rng = torch.get_rng_state().clone()
            audit = initialization_audit(models)
            self.assertTrue(audit['common_tensors_exact'])
            self.assertTrue(audit['initial_outputs_expected_equal'])
            self.assertEqual(audit['train']['max_prediction_difference'],0.)
            self.assertEqual(audit['eval']['max_prediction_difference'],0.)
            self.assertEqual(audit['additional_parameters'],512)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            x, y = torch.randn(2, 96, 7), torch.randn(2, 96, 7)
            for arm in ('control', 'residual'):
                model = models[arm]
                reference = copy.deepcopy(model)
                predictions, rngs = [], []
                for instance, capture in ((model, True), (reference, False)):
                    torch.manual_seed(123)
                    statistics = FeatureStatistics(instance) if capture else None
                    prediction = instance(x)
                    predictions.append(prediction)
                    mse = (prediction-y).square().mean()
                    distance, _ = parts(prediction, y)
                    if capture:
                        rng = torch.get_rng_state().clone()
                        result = gradient_probe(instance, mse, distance)
                        features = statistics.summary()
                        statistics.close()
                        self.assertEqual(len(features), 4)
                        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                        self.assertTrue(all(p.grad is None for p in instance.parameters()))
                        self.assertGreater(result['readout_hidden']['mse_norm'], 0.)
                        self.assertEqual(result['output_skip']['parameter_tensors'],4 if arm=='residual' else 0)
                        if arm=='residual':self.assertGreater(result['output_skip']['mse_norm'],0.)
                        for feature in features.values():
                            self.assertEqual(feature['bypass_prediction_rms'],0.)
                            self.assertEqual(feature['skip_weight_norm'],0.)
                            self.assertLess(feature['reconstruction_max_abs'],2e-5)
                    (mse+distance).backward()
                    rngs.append(torch.get_rng_state().clone())
                torch.testing.assert_close(*predictions, rtol=0, atol=0)
                self.assertTrue(torch.equal(*rngs))
                for a, b in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous)

    def test_joint_strict_stopping(self):
        curves = []
        scores = {'control': [.3, .4, .4, .4], 'residual': [.4, .3, .3, .3]}
        for epoch in range(1, 5):
            for arm in scores:
                curves.append(dict(epoch=epoch, arm=arm, learning_rate=learning_rate(epoch),
                    train_loss=.5, train_mse=.4, train_d=.1, val={'mse': scores[arm][epoch-1]}))
        last = dict(curves=curves, best={'control': .3, 'residual': .3}, best_epoch={'control': 1, 'residual': 2},
            epoch=4, stale=2, updates_per_arm=8,
            optimizers={a: {'state': {0: {'step': torch.tensor(8)}}} for a in scores})
        self.assertEqual(verify_policy(last, 33, 4)['joint_stale'], 2)
        last['stale'] = 3
        with self.assertRaises(AssertionError):
            verify_policy(last, 33, 4)


if __name__ == '__main__':
    unittest.main()
