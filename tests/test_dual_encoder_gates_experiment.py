"""Same-forward gate probes and joint early stopping semantics."""
import copy
import unittest

import torch

from test_pdm_direct_patch import cfg
from scripts.run_timemix_dual_encoder_gates_experiment import (
    initialize_pair, initialization_audit, gradient_probe, verify_policy)
from scripts.run_timemix_dual_encoder_experiment import learning_rate
from utils.patch_covariance_loss import parts


class GateExperimentTests(unittest.TestCase):
    def test_same_forward_probe_and_initial_audit(self):
        previous = torch.get_num_threads(); torch.set_num_threads(2)
        try:
            torch.manual_seed(2021)
            models = initialize_pair(cfg())
            rng = torch.get_rng_state().clone()
            audit = initialization_audit(models)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(audit['common_tensors_exact'])
            model = models['gated']; model.enable_gate_statistics(True)
            reference = copy.deepcopy(model)
            x, y = torch.randn(2, 96, 7), torch.randn(2, 96, 7)
            for instance, capture in ((model, True), (reference, False)):
                torch.manual_seed(123)
                prediction = instance(x)
                mse = (prediction-y).square().mean(); distance, _ = parts(prediction, y)
                if capture:
                    rng = torch.get_rng_state().clone()
                    probe = gradient_probe(instance, mse, distance)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertTrue(all(p.grad is None for p in instance.parameters()))
                    self.assertGreater(probe['forget_gate']['mse_norm'], 0.)
                    self.assertGreater(probe['write_gate']['mse_norm'], 0.)
                (mse+distance).backward()
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous)

    def test_any_arm_improvement_and_strict_selection(self):
        curves = []
        scores = {'ungated': [.3, .4, .4, .4], 'gated': [.4, .3, .3, .3]}
        for epoch in range(1, 5):
            for arm in scores:
                curves.append(dict(epoch=epoch, arm=arm, learning_rate=learning_rate(epoch),
                    train_loss=.5, train_mse=.4, train_d=.1, val={'mse':scores[arm][epoch-1]}))
        last = dict(curves=curves, best={'ungated':.3,'gated':.3}, best_epoch={'ungated':1,'gated':2},
                    epoch=4, stale=2, updates_per_arm=8,
                    optimizers={a:{'state':{0:{'step':torch.tensor(8)}}} for a in scores})
        result = verify_policy(last, 33, 4)
        self.assertEqual(result['joint_stale'], 2)
        self.assertEqual(result['selected_epochs'], {'ungated':1,'gated':2})
        last['stale'] = 3
        with self.assertRaises(AssertionError):
            verify_policy(last, 33, 4)


if __name__ == '__main__':
    unittest.main()
