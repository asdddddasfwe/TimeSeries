import unittest
from types import SimpleNamespace
import torch
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_autocorr import (ARMS, initialize_pair, parameter_groups,
    training_losses, loss_configuration, selection_score, dataset, joint_stale_count)
from utils.joint_gaussian_autocorr_loss import joint_gaussian_autocorr_kl_loss


class PDMAutocorrTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2021)
        self.cfg = config(seq_len=96, pred_len=96, moving_avg=25, dropout=.2)
        self.cfg.e_layers = 2
        self.cfg.seasonal_normalize = False
        self.loss_cfg = loss_configuration(SimpleNamespace(variance_floor=.01, kl_jitter=.01))

    def test_initial_models_predictions_rng_and_optimizer_equal(self):
        models = initialize_pair(self.cfg)
        states = [m.state_dict() for m in models.values()]
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
        x = torch.randn(2, 96, self.cfg.enc_in)
        for train in [False, True]:
            preds, rng = [], []
            for model in models.values():
                model.train(train)
                torch.manual_seed(7)
                with torch.no_grad(): preds.append(model(x))
                rng.append(torch.get_rng_state())
            self.assertTrue(torch.equal(preds[0], preds[1]))
            self.assertTrue(torch.equal(rng[0], rng[1]))
        policies = []
        for arm, model in models.items():
            groups = parameter_groups(model, arm)
            wd = {id(p):g['weight_decay'] for g in groups for p in g['params']}
            policies.append({n:wd[id(p)] for n,p in model.named_parameters()})
        self.assertEqual(policies[0], policies[1])

    def test_loss_gradient_is_mse_plus_differentiable_correlation(self):
        h = torch.randn(2,96,2,dtype=torch.double,requires_grad=True)
        y = torch.randn_like(h,requires_grad=True)
        for arm in ARMS:
            p = torch.randn_like(h,requires_grad=True)
            got = training_losses(p,y,arm,self.loss_cfg,h)
            expected = (p-y.detach()).square().mean()
            if arm == 'autocorr_kl':
                expected = expected + joint_gaussian_autocorr_kl_loss(h,p,y)
            torch.testing.assert_close(got['loss'],expected)
            g = torch.autograd.grad(got['loss'],(p,h,y),allow_unused=True,retain_graph=True)
            e = torch.autograd.grad(expected,p)[0]
            torch.testing.assert_close(g[0],e)
            self.assertIsNone(g[1]);self.assertIsNone(g[2])
            self.assertTrue(torch.isfinite(g[0]).all())
            self.assertFalse(got['kl'].requires_grad)
            if arm == 'mse':self.assertEqual(float(got['auxiliary_kl']),0)
            else:
                self.assertGreater(float(got['auxiliary_kl']),0)
                self.assertGreater(float((g[0]-2*(p-y.detach())/p.numel()).norm()),0)

    def test_mse_selection_joint_stop_and_no_test(self):
        self.assertEqual(selection_score(dict(neural_mse=.5,kl=0)),.5)
        self.assertEqual(joint_stale_count(4,True),0)
        self.assertEqual(joint_stale_count(4,False),5)
        with self.assertRaises(ValueError):dataset(self.cfg,'test',0)
        with self.assertRaises(ValueError):training_losses(None,None,'bad',self.loss_cfg,None)
        self.assertEqual(self.loss_cfg['kl_weight_by_arm'],{'mse':0.0,'autocorr_kl':1.0})


if __name__ == '__main__':unittest.main()
