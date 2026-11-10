"""CPU structural checks for the frozen trained-control correction diagnostic."""
from pathlib import Path
import sys
import unittest
import torch
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from models.OscillatorPDMStatePatchMAFirstFrozenCorrection import load_trained_control
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups

class FrozenCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model, _ = load_trained_control(R, R/"tmp/mafirst_frozen_correction_unit_logs_20261020")
        torch.manual_seed(1234)
        cls.x = torch.randn(2, 96, 7)
        cls.y = torch.randn(2, 720, 7)

    def test_01_zero_correction_exact_control(self):
        m = self.model.eval()
        with torch.no_grad():
            original = m.base(self.x)
            prediction, details = m(self.x, return_details=True)
        self.assertTrue(torch.equal(original, prediction))
        self.assertEqual(float(details["corrections"].abs().max()), 0.)
        self.assertEqual(tuple(prediction.shape), (2,720,7))

    def test_02_train_keeps_backbone_eval_and_frozen(self):
        m = self.model.train()
        self.assertTrue(m.corrections.training)
        self.assertTrue(all(not child.training for child in m.base.modules()))
        self.assertTrue(all(not p.requires_grad for p in m.base.parameters()))
        self.assertEqual(sum(p.numel() for p in m.parameters()), 734722)
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad), 414272)
        self.assertEqual(len(m.state_dict()), 312)

    def test_03_optimizer_exact_trainable_coverage(self):
        m = self.model
        groups = norm_bias_no_decay_groups(m.corrections, 1e-4)
        ids = [id(p) for g in groups for p in g["params"]]
        expected = {id(p) for p in m.parameters() if p.requires_grad}
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),expected)
        self.assertFalse(set(ids) & {id(p) for p in m.base.parameters()})

    def test_04_units_and_unchanged_recurrent_control(self):
        m = self.model
        before = {k:v.clone() for k,v in m.base.state_dict().items()}
        m.train()
        with torch.no_grad():
            p,d = m(self.x,return_details=True)
            reconstructed = d["components"][...,:4].sum(-1)+d["components"][...,4:].sum(-1)+d["history_mean"]
        self.assertLessEqual(float((p-reconstructed).abs().max()),2e-5)
        self.assertTrue(all(torch.equal(v,m.base.state_dict()[k]) for k,v in before.items()))

    def test_05_two_updates_learnable_finite_backbone_unchanged(self):
        m = self.model
        initial = {k:v.clone() for k,v in m.state_dict().items()}
        baseline = {k:v.clone() for k,v in m.base.state_dict().items()}
        m.train()
        opt = torch.optim.Adam(norm_bias_no_decay_groups(m.corrections,1e-4),lr=.001)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            loss = (m(self.x)-self.y).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.corrections.parameters()))
            self.assertTrue(all(p.grad is None for p in m.base.parameters()))
            if step == 1:
                self.assertGreater(sum(float(p.grad.square().sum()) for n,p in m.corrections.named_parameters()
                    if ".transformer." in n),0.)
            opt.step()
        self.assertTrue(all(torch.equal(v,m.base.state_dict()[k]) for k,v in baseline.items()))
        with torch.no_grad():
            p,d=m(self.x,return_details=True)
        self.assertGreater(float(d["corrections"].abs().max()),0.)
        m.load_state_dict(initial,strict=True)

if __name__ == "__main__":
    unittest.main(verbosity=2)
