"""组合模型的初始化、参数分组、梯度和冻结适配器回归。"""
import copy
import unittest
from unittest.mock import patch
import torch
from test_oscillator_mixer import config
from models.OscillatorTimeMix import Model as Control
from models.OscillatorLowRankLocalTimeMix import Model as Combined
from models.OscillatorCombinedLinearBlend import Model as Blend
from layers.LinearForecastBlend import LinearForecastBlend
from scripts.train_timemix_combined import parameter_groups, dataset


class CombinedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2021)
        self.cfg = config(seq_len=96, pred_len=96, is_training=0)

    def test_svd_affine_product_and_shared_state(self):
        control = Control(self.cfg)
        candidate = Combined(self.cfg)
        candidate.initialize_from_control(control)
        dense = control.seasonals[0].temporal_projection
        first, second = candidate.seasonals[0].temporal_projection
        u, s, vh = torch.linalg.svd(dense.weight.double(), full_matrices=False)
        expected = (u[:, :16] * s[:16]) @ vh[:16]
        torch.testing.assert_close(second.weight @ first.weight, expected.float(), atol=2e-7, rtol=2e-6)
        x = torch.randn(2, 3, 96)
        torch.testing.assert_close(candidate.seasonals[0].temporal_projection(x),
            torch.nn.functional.linear(x, second.weight @ first.weight, dense.bias), atol=3e-7, rtol=2e-5)
        torch.testing.assert_close(first.bias, torch.zeros_like(first.bias), atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in control.parameters())-sum(p.numel() for p in candidate.parameters()),6128)
        for name, value in candidate.state_dict().items():
            if not name.startswith('seasonals.0.temporal_projection.'):
                torch.testing.assert_close(value,control.state_dict()[name],atol=0,rtol=0)

    def test_finite_gradients_decay_and_factor_update(self):
        model = Combined(self.cfg)
        groups = parameter_groups(model,'combined')
        decay = {id(p):g['weight_decay'] for g in groups for p in g['params']}
        optimizer = torch.optim.Adam(groups,lr=.001)
        before = model.seasonals[0].temporal_projection[0].weight.detach().clone()
        loss = (model(torch.randn(2,96,2))-torch.randn(2,96,2)).square().mean()
        loss.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        for module in model.seasonals[0].temporal_projection:
            self.assertEqual(decay[id(module.weight)],1e-4)
            self.assertEqual(decay[id(module.bias)],0)
        optimizer.step()
        self.assertFalse(torch.equal(before,model.seasonals[0].temporal_projection[0].weight))

    def test_dropout_rng_and_normalization(self):
        cfg = config(seq_len=96,pred_len=96,dropout=.2)
        control = Control(cfg).train()
        candidate = Combined(cfg).train()
        candidate.initialize_from_control(control)
        x = torch.randn(2,96,2)
        torch.manual_seed(52); control(x); reference_rng = torch.get_rng_state()
        torch.manual_seed(52)
        with patch.object(candidate.normalize,'forward',wraps=candidate.normalize.forward) as norm:
            candidate(x)
        self.assertTrue(torch.equal(reference_rng,torch.get_rng_state()))
        self.assertEqual([c.args[1] for c in norm.call_args_list],['norm','denorm'])

    def test_adapter_roundtrip_future_isolation_and_no_training(self):
        neural = Combined(self.cfg).eval()
        direct = LinearForecastBlend(neural,torch.randn(2,97,96)*.01).eval()
        adapter = Blend(self.cfg)
        x = torch.randn(2,96,2)
        with self.assertRaises(RuntimeError): adapter(x)
        adapter.load_state_dict(copy.deepcopy(direct.state_dict()),strict=True)
        with torch.no_grad():
            torch.testing.assert_close(adapter(x),direct(x),atol=2e-6,rtol=1e-5)
            poison = torch.full((2,96,2),float('nan'))
            torch.testing.assert_close(adapter(x,poison,poison,poison),adapter(x),atol=0,rtol=0)
        with self.assertRaises(ValueError): adapter.train()
        with self.assertRaises(ValueError): adapter.load_state_dict({},strict=False)
        with self.assertRaises(ValueError): dataset(None,'test',0)


if __name__=='__main__': unittest.main()
