"""Mean-patch direct path: identical states, exact bypass and original A24 scan."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from test_pdm_state_patch import cfg
from models.OscillatorPDMStatePatch import Model as Control
from models import OscillatorPDMStatePatchOutputSkip as module
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts
from scripts.run_timemix_pdm_state_patch_output_skip_experiment import initialize_pair


class OutputSkipTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_state_rng_optimizer_and_preserved_condition(self):
        control = Control(cfg())
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2021)
        residual = module.Model(cfg())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in residual.parameters()), 137058)
        self.assertEqual(set(residual.state_dict())-set(control.state_dict()), {f'seasonals.{i}.decoder.readout.skip_projection.weight' for i in range(4)})
        for encoder in residual.seasonals:
            head=encoder.decoder.readout
            self.assertIsNone(head.skip_projection.bias)
            self.assertEqual(torch.count_nonzero(head.skip_projection.weight),0)
            self.assertNotEqual(head.skip_projection.weight.data_ptr(),head.output_projection.weight.data_ptr())
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, residual.state_dict()[name]), name)
            self.assertNotEqual(value.data_ptr(), residual.state_dict()[name].data_ptr(), name)
        self.assertEqual(sum(n.endswith('.trend_condition.weight') for n in residual.state_dict()), 4)
        for model in (control, residual):
            groups = norm_bias_no_decay_groups(model, 1e-4)
            ids = [id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters()})
            wd = {id(p): g['weight_decay'] for g in groups for p in g['params']}
            for n, p in model.named_parameters():
                if '.trend_condition.' in n:
                    self.assertEqual(wd[id(p)], 0.)

    def test_readout_formula_and_direct_gradient(self):
        head=initialize_pair(cfg())['residual'].seasonals[0].decoder.readout.double()
        h=torch.randn(2,24,16,dtype=torch.double,requires_grad=True)
        u=torch.randn(2,24,32,dtype=torch.double,requires_grad=True)
        old=head.output_projection(head.activation(head.input_projection(h)))
        torch.testing.assert_close(head(h,u),old,rtol=0,atol=0)
        du,dw=torch.autograd.grad(head(h,u).square().mean(),(u,head.skip_projection.weight))
        self.assertEqual(torch.count_nonzero(du),0)
        self.assertGreater(float(dw.norm()),0.)
        with torch.no_grad():head.skip_projection.weight.add_(dw,alpha=-.01)
        actual=head(h,u)
        expected=old+F.linear(u,head.skip_projection.weight)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        du,=torch.autograd.grad(actual.sum(),u)
        torch.testing.assert_close(du,head.skip_projection.weight.sum(0).expand_as(u),rtol=0,atol=0)
        self.assertGreater(float(du.norm()),0.)
        with torch.no_grad():
            head.output_projection.weight.zero_();head.output_projection.bias.zero_()
        torch.testing.assert_close(head(h,u),F.linear(u,head.skip_projection.weight),rtol=0,atol=0)

    def test_serial_original_transition_and_gradient(self):
        model = initialize_pair(cfg())['residual'].double().eval()
        with torch.no_grad():
            for encoder in model.seasonals: encoder.decoder.readout.skip_projection.weight.normal_(0,.03)
        x = torch.randn(2, 96, 7, dtype=torch.double)
        scales, (histories, trends) = model.encode_multiscale(model.downsample(x))
        alpha, beta = model.branch_gains()
        encoder = model.seasonals[0]
        history = histories[0].detach().requires_grad_()
        trend = trends[0].detach().requires_grad_()
        captures = []
        original = module.complex_oscillator_scan
        def capture(a, b, h):
            captures.append((a, b, h))
            return original(a, b, h)
        with patch.object(module, 'complex_oscillator_scan', capture):
            actual = module.output_skip_patch_forecast(encoder, scales[0], alpha, beta, history, trend)
        a, b, h = captures[0]
        self.assertEqual(a.shape, (14, 24, 8, 2))
        transition, initial, _, _ = encoder.decoder.dynamics(history, trend)
        torch.testing.assert_close(a, transition[:, None].expand_as(a), rtol=0, atol=0)
        torch.testing.assert_close(h, initial*alpha.repeat(2)[:, None, None], rtol=0, atol=0)
        future = encoder.temporal_projection(history.transpose(1, 2)).transpose(1, 2)+encoder.future_positions
        pooled = torch.stack([future[:, i:i+4].mean(1) for i in range(0, 96, 4)], 1)
        drive = encoder.decoder.future_drive(pooled, history).reshape(14, 24, 8, 2)*beta.repeat(2)[:, None, None, None]
        torch.testing.assert_close(b, drive, rtol=0, atol=0)
        states = []
        for t in range(24):
            ar, ai = a[:, t].unbind(-1); hr, hi = h.unbind(-1)
            h = torch.stack((ar*hr-ai*hi, ar*hi+ai*hr), -1)+b[:, t]
            states.append(h)
        expected = encoder.decoder.readout(torch.stack(states, 1).flatten(2), pooled).reshape(2, 7, 96).transpose(1, 2)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        variables = (history, trend, encoder.decoder.log_decay, encoder.decoder.trend_condition.weight,
                     next(encoder.temporal_projection.parameters()), encoder.decoder.readout.output_projection.weight, encoder.decoder.readout.skip_projection.weight)
        left = torch.autograd.grad(actual.square().mean(), variables, retain_graph=True)
        right = torch.autograd.grad(expected.square().mean(), variables)
        for l, r in zip(left, right):
            torch.testing.assert_close(l, r, rtol=1e-9, atol=1e-10)

    def check_full(self, device):
        models = {k: v.to(device) for k, v in initialize_pair(cfg()).items()}
        x = torch.randn(2, 96, 7, device=device)
        y = torch.randn_like(x, requires_grad=True)
        outputs, rngs = [], []
        for model in models.values():
            torch.manual_seed(71)
            outputs.append(model(x))
            rngs.append(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state())
        self.assertTrue(torch.equal(*rngs))
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
        for p in outputs:
            d,_=parts(p,y)
            ((p-y.detach()).square().mean()+d).backward()
        cparams=dict(models['control'].named_parameters());rparams=dict(models['residual'].named_parameters())
        for name,parameter in cparams.items():
            torch.testing.assert_close(parameter.grad,rparams[name].grad,rtol=0,atol=0,msg=name)
        for encoder in models['residual'].seasonals:
            self.assertGreater(float(encoder.decoder.readout.skip_projection.weight.grad.norm()),0.)
        for m in models.values():m.zero_grad(set_to_none=True)
        model = models['residual']; optimizer = torch.optim.Adam(norm_bias_no_decay_groups(model, 1e-4), lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            p = model(x); d, _ = parts(p, y)
            self.assertEqual(p.shape, x.shape)
            ((p-y.detach()).square().mean()+d).backward()
            self.assertIsNone(y.grad)
            for n, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, n)
                self.assertTrue(torch.isfinite(parameter.grad).all(), n)
            for encoder in model.seasonals:
                self.assertGreater(float(encoder.decoder.trend_condition.weight.grad.norm()), 0.)
                self.assertGreater(float(encoder.decoder.readout.input_projection.weight.grad.norm()), 0.)
                self.assertGreater(float(encoder.decoder.readout.output_projection.weight.grad.norm()), 0.)
                self.assertGreater(float(encoder.decoder.readout.skip_projection.weight.grad.norm()), 0.)
            optimizer.step()
        model.eval()
        restored = module.Model(cfg()).to(device).eval()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_cpu_training(self):
        self.check_full('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_training(self):
        self.check_full('cuda')


if __name__ == '__main__':
    unittest.main()
