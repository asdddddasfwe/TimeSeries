"""Numerical and full-backbone invariants for a main-path readout intervention."""
import copy
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from models.OscillatorPDMStatePatchReadoutDiagnostic import Model, constant_drive_substates
from models.OscillatorPDMStatePatchMAKernel9 import Model as Baseline
from layers.OscillatorTimeMix import complex_oscillator_scan

ROOT = Path(__file__).resolve().parents[1]


class ReadoutDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = json.loads((ROOT / "tmp/mafirst_kernel9_point9_ettm1_h720_20261030_full_seed2021/model_config.json").read_text())

    def build(self, policy):
        cfg = SimpleNamespace(**(self.cfg | {"state_readout_policy": policy}))
        with torch.random.fork_rng():
            torch.manual_seed(2021)
            return Model(cfg)

    def test_full_backbone_endpoint_forward_and_gradients_match(self):
        a = self.build("endpoint")
        cfg = SimpleNamespace(**self.cfg)
        b = Baseline(cfg)
        b.load_state_dict(a.state_dict())
        a.eval(); b.eval()
        x = torch.randn(2, 96, 7)
        pa, pb = a(x), b(x)
        self.assertTrue(torch.equal(pa, pb))
        pa.square().mean().backward(); pb.square().mean().backward()
        for (na, va), (nb, vb) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(na, nb)
            self.assertTrue(torch.equal(va.grad, vb.grad), na)

    def test_substate_preserves_parameter_state_and_constructor_rng(self):
        torch.manual_seed(2021)
        a = self.build("endpoint")
        before = torch.get_rng_state().clone()
        b = self.build("substate")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in a.parameters()), 345786)
        self.assertEqual(len(a.state_dict()), 244)
        self.assertEqual(len(list(a.parameters())), 212)
        self.assertEqual(len(list(a.buffers())), 32)
        self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))

    def test_endpoints_and_block_map_gradients_unchanged(self):
        radius = torch.tensor([[.2, .99, .9999]], dtype=torch.float64, requires_grad=True)
        freq = torch.tensor([[.1, 1.2, 3.0]], dtype=torch.float64, requires_grad=True)
        transition = torch.stack((radius * freq.cos(), radius * freq.sin()), -1)
        drive = torch.randn(1, 7, 3, 2, dtype=torch.float64, requires_grad=True)
        initial = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
        endpoints = complex_oscillator_scan(transition[:, None].expand(-1, 7, -1, -1), drive, initial)
        sub, rebuilt = constant_drive_substates(transition, radius, freq, drive, initial, endpoints)
        self.assertIs(sub[-1], endpoints)
        self.assertLess(float((rebuilt - endpoints).detach().abs().max()), 1e-12)
        ga = torch.autograd.grad(rebuilt.square().sum(), (radius, freq, drive, initial), retain_graph=True)
        gb = torch.autograd.grad(endpoints.square().sum(), (radius, freq, drive, initial))
        for a, b in zip(ga, gb):
            self.assertTrue(torch.allclose(a, b, rtol=1e-9, atol=1e-9))

    def test_original_block_clock_preserved_not_replaced_by_fourth_power(self):
        radius = torch.tensor([[.8]], dtype=torch.float64)
        freq = torch.tensor([[.6]], dtype=torch.float64)
        transition = torch.stack((radius * freq.cos(), radius * freq.sin()), -1)
        initial = torch.tensor([[[1., 0.]]], dtype=torch.float64)
        drive = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
        end = complex_oscillator_scan(transition[:, None], drive, initial)
        sub, rebuilt = constant_drive_substates(transition, radius, freq, drive, initial, end)
        self.assertTrue(torch.equal(sub[3], end))
        self.assertTrue(torch.allclose(sub[0], torch.tensor([[[[.8**.25*np.cos(.15), .8**.25*np.sin(.15)]]]])))
        self.assertTrue(torch.allclose(rebuilt, end))

    def test_substate_full_loss_gradients_all_finite_and_rng_calls_match(self):
        a, b = self.build("endpoint"), self.build("substate")
        a.train(); b.train()
        x = torch.randn(2, 96, 7)
        rng = torch.get_rng_state().clone()
        pa = a(x); aftera = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        pb = b(x); afterb = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(aftera, afterb))
        self.assertFalse(torch.equal(pa, pb))
        from tmp.mafirst_fullbatch_mmd_recompute_20261028 import parts
        target = torch.randn_like(pb)
        value = (pb-target).square().mean() + parts(pb, target)[0]
        before = torch.get_rng_state().clone()
        value.backward()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for name, p in b.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        for diag in b.last_state_readout_diagnostics:
            self.assertTrue(diag["selected_endpoint_bitwise_exact"])
            self.assertLess(float(diag["substate_reconstruction_relative"]), 1e-5)

    def test_four_rows_read_respective_coordinates_no_extra_path(self):
        model = self.build("substate")
        self.assertEqual(list(model.initial_decomposition.kernels.values()), [9]*4)
        self.assertFalse(any("correction" in name or "residual_head" in name for name in model.state_dict()))
        model.eval()
        comp = model.predict_components(torch.randn(1, 96, 7))
        self.assertEqual(comp.shape, (1, 720, 7, 8))


if __name__ == "__main__":
    unittest.main()
