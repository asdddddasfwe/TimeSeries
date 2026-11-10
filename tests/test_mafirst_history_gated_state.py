"""Behavioral and numerical checks for MAFirst gated recurrence; synthetic data only."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from layers.HistoryGatedState import HistoryGatedStateDecoder
from models.MAFirstHistoryGatedState import Model
from models.OscillatorPDMStatePatchMAFirst import Model as Reference
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups


def config(horizon=16):
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=96, pred_len=horizon,
        enc_in=2, c_out=2, features="M", moving_avg=25, d_model=8,
        d_ff=16, e_layers=1, dropout=0.0, ssm_modes=2, top_k=3,
        down_sampling_layers=3, down_sampling_window=2,
        down_sampling_method="avg", channel_independence=1,
        use_norm=1, decomp_method="moving_avg", tensorboard=False,
    )


def decoder(horizon=12):
    return HistoryGatedStateDecoder(
        length=3, horizon=horizon, width=4, state_size=4,
        stride=2, readout=nn.Linear(4, 4), dropout=0.0,
    )


class GatedStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(41)

    def test_cell_can_retain_or_replace_state(self):
        core = decoder()
        with torch.no_grad():
            for parameter in core.cell.parameters():
                parameter.zero_()
            # PyTorch GRU packs reset/update/candidate in this order.
            core.cell.bias_ih[4:8].fill_(30)
        state = torch.tensor([[.2, -.3, .4, -.5]])
        x = torch.randn(1, 4)
        torch.testing.assert_close(core.cell(x, state), state)
        with torch.no_grad():
            core.cell.bias_ih[4:8].fill_(-30)
        torch.testing.assert_close(core.cell(x, state), torch.zeros_like(state),
                                   rtol=0, atol=1e-12)

    def test_hidden_coordinates_are_coupled(self):
        cell = decoder().double().cell
        x, state = torch.randn(1, 4, dtype=torch.float64), torch.randn(1, 4, dtype=torch.float64)
        jacobian = torch.autograd.functional.jacobian(
            lambda h: cell(x, h).flatten(), state
        ).reshape(4, 4)
        off_diagonal = jacobian - torch.diag(torch.diag(jacobian))
        self.assertGreater(float(off_diagonal.abs().max()), 1e-4)

    def test_context_changes_with_current_state(self):
        core = decoder()
        memory = torch.randn(2, 3, 4)
        first = torch.zeros(2, 4)
        second = torch.ones(2, 4)
        _, a = core.decode(memory, first, steps=1, return_details=True)
        _, b = core.decode(memory, second, steps=1, return_details=True)
        self.assertGreater(float((a["attention"] - b["attention"]).detach().abs().max()), 1e-5)
        torch.testing.assert_close(a["attention"].sum(-1), torch.ones(2, 1))
        self.assertTrue((a["attention"] >= 0).all())

    def test_future_prefix_and_no_prediction_feedback(self):
        core = decoder().eval()
        memory, initial = torch.randn(2, 3, 4), torch.randn(2, 4)
        full, details = core.decode(memory, initial, return_details=True)
        prefix = core.decode(memory, initial, steps=2)
        torch.testing.assert_close(full[:, :8], prefix, rtol=0, atol=0)
        hook = core.readout.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output)
        )
        try:
            changed, changed_details = core.decode(memory, initial, return_details=True)
        finally:
            hook.remove()
        self.assertEqual(tuple(details["states"].shape), (2, 3, 4))
        self.assertTrue(torch.equal(changed, torch.zeros_like(changed)))
        torch.testing.assert_close(details["states"], changed_details["states"], rtol=0, atol=0)

    def test_fp64_gradient_to_both_history_streams(self):
        core = decoder(horizon=8).double()
        season = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
        trend = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda s, t: core(s, t), (season, trend),
            eps=1e-6, atol=1e-5, rtol=1e-4,
        ))

    def test_reference_features_heads_and_sum_are_preserved(self):
        cfg = config()
        reference, model = Reference(cfg).eval(), Model(cfg).eval()
        audit = model.load_reference_initialization(reference.state_dict())
        self.assertGreater(len(audit["fresh"]), 0)
        self.assertTrue(all(torch.equal(model.state_dict()[k], reference.state_dict()[k])
                            for k in audit["copied"]))
        x = torch.randn(2, 96, 2)
        with torch.no_grad():
            _, (s1, t1) = reference.encode_multiscale(reference.downsample(x))
            _, (s2, t2) = model.encode_multiscale(model.downsample(x))
            for a, b in zip(s1 + t1, s2 + t2):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            parts = model.predict_components(x)
            old_parts = reference.predict_components(x)
            torch.testing.assert_close(parts[..., 4:], old_parts[..., 4:], rtol=0, atol=0)
            expected = model.normalize(parts[..., :4].sum(-1) + parts[..., 4:].sum(-1), "denorm")
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        self.assertEqual(tuple(parts.shape), (2, 16, 2, 8))
        for old, new in zip(reference.seasonals, model.seasonals):
            for key, value in old.decoder.readout.state_dict().items():
                self.assertTrue(torch.equal(value, new.decoder.readout.state_dict()[key]))
            self.assertFalse(hasattr(new, "temporal_projection"))
            self.assertFalse(hasattr(new.decoder, "log_decay"))

    def test_h720_backward_and_optimizer_cover_every_parameter(self):
        model = Model(config(720)).train()
        groups = norm_bias_no_decay_groups(model, 1e-4)
        parameters = [p for group in groups for p in group["params"]]
        self.assertEqual(len({id(p) for p in parameters}), len(parameters))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters()})
        decay = {id(p): group["weight_decay"] for group in groups for p in group["params"]}
        for encoder in model.seasonals:
            self.assertEqual(decay[id(encoder.decoder.cell.bias_ih)], 0.0)
            self.assertEqual(decay[id(encoder.decoder.cell.bias_hh)], 0.0)
        x = torch.randn(2, 96, 2)
        prediction = model(x)
        self.assertEqual(tuple(prediction.shape), (2, 720, 2))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_targets_are_not_an_input(self):
        model = Model(config()).eval()
        x = torch.randn(2, 96, 2)
        with torch.no_grad():
            first = model(x, x_dec=torch.zeros(2, 16, 2))
            second = model(x, x_dec=torch.full((2, 16, 2), float("nan")))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_shape_errors_fail_explicitly(self):
        core = decoder()
        with self.assertRaises(ValueError):
            core(torch.randn(2, 4, 4), torch.randn(2, 4, 4))
        with self.assertRaises(ValueError):
            core.decode(torch.randn(2, 3, 4), torch.randn(2, 3))
        with self.assertRaises(ValueError):
            core.decode(torch.randn(2, 3, 4), torch.randn(2, 4), steps=0)
        with self.assertRaises(ValueError):
            decoder(horizon=7)


if __name__ == "__main__":
    unittest.main()
