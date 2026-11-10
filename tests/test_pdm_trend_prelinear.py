import io
import unittest

import torch

from test_oscillator_mixer import config
from models.OscillatorPDMReadoutSplit import Model as Control
from models.OscillatorPDMTrendPreLinear import Model, PreLinearTrendMixing
from scripts.train_timemix_pdm_readout_split import parameter_groups


class TrendPreLinearTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = config(seq_len=96, pred_len=24, moving_avg=25,
                          e_layers=2, dropout=.2)
        self.cfg.seasonal_normalize = False

    def pair(self):
        torch.manual_seed(2021)
        control = Control(self.cfg, variant="seasonal")
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2021)
        model = Model(self.cfg)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        return control, model

    def test_common_weights_identity_and_independent_automatic_scales(self):
        control, model = self.pair()
        for k, v in control.state_dict().items():
            torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
        weights = []
        for block in model.pdm_blocks:
            mix = block.mixing_multi_scale_trend
            self.assertEqual(mix.lengths, (96, 48, 24, 12))
            for layer, length in zip(mix.pre_linears, mix.lengths):
                torch.testing.assert_close(layer.weight, torch.eye(length), rtol=0, atol=0)
                self.assertEqual(layer.bias.count_nonzero(), 0)
                weights.append(layer.weight.data_ptr())
        self.assertEqual(len(set(weights)), 8)
        extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in control.parameters())
        self.assertEqual(extra, 2 * sum(n*n+n for n in model.lengths))
        cfg = config(seq_len=64, pred_len=16, moving_avg=25, e_layers=1)
        cfg.seasonal_normalize = False
        other = Model(cfg)
        self.assertEqual(other.pdm_blocks[0].mixing_multi_scale_trend.lengths, (64,32,16,8))

    def test_initial_prediction_rng_and_twelve_components_match(self):
        control, model = self.pair()
        x = torch.randn(2, 96, 2)
        for train in [False, True]:
            outputs, rngs = [], []
            for m in [control, model]:
                m.train(train)
                torch.manual_seed(97)
                with torch.no_grad():outputs.append(m(x))
                rngs.append(torch.get_rng_state())
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=2e-6)
            self.assertTrue(torch.equal(*rngs))
        model.eval(); control.eval()
        with torch.no_grad():
            a, b = control.predict_components(x), model.predict_components(x)
        self.assertEqual(b.shape, (2,24,2,12))
        torch.testing.assert_close(a, b, rtol=0, atol=2e-6)

    def test_temporal_transform_precedes_mixing(self):
        control, model = self.pair()
        old = control.pdm_blocks[0].mixing_multi_scale_trend
        new = model.pdm_blocks[0].mixing_multi_scale_trend
        data = [torch.randn(3, 4, n) for n in model.lengths]
        with torch.no_grad():
            for i, linear in enumerate(new.pre_linears):
                linear.weight.copy_(torch.eye(linear.in_features).roll(1, dims=0)*(i+1))
                linear.bias.fill_(.1*i)
            transformed = [linear(t) for linear,t in zip(new.pre_linears,data)]
            expected, actual = old(transformed), new(data)
        for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertFalse(torch.equal(actual[0], old(data)[0]))
        with self.assertRaises(ValueError):new(data[:-1])
        with self.assertRaises(ValueError):new([v.transpose(1,2) for v in data])

    def test_all_added_layers_receive_gradients_and_update(self):
        _, model = self.pair()
        groups = parameter_groups(model, 'seasonal_trend')
        params = [p for g in groups for p in g['params']]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
        rates = {id(p):g['weight_decay'] for g in groups for p in g['params']}
        layers = [l for b in model.pdm_blocks for l in b.mixing_multi_scale_trend.pre_linears]
        optimizer = torch.optim.Adam(groups, lr=.001)
        x, y = torch.randn(2,96,2), torch.randn(2,24,2)
        loss = (model(x)-y).square().mean();self.assertTrue(torch.isfinite(loss));loss.backward()
        for l in layers:
            self.assertEqual(rates[id(l.weight)], 1e-4)
            self.assertEqual(rates[id(l.bias)], 0)
            for p in [l.weight,l.bias]:
                self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(),0)
        optimizer.step()
        for l in layers:self.assertFalse(torch.equal(l.weight,torch.eye(l.in_features)))
        model.eval()
        with torch.no_grad():expected=model(x)
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        restored=Model(self.cfg).eval();restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
        with torch.no_grad():torch.testing.assert_close(expected,restored(x),rtol=0,atol=0)

    def test_future_ignored_and_single_inverse_normalization(self):
        _,m=self.pair();m.eval();x=torch.randn(2,96,2);events=[]
        h=m.normalize.register_forward_pre_hook(lambda mod,args:events.append(args[1]))
        with torch.no_grad():p=m(x)
        h.remove();self.assertEqual(events,['norm','denorm'])
        nan=torch.full((2,24,2),float('nan'))
        with torch.no_grad():torch.testing.assert_close(p,m(x,nan,nan,nan),rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
