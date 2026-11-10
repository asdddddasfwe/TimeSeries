"""Single-factor local-drive equation, isolation and training regression tests."""
import copy
import unittest
from unittest.mock import patch
import torch
from test_oscillator_mixer import config
from models.OscillatorTimeMix import Model as Control
from models.OscillatorLocalDriveTimeMix import Model, history_response_plus_local_drive
from layers.OscillatorCalibratedMixer import calibrated_seasonal_forecast
from layers.OscillatorMixer import complex_product
from scripts.train_timemix_local_drive import dataset, parameter_groups, build_optimizer


def serial_oracle(a, b, initial):
    history = initial
    values = []
    for h in range(b.shape[1]):
        history = complex_product(a[:, h], history)
        values.append(history + b[:, h])
    return torch.stack(values, 1)


class LocalDriveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2021)

    def test_equation_gradients_and_no_previous_drive(self):
        a=(torch.randn(2,7,3,2,dtype=torch.float64)*.2).requires_grad_()
        b=torch.randn_like(a,requires_grad=True)
        z=torch.randn(2,3,2,dtype=torch.float64,requires_grad=True)
        actual=history_response_plus_local_drive(a,b,z)
        expected=serial_oracle(a,b,z)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        for ga,ge in zip(torch.autograd.grad(actual.square().sum(),(a,b,z),retain_graph=True),torch.autograd.grad(expected.square().sum(),(a,b,z))):
            torch.testing.assert_close(ga,ge,atol=1e-11,rtol=1e-11)
        gb=torch.autograd.grad(actual[:,-1].sum(),b)[0]
        torch.testing.assert_close(gb[:,:-1],torch.zeros_like(gb[:,:-1]),atol=0,rtol=0)
        torch.testing.assert_close(gb[:,-1],torch.ones_like(gb[:,-1]),atol=0,rtol=0)

    def test_only_finest_forecast_changes_and_normalization_once(self):
        base=Control(config()).eval(); candidate=Model(config()).eval()
        candidate.load_state_dict(base.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in base.parameters()),sum(p.numel() for p in candidate.parameters()))
        x=torch.randn(3,16,2)*2+4
        def intervened(encoder,*args,**kwargs):
            if encoder is base.seasonals[0]: kwargs['scan_function']=serial_oracle
            return calibrated_seasonal_forecast(encoder,*args,**kwargs)
        with torch.no_grad(),patch('models.OscillatorTimeMix.calibrated_seasonal_forecast',side_effect=intervened):
            expected=base(x)
        with torch.no_grad(),patch.object(candidate.normalize,'forward',wraps=candidate.normalize.forward) as norm:
            actual=candidate(x)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
        self.assertEqual([c.args[1] for c in norm.call_args_list],['norm','denorm'])
        with torch.no_grad():
            s,t=base.decomposition(base.normalize(x,'norm'))
            control_scales=base.predict_scales(s,t)
            changed_scales=candidate.predict_scales(s,t)
        torch.testing.assert_close(control_scales[...,1:],changed_scales[...,1:],atol=0,rtol=0)
        self.assertGreater(float(abs(control_scales[...,0]-changed_scales[...,0]).max()),1e-6)

    def test_finite_gradients_optimizer_and_matching_groups(self):
        model=Model(config(dropout=.1)); control=Control(config(dropout=.1))
        def audit(m):
            groups=parameter_groups(m,'control')
            by_id={id(p):g['weight_decay'] for g in groups for p in g['params']}
            return {k:by_id[id(p)] for k,p in m.named_parameters()}
        self.assertEqual(audit(model),audit(control))
        optimizer=build_optimizer(parameter_groups(model,'finest_local_drive'),'finest_local_drive')
        before={k:p.clone() for k,p in model.named_parameters()}
        for _ in range(2):
            optimizer.zero_grad()
            loss=(model(torch.randn(3,16,2))-torch.randn(3,11,2)).square().mean()
            loss.backward()
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            optimizer.step()
        for name in ['seasonals.0.decoder.log_decay','seasonals.0.temporal_projection.weight','seasonals.0.decoder.drive_projection.weight','trend_heads.0.weight']:
            self.assertFalse(torch.equal(before[name],dict(model.named_parameters())[name]),name)

    def test_batch_future_ms_and_strict_roundtrip(self):
        model=Model(config()).eval(); x=torch.randn(3,16,2)
        with torch.no_grad():
            y=model(x)
            torch.testing.assert_close(y,torch.cat([model(v[None]) for v in x]),atol=3e-6,rtol=1e-5)
            nan=torch.full((3,11,2),float('nan'))
            torch.testing.assert_close(y,model(x,nan,nan,nan),atol=0,rtol=0)
            restored=Model(config()).eval(); restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
            torch.testing.assert_close(y,restored(x),atol=0,rtol=0)
            ms=Model(config(features='MS',c_out=1)).eval(); ms.load_state_dict(model.state_dict(),strict=True)
            torch.testing.assert_close(ms(x),y[...,-1:],atol=0,rtol=0)
            self.assertTrue(torch.isfinite(model(torch.ones_like(x)*4)).all())
        with self.assertRaises(ValueError): dataset(None,'test',0)

if __name__=='__main__': unittest.main()
