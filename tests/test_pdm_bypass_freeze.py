import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from test_oscillator_mixer import config
from scripts.pdm_bypass_freeze_study import (ARMS, initialize_pair, parameter_groups,
    targeted, raw_state, bypass_audit, PredictionGroupCapture)
from scripts.train_timemix_pdm_trend_smooth_residual import initialize_pair as old_pair
from utils.pdm_trend_smooth_residual_diagnostics import SelectedValidationCapture

class BypassFreezeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=2,dropout=.2)
        self.cfg.seasonal_normalize=False
        torch.manual_seed(2021)
        self.models=initialize_pair(self.cfg)

    def test_exact_initial_state_rng_and_corresponding_predictions(self):
        rng=torch.get_rng_state();torch.manual_seed(2021);old=old_pair(self.cfg)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        x=torch.randn(2,96,2)
        for a,m in self.models.items():
            base=old[a.removesuffix('_fixed')]
            self.assertEqual(m.state_dict().keys(),base.state_dict().keys())
            for k,v in base.state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[k]))
            for training in [True,False]:
                m.train(training);base.train(training)
                torch.manual_seed(11);p=m(x);s=torch.get_rng_state()
                torch.manual_seed(11);q=base(x)
                torch.testing.assert_close(p,q,rtol=0,atol=2e-6 if a.endswith('_fixed') else 0)
                self.assertTrue(torch.equal(s,torch.get_rng_state()))

    def test_optimizer_coverage_and_freeze_scope(self):
        for a,m in self.models.items():
            groups=parameter_groups(m,a);ids=[id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(set(ids),{id(p) for p in m.parameters() if p.requires_grad})
            for n,p in m.named_parameters():
                self.assertEqual(p.requires_grad,not(a.endswith('_fixed') and targeted(n)))
            self.assertEqual(sum(p.numel() for n,p in m.named_parameters() if targeted(n)),36144)

    def test_fixed_function_unchanged_and_remaining_parameters_train(self):
        x=torch.randn(2,96,2);y=torch.randn_like(x)
        for a in ARMS[2:]:
            m=self.models[a];initial=raw_state(m)
            other={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
            m.eval()
            with torch.no_grad():before=m.predict_components(x)[...,8:].clone()
            opt=torch.optim.Adam(parameter_groups(m,a),lr=.001)
            for step in range(2):
                m.train();opt.zero_grad(set_to_none=True);(m(x)-y).square().mean().backward();opt.step()
                self.assertEqual(bypass_audit(m,initial,True)['distance_from_initial'],0)
                self.assertTrue(all(p not in opt.state for n,p in m.named_parameters() if targeted(n)))
            m.eval()
            with torch.no_grad():after=m.predict_components(x)[...,8:]
            torch.testing.assert_close(before,after,rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(other[n],p.detach()) for n,p in m.named_parameters() if n in other))

    def test_adaptive_controls_reproduce_old_updates(self):
        torch.manual_seed(2021);old=old_pair(self.cfg)
        x=torch.randn(2,96,2);y=torch.randn_like(x)
        for a in ARMS[:2]:
            m=self.models[a];o=old[a]
            for model in [m,o]:
                model.train();torch.manual_seed(98)
                opt=torch.optim.Adam(parameter_groups(model,a),lr=.001)
                (model(x)-y).square().mean().backward();opt.step()
            for k,v in m.state_dict().items():torch.testing.assert_close(v,o.state_dict()[k],rtol=0,atol=0)

    def test_diagnostic_wrapper_keeps_forward_backward_rng(self):
        m=self.models['smooth_residual_fixed'];x=torch.randn(2,96,2)
        torch.manual_seed(19);p=m(x);p.square().mean().backward()
        grads={n:v.grad.clone() for n,v in m.named_parameters() if v.requires_grad};rng=torch.get_rng_state()
        m.zero_grad(set_to_none=True);torch.manual_seed(19)
        with PredictionGroupCapture(m) as cap:q=m(x)
        q.square().mean().backward()
        torch.testing.assert_close(p,q,rtol=0,atol=0);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for n,v in m.named_parameters():
            if v.requires_grad:torch.testing.assert_close(v.grad,grads[n],rtol=0,atol=0)
        self.assertEqual(np.array(cap.summary['rms_per_channel']).shape,(2,3))
        self.assertNotIn('predict_components',m.__dict__)

    def test_selected_capture_and_corruption_detection(self):
        m=self.models['smooth_residual_fixed'].eval();initial=raw_state(m);x=torch.randn(2,96,2)
        with tempfile.TemporaryDirectory() as folder:
            with SelectedValidationCapture(m,Path(folder),'smooth_residual_fixed',2) as cap:
                with torch.no_grad():p=m(x).numpy()
            report=cap.report(p,np.zeros_like(p))
            self.assertLess(report['reconstruction_max_abs_error'],2e-5)
        with torch.no_grad():m.raw_trend_heads[0].bias.add_(1)
        with self.assertRaises(RuntimeError):bypass_audit(m,initial,True)

if __name__=='__main__':unittest.main()
