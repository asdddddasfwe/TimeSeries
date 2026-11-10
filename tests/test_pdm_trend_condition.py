"""Trend-context routing, identical initialization, gradients and capture."""
import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_trend_condition import initialize_pair, parameter_groups
from scripts.train_timemix_pdm_segment_pe import initialize_pair as previous_pair
from utils.pdm_trend_condition_diagnostics import SelectedValidationCapture


class TrendConditionTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=2,dropout=.2,
                   down_sampling_layers=3,down_sampling_window=2)
        cfg.seasonal_normalize=False
        self.cfg=cfg;torch.manual_seed(2021);self.models=initialize_pair(cfg)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_common_initial_state_and_control_reproduction(self):
        c,t=self.models.values();source=c.state_dict()
        for k,v in t.state_dict().items():
            if k in source:torch.testing.assert_close(v,source[k],rtol=0,atol=0)
            else:self.assertTrue(k.endswith('.trend_condition.weight'));self.assertEqual(torch.count_nonzero(v),0)
        extra=sum(p.numel() for n,p in t.named_parameters() if n.endswith('.trend_condition.weight'))
        self.assertEqual(extra,4*2*self.cfg.d_model*2*self.cfg.ssm_modes)
        torch.manual_seed(2021);old=previous_pair(self.cfg)['segment_pe']
        for k,v in old.state_dict().items():torch.testing.assert_close(source[k],v,rtol=0,atol=0)

    def test_initial_train_eval_function_rng_and_backward(self):
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        for train in (False,True):
            outputs=[];rng=[]
            for m in self.models.values():
                m.train(train);torch.manual_seed(42);outputs.append(m(x));rng.append(torch.get_rng_state())
            torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=2e-6)
            self.assertTrue(torch.equal(*rng))
        (outputs[1]-y).square().mean().backward()
        for n,p in self.models['trend_condition'].named_parameters():
            self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
            if n.endswith('.trend_condition.weight'):self.assertGreater(float(p.grad.norm()),0.)

    def test_trend_changes_only_dynamics_entry_and_gradient_reaches_trend(self):
        d=self.models['trend_condition'].seasonals[0].decoder.eval();width=d.drive_projection.in_features
        h=torch.randn(2,d.length,width);t=torch.randn_like(h,requires_grad=True)
        with torch.no_grad():d.trend_condition.weight.normal_(0,.02)
        a=d.dynamics(h,t);b=d.dynamics(h,t.flip(1))
        self.assertFalse(torch.equal(a[0],b[0]));self.assertTrue((a[2]>0).all() and (a[2]<1).all())
        a[0].square().sum().backward();self.assertGreater(float(t.grad.norm()),0.)
        with self.assertRaises(ValueError):d.dynamics(h)

    def test_optimizer_covers_all_and_preserves_common_decay(self):
        maps={}
        for arm,m in self.models.items():
            gs=parameter_groups(m,arm);ids=[id(p) for g in gs for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.parameters()})
            wd={id(p):g['weight_decay'] for g in gs for p in g['params']};maps[arm]={n:wd[id(p)] for n,p in m.named_parameters()}
        for n,wd in maps['segment_pe'].items():self.assertEqual(wd,maps['trend_condition'][n])
        for n,wd in maps['trend_condition'].items():
            if n.endswith('.trend_condition.weight'):self.assertEqual(wd,0.)

    def test_capture_no_extra_forward_and_valid_reconstruction(self):
        x=torch.randn(2,96,2)
        with tempfile.TemporaryDirectory() as directory:
            for arm,m in self.models.items():
                m.eval()
                with torch.no_grad():expected=m(x)
                with SelectedValidationCapture(m,Path(directory),arm,2) as cap:
                    with torch.no_grad():actual=m(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                audit=cap.report(actual.numpy(),np.zeros_like(actual.numpy()))
                self.assertLess(audit['reconstruction_max_abs_error'],2e-5)
                self.assertEqual(audit['initial_split_calls'],4)
                self.assertEqual(len(audit['component_order']),8)
                self.assertTrue(audit['segment_position_added']);self.assertTrue(audit['future_position_added'])
                self.assertEqual(set(audit['fft_routing']),{'96','48','24','12'})
                if arm=='trend_condition':
                    self.assertEqual(len(audit['trend_logit_shift']),4)
                    self.assertTrue(all(v['decay_rms']==v['frequency_rms']==0 for v in audit['trend_logit_shift'].values()))


if __name__=='__main__':unittest.main()
