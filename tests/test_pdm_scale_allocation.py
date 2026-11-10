"""Numerical, gradient, and stopping contracts for scale allocation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import TensorDataset
from test_oscillator_mixer import config
from models.OscillatorFeaturePDMTrendBypass import Model as Previous
from models.OscillatorPDMScaleAllocation import Model
import models.OscillatorPDMScaleAllocation as module
from scripts import train_timemix_pdm_scale_allocation as train
from scripts.train_timemix_pdm_trend_bypass import initialize_pair as previous_pair


class ScaleAllocationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2021)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,dropout=.2)
        self.cfg.e_layers=2;self.cfg.seasonal_normalize=False

    def test_shared_initialization_and_exact_baseline_and_gates(self):
        torch.manual_seed(2021);ref=previous_pair(self.cfg)['pdm_trend_bypass']
        torch.manual_seed(2021);models=train.initialize_pair(self.cfg)
        for model in models.values():
            for key,value in ref.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        x=torch.randn(2,96,2)
        for mode in [False,True]:
            ref.train(mode);torch.manual_seed(31);expected=ref(x);state=torch.get_rng_state()
            for key in ['bypass','channel_weights']:
                model=models[key].train(mode);torch.manual_seed(31)
                torch.testing.assert_close(expected,model(x),rtol=0,atol=0)
                self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_drop_is_only_final_readout_and_normalize_once(self):
        models=train.initialize_pair(self.cfg);base=models['bypass'].eval();drop=models['drop_pdm12'].eval();x=torch.randn(2,96,2)
        original=module.calibrated_seasonal_forecast;outputs=[]
        def capture(*a,**kw):
            y=original(*a,**kw);outputs.append(y);return y
        calls=[];h=drop.normalize.register_forward_pre_hook(lambda m,args:calls.append(args[1]))
        with patch.object(module,'calibrated_seasonal_forecast',side_effect=capture):expected=base(x)-outputs[3]*base.normalize.stdev
        torch.testing.assert_close(drop(x),expected,rtol=1e-5,atol=1e-6)
        self.assertEqual(calls,['norm','denorm']);h.remove()

    def test_weights_broadcast_per_channel_and_finite_gradient(self):
        m=Model(self.cfg,variant='channel_weights').eval();x=torch.randn(2,96,2)
        original=module.calibrated_seasonal_forecast;outputs=[]
        def capture(*a,**kw):
            y=original(*a,**kw);outputs.append(y);return y
        with patch.object(module,'calibrated_seasonal_forecast',side_effect=capture):base=m(x)
        with torch.no_grad():m.scale_gate_logits[3,1]=np.log(3.)
        expected=base.clone();expected[:,:,1]+=0.5*outputs[3][:,:,1]*m.normalize.stdev[:,:,1]
        torch.testing.assert_close(m(x),expected,rtol=1e-5,atol=1e-6)
        for variant in train.ARMS:
            m=Model(self.cfg,variant=variant);(m(x)-torch.randn_like(x)).square().mean().backward()
            self.assertEqual([n for n,p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()],[])
            if variant=='channel_weights':
                self.assertGreater(float(m.scale_gate_logits.grad.abs().sum()),0.)
                groups=train.parameter_groups(m,variant)
                self.assertEqual(next(g['weight_decay'] for g in groups if any(p is m.scale_gate_logits for p in g['params'])),.0001)

    def test_serialization_and_no_future_inputs(self):
        m=Model(self.cfg,variant='channel_weights').eval();x=torch.randn(2,96,2)
        cfg=copy.copy(self.cfg);cfg.scale_variant='channel_weights';other=Model(cfg).eval();other.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(m(x),other(x),rtol=0,atol=0)
        torch.testing.assert_close(m(x),m(x,None,torch.randn_like(x),None),rtol=0,atol=0)

    def test_three_arm_driver_joint_stop(self):
        cfg=copy.copy(self.cfg);cfg.enc_in=cfg.c_out=2
        data=TensorDataset(torch.randn(4,96,2),torch.randn(4,96,2),torch.zeros(4,96,1),torch.zeros(4,96,1))
        framework=train.framework_args
        def settings(args):
            actual=framework(args);vars(actual).update(vars(cfg));return actual
        vals=iter([.1]*3+[.2]*3+[.3]*3)
        def validation(*args):
            v=next(vals);return dict(neural_mse=v,neural_mae=v)
        def selected(*args,**kwargs):
            p=np.full((4,96,2),np.sqrt(.1));return dict(mse=.1),p,np.zeros_like(p)
        with tempfile.TemporaryDirectory(prefix='scale_earlystop_',dir=Path(train.ROOT)/'tmp') as tmp:
            out=Path(tmp)/'training';argv=['train','--output',str(out),'--device','cpu','--epochs','8','--patience','2','--train-limit','4','--val-limit','4']
            with patch('sys.argv',argv),patch.object(train,'framework_args',side_effect=settings),patch.object(train,'dataset',return_value=data),patch.object(train,'evaluate_validation',side_effect=validation),patch.object(train,'evaluate',side_effect=selected):train.main()
            p=json.loads((out/'2021/progress.json').read_text());report=json.loads((out/'report.json').read_text())
            self.assertEqual(p['epoch'],3);self.assertEqual(p['stale'],2)
            self.assertEqual(p['best_epoch'],{name:1 for name in train.ARMS})
            self.assertEqual(len(report['runs']),3)
            self.assertTrue(all(v['stop_reason']=='joint_patience' for v in report['runs']))
            ck=torch.load(out/'2021/last.pt',weights_only=True);self.assertEqual(ck['completed_updates'],3)
