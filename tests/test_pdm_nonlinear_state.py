"""Paired study controls, same-forward diagnostics, and correction gradients."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from test_oscillator_mixer import config
from scripts import train_timemix_pdm_nonlinear_state as study
from utils.pdm_nonlinear_state_diagnostics import SolverCapture, SelectedValidationCapture


class NonlinearStudyTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=1,dropout=.2,
                        down_sampling_layers=3,down_sampling_window=2)
        self.cfg.seasonal_normalize=False
        torch.manual_seed(2021);self.models=study.initialize_pair(self.cfg)
    def tearDown(self):torch.set_num_threads(self.threads)

    def test_initial_functions_rng_and_new_parameters(self):
        a,b=self.models.values();sa,sb=a.state_dict(),b.state_dict()
        self.assertEqual(len(set(sb)-set(sa)),16)
        for k,v in sa.items():torch.testing.assert_close(v,sb[k],rtol=0,atol=0)
        x=torch.randn(2,96,2)
        for training in (False,True):
            outputs=[];rng=[]
            for model in self.models.values():
                model.train(training);torch.manual_seed(77);outputs.append(model(x));rng.append(torch.get_rng_state())
            torch.testing.assert_close(*outputs,rtol=0,atol=0);self.assertTrue(torch.equal(*rng))
        for arm,model in self.models.items():
            cfg=study.arm_config(self.cfg,arm);restored=study.VARIANTS[arm](cfg)
            restored.load_state_dict(model.state_dict(),strict=True)
            self.assertEqual(cfg.learning_rate,.001)
            self.assertEqual(cfg.nonlinear_iterations,4)

    def test_optimizer_and_mse_gradients(self):
        policies={}
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        cfg=study.loss_configuration(SimpleNamespace(variance_floor=.01,kl_jitter=.01))
        for arm,model in self.models.items():
            groups=study.parameter_groups(model,arm);byid={id(p):g['weight_decay'] for g in groups for p in g['params']}
            policies[arm]={k:byid[id(p)] for k,p in model.named_parameters()}
            opt=study.build_optimizer(groups,arm)
            for step in range(2):
                opt.zero_grad();p=model(x);parts=study.training_losses(p,y,arm,cfg,x)
                torch.testing.assert_close(parts['loss'],(p-y).square().mean(),rtol=0,atol=0)
                self.assertEqual(parts['auxiliary_kl'],0);parts['loss'].backward()
                for k,param in model.named_parameters():
                    self.assertIsNotNone(param.grad,k);self.assertTrue(torch.isfinite(param.grad).all(),k)
                    if '.state_correction.input.weight' in k and step==1:self.assertGreater(float(param.grad.norm()),0)
                opt.step()
        for k,v in policies['oscillator'].items():self.assertEqual(v,policies['nonlinear4'][k])
        for k,v in policies['nonlinear4'].items():
            if k not in policies['oscillator']:self.assertEqual(v,0. if k.endswith('.bias') else 1e-4)
        for epoch in range(31):self.assertEqual(study.learning_rate('oscillator',epoch,30),study.learning_rate('nonlinear4',epoch,30))

    def test_diagnostics_do_not_change_prediction_gradient_rng(self):
        model=self.models['nonlinear4']
        with torch.no_grad():
            for encoder in model.seasonals:encoder.decoder.state_correction.output.weight.normal_(0,.003)
        x=torch.randn(2,96,2);values=[];gradients=[];rng=[]
        for diagnostics in (False,True):
            model.zero_grad();torch.manual_seed(52)
            if diagnostics:
                with SolverCapture(model,reference_windows=2) as capture:p=model(x)
                report=capture.report();self.assertEqual(len(report),4)
                for v in report.values():self.assertEqual(v['reference_series'],4);self.assertGreaterEqual(v['residual_max'],0)
            else:p=model(x)
            p.square().mean().backward();values.append(p.detach());rng.append(torch.get_rng_state())
            gradients.append({k:p.grad.clone() for k,p in model.named_parameters()})
        torch.testing.assert_close(*values,rtol=0,atol=0);self.assertTrue(torch.equal(*rng))
        for k,v in gradients[0].items():torch.testing.assert_close(v,gradients[1][k],rtol=0,atol=0)

    def test_capture_reconstructs_nonlinear_and_solver_statistics(self):
        model=self.models['nonlinear4'].eval();x=torch.randn(2,96,2)
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            with SelectedValidationCapture(model,Path(tmp),'nonlinear4',2) as capture:p=model(x)
            report=capture.report(p.numpy(),p.numpy())
            self.assertLess(report['reconstruction_max_abs_error'],2e-5)
            self.assertEqual(report['seasonal_decoder'],'nonlinear_history_corrected_oscillator')
            self.assertEqual(report['nonlinear_iterations'],4)
            self.assertEqual(len(report['solver_diagnostics']),4)
            self.assertEqual(sum(v['reference_series'] for v in report['solver_diagnostics'].values()),16)

if __name__=='__main__':unittest.main()
