"""Paired optimizer, initialization and same-forward capture checks."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_no_history_pe import initialize_pair, parameter_groups, training_losses, loss_configuration
from utils.pdm_no_history_pe_diagnostics import SelectedValidationCapture
from types import SimpleNamespace


class SingleFFTStudyTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.cfg = config(seq_len=96,pred_len=96,moving_avg=25,e_layers=2,dropout=.2,
                          down_sampling_layers=3,down_sampling_window=2)
        self.cfg.seasonal_normalize=False
        torch.manual_seed(2021)
        self.models=initialize_pair(self.cfg)
    def tearDown(self):
        torch.set_num_threads(self.threads)
    def test_common_initial_state_and_all_gradients_and_optimizer_coverage(self):
        ref=self.models['single_fft'].state_dict()
        for key,value in self.models['no_history_pe'].state_dict().items():
            torch.testing.assert_close(value,ref[key],rtol=0,atol=0)
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        loss_cfg=loss_configuration(SimpleNamespace(variance_floor=.01,kl_jitter=.01))
        for name,m in self.models.items():
            groups=parameter_groups(m,name)
            ids=[id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(set(ids),{id(p) for p in m.parameters()})
            losses=training_losses(m(x),y,name,loss_cfg,x)
            torch.testing.assert_close(losses['loss'],losses['mse'],rtol=0,atol=0)
            losses['loss'].backward()
            for p in m.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
    def test_same_forward_capture_and_reconstruction(self):
        x=torch.randn(3,96,2)
        with tempfile.TemporaryDirectory() as tmp:
            for name,m in self.models.items():
                m.eval()
                with torch.no_grad():expected=m(x)
                with SelectedValidationCapture(m,Path(tmp),name,3) as cap:
                    with torch.no_grad():actual=m(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                report=cap.report(actual.numpy(),np.zeros_like(actual.numpy()))
                self.assertEqual(len(report['component_order']),8)
                self.assertLessEqual(report['reconstruction_max_abs_error'],2e-5)
                restored=np.asarray(cap.components,dtype='float64').sum(-1)+cap.means
                np.testing.assert_allclose(restored,actual.numpy(),rtol=0,atol=2e-5)
                self.assertEqual(report['initial_split_calls'],4)
                self.assertTrue(all(v['series']==6 for v in report['fft_routing'].values()))


    def test_no_history_positions_matches_zero_position_control(self):
        control=self.models['single_fft'].eval();treatment=self.models['no_history_pe'].eval()
        for enc in control.seasonals:enc.positions.zero_()
        x=torch.randn(2,96,2)
        with torch.no_grad():
            torch.testing.assert_close(control(x),treatment(x),rtol=0,atol=0)

    def test_history_position_buffer_unused_future_position_retained(self):
        model=self.models['no_history_pe'].eval();x=torch.randn(2,96,2)
        with torch.no_grad():
            before=model(x)
            for enc in model.seasonals:enc.positions.normal_(0,100)
            torch.testing.assert_close(model(x),before,rtol=0,atol=0)
            for enc in model.seasonals:enc.future_positions.zero_()
            self.assertFalse(torch.equal(model(x),before))

    def test_initial_split_receives_only_embedded_observations(self):
        model=self.models['no_history_pe'].eval();embedded=[];inputs=[];handles=[]
        for enc in model.seasonals:
            handles.append(enc.value_embedding.register_forward_hook(lambda m,args,out:embedded.append(out.detach().clone())))
        handles.append(model.initial_decomposition.register_forward_pre_hook(lambda m,args:inputs.append(args[0].detach().clone())))
        with torch.no_grad():model(torch.randn(2,96,2))
        for h in handles:h.remove()
        self.assertEqual(len(inputs),4)
        for x,y in zip(embedded,inputs):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in self.models['single_fft'].parameters()))

if __name__=='__main__':unittest.main()

