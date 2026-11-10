"""Paired optimizer, initialization and same-forward capture checks."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_single_fft import initialize_pair, parameter_groups, training_losses, loss_configuration
from utils.pdm_single_fft_diagnostics import SelectedValidationCapture
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
        ref=self.models['seasonal_trend'].state_dict()
        for key,value in self.models['single_fft'].state_dict().items():
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
                self.assertEqual(len(report['component_order']),8 if name=='single_fft' else 12)
                self.assertLessEqual(report['reconstruction_max_abs_error'],2e-5)
                restored=np.asarray(cap.components,dtype='float64').sum(-1)+cap.means
                np.testing.assert_allclose(restored,actual.numpy(),rtol=0,atol=2e-5)
                if name=='single_fft':self.assertEqual(report['initial_split_calls'],4)

if __name__=='__main__':unittest.main()
