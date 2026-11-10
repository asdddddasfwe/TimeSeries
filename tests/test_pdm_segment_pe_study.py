"""Paired segment PE initialization, optimizer and capture checks."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_segment_pe import initialize_pair, parameter_groups, training_losses, loss_configuration
from utils.pdm_segment_pe_diagnostics import SelectedValidationCapture


class SegmentPEStudyTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.cfg = config(seq_len=96,pred_len=96,moving_avg=25,e_layers=2,dropout=.2,
                          down_sampling_layers=3,down_sampling_window=2)
        self.cfg.seasonal_normalize = False
        torch.manual_seed(2021)
        self.models = initialize_pair(self.cfg)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_all_initial_tensors_and_optimizer_coverage(self):
        reference = self.models['no_history_pe'].state_dict()
        self.assertEqual(set(reference), set(self.models['segment_pe'].state_dict()))
        for k,v in self.models['segment_pe'].state_dict().items():
            torch.testing.assert_close(v,reference[k],rtol=0,atol=0)
        for arm,model in self.models.items():
            groups = parameter_groups(model,arm)
            ids = [id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(set(ids),{id(p) for p in model.parameters()})

    def test_pure_mse_and_finite_gradients(self):
        x,y = torch.randn(2,96,2),torch.randn(2,96,2)
        cfg = loss_configuration(SimpleNamespace(variance_floor=.01,kl_jitter=.01))
        for arm,model in self.models.items():
            losses = training_losses(model(x),y,arm,cfg,x)
            torch.testing.assert_close(losses['loss'],losses['mse'],rtol=0,atol=0)
            losses['loss'].backward()
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_capture_preserves_forward_and_records_both_routing_paths(self):
        x = torch.randn(3,96,2)
        with tempfile.TemporaryDirectory() as directory:
            for arm,model in self.models.items():
                model.eval()
                with torch.no_grad():expected = model(x)
                with SelectedValidationCapture(model,Path(directory),arm,3) as capture:
                    with torch.no_grad():actual = model(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                report = capture.report(actual.numpy(),np.zeros_like(actual.numpy()))
                self.assertEqual(len(report['component_order']),8)
                self.assertLess(report['reconstruction_max_abs_error'],2e-5)
                self.assertFalse(report['history_position_added'])
                self.assertEqual(report['segment_position_added'],arm=='segment_pe')
                self.assertTrue(report['future_position_added'])
                self.assertEqual(set(report['fft_routing']),{'96','48','24','12'})
                for row in report['fft_routing'].values():
                    self.assertEqual(row['series'],6)
                    self.assertEqual(sum(row['top1_counts'].values()),6)
                    self.assertAlmostEqual(sum(row['weighted_period_mass'].values()),6,places=5)


if __name__ == '__main__':unittest.main()
