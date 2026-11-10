import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_scaled_ma import (ARMS, initialize_pair, parameter_groups, joint_stale_count, learning_rate, dataset)
from utils.pdm_scaled_ma_diagnostics import SelectedValidationCapture


class ScaledMAStudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2021)
        self.cfg = config(seq_len=96, pred_len=96, moving_avg=25, dropout=.2)
        self.cfg.e_layers = 2
        self.cfg.seasonal_normalize = False

    def test_common_initialization_and_optimizer_policy(self):
        models = initialize_pair(self.cfg)
        reference = models['ma25'].state_dict()
        audits = []
        for arm, model in models.items():
            self.assertEqual(set(reference), set(model.state_dict()))
            for key, value in reference.items():
                torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
            groups = parameter_groups(model, arm)
            decay = {id(p): g['weight_decay'] for g in groups for p in g['params']}
            audits.append({k:decay[id(p)] for k,p in model.named_parameters()})
        self.assertTrue(all(a == audits[0] for a in audits))
        self.assertEqual(joint_stale_count(4, True), 0)
        self.assertEqual(joint_stale_count(4, False), 5)
        self.assertEqual(learning_rate(ARMS[0], 30, 30), 0.)
        with self.assertRaises(ValueError):
            dataset(self.cfg, 'test', 0)

    def test_saved_branch_reconstruction_and_all_scales(self):
        model = initialize_pair(self.cfg)['scaled_ma'].eval()
        x = torch.randn(3, 96, self.cfg.enc_in)
        with tempfile.TemporaryDirectory() as folder:
            with SelectedValidationCapture(model, Path(folder), 'scaled_ma', 3) as capture:
                with torch.no_grad():
                    p = torch.cat([model(x[:2]), model(x[2:])]).numpy()
            result = capture.report(p, np.zeros_like(p))
            self.assertLess(result['reconstruction_max_abs_error'], 2e-5)
            self.assertEqual(len(result['all_scale_decomposition']), 8)
            self.assertIsNone(capture.model)
            self.assertEqual(capture.cache, [])
            reconstructed = np.asarray(capture.components).sum(-1) + np.asarray(capture.means)
            np.testing.assert_allclose(reconstructed, p, rtol=0, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
