import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_readout_split import initialize_pair as old_pair
from scripts.train_timemix_pdm_trend_prelinear import initialize_pair, parameter_groups, dataset, training_losses, learning_rate, joint_stale_count
from utils.pdm_trend_prelinear_diagnostics import SelectedValidationCapture, prelinear_parameter_audit


class TrendPreLinearStudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = config(seq_len=96, pred_len=96, moving_avg=25, e_layers=2, dropout=.2)
        self.cfg.seasonal_normalize = False

    def test_exact_old_control_common_states_and_optimizer_policy(self):
        torch.manual_seed(2021)
        expected = old_pair(self.cfg)['seasonal_trend']
        torch.manual_seed(2021)
        pair = initialize_pair(self.cfg)
        a,b = pair.values()
        for key,value in expected.state_dict().items():
            self.assertTrue(torch.equal(value,a.state_dict()[key]),key)
            self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
        for arm,m in pair.items():
            groups = parameter_groups(m,arm)
            params = [p for g in groups for p in g['params']]
            self.assertEqual(len(params),len({id(p) for p in params}))
            self.assertEqual({id(p) for p in params},{id(p) for p in m.parameters()})
            rates = {id(p):g['weight_decay'] for g in groups for p in g['params']}
            for name,p in m.named_parameters():
                if 'pre_linears' in name:
                    self.assertEqual(rates[id(p)],0 if name.endswith('bias') else 1e-4)

    def test_same_forward_capture_and_transform_statistics(self):
        for arm,m in initialize_pair(self.cfg).items():
            m.eval()
            x = torch.randn(3,96,2)
            with torch.no_grad():expected=m(x).numpy()
            with tempfile.TemporaryDirectory() as folder:
                with SelectedValidationCapture(m,Path(folder),arm,3) as cap:
                    with torch.no_grad():actual=m(x).numpy()
                report=cap.report(actual,np.zeros_like(actual))
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_allclose(cap.components.sum(-1)+cap.means,actual,rtol=0,atol=2e-5)
                self.assertEqual(cap.components.shape[-1],12)
                self.assertEqual(len(report['prelinear_transforms']),8 if arm=='trend_prelinear' else 0)
                for stats in report['prelinear_transforms'].values():
                    self.assertLess(stats['change_rms'],2e-6)
                self.assertIsNone(cap.model)
                self.assertFalse(any(mod._forward_hooks for mod in m.modules()))

    def test_pure_mse_gradient_trace_and_selection_rules(self):
        m=initialize_pair(self.cfg)['trend_prelinear']
        x=torch.randn(2,96,2);target=torch.randn_like(x)
        cfg={'offsets':[0,1,2,4,8,12,24,48,96], 'variance_floor':.01, 'jitter':.01}
        prediction=m(x)
        loss=training_losses(prediction,target,'trend_prelinear',cfg,x)
        torch.testing.assert_close(loss['loss'],(prediction-target).square().mean(),rtol=0,atol=0)
        self.assertEqual(loss['auxiliary_kl'].item(),0)
        loss['loss'].backward()
        trace=prelinear_parameter_audit(m)
        self.assertEqual(len(trace),8)
        self.assertTrue(all(v['weight_gradient_norm']>0 for v in trace.values()))
        self.assertEqual(joint_stale_count(4,True),0)
        self.assertEqual(joint_stale_count(4,False),5)
        self.assertEqual(learning_rate('trend_prelinear',30,30),0)
        with self.assertRaises(ValueError):dataset(self.cfg,'test',0)


if __name__=='__main__':
    unittest.main()
