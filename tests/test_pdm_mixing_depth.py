"""One/two mixing depth: retained initialization, restoration, gradients and capture."""
import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from test_oscillator_mixer import config
from models.OscillatorPDMTrendCondition import Model
from scripts.train_timemix_pdm_mixing_depth import initialize_pair, parameter_groups, arm_config
from scripts.train_timemix_pdm_trend_condition import initialize_pair as prior_pair
from utils.pdm_mixing_depth_diagnostics import SelectedValidationCapture

class MixingDepthTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=2,dropout=.2,down_sampling_layers=3,down_sampling_window=2)
        self.cfg.seasonal_normalize=False
        torch.manual_seed(2021);self.models=initialize_pair(self.cfg)
    def tearDown(self):torch.set_num_threads(self.threads)
    def test_retained_initialization_and_old_control(self):
        c,t=self.models.values();self.assertEqual(len(c.pdm_blocks),2);self.assertEqual(len(t.pdm_blocks),1)
        removed=set(c.state_dict())-set(t.state_dict());self.assertTrue(removed)
        self.assertTrue(all(k.startswith('pdm_blocks.1.') for k in removed))
        for k,v in t.state_dict().items():torch.testing.assert_close(v,c.state_dict()[k],rtol=0,atol=0)
        torch.manual_seed(2021);old=prior_pair(self.cfg)['trend_condition']
        for k,v in old.state_dict().items():torch.testing.assert_close(v,c.state_dict()[k],rtol=0,atol=0)
    def test_config_restore_and_finite_backward(self):
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        for arm,m in self.models.items():
            restored=Model(arm_config(self.cfg,arm));restored.load_state_dict(m.state_dict(),strict=True)
            m.eval();restored.eval()
            with torch.no_grad():torch.testing.assert_close(m(x),restored(x),rtol=0,atol=0)
            m.train();loss=(m(x)-y).square().mean();loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for n,p in m.named_parameters():self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
    def test_optimizer_retained_policy_and_coverage(self):
        maps={}
        for arm,m in self.models.items():
            gs=parameter_groups(m,arm);ids=[id(p) for g in gs for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.parameters()})
            wd={id(p):g['weight_decay'] for g in gs for p in g['params']};maps[arm]={n:wd[id(p)] for n,p in m.named_parameters()}
            for n,v in maps[arm].items():
                if n.endswith('.trend_condition.weight'):self.assertEqual(v,0.)
        for n,v in maps['depth1'].items():self.assertEqual(v,maps['depth2'][n])
    def test_capture_depth_and_reconstruction_without_output_change(self):
        x=torch.randn(2,96,2)
        with tempfile.TemporaryDirectory() as directory:
            for arm,m in self.models.items():
                m.eval()
                with torch.no_grad():expected=m(x)
                with SelectedValidationCapture(m,Path(directory),arm,2) as cap:
                    with torch.no_grad():actual=m(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                a=cap.report(actual.numpy(),np.zeros_like(actual.numpy()))
                self.assertEqual(a['mixing_depth'],int(arm[-1]));self.assertEqual(a['initial_split_calls'],4)
                self.assertEqual(len(a['component_order']),8);self.assertLess(a['reconstruction_max_abs_error'],2e-5)
                self.assertEqual(len(a['stream_statistics']),int(arm[-1])*8)
                self.assertEqual(len(a['trend_logit_shift']),4);self.assertEqual(set(a['fft_routing']),{'96','48','24','12'})
                self.assertFalse(a['history_position_added']);self.assertTrue(a['segment_position_added'] and a['future_position_added'])

if __name__=='__main__':unittest.main()
