import tempfile
import unittest
from pathlib import Path
import torch
import numpy as np
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_seasonal_head import initialize_pair, parameter_groups, dataset, learning_rate, joint_stale_count
from utils.pdm_seasonal_head_diagnostics import SelectedValidationCapture

class SeasonalHeadStudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2021)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=2,dropout=.2)
        self.cfg.seasonal_normalize=False

    def test_common_initialization_policy_and_selection(self):
        models=initialize_pair(self.cfg)
        a,b=models.values()
        for k,v in a.state_dict().items():
            torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        policies=[]
        for name,model in models.items():
            groups=parameter_groups(model,name)
            ids=[id(v) for g in groups for v in g['params']]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(set(ids),{id(v) for v in model.parameters()})
            rates={id(v):g['weight_decay'] for g in groups for v in g['params']}
            policies.append({k:rates[id(v)] for k,v in model.named_parameters()})
        for k,v in policies[0].items(): self.assertEqual(v,policies[1][k])
        self.assertEqual(joint_stale_count(4,False),5)
        self.assertEqual(joint_stale_count(4,True),0)
        self.assertEqual(learning_rate('seasonal_head',30,30),0)
        with self.assertRaises(ValueError):dataset(self.cfg,'test',0)

    def test_same_forward_capture_and_cleanup_both_arms(self):
        for name,model in initialize_pair(self.cfg).items():
            model.eval();x=torch.randn(3,96,self.cfg.enc_in)
            with torch.no_grad(): expected=model(x).numpy()
            with tempfile.TemporaryDirectory() as folder:
                with SelectedValidationCapture(model,Path(folder),name,3) as capture:
                    with torch.no_grad():p=torch.cat([model(x[:2]),model(x[2:])]).numpy()
                np.testing.assert_allclose(p,expected,rtol=1e-5,atol=2e-5)
                result=capture.report(p,np.zeros_like(p))
                self.assertEqual(len(result['component_order']),12 if name=='seasonal_head' else 8)
                np.testing.assert_allclose(capture.components.sum(-1)+capture.means,p,rtol=0,atol=2e-5)
                self.assertIsNone(capture.model)
                self.assertEqual(capture.cache,[])
                self.assertIsNone(capture.patch)
                self.assertEqual(len(result['all_scale_decomposition']),8)

if __name__=='__main__':unittest.main()
