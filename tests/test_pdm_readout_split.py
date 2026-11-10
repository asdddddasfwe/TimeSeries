import tempfile
import unittest
from pathlib import Path
import torch
import numpy as np
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_readout_split import initialize_pair,parameter_groups,joint_stale_count,learning_rate,dataset
from utils.pdm_readout_split_diagnostics import SelectedValidationCapture

class ReadoutSplitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2021)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=2,dropout=.2)
        self.cfg.seasonal_normalize=False

    def test_matched_states_and_unchanged_pdm(self):
        models=initialize_pair(self.cfg)
        base,full,season=models.values()
        for k,v in base.state_dict().items():torch.testing.assert_close(v,full.state_dict()[k],rtol=0,atol=0)
        for k,v in full.state_dict().items():torch.testing.assert_close(v,season.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in full.parameters()),sum(p.numel() for p in season.parameters()))
        for x,y in zip(base.pdm_blocks,season.pdm_blocks):self.assertIs(type(x),type(y))
        for m in [full,season]:self.assertEqual(m.readout_decomposition.kernels,{96:25,48:13,24:7,12:3})

    def test_initial_full_control_matches_baseline(self):
        models=initialize_pair(self.cfg);x=torch.randn(2,96,2)
        for train in [False,True]:
            outputs=[]
            for m in models.values():
                m.train(train);torch.manual_seed(28)
                with torch.no_grad():outputs.append(m(x))
            torch.testing.assert_close(outputs[0],outputs[1],atol=2e-6,rtol=0)
            self.assertGreater((outputs[1]-outputs[2]).abs().max().item(),1e-5)
        for name in ['full_trend','seasonal_trend']:
            with torch.no_grad():self.assertEqual(models[name].predict_components(x)[...,4:8].count_nonzero(),0)

    def test_only_decoder_route_changes_and_split_reconstructs(self):
        _,full,season=initialize_pair(self.cfg).values();x=torch.randn(2,96,2)
        full.eval();season.eval()
        with torch.no_grad():
            _,hf=full.encode_multiscale(full.downsample(x));_,hs=season.encode_multiscale(season.downsample(x))
            for h,k in zip(hf,hs):torch.testing.assert_close(h,k,rtol=0,atol=0)
            rf,tf=full.decoder_histories(hf);rs,ts=season.decoder_histories(hs)
            for h,u,s,t,v in zip(hf,rf,rs,tf,ts):
                torch.testing.assert_close(h,u,rtol=0,atol=0)
                torch.testing.assert_close(t,v,rtol=0,atol=0)
                torch.testing.assert_close(s+t,h,atol=2e-6,rtol=0)

    def test_optimizer_coverage_finite_backward_and_new_head_learns(self):
        models=initialize_pair(self.cfg);policies=[];x=torch.randn(2,96,2);y=torch.randn_like(x)
        for name,m in models.items():
            groups=parameter_groups(m,name);ids=[id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.parameters()})
            rates={id(p):g['weight_decay'] for g in groups for p in g['params']};policies.append({k:rates[id(p)] for k,p in m.named_parameters()})
            opt=torch.optim.Adam(groups,lr=.001);(m(x)-y).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            if hasattr(m,'feature_trend_heads'):self.assertTrue(any(h.readout.weight.grad.abs().sum()>0 for h in m.feature_trend_heads))
            opt.step()
        self.assertEqual(policies[1],policies[2])
        for k,v in policies[0].items():self.assertEqual(v,policies[1][k])

    def test_selected_capture_has_no_prediction_change_and_cleans_up(self):
        for name,m in initialize_pair(self.cfg).items():
            m.eval();x=torch.randn(3,96,2)
            with torch.no_grad():expected=m(x).numpy()
            with tempfile.TemporaryDirectory() as folder:
                with SelectedValidationCapture(m,Path(folder),name,3) as cap:
                    with torch.no_grad():p=torch.cat([m(x[:2]),m(x[2:])]).numpy()
                result=cap.report(p,np.zeros_like(p))
                np.testing.assert_allclose(p,expected,atol=2e-5,rtol=0)
                np.testing.assert_allclose(cap.components.sum(-1)+cap.means,p,atol=2e-5,rtol=0)
                self.assertEqual(cap.components.shape[-1],8 if name=='baseline' else 12)
                self.assertIsNone(cap.model);self.assertIsNone(cap.patch);self.assertEqual(cap.cache,[])
                if name!='baseline':self.assertEqual(len(result['decoder_entrance']),4)

    def test_no_future_input_and_one_denormalization(self):
        m=initialize_pair(self.cfg)['seasonal_trend'].eval();x=torch.randn(2,96,2);events=[]
        h=m.normalize.register_forward_pre_hook(lambda mod,args:events.append(args[1]))
        with torch.no_grad():p=m(x)
        h.remove();self.assertEqual(events,['norm','denorm'])
        nan=torch.full_like(x,float('nan'))
        with torch.no_grad():torch.testing.assert_close(p,m(x,nan,nan,nan),atol=0,rtol=0)
        with self.assertRaises(ValueError):dataset(self.cfg,'test',0)
        self.assertEqual(joint_stale_count(4,False),5);self.assertEqual(joint_stale_count(4,True),0)
        self.assertEqual(learning_rate('seasonal_trend',30,30),0)

if __name__=='__main__':unittest.main()
