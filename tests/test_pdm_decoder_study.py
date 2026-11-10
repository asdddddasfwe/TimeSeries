"""Direct-head ablation: exact retained encoder, gradients, restore, capture."""
import unittest, tempfile
from pathlib import Path
import numpy as np
import torch
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_decoder_study import initialize_pair, arm_config, parameter_groups, VARIANTS, allowed_removed
from scripts.train_timemix_pdm_mixing_depth import initialize_pair as prior_pair
from utils.pdm_decoder_study_diagnostics import SelectedValidationCapture, setup_hooks, trace

class DecoderStudyTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=1,dropout=.2,down_sampling_layers=3,down_sampling_window=2)
        self.cfg.seasonal_normalize=False
        torch.manual_seed(2021);self.models=initialize_pair(self.cfg)
    def tearDown(self):torch.set_num_threads(self.threads)
    def test_common_initial_state_and_old_control(self):
        c,t=self.models.values();cs,ts=c.state_dict(),t.state_dict()
        self.assertTrue(set(cs)-set(ts));self.assertTrue(set(ts)-set(cs))
        self.assertTrue(all(allowed_removed(k) for k in set(cs)-set(ts)))
        self.assertTrue(all('.direct_readout.' in k for k in set(ts)-set(cs)))
        for k in set(cs)&set(ts):torch.testing.assert_close(cs[k],ts[k],rtol=0,atol=0)
        torch.manual_seed(2021);old=prior_pair(self.cfg)['depth1']
        for k,v in old.state_dict().items():torch.testing.assert_close(v,cs[k],rtol=0,atol=0)
    def test_same_initial_encoder_and_normalization(self):
        x=torch.randn(2,96,2);outputs=[]
        for model in self.models.values():
            model.eval()
            with torch.no_grad():outputs.append(model.encode_multiscale(model.downsample(x)))
        for aa,bb in zip(outputs[0][0],outputs[1][0]):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
        for group1,group2 in zip(outputs[0][1],outputs[1][1]):
            for aa,bb in zip(group1,group2):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
        for model in self.models.values():
            normal=model.normalize(x,'norm');restored=model.normalize(normal,'denorm')
            torch.testing.assert_close(restored,x,rtol=1e-6,atol=1e-6)
    def test_config_restore_and_finite_backward(self):
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        for arm,m in self.models.items():
            restored=VARIANTS[arm](arm_config(self.cfg,arm));restored.load_state_dict(m.state_dict(),strict=True)
            m.eval();restored.eval()
            with torch.no_grad():torch.testing.assert_close(m(x),restored(x),rtol=0,atol=0)
            m.train();loss=(m(x)-y).square().mean();loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for name,p in m.named_parameters():self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
    def test_optimizer_and_no_dormant_decoder(self):
        maps={}
        for arm,m in self.models.items():
            gs=parameter_groups(m,arm);ids=[id(p) for g in gs for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.parameters()})
            wd={id(p):g['weight_decay'] for g in gs for p in g['params']};maps[arm]={n:wd[id(p)] for n,p in m.named_parameters()}
        for n in maps['direct'].keys()&maps['oscillator'].keys():self.assertEqual(maps['direct'][n],maps['oscillator'][n])
        self.assertFalse(any('.decoder.' in n or 'gain_offset' in n for n in maps['direct']))
        for n,w in maps['direct'].items():
            if n.endswith('direct_readout.weight'):self.assertEqual(w,1e-4)
            if n.endswith('direct_readout.bias'):self.assertEqual(w,0.)
    def test_direct_equation_and_prediction_reconstruction(self):
        m=self.models['direct'].eval();x=torch.randn(2,96,2)
        with torch.no_grad():
            _,(ss,ts)=m.encode_multiscale(m.downsample(x));pieces=[]
            for e,h in zip(m.seasonals,ss):
                out=e.direct_readout(e.temporal_projection(h.transpose(1,2)).transpose(1,2)+e.future_positions)
                pieces.append(out.squeeze(-1).reshape(2,2,96).transpose(1,2))
            pieces += [head(h).reshape(2,2,96).transpose(1,2) for head,h in zip(m.feature_trend_heads,ts)]
            expected=torch.stack(pieces,-1)
            torch.testing.assert_close(m.predict_components(x),expected,rtol=0,atol=0)
            torch.testing.assert_close(m(x),m.normalize(expected[...,:4].sum(-1)+expected[...,4:].sum(-1),'denorm'),rtol=0,atol=0)
    def test_capture_and_training_hooks_preserve_output(self):
        x=torch.randn(2,96,2)
        with tempfile.TemporaryDirectory() as directory:
            for arm,m in self.models.items():
                m.eval()
                with torch.no_grad():expected=m(x)
                with SelectedValidationCapture(m,Path(directory),arm,2) as cap:
                    with torch.no_grad():actual=m(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                a=cap.report(actual.numpy(),np.zeros_like(actual.numpy()))
                self.assertEqual(a['mixing_depth'],1);self.assertEqual(a['initial_split_calls'],4)
                self.assertEqual(len(a['component_order']),8);self.assertLess(a['reconstruction_max_abs_error'],2e-5)
                self.assertEqual(len(a['stream_statistics']),8);self.assertEqual(len(a['trend_logit_shift']),4 if arm=='oscillator' else 0)
                self.assertEqual(a['trend_conditioned_dynamics'],arm=='oscillator')
                self.assertEqual(set(a['fft_routing']),{'96','48','24','12'})
                self.assertFalse(a['history_position_added']);self.assertTrue(a['segment_position_added'] and a['future_position_added'])
                capvalues,handles=setup_hooks(m)
                try:loss=m(x).square().mean();loss.backward()
                finally:
                    for h in handles:h.remove()
                wd={id(p):g['weight_decay'] for g in parameter_groups(m,arm) for p in g['params']}
                tr=trace(m,capvalues,1,loss,wd);self.assertEqual(len(tr['scales']),4)
                for scale in tr['scales']:self.assertIn('output',scale['activations'])

if __name__=='__main__':unittest.main()
