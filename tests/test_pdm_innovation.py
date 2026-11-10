"""Independent filter mathematics, retained initialization and capture checks."""
import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from test_oscillator_mixer import config
from models.OscillatorPDMInnovation import filter_states
from scripts.train_timemix_pdm_innovation import initialize_pair,arm_config,parameter_groups,VARIANTS
from scripts.train_timemix_pdm_mixing_depth import initialize_pair as prior_pair
from utils.pdm_innovation_diagnostics import SelectedValidationCapture,setup_hooks,trace


class InnovationMathTests(unittest.TestCase):
    def test_independent_matrix_kalman_reference(self):
        torch.manual_seed(7)
        obs=torch.randn(2,9,3,2,dtype=torch.float64)
        angles=torch.rand(2,3,dtype=torch.float64);radius=.7+.2*torch.rand(2,3,dtype=torch.float64)
        transition=torch.stack((radius*angles.cos(),radius*angles.sin()),-1);q=torch.tensor([.01,.1,.3],dtype=torch.float64)
        for stride in [1,2,4,8]:
            states,gains,ps=filter_states(obs,transition,q,stride)
            expected=np.zeros(states.shape)
            for n in range(2):
                for j in range(3):
                    ar,ai=transition[n,j].numpy();A=np.array([[ar,-ai],[ai,ar]])
                    Af=(radius[n,j].item()**(1/stride))*np.array([[np.cos(angles[n,j].item()/stride),-np.sin(angles[n,j].item()/stride)],[np.sin(angles[n,j].item()/stride),np.cos(angles[n,j].item()/stride)]])
                    Q=sum(np.linalg.matrix_power(Af,i) @ (q[j].item()*np.eye(2)) @ np.linalg.matrix_power(Af,i).T for i in range(stride))
                    P=np.eye(2);z=np.zeros(2)
                    for t in range(9):
                        zp=A@z;prior=A@P@A.T+Q;K=prior@np.linalg.inv(prior+np.eye(2))
                        z=zp+K@(obs[n,t,j].numpy()-zp);P=(np.eye(2)-K)@prior@(np.eye(2)-K).T+K@K.T
                        expected[n,t,j]=z
                        np.testing.assert_allclose(K,np.eye(2)*gains[n,t,j].item(),rtol=1e-12,atol=1e-12)
                        np.testing.assert_allclose(P,np.eye(2)*ps[n,t,j].item(),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(states.numpy(),expected,rtol=1e-12,atol=1e-12)
    def test_gradcheck(self):
        torch.manual_seed(1)
        obs=torch.randn(1,4,2,2,dtype=torch.float64,requires_grad=True)
        a=torch.tensor([[[.7,.2],[.5,.3]]],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([.1,.2],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda o,a,q:filter_states(o,a,q,4)[0],(obs,a,q),eps=1e-6,atol=2e-5))
    def test_extremes_zero_and_observation_weight(self):
        for radius in [0.,1e-8,.999999]:
            obs=torch.zeros(2,96,3,2,requires_grad=True);a=torch.zeros(2,3,2);a[...,0]=radius;a.requires_grad_();q=torch.tensor([1e-6,.01,1e6],requires_grad=True)
            z,k,p=filter_states(obs,a,q,8)
            self.assertTrue(torch.isfinite(z).all() and torch.isfinite(p).all())
            self.assertTrue((p>0).all() and (k>=0).all() and (k<=1).all())
            self.assertEqual(torch.count_nonzero(z).item(),0)
            (z.sum()+p.sum()).backward()
            for x in [obs,a,q]:self.assertTrue(torch.isfinite(x.grad).all())
        obs=torch.ones(1,4,1,2);a=torch.tensor([[[.8,.2]]])
        _,low,_=filter_states(obs,a,torch.tensor([.001]));_,high,_=filter_states(obs,a,torch.tensor([10.]))
        self.assertTrue((high>low).all())


class InnovationModelTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads();torch.set_num_threads(1)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=1,dropout=.2,down_sampling_layers=3,down_sampling_window=2)
        self.cfg.seasonal_normalize=False;torch.manual_seed(2021);self.models=initialize_pair(self.cfg)
    def tearDown(self):torch.set_num_threads(self.threads)
    def test_common_state_control_and_optimizer(self):
        c,t=self.models.values();cs,ts=c.state_dict(),t.state_dict()
        self.assertTrue(set(cs)<set(ts));added=set(ts)-set(cs)
        self.assertEqual(len(added),4);self.assertEqual(sum(ts[k].numel() for k in added),4*c.seasonals[0].decoder.modes)
        for k in cs:torch.testing.assert_close(cs[k],ts[k],rtol=0,atol=0)
        torch.manual_seed(2021);old=prior_pair(self.cfg)['depth1']
        for k,v in old.state_dict().items():torch.testing.assert_close(v,cs[k],rtol=0,atol=0)
        policies={}
        for arm,m in self.models.items():
            gs=parameter_groups(m,arm);ids=[id(p) for g in gs for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.parameters()})
            wd={id(p):g['weight_decay'] for g in gs for p in g['params']};policies[arm]={n:wd[id(p)] for n,p in m.named_parameters()}
        for k in policies['oscillator']:self.assertEqual(policies['oscillator'][k],policies['innovation'][k])
        for k in added:self.assertEqual(policies['innovation'][k],0.)
    def test_encoder_normalization_rng_restore_and_backward(self):
        x=torch.randn(2,96,2);enc=[];rng=[]
        for arm,m in self.models.items():
            m.eval()
            with torch.no_grad():enc.append(m.encode_multiscale(m.downsample(x)))
            restored=VARIANTS[arm](arm_config(self.cfg,arm));restored.load_state_dict(m.state_dict());restored.eval()
            with torch.no_grad():torch.testing.assert_close(m(x),restored(x),rtol=0,atol=0)
            m.train();torch.manual_seed(4);loss=m(x).square().mean();rng.append(torch.get_rng_state());loss.backward()
            for n,p in m.named_parameters():self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
            if arm=='innovation':self.assertTrue(any(p.grad.abs().sum()>0 for n,p in m.named_parameters() if n.endswith('.log_process_variance')))
        self.assertTrue(torch.equal(*rng))
        for group in [0,1]:
            aa,bb=enc[0][group],enc[1][group]
            if group==0:
                for u,v in zip(aa,bb):torch.testing.assert_close(u,v,rtol=0,atol=0)
            else:
                for us,vs in zip(aa,bb):
                    for u,v in zip(us,vs):torch.testing.assert_close(u,v,rtol=0,atol=0)
        self.assertIs(type(self.models['oscillator'].normalize),type(self.models['innovation'].normalize))
    def test_same_forward_capture_and_reconstruction(self):
        x=torch.randn(2,96,2)
        with tempfile.TemporaryDirectory() as directory:
            for arm,m in self.models.items():
                m.eval()
                with torch.no_grad():expected=m(x)
                with SelectedValidationCapture(m,Path(directory),arm,2) as cap:
                    with torch.no_grad():actual=m(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                row=cap.report(actual.numpy(),np.zeros_like(actual.numpy()))
                self.assertLess(row['reconstruction_max_abs_error'],2e-5);self.assertEqual(len(row['component_order']),8)
                self.assertEqual(len(row['history_filter']),4 if arm=='innovation' else 0)
                captured,handles=setup_hooks(m)
                try:loss=m(x).square().mean();loss.backward()
                finally:
                    for handle in handles:handle.remove()
                wd={id(p):g['weight_decay'] for g in parameter_groups(m,arm) for p in g['params']}
                tr=trace(m,captured,1,loss,wd);self.assertEqual(len(tr['filter_parameters']),4 if arm=='innovation' else 0)
                if arm=='innovation':self.assertTrue(all(e.decoder.filter_observer is None for e in m.seasonals))

if __name__=='__main__':unittest.main()
