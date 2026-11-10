"""42 inherited invariants plus12 tests for the one global joint seasonal head."""
from test_mafirst_kernel_h720_dropoutgrammse_20261024 import AffineTests
from pathlib import Path
import json,tempfile,unittest
from types import SimpleNamespace
import numpy as np,torch
from torch import nn
from tmp.mafirst_kernel_h720_globaljoint_20261024_factory import fresh,optimizer_groups,G,R
from models.MAFirstKernelH720DirectGram import Model as Base
from utils.global_joint_output_gradients_20261024 import training_terms,gradient_summary,TERM_NAMES
from utils.drop_output_gram_mse_20261024 import training_terms as control_terms
class GlobalJointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(9931)
        self.temp=tempfile.TemporaryDirectory(prefix='globaljoint_cpu_',dir=R/'tmp');self.addCleanup(self.temp.cleanup)
    def model(self):return fresh(720,'h720_global_joint',Path(self.temp.name)/'logs')
    def base(self,cfg):
        with torch.random.fork_rng(devices=[]):torch.manual_seed(2021);m=Base(cfg)
        m.load_state_dict(torch.load(G/'reference_initial/shared258.pt',map_location='cpu',weights_only=True));return m
    def test_global01_counts_reference_and_independent_shared258(self):
        m,cfg=self.model();s=torch.load(G/'reference_initial/shared258.pt',map_location='cpu',weights_only=True)
        self.assertEqual((sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers()))),(906416,227,259,32))
        self.assertEqual(set(m.state_dict())-set(s),{'global_seasonal_delta'});self.assertEqual(torch.count_nonzero(m.global_seasonal_delta).item(),0)
        for n,v in s.items():self.assertTrue(torch.equal(v,m.state_dict()[n]));self.assertNotEqual(v.data_ptr(),m.state_dict()[n].data_ptr())
    def test_global02_initial_prediction_exact_control_C(self):
        m,cfg=self.model();b=self.base(cfg);x=torch.randn(2,96,7);m.eval();b.eval()
        with torch.no_grad():a=m(x);z=b(x)
        self.assertTrue(torch.equal(a,z));self.assertEqual(tuple(a.shape),(2,720,7))
    def test_global03_trends_unchanged_and_seasonal_identity(self):
        m,cfg=self.model();b=self.base(cfg);x=torch.randn(1,96,7);m.eval();b.eval()
        with torch.no_grad():a=m.predict_components(x);z=b.predict_components(x)
        self.assertTrue(torch.equal(a,z))
        with torch.no_grad():m.global_seasonal_delta[700,0]=.2;a=m.predict_components(x)
        self.assertTrue(torch.equal(a[...,4:],z[...,4:]));self.assertTrue(torch.equal(a[:,:700,:,:4],z[:,:700,:,:4]));self.assertGreater(float((a[:,700,:,:4]-z[:,700,:,:4]).abs().max()),0)
    def test_global04_dense_map_matches_independent_numpy(self):
        m,_=self.model();m.double();z=torch.randn(2,720,7,4,dtype=torch.double)
        with torch.no_grad():m.global_seasonal_delta.copy_(torch.randn(720,720,dtype=torch.double)*.002)
        a=m.mix_seasonal(z).detach().numpy();x=z.numpy().transpose(0,2,3,1);expected=(x+np.matmul(x,m.global_seasonal_delta.detach().numpy().T)).transpose(0,3,1,2)
        np.testing.assert_allclose(a,expected,atol=1e-12,rtol=1e-12)
    def test_global05_far_output_has_earlier_input_gradient(self):
        m,_=self.model();z=torch.randn(1,720,7,4,requires_grad=True)
        with torch.no_grad():m.global_seasonal_delta[700,0]=.3
        a=m.mix_seasonal(z);g=torch.autograd.grad(a[:,700].sum(),z)[0]
        self.assertTrue(torch.equal(g[:,0],torch.full_like(g[:,0],.3)));self.assertTrue(torch.equal(g[:,700],torch.ones_like(g[:,700])))
    def test_global06_no_batch_variable_or_scale_mixing(self):
        m,_=self.model();z=torch.zeros(2,720,7,4);z[0,0,2,3]=1
        with torch.no_grad():m.global_seasonal_delta[701,0]=.4
        a=m.mix_seasonal(z);self.assertEqual(torch.count_nonzero(a).item(),2);self.assertEqual(a[0,701,2,3].item(),torch.tensor(.4).item());self.assertEqual(torch.count_nonzero(a[1]).item(),0)
    def test_global07_complete_normalization_once_no_new_affine(self):
        m,_=self.model();calls={'norm':0,'denorm':0}
        def hook(mod,args,out):calls[args[1]]+=1
        h=m.normalize.register_forward_hook(hook);m(torch.randn(2,96,7));h.remove()
        self.assertEqual(calls,{'norm':1,'denorm':1});self.assertFalse(hasattr(m,'coarse_normalizers'));self.assertEqual(len([n for n,p in m.named_parameters() if n.endswith(('affine_weight','affine_bias'))]),2)
    def test_global08_four_loss_values_and_old_gradients_exact_at_identity(self):
        m,cfg=self.model();b=self.base(cfg);m.eval();b.eval();x=torch.randn(1,96,7);y=torch.randn(1,720,7)
        p=m(x);t,_=training_terms(p,y,m.predicted_gram,'none');q=b(x);u,_=control_terms(q,y,b.predicted_gram,'none')
        self.assertEqual(tuple(t),TERM_NAMES)
        for n in t:self.assertTrue(torch.equal(t[n],u[n]))
        mg=torch.autograd.grad(sum(t.values()),list(m.parameters()),allow_unused=True);bg=torch.autograd.grad(sum(u.values()),list(b.parameters()),allow_unused=True);gm=dict(zip([n for n,p in m.named_parameters()],mg));gb=dict(zip([n for n,p in b.named_parameters()],bg))
        for n,g in gb.items():self.assertTrue(torch.equal(gm[n],g),n)
    def test_global09_new_head_has_output_gradients_but_not_direct_head_loss(self):
        m,_=self.model();p=m(torch.randn(2,96,7));t,d=training_terms(p,torch.randn_like(p),m.predicted_gram,'none')
        for n in ['point_mse','output_local_cs']:
            g=torch.autograd.grad(t[n],m.global_seasonal_delta,retain_graph=True)[0];self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)
        for n in ['head_gram_mse','head_local_cs']:self.assertIsNone(torch.autograd.grad(t[n],m.global_seasonal_delta,retain_graph=True,allow_unused=True)[0])
        stats=gradient_summary(t,list(m.named_parameters()));self.assertEqual(len(stats['global_output_head']['pairs']),6);self.assertEqual(stats['global_output_head']['norms']['head_gram_mse'],0)
    def test_global10_optimizer_retains226_rules_newzero_delta_decay(self):
        m,_=self.model();groups=optimizer_groups(m);opt=torch.optim.Adam(groups,lr=.001);m.global_seasonal_delta.grad=torch.zeros_like(m.global_seasonal_delta);opt.step();self.assertEqual(torch.count_nonzero(m.global_seasonal_delta).item(),0)
        names={id(p):n for n,p in m.named_parameters()};v={names[id(p)]:g['weight_decay'] for g in groups for p in g['params']};self.assertEqual(len(v),227);self.assertEqual(v['global_seasonal_delta'],1e-4)
    def test_global11_updates_finite_and_actual_far_drive_reaches_early_blocks(self):
        m,_=self.model();opt=torch.optim.Adam(optimizer_groups(m),lr=.001);x=torch.randn(2,96,7);y=torch.randn(2,720,7)
        for _ in range(2):
            opt.zero_grad(set_to_none=True);p=m(x);t,_=training_terms(p,y,m.predicted_gram,'none');sum(t.values()).backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()));opt.step()
        self.assertGreater(float((m.global_seasonal_delta-torch.diag_embed(m.global_seasonal_delta.diagonal())).norm()),0)
        from tmp.run_mafirst_kernel_h720_globaljoint_20261024 import gradient_capture
        from models import MAFirstKernelTransferAblation as module
        orig=module.parallel_future_states;gradient_capture(m,x,y,Path(self.temp.name),'g');self.assertIs(module.parallel_future_states,orig)
        z=np.load(Path(self.temp.name)/'g.npz');self.assertTrue(all(np.any(z[f's{i}_adjoint_rms'][:135]>0) for i in range(4)))
    def test_global12_frozen_protocol_unchanged_losses_and_reference259(self):
        m,_=self.model();s=torch.load(G/'reference_initial/h720.pt',map_location='cpu',weights_only=True);self.assertEqual(len(s),259)
        for n,v in m.state_dict().items():self.assertTrue(torch.equal(v,s[n]))
        plan=json.loads((R/'tmp/mafirst_kernel_h720_globaljoint_plan_20261024.json').read_text());self.assertEqual(len(plan['jobs']),1);self.assertEqual(set(plan['training']['applied_weights'].values()),{1.});self.assertEqual(set(plan['training']['diagnostic_weights'].values()),{0.})
        s=(R/'tmp/run_mafirst_kernel_h720_globaljoint_20261024.py').read_text();self.assertIn('early_drive_zero_expected=False',s);self.assertIn('only_global_seasonal_joint_head_added=True',s);self.assertIn('loss=sum(terms.values())',s)
if __name__=='__main__':unittest.main()