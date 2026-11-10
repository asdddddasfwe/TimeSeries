"""Ten CPU checks for the authorized value-only kernel-feedback study."""
import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from utils.value_patch_mmd import parts,patch_parts,numpy_parts
from utils.time_position_patch_mmd import patch_parts as timed
from models.MAFirstKernelValueOnly336 import kernel_supervision,value_gram,parallel_future_states,recurrent_future_states
from tmp.mafirst_kernel_valueonly_h336_20261022_factory import fresh,optimizer_groups,BASE,CONTROL

torch.set_num_threads(2)
class StudyTests(unittest.TestCase):
    def sample(self):return torch.randn(2,96,7,generator=torch.Generator().manual_seed(127))
    def model(self,arm):return fresh(arm,tempfile.gettempdir()+'/mafirst_valueonly_unit')[0]
    def test_01_numpy_formula(self):
        p=torch.randn(2,12,3,dtype=torch.float64,generator=torch.Generator().manual_seed(1));y=torch.randn(2,12,3,dtype=torch.float64,generator=torch.Generator().manual_seed(2))
        a,d=parts(p,y);b=numpy_parts(p.numpy(),y.numpy());self.assertAlmostEqual(float(a),b['value_mmd'],places=13)
        self.assertAlmostEqual(float(a),d['mmd_prediction_self']+d['mmd_target_self']-2*d['mmd_cross'],places=13)
    def test_02_equals_time_kernel_all_ones(self):
        p=torch.randn(1,2,5,4,dtype=torch.float64,requires_grad=True);y=torch.randn_like(p)
        a=patch_parts(p,y)[0];b=timed(p,y,sigma_time=1e30)[0]
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(torch.autograd.grad(a,p,retain_graph=True)[0],torch.autograd.grad(b,p)[0]))
    def test_03_patch_permutation_invariant_but_time_loss_not(self):
        g=torch.Generator().manual_seed(19);p=torch.randn(1,1,6,4,dtype=torch.float64,generator=g);y=p+.1*torch.randn(p.shape,dtype=p.dtype,generator=g);ix=torch.tensor([5,2,0,4,1,3])
        self.assertAlmostEqual(float(patch_parts(p,y)[0]),float(patch_parts(p[:,:,ix],y)[0]),places=13)
        self.assertGreater(abs(float(timed(p,y)[0])-float(timed(p[:,:,ix],y)[0])),1e-5)
    def test_04_gradcheck_targetdetach_isolation(self):
        p=torch.randn(1,1,3,4,dtype=torch.float64,requires_grad=True);y=torch.randn_like(p,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a:patch_parts(a,y)[0],(p,),eps=1e-6,atol=1e-5))
        self.assertIsNone(torch.autograd.grad(patch_parts(p,y)[0],y,allow_unused=True)[0]);self.assertEqual(float(patch_parts(p,p.detach())[0]),0.)
        a=torch.cat([p.detach(),p.detach()+4],0);b=torch.cat([y.detach(),y.detach()-2],0)
        self.assertAlmostEqual(float(patch_parts(a,b)[0]),sum(float(patch_parts(a[j:j+1],b[j:j+1])[0]) for j in range(2))/2,places=13)
    def test_05_all256_initial_and_all224_decay_exact(self):
        initial=torch.load(BASE/'initial.pt',map_location='cpu',weights_only=True);old={n:g['weight_decay'] for g in json.loads((BASE/'optimizer_groups.json').read_text()) for n in g['names']}
        for arm in ['recurrent','parallel']:
            m=self.model(arm);self.assertEqual((sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers()))),(240786,224,256,32))
            self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items()));names={id(p):n for n,p in m.named_parameters()};self.assertEqual({names[id(p)]:g['weight_decay'] for g in optimizer_groups(m) for p in g['params']},old)
    def test_06_recurrent_matches_kernel_baseline_outputs_and_rng(self):
        from models.MAFirstAffineKernel336 import Model
        cfg=json.loads((BASE/'model_config.json').read_text());b=Model(SimpleNamespace(**cfg));b.load_state_dict(torch.load(BASE/'initial.pt',map_location='cpu',weights_only=True));m=self.model('recurrent');x=self.sample()
        for training in [False,True]:
            outputs=[];rng=[]
            for a in [b,m]:a.train(training);torch.manual_seed(17);outputs.append(a(x));rng.append(torch.get_rng_state())
            self.assertTrue(torch.equal(*outputs));self.assertTrue(torch.equal(*rng))
    def test_07_parallel_zero_feedback_matches_previous_parallel(self):
        from models.MAFirstParallelDecoder import Model
        cfg=json.loads((CONTROL/'protocol.json').read_text())['model_config'];cfg.update(log_dir=tempfile.gettempdir()+'/mafirst_valueonly_unit',tensorboard=False);b=Model(SimpleNamespace(**cfg));b.load_state_dict(torch.load(CONTROL/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd'])
        m=self.model('parallel');m.eval();b.eval();x=self.sample();self.assertTrue(torch.equal(m(x),b(x)))
        for n,v in b.state_dict().items():
            if 'position' in n:self.assertTrue(torch.equal(v,m.state_dict()[n]))
    def test_08_feedback_both_routes_and_no_future_input(self):
        x=self.sample();y=torch.randn(2,336,7,generator=torch.Generator().manual_seed(133))
        for arm in ['recurrent','parallel']:
            m=self.model(arm);m.eval();opt=torch.optim.Adam(optimizer_groups(m),lr=.001)
            for _ in range(4):
                opt.zero_grad();p=m(x);loss=(p-y).square().mean()+parts(p,y)[0]+kernel_supervision(m.predicted_gram,y)[0];loss.backward();opt.step()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            baseline=m.predict_components(x);handle=m.kernel_head.register_forward_hook(lambda mod,args,out:torch.eye(out.shape[-1]).expand_as(out))
            try:altered=m.predict_components(x)
            finally:handle.remove()
            self.assertFalse(torch.equal(baseline[...,:4],altered[...,:4]));self.assertFalse(torch.equal(baseline[...,4:],altered[...,4:]))
            p=m(x);target=y.clone().requires_grad_();aux,gold=kernel_supervision(m.predicted_gram,target);self.assertFalse(gold.requires_grad);self.assertIsNone(torch.autograd.grad(aux,target,allow_unused=True)[0]);self.assertTrue(torch.equal(p,m(x)))
            c=m.predict_components(x);scale,offset=m.inverse_coordinates();physical=c*scale.unsqueeze(-1);out=m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm');self.assertLess(float((physical.sum(-1)+offset-out).abs().max()),2e-5)
    def test_09_operator_gradients_and_gram_properties(self):
        a=torch.randn(2,3,2,dtype=torch.float64);u=torch.randn(2,8,3,2,dtype=torch.float64,requires_grad=True);h=torch.randn_like(a,requires_grad=True)
        p=parallel_future_states(a,u,h);g=torch.autograd.grad(p[:,-2:].square().sum(),u)[0];self.assertEqual(torch.count_nonzero(g[:,:6]),0);self.assertEqual(recurrent_future_states(a,u,h).shape,p.shape)
        v=value_gram(torch.randn(3,84,4,dtype=torch.float64));self.assertTrue(torch.equal(v,v.transpose(-1,-2)));self.assertGreater(float(torch.linalg.eigvalsh(v).min()),-1e-10);self.assertTrue(torch.equal(v.diagonal(dim1=-2,dim2=-1),torch.ones(3,84,dtype=v.dtype)))
    def test_10_data_same_standard_prefix(self):
        from tmp.run_mafirst_kernel_valueonly_h336_20261022 import data_for,read,PLAN
        plan=read(PLAN);self.assertEqual(len(plan['jobs']),2);tr,va,scaler,data=data_for(plan['jobs'][0]);self.assertEqual((len(tr),len(va)),(34129,11185))
        old=json.loads((BASE/'data_manifest.json').read_text());self.assertEqual(scaler.mean_.tolist(),old['scaler_mean']);self.assertEqual(scaler.scale_.tolist(),old['scaler_scale']);self.assertEqual(tuple(tr[0][0].shape),(96,7));self.assertEqual(tuple(va[len(va)-1][1].shape),(336,7))
if __name__=='__main__':unittest.main()
