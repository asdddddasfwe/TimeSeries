"""Isolated CPU tests for the new authorized H336 study only."""
import unittest, tempfile, json
from pathlib import Path
import torch
from models.MAFirstAffineKernel336 import value_gram, kernel_supervision
from models.OscillatorPDMStatePatchMAFirst import Model as Original
from tmp.mafirst_affine_kernel_h336_20261022_factory import fresh,optimizer_groups,CONTROL
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups

torch.set_num_threads(2)
class StudyTests(unittest.TestCase):
    def original(self):
        from types import SimpleNamespace
        cfg=json.loads((CONTROL/'protocol.json').read_text())['model_config']
        cfg.update(log_dir=tempfile.gettempdir()+'/mafirst336_unit_original',tensorboard=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20211014);m=Original(SimpleNamespace(**cfg))
        m.load_state_dict(torch.load(CONTROL/'initial.pt',weights_only=True,map_location='cpu')['time_position_mmd'])
        return m
    def sample(self):return torch.randn(2,96,7,generator=torch.Generator().manual_seed(124))
    def test_01_old236_exact_and_added_reference_deterministic(self):
        source=torch.load(CONTROL/'initial.pt',weights_only=True,map_location='cpu')['time_position_mmd']
        for arm,affine in [('affine',True),('parallel',False),('parallel',True),('kernel',False),('kernel',True)]:
            a,_=fresh(arm,affine,tempfile.gettempdir()+'/mafirst336_unit_a');b,_=fresh(arm,affine,tempfile.gettempdir()+'/mafirst336_unit_b')
            self.assertTrue(all(torch.equal(a.state_dict()[k],v) for k,v in source.items()))
            self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
            if arm=='affine':
                self.assertEqual(sum(p.numel() for p in a.parameters()),209914)
                self.assertEqual(len(a.state_dict()),244)
    def test_02_affine_identity_outputs_rng_train_eval(self):
        a=self.original();b,_=fresh('affine',True,tempfile.gettempdir()+'/mafirst336_unit');x=self.sample()
        for train in (False,True):
            outputs=[];rng=[]
            for m in (a,b):
                m.train(train);torch.manual_seed(314)
                outputs.append(m(x));rng.append(torch.get_rng_state())
            self.assertTrue(torch.equal(*outputs));self.assertTrue(torch.equal(*rng))
    def test_03_nonidentity_inverse_and_components(self):
        m,_=fresh('affine',True,tempfile.gettempdir()+'/mafirst336_unit');m.eval();x=self.sample()
        with torch.no_grad():
            m.normalize.affine_weight.copy_(torch.linspace(.6,1.4,7));m.normalize.affine_bias.copy_(torch.linspace(-.3,.3,7))
            z=m.normalize(x,'norm');self.assertTrue(torch.allclose(m.normalize(z,'denorm'),x,atol=1e-6))
            c=m.predict_components(x);p=m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm')
            scale,offset=m.inverse_coordinates();physical=c*scale.unsqueeze(-1)
            self.assertLess(float((physical[...,:4].sum(-1)+physical[...,4:].sum(-1)+offset-p).abs().max()),2e-5)
            self.assertTrue(torch.equal(m.normalize.mean,x.mean(1,keepdim=True)))
    def test_04_old_optimizer_rules_exact_and_affine_exempt(self):
        old=self.original();base={n:g['weight_decay'] for g in norm_bias_no_decay_groups(old,1e-4) for n,p in old.named_parameters() if any(p is q for q in g['params'])}
        for arm,affine in [('affine',True),('parallel',False),('kernel',True)]:
            m,_=fresh(arm,affine,tempfile.gettempdir()+'/mafirst336_unit');gs=optimizer_groups(m)
            decay={id(p):g['weight_decay'] for g in gs for p in g['params']}
            for n,p in m.named_parameters():
                if n in base:self.assertEqual(decay[id(p)],base[n])
                if n.endswith(('affine_weight','affine_bias')):self.assertEqual(decay[id(p)],0.)
    def test_05_gram_symmetry_psd_diag_raw_units(self):
        u=torch.randn(3,84,4,dtype=torch.float64,generator=torch.Generator().manual_seed(9));g=value_gram(u)
        self.assertTrue(torch.equal(g,g.transpose(-1,-2)));self.assertTrue(torch.equal(g.diagonal(dim1=-2,dim2=-1),torch.ones(3,84,dtype=g.dtype)))
        self.assertGreater(float(torch.linalg.eigvalsh(g).min()),-1e-10)
        ref=torch.exp(-.5*(u[:,4]-u[:,19]).square().sum(-1));self.assertTrue(torch.equal(g[:,4,19],ref))
        self.assertTrue(torch.allclose(g,value_gram(u+3.2),atol=1e-14));self.assertFalse(torch.allclose(g,value_gram(u*2)))
    def test_06_no_teacher_forcing_aux_target_detached(self):
        m,_=fresh('kernel',True,tempfile.gettempdir()+'/mafirst336_unit');m.eval();x=self.sample();p=m(x);g=m.predicted_gram.clone()
        y=torch.randn(2,336,7,requires_grad=True);loss,gold=kernel_supervision(m.predicted_gram,y)
        self.assertFalse(gold.requires_grad);self.assertIsNone(torch.autograd.grad(loss,y,allow_unused=True,retain_graph=True)[0])
        self.assertTrue(torch.equal(p,m(x)));self.assertTrue(torch.equal(g,m.predicted_gram))
        self.assertEqual(tuple(g.shape),(14,84,84));self.assertTrue(torch.isfinite(loss))
    def test_07_kernel_initial_forecast_equals_base_both_affines(self):
        x=self.sample()
        for affine in (False,True):
            base,_=fresh('affine',affine,tempfile.gettempdir()+'/mafirst336_unit');head,_=fresh('kernel',affine,tempfile.gettempdir()+'/mafirst336_unit')
            base.eval();head.eval();self.assertTrue(torch.equal(base(x),head(x)))
    def test_08_feedback_both_branches_and_all_gradients(self):
        from utils.time_position_patch_mmd import parts
        m,_=fresh('kernel',True,tempfile.gettempdir()+'/mafirst336_unit');m.eval();x=self.sample();y=torch.randn(2,336,7)
        opt=torch.optim.Adam(optimizer_groups(m),lr=.001)
        # The original trend readout starts at zero; its upstream route opens after one update.
        for step in range(4):
            opt.zero_grad();p=m(x);loss=(p-y).square().mean()+parts(p,y)[0]+kernel_supervision(m.predicted_gram,y)[0];loss.backward()
            opt.step()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertGreater(sum(float(p.grad.square().sum()) for p in m.kernel_head.parameters()),0.)
        for branch in (m.kernel_season_feedback,m.kernel_trend_feedback):
            self.assertGreater(sum(float(p.grad.square().sum()) for p in branch.parameters()),0.)
        for branch in (m.kernel_season_feedback,m.kernel_trend_feedback):
            with torch.no_grad():branch[0].bias.fill_(.1)
            c=m.predict_components(x);a=c[...,:4] if branch is m.kernel_season_feedback else c[...,4:]
            with torch.no_grad():branch[0].bias.zero_()
            baseline=m.predict_components(x);b=baseline[...,:4] if branch is m.kernel_season_feedback else baseline[...,4:]
            self.assertFalse(torch.equal(a,b))
        baseline=m.predict_components(x)
        handle=m.kernel_head.register_forward_hook(lambda module,args,out:torch.eye(out.shape[-1],device=out.device).expand_as(out))
        try:intervened=m.predict_components(x)
        finally:handle.remove()
        self.assertFalse(torch.equal(baseline[...,:4],intervened[...,:4]))
        self.assertFalse(torch.equal(baseline[...,4:],intervened[...,4:]))
    def test_09_parallel_matches_completed_operator(self):
        from models.MAFirstParallelDecoder import Model as Parallel
        x=self.sample();m,cfg=fresh('parallel',False,tempfile.gettempdir()+'/mafirst336_unit');m.eval()
        with torch.random.fork_rng(devices=[]):b=Parallel(cfg)
        b.load_state_dict(m.state_dict());b.eval();self.assertTrue(torch.equal(m(x),b(x)))
if __name__=='__main__':unittest.main()
