"""Focused numerical/model checks for the future-only parallel intervention."""
import unittest,tempfile,torch,json
from pathlib import Path
from models.MAFirstParallelDecoder import parallel_future_states
from models.OscillatorPDMStatePatchMAFirst import Model as Original
from layers.OscillatorTimeMix import complex_oscillator_scan
from tmp.mafirst_parallel_decoder_factory_20261022 import load_original_scratch,optimizer_groups
from types import SimpleNamespace
R=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
class ParallelTests(unittest.TestCase):
    def test_operator_reference(self):
        torch.manual_seed(2);a=torch.randn(2,3,2,dtype=torch.float64);u=torch.randn(2,9,3,2,dtype=torch.float64);h=torch.randn_like(a)
        s=parallel_future_states(a,u,h)
        reference=torch.stack([torch.view_as_real(torch.view_as_complex(a)*torch.view_as_complex(h))+u[:,k] for k in range(9)],1)
        self.assertTrue(torch.equal(s,reference))
        old=complex_oscillator_scan(a[:,None].expand_as(u),u,h)
        torch.testing.assert_close(s[:,0],old[:,0],rtol=0,atol=0)
        self.assertGreater(float((s[:,1:]-old[:,1:]).abs().max()),1e-4)
    def test_gradient_and_no_future_chain(self):
        a=torch.randn(1,2,2,dtype=torch.float64,requires_grad=True);h=torch.randn_like(a,requires_grad=True);u=torch.randn(1,5,2,2,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(parallel_future_states,(a,u,h),fast_mode=True))
        gu=torch.autograd.grad(parallel_future_states(a,u,h)[:,-1].sum(),u)[0]
        self.assertTrue(torch.equal(gu[:,:-1],torch.zeros_like(gu[:,:-1])))
        self.assertTrue(torch.equal(gu[:,-1],torch.ones_like(gu[:,-1])))
    def test_original_state_and_budget(self):
        m,_=load_original_scratch(R,R/'tmp/parallel_unit_logs')
        self.assertEqual(sum(p.numel() for p in m.parameters()),320450)
        self.assertEqual(len(list(m.parameters())),204);self.assertEqual(len(m.state_dict()),236)
        self.assertEqual(m.initialization_audit['copied_tensors'],236)
    def test_shape_and_reconstruction(self):
        m,_=load_original_scratch(R,R/'tmp/parallel_unit_logs');m.eval();x=torch.randn(2,96,7)
        with torch.no_grad():
            p=m(x);c=m.predict_components(x);q=m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm')
        self.assertEqual(tuple(p.shape),(2,720,7));self.assertEqual(tuple(c.shape),(2,720,7,8))
        self.assertTrue(torch.equal(p,q));self.assertTrue(torch.isfinite(p).all())
        with self.assertRaises(ValueError):m(torch.randn(2,95,7))
    def test_historical_and_trend_paths_identical(self):
        m,cfg=load_original_scratch(R,R/'tmp/parallel_unit_logs')
        with torch.random.fork_rng(devices=[]):old=Original(SimpleNamespace(**cfg))
        old.load_state_dict(m.state_dict(),strict=True);old.eval();m.eval();x=torch.randn(2,96,7)
        with torch.no_grad():
            ns,nt=m.encode_multiscale(m.downsample(x));os,ot=old.encode_multiscale(old.downsample(x))
            for aa,bb in zip(nt[0]+nt[1],ot[0]+ot[1]):self.assertTrue(torch.equal(aa,bb))
            for e,oe,s,t in zip(m.seasonals,old.seasonals,nt[0],nt[1]):
                for a,b in zip(e.decoder.dynamics(s,t),oe.decoder.dynamics(s,t)):self.assertTrue(torch.equal(a,b))
            a=m.predict_components(x);b=old.predict_components(x)
        self.assertTrue(torch.equal(a[...,4:],b[...,4:]))
        self.assertGreater(float((a[...,:4]-b[...,:4]).abs().max()),0)
    def test_all204_gradients_finite_and_used(self):
        m,cfg=load_original_scratch(R,R/'tmp/parallel_unit_logs')
        old=Original(SimpleNamespace(**cfg));old.load_state_dict(m.state_dict(),strict=True)
        zero_sets=[]
        for model in (m,old):
            model.train();torch.manual_seed(13)
            p=model(torch.randn(3,96,7));(p-torch.randn_like(p)).square().mean().backward()
            self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()))
            zero_sets.append({n for n,v in model.named_parameters() if float(v.grad.abs().sum())==0})
        # Original zero-initialized gates/heads give legitimate zero first-step gradients.
        self.assertEqual(zero_sets[0],zero_sets[1])
        self.assertTrue(all(float(v.grad.abs().sum())>0 for n,v in m.named_parameters()
                            if '.decoder.drive_projection.weight' in n or '.decoder.log_decay' in n))
    def test_optimizer_groups_exact(self):
        m,_=load_original_scratch(R,R/'tmp/parallel_unit_logs');groups=optimizer_groups(m);names={id(p):n for n,p in m.named_parameters()}
        actual=[dict(group_name=g.get('group_name'),weight_decay=g['weight_decay'],names=[names[id(p)] for p in g['params']]) for g in groups]
        expected=json.loads((R/'tmp/mafirst_memory_factorial_20261017_full_h720_control_seed2021/optimizer_groups.json').read_text())
        self.assertEqual(actual,expected)
        ids=[id(p) for g in groups for p in g['params']];self.assertEqual(len(ids),len(set(ids)));self.assertEqual(len(ids),204)
if __name__=='__main__':unittest.main()
