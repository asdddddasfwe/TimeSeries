"""Mechanism tests for a parameter-free entrance-boundary intervention."""
import unittest,json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from models.OscillatorPDMStatePatchBoundaryDiagnostic import Model,BoundaryDecomposition
from models.OscillatorPDMStatePatchDriveDiagnostic import Model as Prior
R=Path(__file__).resolve().parents[1]
class BoundaryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2)
  cls.cfg=json.loads((R/'tmp/mafirst_drive_mean_ettm1_fold0_h720_20261031_group/plan.json').read_text())['models']['mafirst_season']['config']
 def build(self,mode):
  torch.manual_seed(2021)
  return Model(SimpleNamespace(**(self.cfg|dict(boundary_padding=mode))))
 def test_constant_and_reconstruction_all_lengths(self):
  for mode in ['reflect','replicate']:
   d=BoundaryDecomposition(mode)
   for length in [96,48,24,12]:
    x=torch.full((14,length,1),3.25,dtype=torch.float64)
    s,t=d(x);self.assertTrue(torch.equal(t,x));self.assertEqual(float(s.abs().max()),0)
    self.assertTrue(torch.equal(s+t,x))
 def test_reflection_values_exclude_endpoint(self):
  d=BoundaryDecomposition('reflect');x=torch.arange(12,dtype=torch.float64)[None,:,None].repeat(7,1,1)
  _,t=d(x)
  self.assertAlmostEqual(float(t[0,0,0]),20/9)
  self.assertAlmostEqual(float(t[0,-1,0]),79/9)
  expected=F.avg_pool1d(F.pad(x.transpose(1,2),(4,4),mode='reflect'),9,1).transpose(1,2)
  self.assertTrue(torch.equal(t,expected))
 def test_interior_bitwise_and_only_border_changes(self):
  a,b=BoundaryDecomposition('replicate'),BoundaryDecomposition('reflect')
  for n in [96,48,24,12]:
   x=torch.randn(14,n,1);sa,ta=a(x);sb,tb=b(x)
   self.assertTrue(torch.equal(ta[:,4:-4],tb[:,4:-4]))
   self.assertTrue(torch.equal(sa[:,4:-4],sb[:,4:-4]))
   self.assertGreater(float((ta-tb).abs().max()),1e-3)
   self.assertLess(float((sb+tb-x).abs().max()),3e-7)
 def test_reflection_gradient_matches_reference(self):
  d=BoundaryDecomposition('reflect')
  for n in [96,48,24,12]:
   x=torch.randn(7,n,1,dtype=torch.float64,requires_grad=True)
   s,t=d(x);e=F.avg_pool1d(F.pad(x.transpose(1,2),(4,4),mode='reflect'),9,1).transpose(1,2)
   a=torch.autograd.grad(t.square().sum()+s.square().sum(),x,retain_graph=True)[0]
   b=torch.autograd.grad(e.square().sum()+(x-e).square().sum(),x)[0]
   self.assertTrue(torch.allclose(a,b,rtol=1e-12,atol=1e-12))
 def test_constructor_state_parameters_and_external_rng(self):
  a=self.build('reflect');rng=torch.get_rng_state().clone();b=self.build('replicate')
  self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  torch.manual_seed(2021);old=Prior(SimpleNamespace(**self.cfg))
  self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  self.assertEqual((sum(p.numel() for p in a.parameters()),len(list(a.parameters())),len(list(a.buffers())),len(a.state_dict())),(345786,212,32,244))
  self.assertTrue(all(torch.equal(v,old.state_dict()[k]) and torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
 def test_replicate_full_forward_and_gradients_exact(self):
  a=self.build('replicate');old=Prior(SimpleNamespace(**self.cfg));old.load_state_dict(a.state_dict())
  a.train();old.train();x=torch.randn(2,96,7);y=torch.randn(2,720,7)
  rng=torch.get_rng_state();p=a(x);after=torch.get_rng_state();torch.set_rng_state(rng);q=old(x)
  self.assertTrue(torch.equal(after,torch.get_rng_state()));self.assertTrue(torch.equal(p,q))
  (p-y).square().mean().backward();(q-y).square().mean().backward()
  self.assertTrue(all(torch.equal(v.grad,dict(old.named_parameters())[k].grad) for k,v in a.named_parameters()))
 def test_reflect_complete_loss_gradients_rng_and_intervention(self):
  a=self.build('reflect');b=self.build('replicate');a.train();b.train()
  x=torch.randn(2,96,7);y=torch.randn(2,720,7);rng=torch.get_rng_state();p=a(x);after=torch.get_rng_state();torch.set_rng_state(rng);q=b(x)
  self.assertTrue(torch.equal(after,torch.get_rng_state()));self.assertFalse(torch.equal(p,q))
  from tmp.mafirst_fullbatch_mmd_recompute_20261028 import parts
  loss=(p-y).square().mean()+parts(p,y)[0];loss.backward()
  self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in a.parameters()))
  self.assertTrue(all(z['interior_trend_change_max']==0 for z in a.initial_decomposition.last_boundary_diagnostics.values()))
  self.assertTrue(all(z['shared_block_dropout_mask'] and z['original_block_clock'] and z['selected_endpoint_bitwise_exact'] for z in a.last_state_readout_diagnostics))
  self.assertTrue(all(float(z['projected_drive_phase_contrast_energy'].max())==0 for z in a.last_state_readout_diagnostics))
 def test_configuration_and_no_added_head(self):
  m=self.build('reflect');self.assertEqual(list(m.initial_decomposition.kernels.values()),[9]*4)
  self.assertEqual(m.state_drive_policy,'mean');self.assertEqual(m.state_readout_policy,'substate')
  self.assertFalse(any('correction' in k or 'residual_head' in k for k in m.state_dict()))
  with self.assertRaises(ValueError):BoundaryDecomposition('circular')
  with self.assertRaises(ValueError):Model(SimpleNamespace(**(self.cfg|dict(state_drive_policy='phase',boundary_padding='reflect'))))
if __name__=='__main__':unittest.main()