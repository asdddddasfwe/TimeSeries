"""Kernel algebra and original-model preservation tests; no test-split access."""
import unittest,math
import numpy as np,torch
from utils.time_position_amplitude_patch_mmd_20261025 import parts,kernel,numpy_parts
from utils.time_position_patch_mmd import parts as original_parts
from tmp.mafirst_original_h720_ampmmd_20261025_common import *
from tmp.mafirst_original_h720_ampmmd_20261025_factory import fresh,optimizer_groups,groups_audit,original
torch.set_num_threads(2)
class KernelTests(unittest.TestCase):
 def setUp(self):
  gen=torch.Generator().manual_seed(43);self.p=torch.randn(2,32,3,generator=gen,dtype=torch.float64,requires_grad=True);self.y=torch.randn(2,32,3,generator=gen,dtype=torch.float64,requires_grad=True)
 def test_01_numpy_all_terms(self):
  v,d=parts(self.p,self.y);n=numpy_parts(self.p.detach().numpy(),self.y.detach().numpy())
  for k in n:self.assertAlmostEqual(d[k],n[k],places=13)
 def test_02_original_when_disabled(self):
  a,ad=parts(self.p,self.y,amplitude=False);b,bd=original_parts(self.p,self.y)
  self.assertTrue(torch.equal(a,b))
  for k in ('mmd_prediction_self','mmd_target_self','mmd_cross'):self.assertEqual(ad[k],bd[k])
 def test_03_self_zero(self):self.assertLess(abs(float(parts(self.p,self.p)[0].detach())),1e-14)
 def test_04_target_detached(self):
  v,_=parts(self.p,self.y);a,b=torch.autograd.grad(v,[self.p,self.y],allow_unused=True);self.assertIsNone(b);self.assertTrue(torch.isfinite(a).all());self.assertGreater(float(a.norm()),0)
 def test_05_gradcheck(self):
  a=self.p[:1,:8,:1];b=self.y[:1,:8,:1];self.assertTrue(torch.autograd.gradcheck(lambda z:parts(z,b)[0],(a,),eps=1e-6,atol=1e-6))
 def test_06_joint_psd(self):
  p=self.p.detach().transpose(1,2).unfold(-1,4,4);y=self.y.detach().transpose(1,2).unfold(-1,4,4)
  a,b,c=kernel(p,p),kernel(p,y),kernel(y,y);joint=torch.cat([torch.cat([a,b],-1),torch.cat([b.transpose(-1,-2),c],-1)],-2)
  self.assertGreaterEqual(float(torch.linalg.eigvalsh(joint).min()),-1e-12)
 def test_07_four_channels(self):
  p=self.p.transpose(1,2).unfold(-1,4,4);q=self.y.transpose(1,2).unfold(-1,4,4)
  base=kernel(p,q,amplitude=False);expected=sum(base*torch.nn.functional.softplus(p[...,r,None])*torch.nn.functional.softplus(q[...,None,:,r]) for r in range(4))/4
  torch.testing.assert_close(kernel(p,q),expected,atol=1e-15,rtol=1e-14)
 def test_08_diagonal(self):
  p=self.p.transpose(1,2).unfold(-1,4,4)
  torch.testing.assert_close(kernel(p,p).diagonal(dim1=-2,dim2=-1),torch.nn.functional.softplus(p).square().mean(-1))
 def test_09_batch_channel_independence(self):
  a=parts(self.p,self.y)[0];b=torch.stack([parts(self.p[i:i+1,:,c:c+1],self.y[i:i+1,:,c:c+1])[0] for i in range(2) for c in range(3)]).mean();torch.testing.assert_close(a,b)
 def test_10_time_position_retained(self):
  p=torch.tensor([0.,0.,0.,0.,1.,1.,1.,1.,3.,3.,3.,3.]).reshape(1,12,1);y=p.flip(1)
  self.assertGreater(float(parts(p,y)[0]),0)
 def test_11_flat_offset_distinguished(self):
  p=torch.zeros(1,720,1);y=torch.ones_like(p);self.assertGreater(float(parts(p,y)[0]),0)
 def test_12_shift_changes_weight(self):
  a=parts(self.p,self.y)[0];b=parts(self.p+2,self.y+2)[0];self.assertGreater(abs(float((a-b).detach())),1e-5)
 def test_13_cross_term_essential(self):
  p=torch.zeros(1,32,1,dtype=torch.float64);y=torch.ones_like(p);v,d=parts(p,y);self.assertAlmostEqual(float(v),d['mmd_prediction_self']+d['mmd_target_self']-2*d['mmd_cross'],places=8)
 def test_14_extreme_finite(self):
  p=torch.full((1,8,1),1000.,requires_grad=True);y=torch.full_like(p,-1000.);v,_=parts(p,y);g=torch.autograd.grad(v,p)[0];self.assertTrue(torch.isfinite(v) and torch.isfinite(g).all())
 def test_15_invalid(self):
  for p in (torch.zeros(1,7,1),torch.full((1,8,1),float('nan'))):
   with self.assertRaises(ValueError):parts(p,p)
class OriginalModelTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m,cls.cfg=fresh(G/'cpu_tests/logs')
 def test_16_optimizer_exact(self):groups_audit(self.m,optimizer_groups(self.m))
 def test_17_state_and_prediction(self):
  self.m.eval();x=torch.randn(2,96,7,generator=torch.Generator().manual_seed(91))
  with torch.no_grad():
   a=self.m(x);c=self.m.predict_components(x);b=self.m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm')
  torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertEqual(a.shape,(2,720,7))
  self.assertFalse(any('global_seasonal_delta' in n for n in self.m.state_dict()));self.assertFalse(self.m.normalize.affine)
 def test_18_dataset_matches_saved(self):
  ds=original.dataset(self.cfg,'val');x,y,_,_=ds[0]
  a=np.load(BASE/'time_position_mmd/val/history.npy',mmap_mode='r');b=np.load(BASE/'time_position_mmd/val/truth.npy',mmap_mode='r')
  np.testing.assert_array_equal(x.astype('float32'),a[0]);np.testing.assert_array_equal(y[-720:].astype('float32'),b[0])
 def test_19_initial_independent(self):
  ref=torch.load(BASE/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
  self.assertTrue(all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in self.m.state_dict().items()))
 def test_20_both_loss_gradients(self):
  self.m.train();original.shared.seed_all(123);x=torch.randn(2,96,7);y=torch.randn(2,720,7);p=self.m(x);mse=(p-y).square().mean();md=parts(p,y)[0]
  for loss in (mse,md):
   g=torch.autograd.grad(loss,list(self.m.parameters()),retain_graph=True);self.assertTrue(all(torch.isfinite(v).all() for v in g));self.assertGreater(sum(float(v.square().sum()) for v in g),0)
if __name__=='__main__':unittest.main(verbosity=2)
