"""RQ algebra, gradients and exact preservation of the original model."""
import unittest
import numpy as np,torch
from utils.time_position_rq_patch_mmd_20261025 import parts,kernel,numpy_parts
from utils.time_position_patch_mmd import parts as original_parts
from tmp.mafirst_original_h720_rqmmd_20261025_common import *
from tmp.mafirst_original_h720_rqmmd_20261025_factory import fresh,optimizer_groups,groups_audit,original
torch.set_num_threads(2)
class KernelTests(unittest.TestCase):
 def setUp(self):
  gen=torch.Generator().manual_seed(43);self.p=torch.randn(2,32,3,generator=gen,dtype=torch.float64,requires_grad=True);self.y=torch.randn(2,32,3,generator=gen,dtype=torch.float64,requires_grad=True)
 def test_01_numpy_all_terms(self):
  v,d=parts(self.p,self.y);n=numpy_parts(self.p.detach().numpy(),self.y.detach().numpy())
  for k in n:self.assertAlmostEqual(d[k],n[k],places=13)
 def test_02_rbf_adapter_exact_values_and_gradients(self):
  a,ad=parts(self.p,self.y,family='rbf');b,bd=original_parts(self.p,self.y);self.assertTrue(torch.equal(a,b))
  self.assertTrue(torch.equal(torch.autograd.grad(a,self.p)[0],torch.autograd.grad(b,self.p)[0]))
  for k in ('mmd_prediction_self','mmd_target_self','mmd_cross'):self.assertEqual(ad[k],bd[k])
 def test_03_self_zero(self):self.assertLess(abs(float(parts(self.p,self.p)[0].detach())),1e-14)
 def test_04_target_detached(self):
  v,_=parts(self.p,self.y);a,b=torch.autograd.grad(v,[self.p,self.y],allow_unused=True);self.assertIsNone(b);self.assertTrue(torch.isfinite(a).all());self.assertGreater(float(a.norm()),0)
 def test_05_gradcheck(self):
  self.assertTrue(torch.autograd.gradcheck(lambda z:parts(z,self.y[:1,:8,:1])[0],(self.p[:1,:8,:1],),eps=1e-6,atol=1e-6))
 def test_06_joint_psd(self):
  p=self.p.detach().transpose(1,2).unfold(-1,4,4);y=self.y.detach().transpose(1,2).unfold(-1,4,4)
  a,b,c=kernel(p,p),kernel(p,y),kernel(y,y);joint=torch.cat([torch.cat([a,b],-1),torch.cat([b.transpose(-1,-2),c],-1)],-2);self.assertGreaterEqual(float(torch.linalg.eigvalsh(joint).min()),-1e-12)
 def test_07_formula(self):
  p=self.p.transpose(1,2).unfold(-1,4,4);q=self.y.transpose(1,2).unfold(-1,4,4);d=((p.unsqueeze(-2)-q.unsqueeze(-3))**2).sum(-1);i=torch.arange(8,dtype=torch.float64);t=torch.exp(-((i[:,None]-i[None,:])**2)/8)
  torch.testing.assert_close(kernel(p,q),t/(1+d/2),rtol=5e-16,atol=2e-16)
 def test_08_unit_diagonal(self):
  p=self.p.transpose(1,2).unfold(-1,4,4);torch.testing.assert_close(kernel(p,p).diagonal(dim1=-2,dim2=-1),torch.ones(2,3,8,dtype=torch.float64),rtol=0,atol=0)
 def test_09_batch_channel_independence(self):
  a=parts(self.p,self.y)[0];b=torch.stack([parts(self.p[i:i+1,:,c:c+1],self.y[i:i+1,:,c:c+1])[0] for i in range(2) for c in range(3)]).mean();torch.testing.assert_close(a,b)
 def test_10_time_position_retained(self):
  p=torch.tensor([0.,0.,0.,0.,1.,1.,1.,1.,3.,3.,3.,3.]).reshape(1,12,1);self.assertGreater(float(parts(p,p.flip(1))[0]),0)
 def test_11_flat_offset_distinguished(self):
  p=torch.zeros(1,720,1);self.assertGreater(float(parts(p,torch.ones_like(p))[0]),0)
 def test_12_shift_and_joint_sign_symmetry(self):
  a=parts(self.p,self.y)[0]
  torch.testing.assert_close(a,parts(self.p+2,self.y+2)[0],rtol=1e-13,atol=1e-13)
  torch.testing.assert_close(a,parts(-self.p,-self.y)[0],rtol=0,atol=0)
 def test_13_cross_term_identity(self):
  p=torch.zeros(1,32,1,dtype=torch.float64);v,d=parts(p,p+1);self.assertAlmostEqual(float(v),d['mmd_prediction_self']+d['mmd_target_self']-2*d['mmd_cross'],places=13)
 def test_14_extreme_finite(self):
  p=torch.full((1,8,1),1000.,requires_grad=True);y=torch.full_like(p,-1000.);v,_=parts(p,y);g=torch.autograd.grad(v,p)[0];self.assertTrue(torch.isfinite(v) and torch.isfinite(g).all())
 def test_15_invalid(self):
  for p in (torch.zeros(1,7,1),torch.full((1,8,1),float('nan'))):
   with self.assertRaises(ValueError):parts(p,p)
  with self.assertRaises(ValueError):parts(self.p,self.y,alpha=0)
 def test_21_rbf_limit(self):
  p=self.p.transpose(1,2).unfold(-1,4,4);q=self.y.transpose(1,2).unfold(-1,4,4)
  torch.testing.assert_close(kernel(p,q,alpha=1e7),kernel(p,q,family='rbf'),rtol=2e-5,atol=1e-7)
 def test_22_tail_gradient_and_selfgram_ambiguity(self):
  a=torch.zeros(1,1,4,dtype=torch.float64,requires_grad=True);b=torch.ones_like(a)*2
  rq=kernel(a,b).sum();rb=kernel(a,b,family='rbf').sum()
  self.assertGreater(float(torch.autograd.grad(rq,a,retain_graph=True)[0].norm()),float(torch.autograd.grad(rb,a)[0].norm()))
  p=self.p.detach().transpose(1,2).unfold(-1,4,4);torch.testing.assert_close(kernel(p,p),kernel(-p,-p),rtol=0,atol=0)
  self.assertGreater(float(parts(self.p,-self.p)[0].detach()),0)
 def test_23_full720_numpy_and_dtype(self):
  g=torch.Generator().manual_seed(8);p=torch.randn(2,720,7,generator=g);y=torch.randn(2,720,7,generator=g);v,d=parts(p,y);self.assertEqual(v.dtype,p.dtype)
  n=numpy_parts(p.numpy(),y.numpy())
  for k in n:self.assertAlmostEqual(d[k] if k!='time_position_mmd' else float(v),n[k],places=8)

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
