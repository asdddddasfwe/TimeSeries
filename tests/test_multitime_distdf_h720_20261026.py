"""Numerical loss checks and fresh model preservation for both authorized runs."""
import unittest
import numpy as np,torch
from tmp.mafirst_multitime_distdf_20261026_common import *
from utils.time_position_multiband_patch_mmd_20261026 import parts,kernel,time_kernel,numpy_parts
from utils.time_position_patch_mmd import parts as old_parts
torch.set_num_threads(2)
class MultiTimeTests(unittest.TestCase):
 def setUp(self):
  g=torch.Generator().manual_seed(43);self.p=torch.randn(2,32,3,generator=g,dtype=torch.float64,requires_grad=True);self.y=torch.randn(2,32,3,generator=g,dtype=torch.float64,requires_grad=True)
 def test_01_numpy(self):
  v,d=parts(self.p,self.y);n=numpy_parts(self.p.detach().numpy(),self.y.detach().numpy())
  for k in n:self.assertAlmostEqual(d[k],n[k],places=12)
 def test_02_single_band_original(self):
  a,_=parts(self.p,self.y,sigmas=(2.,),weights=(1.,));b,_=old_parts(self.p,self.y)
  self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(torch.autograd.grad(a,self.p)[0],torch.autograd.grad(b,self.p)[0]))
 def test_03_psd_and_mass(self):
  for n in (2,8,180):
   t=time_kernel(n);i=torch.arange(n,dtype=torch.float64);b=torch.exp(-((i[:,None]-i[None,:])**2)/8)
   self.assertAlmostEqual(float(t.mean()),float(b.mean()),places=14);self.assertGreaterEqual(float(torch.linalg.eigvalsh(t).min()),-1e-12);self.assertTrue(torch.all(t>=0));torch.testing.assert_close(t,t.T)
 def test_04_joint_psd(self):
  p=self.p.transpose(1,2).unfold(-1,4,4);y=self.y.transpose(1,2).unfold(-1,4,4)
  a,b,c=kernel(p,p),kernel(p,y),kernel(y,y);j=torch.cat([torch.cat([a,b],-1),torch.cat([b.transpose(-1,-2),c],-1)],-2)
  self.assertGreaterEqual(float(torch.linalg.eigvalsh(j).min()),-1e-12)
 def test_05_detach_and_gradcheck(self):
  a=parts(self.p,self.y)[0];g,t=torch.autograd.grad(a,[self.p,self.y],allow_unused=True);self.assertIsNone(t);self.assertTrue(torch.isfinite(g).all())
  self.assertTrue(torch.autograd.gradcheck(lambda z:parts(z,self.y[:1,:8,:1])[0],(self.p[:1,:8,:1],),atol=1e-6))
 def test_06_self_and_flat_offset(self):
  self.assertLess(abs(float(parts(self.p,self.p)[0].detach())),1e-14)
  p=torch.zeros(1,720,1);self.assertGreater(float(parts(p,p+1)[0]),0)
 def test_07_linear_combination(self):
  a=parts(self.p,self.y)[0];n=self.p.shape[1]//4;mass=time_kernel(n,sigmas=(2.,),weights=(1.,)).mean()
  values=[]
  for s in (2.,24.,90.):
   # Single-band helper already matches sigma2 mass.
   values.append(parts(self.p,self.y,sigmas=(s,),weights=(1.,))[0])
  torch.testing.assert_close(a,torch.stack(values).mean(),rtol=1e-13,atol=1e-13)
 def test_08_full720_dtype(self):
  g=torch.Generator().manual_seed(8);p=torch.randn(2,720,7,generator=g);y=torch.randn(2,720,7,generator=g);v,d=parts(p,y);self.assertEqual(v.dtype,p.dtype);n=numpy_parts(p.numpy(),y.numpy())
  for k in n:self.assertAlmostEqual(d[k],n[k],places=8)
 def test_09_invalid(self):
  with self.assertRaises(ValueError):parts(torch.zeros(1,7,1),torch.zeros(1,7,1))
  with self.assertRaises(ValueError):time_kernel(180,weights=(-1.,1.,1.))
 def test_10_long_pair_preserved(self):
  self.assertGreater(float(time_kernel(180)[0,24]),float(time_kernel(180,sigmas=(2.,),weights=(1.,))[0,24]))
class ModelTests(unittest.TestCase):
 def test_11_original_preservation(self):
  from tmp.mafirst_multitime_20261026_factory import fresh,optimizer_groups,groups_audit,original
  original.shared.seed_all(2021);m,c=fresh(G/'cpu_tests/logs');groups_audit(m,optimizer_groups(m));self.assertEqual(len(m.state_dict()),236);self.assertEqual(sum(p.numel() for p in m.parameters()),320450)
  m.eval();x=torch.randn(2,96,7)
  with torch.no_grad():
   p=m(x);parts_=m.predict_components(x);q=m.normalize(parts_[...,:4].sum(-1)+parts_[...,4:].sum(-1),'denorm')
  self.assertTrue(torch.equal(p,q));self.assertEqual(tuple(p.shape),(2,720,7))
 def test_12_distdf_preservation_and_loss(self):
  from tmp.distdf_h720_matched_20261026_factory import fresh,optimizer_groups,groups_audit,distdf_loss,source
  source.seed(2021);m,c=fresh(G/'cpu_tests/logs_distdf');groups_audit(m,optimizer_groups(m));x=torch.randn(2,96,7);y=torch.randn(2,720,7,requires_grad=True);m.train();p=m(x);v=distdf_loss(p,y,x);loss=.5*(p-y.detach()).square().mean()+.5*v
  self.assertEqual(tuple(p.shape),(2,720,7));self.assertTrue(torch.isfinite(loss))
  ps=list(m.parameters());gs=torch.autograd.grad(loss,ps+[y],allow_unused=True);self.assertIsNone(gs[-1]);inactive=[n for (n,p),g in zip(m.named_parameters(),gs[:-1]) if g is None]
  self.assertEqual(inactive,read(G/'reference_initial/distdf_inactive.json'));self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gs))
 def test_13_data_same_for_two_models(self):
  from tmp.mafirst_multitime_20261026_factory import original
  from tmp.distdf_h720_matched_20261026_factory import config
  cfg=config();v=original.dataset(cfg,'val');x,y,_,_=v[0];b=BASE/'time_position_mmd/val'
  np.testing.assert_array_equal(x.astype('float32'),np.load(b/'history.npy',mmap_mode='r')[0]);np.testing.assert_array_equal(y[-720:].astype('float32'),np.load(b/'truth.npy',mmap_mode='r')[0])
if __name__=='__main__':unittest.main(verbosity=2)
