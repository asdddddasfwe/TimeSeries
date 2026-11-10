import unittest,torch,numpy as np
from tmp.mafirst_original_h720_wideband_20261026_common import *
from tmp.mafirst_original_h720_wideband_20261026_factory import fresh,optimizer_groups,groups_audit,original
from tmp.mafirst_wideband_kernel_20261026 import parts,numpy_parts
from utils.time_position_patch_mmd import parts as base
torch.set_num_threads(2)
class Kernel(unittest.TestCase):
 def setUp(self):
  g=torch.Generator().manual_seed(37);self.p=torch.randn(2,32,3,dtype=torch.float64,generator=g,requires_grad=True);self.y=torch.randn(2,32,3,dtype=torch.float64,generator=g,requires_grad=True)
 def test_01_all_terms_independent_numpy(self):
  v,d=parts(self.p,self.y);n=numpy_parts(self.p.detach().numpy(),self.y.detach().numpy())
  for k in n:self.assertAlmostEqual(d[k],n[k],places=13)
 def test_02_exact_original_parameter_adapter(self):
  a,d=parts(self.p,self.y);b,e=base(self.p,self.y,sigma_value=2.,sigma_time=24.)
  self.assertTrue(torch.equal(a,b));self.assertEqual(d,e);self.assertTrue(torch.equal(torch.autograd.grad(a,self.p)[0],torch.autograd.grad(b,self.p)[0]))
 def test_03_target_detached(self):
  a,b=torch.autograd.grad(parts(self.p,self.y)[0],(self.p,self.y),allow_unused=True);self.assertIsNone(b);self.assertTrue(torch.isfinite(a).all());self.assertGreater(float(a.norm()),0)
 def test_04_selfzero(self):self.assertLess(abs(float(parts(self.p,self.p)[0].detach())),1e-14)
 def test_05_gradcheck(self):self.assertTrue(torch.autograd.gradcheck(lambda p:parts(p,self.y[:1,:8,:1])[0],(self.p[:1,:8,:1],),atol=1e-6))
 def test_06_joint_psd_unit_diagonal(self):
  p=self.p.detach().transpose(1,2).unfold(-1,4,4);y=self.y.detach().transpose(1,2).unfold(-1,4,4);z=torch.cat([p,y],-2);i=torch.arange(8).repeat(2);t=torch.exp(-.5*((i[:,None]-i[None,:])/24.)**2).double()
  k=t*torch.exp(-.5*((z.unsqueeze(-2)-z.unsqueeze(-3))/2.).square().sum(-1))
  self.assertGreater(float(torch.linalg.eigvalsh(k).min()),-1e-7);torch.testing.assert_close(k.diagonal(dim1=-2,dim2=-1),torch.ones(2,3,16,dtype=torch.float64),rtol=0,atol=0)
 def test_07_batch_variable_independence(self):
  a=parts(self.p,self.y)[0];b=torch.stack([parts(self.p[b:b+1,:,c:c+1],self.y[b:b+1,:,c:c+1])[0] for b in range(2) for c in range(3)]).mean();torch.testing.assert_close(a,b)
 def test_08_full720_dtype_numpy(self):
  g=torch.Generator().manual_seed(3);p=torch.randn(2,720,7,generator=g);y=torch.randn(2,720,7,generator=g);a,d=parts(p,y);n=numpy_parts(p.numpy(),y.numpy());self.assertEqual(a.dtype,p.dtype)
  for k in n:self.assertAlmostEqual(n[k],d[k],places=7)
 def test_09_invalid(self):
  with self.assertRaises(ValueError):parts(torch.zeros(1,7,1),torch.zeros(1,7,1))
 def test_10_time_and_value_weights_grow(self):
  for dt in (8.,24.,48.):self.assertGreater(np.exp(-.5*(dt/24)**2),np.exp(-.5*(dt/2)**2))
  for d in (1.,4.,10.):self.assertGreater(np.exp(-d/8),np.exp(-d/2))
class Model(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m,cls.cfg=fresh(G/'cpu_tests/logs')
 def test_11_scratch236_independent(self):
  ref=torch.load(G/'reference_initial.pt',map_location='cpu',weights_only=True);self.assertEqual(len(ref),236)
  self.assertTrue(all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in self.m.state_dict().items()))
 def test_12_optimizer204_original_rules(self):groups_audit(self.m,optimizer_groups(self.m))
 def test_13_components_and_shape(self):
  self.m.eval();x=torch.randn(2,96,7)
  with torch.no_grad():p=self.m(x);c=self.m.predict_components(x);q=self.m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm')
  self.assertTrue(torch.equal(p,q));self.assertEqual(p.shape,(2,720,7))
 def test_14_two_losses_all204_finite(self):
  self.m.train();original.shared.seed_all(123);p=self.m(torch.randn(2,96,7));y=torch.randn(2,720,7)
  for t in ((p-y).square().mean(),parts(p,y)[0]):
   gs=torch.autograd.grad(t,list(self.m.parameters()),retain_graph=True);self.assertTrue(all(torch.isfinite(g).all() for g in gs))
 def test_15_data_first_last_validation_exact(self):
  ds=original.dataset(self.cfg,'val');a=np.load(BASE/'time_position_mmd/val/history.npy',mmap_mode='r');b=np.load(BASE/'time_position_mmd/val/truth.npy',mmap_mode='r')
  for index in (0,len(ds)-1):
   x,y,_,_=ds[index];np.testing.assert_array_equal(x.astype('float32'),a[index]);np.testing.assert_array_equal(y[-720:].astype('float32'),b[index])
if __name__=='__main__':unittest.main(verbosity=2)
