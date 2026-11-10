"""Focused numerical and protocol tests for same-model point-only control."""
import unittest,tempfile,json
from pathlib import Path
import torch,numpy as np
from tmp.run_mafirst_h720_pointonly_20261025 import objective,point_metrics,fixed_data,data_for,hook_calls,small_passed,groups_audit,batches,R,G,PLAN,read
from tmp.mafirst_h720_pointonly_20261025_factory import fresh,optimizer_groups
class PointOnlyTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.manual_seed(47)
 def test01_value_equals_independent_numpy(self):
  p=torch.randn(2,720,7);y=torch.randn_like(p);self.assertAlmostEqual(float(objective(p,y)),float(((p.numpy()-y.numpy())**2).mean()),places=6)
 def test02_point_gradient_and_target_detach(self):
  p=torch.randn(1,720,7,requires_grad=True);y=torch.randn_like(p,requires_grad=True);g,gy=torch.autograd.grad(objective(p,y),(p,y),allow_unused=True);self.assertIsNone(gy);torch.testing.assert_close(g,2*(p-y.detach())/p.numel())
 def test03_flat_output_has_nonzero_point_gradient(self):
  p=torch.zeros(1,720,7,requires_grad=True);y=torch.ones_like(p);g=torch.autograd.grad(objective(p,y),p)[0];self.assertTrue(torch.all(g!=0))
 def test04_sign_offset_and_fourphase_errors_penalized(self):
  y=torch.randn(1,720,7);self.assertEqual(float(objective(y,y)),0.);self.assertGreater(float(objective(-y+3,y)),1.);phase=torch.tensor([1.,-1.,1.,-1.]).repeat(180)[None,:,None];self.assertAlmostEqual(float(objective(y+phase,y)),1.,places=6)
 def test05_fixed_windows_training_only(self):
  x,y=fixed_data();self.assertEqual(x.shape,(4,96,7));self.assertEqual(y.shape,(4,720,7));self.assertTrue(all(i+816<=34560 for i in read(PLAN)['small_sample']['origins']))
 def test06_split_scaler_and_windows_match_D(self):
  train,val,sc,a=data_for();self.assertEqual((len(train),len(val)),(33745,10801));d=read(R/'tmp/mafirst_kernel_h720_globaljoint_20261024_h720_global_joint_full_seed2021/data_manifest.json');np.testing.assert_array_equal(sc.mean_,d['scaler_mean']);np.testing.assert_array_equal(sc.scale_,d['scaler_scale']);np.testing.assert_array_equal(train[0][0],a[:96]);np.testing.assert_array_equal(val[0][1],a[34560:35280])
 def test07_same_reference_and_optimizer(self):
  with tempfile.TemporaryDirectory(dir=R/'tmp') as t:
   m,c=fresh(Path(t));self.assertEqual((sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers()))),(906416,227,259,32));v=groups_audit(m,optimizer_groups(m));d=read(R/'tmp/mafirst_kernel_h720_globaljoint_20261024_h720_global_joint_full_seed2021/optimizer_groups.json');self.assertEqual(v,d);self.assertEqual(torch.count_nonzero(m.global_seasonal_delta),0)
 def test08_real_model_point_backward_and_normalization(self):
  with tempfile.TemporaryDirectory(dir=R/'tmp') as t:
   m,c=fresh(Path(t));m.eval();x,y=fixed_data();fw,h=hook_calls(m);p=fw(x[:1]);objective(p,y[:1]).backward();h.remove();self.assertEqual(p.shape,(1,720,7));self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()));self.assertGreater(float(m.global_seasonal_delta.grad.norm()),0.);self.assertTrue(all(torch.isfinite(p).all() for n,p in m.named_parameters() if n.endswith(('affine_weight','affine_bias'))))
 def test09_sampler_order_matches_D(self):
  train,_,_,_=data_for();loader=batches(train,1);h=__import__('hashlib').sha256(np.asarray(loader.dataset.indices,dtype=np.int64).tobytes()).hexdigest();d=read(R/'tmp/mafirst_kernel_h720_globaljoint_20261024_h720_global_joint_full_seed2021/curves.json');self.assertEqual(h,d[0]['order_sha256']);self.assertEqual(len(loader),1055)
 def test10_both_gate_thresholds_required(self):
  p=read(PLAN)['small_sample'];self.assertFalse(small_passed(dict(mse=1e-5,max_channel_mse=1e-4,max_abs_error=.3),p));self.assertFalse(small_passed(dict(mse=1e-5,max_channel_mse=.002,max_abs_error=.01),p));self.assertFalse(small_passed(dict(mse=.0002,max_channel_mse=.0005,max_abs_error=.01),p));self.assertTrue(small_passed(dict(mse=1e-4,max_channel_mse=.001,max_abs_error=.2),p))
 def test11_independent_per_channel_metrics(self):
  p=torch.randn(4,720,7);y=torch.randn_like(p);m=point_metrics(p,y);e=p.double().numpy()-y.double().numpy();np.testing.assert_allclose(m['per_channel_mse'],(e*e).mean(1),atol=1e-12);self.assertAlmostEqual(m['mse'],(e*e).mean(),places=12)
if __name__=='__main__':unittest.main()
