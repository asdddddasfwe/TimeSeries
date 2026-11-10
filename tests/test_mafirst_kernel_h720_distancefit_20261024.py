"""54 inherited regressions plus20 tests for one new distance-supervision trial."""
from test_mafirst_kernel_h720_globaljoint_20261024 import AffineTests,GlobalJointTests
import json,unittest,tempfile
from pathlib import Path
import numpy as np,torch
from utils.patch_distance_huber_20261024 import squared_patch_distances,head_distances,distance_parts,TERM_NAMES
from tmp.mafirst_kernel_h720_distancefit_20261024_losses import training_terms,gradient_summary,SCALES
from tmp.mafirst_kernel_h720_distancefit_20261024_factory import fresh,optimizer_groups,G,R
class DistanceTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.manual_seed(47)
 def sample(self):
  p=torch.randn(2,24,7,requires_grad=True);y=torch.randn_like(p,requires_grad=True);h=torch.sigmoid(torch.randn(14,6,6)).detach().requires_grad_();return p,y,h
 def test_distance01_numpy_distances(self):
  p,_,_=self.sample();a=p.detach().numpy().transpose(0,2,1).reshape(14,6,4).astype(float);z=((a[:,:,None]-a[:,None,:])**2).sum(-1);np.testing.assert_allclose(squared_patch_distances(p).detach().numpy(),z,atol=1e-12,rtol=1e-12)
 def test_distance02_inverse_RBF(self):
  p,_,_=self.sample();d=squared_patch_distances(p.detach());self.assertTrue(torch.allclose(head_distances(torch.exp(-d*.5)),d.clamp_max(-2*np.log(1e-12)),atol=1e-12,rtol=1e-12))
 def test_distance03_numpy_Huber(self):
  p,y,h=self.sample();z=distance_parts(p,y,h,SCALES);mask=~np.eye(6,dtype=bool);sc=np.tile(SCALES,2).reshape(-1,1,1)
  for key,loss in [('head_distance','head_huber'),('output_distance','output_huber')]:
   e=(z[key].detach().numpy()-z['target_distance'].numpy())/sc;q=np.where(abs(e)<=1,.5*e**2,abs(e)-.5);self.assertAlmostEqual(float(z[loss].detach()),q[:,mask].mean(),places=5)
 def test_distance04_symmetry_diag(self):
  p,_,_=self.sample();d=squared_patch_distances(p);self.assertTrue(torch.equal(d,d.transpose(-1,-2)));self.assertEqual(torch.count_nonzero(d.diagonal(dim1=-1,dim2=-2)).item(),0)
 def test_distance05_batch_channel_scale_order(self):
  p,y,h=self.sample();a=distance_parts(p,y,h,SCALES);self.assertEqual(a['scales'][:,0,0].tolist(),list(SCALES)*2)
 def test_distance06_target_detached(self):
  p,y,h=self.sample();z=distance_parts(p,y,h,SCALES);self.assertFalse(z['target_distance'].requires_grad);self.assertIsNone(torch.autograd.grad(z['head_huber']+z['output_huber'],y,allow_unused=True)[0])
 def test_distance07_finite_nonzero_gradients(self):
  p,y,h=self.sample();z=distance_parts(p,y,h,SCALES);g=torch.autograd.grad(z['head_huber']+z['output_huber'],[p,h]);self.assertTrue(all(torch.isfinite(x).all() and x.norm()>0 for x in g))
 def test_distance08_flat_output_stationary_expected(self):
  p,y,h=self.sample();p=torch.ones_like(p,requires_grad=True);z=distance_parts(p,y,h,SCALES);g=torch.autograd.grad(z['output_huber'],p)[0];self.assertEqual(torch.count_nonzero(g).item(),0)
 def test_distance09_nonflat_output_not_stationary(self):
  p,y,h=self.sample();p=(torch.ones_like(p)+torch.randn_like(p)*1e-3).detach().requires_grad_();z=distance_parts(p,y,h,SCALES);self.assertGreater(float(torch.autograd.grad(z['output_huber'],p)[0].norm()),0)
 def test_distance10_clamp_boundaries_finite(self):
  h=torch.tensor([0.,1e-20,1e-12,.5,1.],requires_grad=True);d=head_distances(h);d.sum().backward();self.assertTrue(torch.isfinite(d).all() and torch.isfinite(h.grad).all());self.assertEqual(float(d[-1]),0.)
 def test_distance11_invalid_scale(self):
  p,y,h=self.sample()
  with self.assertRaises(ValueError):distance_parts(p,y,h,[0.]*7)
 def test_distance12_reference_model_and_optimizer(self):
  with tempfile.TemporaryDirectory(dir=R/'tmp') as t:
   m,c=fresh(720,'h720_distance_fit',Path(t)/'logs');s=torch.load(G/'reference_initial/h720.pt',weights_only=True);self.assertEqual(len(s),259);self.assertTrue(all(torch.equal(v,s[n]) for n,v in m.state_dict().items()));groups=optimizer_groups(m);self.assertEqual(sum(len(x['params']) for x in groups),227)
 def test_distance13_three_terms_full_model_finite(self):
  with tempfile.TemporaryDirectory(dir=R/'tmp') as t:
   m,_=fresh(720,'h720_distance_fit',Path(t)/'logs');p=m(torch.randn(1,96,7));terms,d=training_terms(p,torch.randn_like(p),m.predicted_gram,'none');self.assertEqual(tuple(terms),TERM_NAMES);sum(terms.values()).backward();self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in m.parameters()))
 def test_distance14_three_raw_pairs(self):
  p,y,h=self.sample();z=distance_parts(p,y,h,SCALES);terms={'point_mse':(p-y.detach()).square().mean(),'head_distance_huber':z['head_huber'],'output_distance_huber':z['output_huber']};g=gradient_summary(terms,[('prediction',p),('head',h)]);self.assertEqual(len(g['all']['pairs']),3)
 def test_distance15_legacy_diagnostic_detached(self):
  p=torch.randn(1,720,7,requires_grad=True);y=torch.randn_like(p);h=torch.sigmoid(torch.randn(7,180,180,requires_grad=True));terms,d=training_terms(p,y,h,'none');self.assertTrue(all(not v.requires_grad for v in d['legacy_terms'].values()));self.assertFalse(d['output']['local_cs'].requires_grad)
 def test_distance16_runner_applied_sum(self):
  s=(R/'tmp/run_mafirst_kernel_h720_distancefit_20261024.py').read_text();self.assertIn('loss=sum(terms.values())',s);self.assertIn("train_head_distance_huber",s);self.assertNotIn("terms['output_local_cs']",s)
 def test_distance17_normalization_and_components(self):
  with tempfile.TemporaryDirectory(dir=R/'tmp') as t:
   m,_=fresh(720,'h720_distance_fit',Path(t)/'logs');calls={'norm':0,'denorm':0}
   def hook(mod,args,out):calls[args[1]]+=1
   hh=m.normalize.register_forward_hook(hook);z=m.predict_components(torch.randn(1,96,7));p=m.normalize(z[...,:4].sum(-1)+z[...,4:].sum(-1),'denorm');hh.remove();self.assertEqual(calls,{'norm':1,'denorm':1});scale,off=m.inverse_coordinates();self.assertLess(float((p-(z*scale.unsqueeze(-1)).sum(-1)-off).detach().abs().max()),2e-5)
 def test_distance18_training_scale_only(self):
  d=json.loads((G/'distance_scale.json').read_text());self.assertFalse(d['validation_or_test_used']);self.assertEqual(len(d['origins']),128);self.assertLessEqual(max(d['origins'])+96+720,34560);self.assertEqual(tuple(d['scales']),SCALES)
 def test_distance19_gradient_finite_difference(self):
  p=torch.randn(1,12,1,dtype=torch.double,requires_grad=True);y=torch.randn_like(p);h=torch.full((1,3,3),.5,dtype=torch.double);loss=distance_parts(p,y,h,[2.])['output_huber'];grad=torch.autograd.grad(loss,p)[0];eps=1e-5;q=p.detach().clone();q[0,0,0]+=eps;v=p.detach().clone();v[0,0,0]-=eps;fd=(distance_parts(q,y,h,[2.])['output_huber']-distance_parts(v,y,h,[2.])['output_huber'])/(2*eps);self.assertAlmostEqual(float(grad[0,0,0]),float(fd),places=7)
 def test_distance20_fixed_scale_and_zero_truth_finite(self):
  p,y,h=self.sample();y=torch.zeros_like(y);z=distance_parts(p,y,h,SCALES);self.assertTrue(torch.isfinite(z['head_huber']+z['output_huber']));self.assertEqual(z['scales'][:,0,0].tolist(),list(SCALES)*2)
if __name__=='__main__':unittest.main()
