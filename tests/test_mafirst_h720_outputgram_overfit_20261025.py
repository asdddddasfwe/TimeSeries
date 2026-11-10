"""Focused tests for actual final-output Gram-only fixed-window diagnosis."""
import json,unittest,tempfile
from pathlib import Path
import numpy as np,torch
from tmp.run_mafirst_h720_outputgram_overfit_20261025 import grams,objective,metrics,data,fresh,optimizer,thresholds_met,G,R
from utils.final_output_gram_losses_20261024 import output_parts
class OutputGramOverfitTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.manual_seed(47)
 def test01_exact_frozen_output_Gram_loss(self):
  p=torch.randn(1,720,7,requires_grad=True);y=torch.randn_like(p);a,g=objective(p,grams(y));old=output_parts(p,y);self.assertTrue(torch.equal(g,old['predicted']));self.assertTrue(torch.equal(a,old['mse']))
 def test02_independent_NumPy_RBF(self):
  p=torch.randn(1,720,7);q=p.numpy().transpose(0,2,1).reshape(7,180,4).astype(float);g=np.exp(-.5*((q[:,:,None]-q[:,None,:])**2).sum(-1));np.testing.assert_allclose(grams(p).numpy(),g,atol=2e-6,rtol=2e-6)
 def test03_truth_zero_and_diagonal_excluded(self):
  p=torch.randn(1,720,7,requires_grad=True);g=grams(p).detach();v,_=objective(p,g);self.assertEqual(float(v.detach()),0.);bad=g.clone();bad.diagonal(dim1=-1,dim2=-2).fill_(99.);z,_=objective(p,bad);self.assertEqual(float(z.detach()),0.)
 def test04_target_detached_gradient(self):
  p=torch.randn(1,720,7,requires_grad=True);y=torch.randn_like(p,requires_grad=True);v,_=objective(p,grams(y));g=torch.autograd.grad(v,[p,y],allow_unused=True);self.assertIsNone(g[1]);self.assertTrue(torch.isfinite(g[0]).all());self.assertGreater(float(g[0].norm()),0)
 def test05_flat_stationary_and_nonflat_finite(self):
  y=torch.randn(1,720,7);p=torch.ones_like(y,requires_grad=True);v,_=objective(p,grams(y));g=torch.autograd.grad(v,p)[0];self.assertEqual(torch.count_nonzero(g).item(),0)
  p=(torch.ones_like(y)+torch.randn_like(y)*.01).requires_grad_();v,_=objective(p,grams(y));self.assertGreater(float(torch.autograd.grad(v,p)[0].norm()),0)
 def test06_train_windows_only_bounds(self):
  x,y,sc=data();self.assertEqual(tuple(x.shape),(4,96,7));self.assertEqual(tuple(y.shape),(4,720,7));plan=json.loads((G/'plan.json').read_text());self.assertEqual(plan['origins'],[0,8000,16000,24000]);self.assertTrue(all(i+816<=34560 for i in plan['origins']));self.assertTrue(plan['no_validation_or_test'])
 def test07_fresh_reference_dropout_and_decay(self):
  with tempfile.TemporaryDirectory(dir=R/'tmp') as t:
   m,c=fresh(Path(t));self.assertFalse(any(v.training for v in m.modules()));self.assertEqual((sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers()))),(906416,227,259,32));opt=optimizer(m);self.assertTrue(all(v['weight_decay']==0 for v in opt.param_groups));self.assertTrue(all(p.requires_grad for p in m.parameters()))
 def test08_actual_model_final_Gram_backward_and_normalization(self):
  with tempfile.TemporaryDirectory(dir=R/'tmp') as t:
   m,c=fresh(Path(t));x,y,_=data();calls={'norm':0,'denorm':0}
   def hook(mod,args,out):calls[args[1]]+=1
   h=m.normalize.register_forward_hook(hook);p=m(x[:1]);loss,g=objective(p,grams(y[:1]).detach());loss.backward();h.remove();self.assertEqual(calls,{'norm':1,'denorm':1});gr=[v.grad for v in m.parameters() if v.grad is not None];self.assertTrue(gr and all(torch.isfinite(v).all() for v in gr));self.assertTrue(m.global_seasonal_delta.grad is not None and torch.isfinite(m.global_seasonal_delta.grad).all())
 def test09_metric_global_mean_and_per_matrix(self):
  p=torch.randn(4,720,7);y=torch.randn_like(p);v,g=objective(p,grams(y));m=metrics(p,y,g,grams(y));self.assertAlmostEqual(m['gram_mse'],float(v),places=7);self.assertEqual(np.asarray(m['per_matrix_mse']).shape,(4,7));self.assertGreaterEqual(m['max_matrix_mse'],m['gram_mse'])
 def test10_translation_sign_invariance_not_point_accuracy(self):
  p=torch.randn(1,720,7);y=torch.randn_like(p);a,_=objective(p,grams(y));b,_=objective(-p+3.,grams(y));self.assertAlmostEqual(float(a),float(b),places=7)
 def test11_success_requires_mean_and_worst_matrix(self):
  plan={'success_mean_MSE':1e-4,'success_max_matrix_MSE':1e-3}
  self.assertFalse(thresholds_met({'gram_mse':1e-5,'max_matrix_mse':.002},plan));self.assertFalse(thresholds_met({'gram_mse':.0002,'max_matrix_mse':.0005},plan));self.assertTrue(thresholds_met({'gram_mse':1e-4,'max_matrix_mse':1e-3},plan))
if __name__=='__main__':unittest.main()
