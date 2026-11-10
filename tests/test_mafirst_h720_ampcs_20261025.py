import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import sys,tempfile,unittest,hashlib
import numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));torch.set_num_threads(2)
from utils.amplitude_kernel_local_cs_20261025 import *
from utils.amplitude_weighted_patch_kernel_20261025 import kernel_matrices
from utils.local_offdiagonal_gram_cs import numpy_parts,parts as cs_parts
from utils.drop_output_gram_mse_20261024 import training_terms as D_terms
from models.MAFirstKernelTransferAblation import value_gram
from tmp.mafirst_h720_ampcs_20261025_factory import fresh,optimizer_groups
from tmp.run_mafirst_h720_ampcs_20261025 import data_for,batches,groups_audit,hook_calls,read,PLAN
from tmp.evaluate_mafirst_h720_ampcs_20261025 import evaluate

def old_builder(x):
 b,h,c=x.shape;k=value_gram(x.transpose(1,2).reshape(b*c,h//4,4));return k[:,None].expand(-1,4,-1,-1)
def sample():
 torch.manual_seed(77);p=torch.randn(1,96,2);y=torch.randn_like(p);return p,y
class Tests(unittest.TestCase):
 def test01_four_unit_terms(self):
  self.assertEqual(WEIGHTS,{n:1. for n in ('point_mse','head_gram_mse','head_local_cs','output_local_cs')})
 def test02_numpy_CS(self):
  p,y=sample();k,t=kernel_matrices(p).numpy(),kernel_matrices(y).numpy();npv=np.mean([numpy_parts(k[:,r],t[:,r])['local_cs_loss'] for r in range(4)]);self.assertAlmostEqual(float(output_parts(p,y)['local_cs']),npv,places=6)
 def test03_four_channels_not_averaged_before_loss(self):
  p,y=sample();k,t=kernel_matrices(p),kernel_matrices(y);a=output_parts(p,y)['local_cs'];b=cs_parts(k,t)[0];wrong=cs_parts(k.mean(1),t.mean(1))[0];self.assertAlmostEqual(float(a),float(b),places=6);self.assertGreater(abs(float(a-wrong)),1e-4)
 def test04_exact_target(self):
  p,y=sample();self.assertLess(abs(float(output_parts(p,p)['local_cs'])),1e-10)
 def test05_diagonal_still_excluded(self):
  p,y=sample()
  def builder(x):
   k=kernel_matrices(x);mask=torch.eye(k.shape[-1],dtype=torch.bool);return torch.where(mask,torch.full_like(k,999),k)
  self.assertAlmostEqual(float(output_parts(p,y)['local_cs']),float(output_parts(p,y,builder)['local_cs']),places=6)
 def test06_wrong_flat_output_not_identified_by_CS(self):
  p=torch.zeros(1,96,2);y=torch.ones_like(p);self.assertLess(abs(float(output_parts(p,y)['local_cs'])),1e-10);self.assertEqual(float((p-y).square().mean()),1.)
 def test07_target_detached(self):
  p,y=sample();p.requires_grad_();y.requires_grad_();head=torch.full((2,24,24),.5,requires_grad=True);terms,_=training_terms(p,y,head);a,b,c=torch.autograd.grad(sum(terms.values()),(p,y,head),allow_unused=True);self.assertIsNone(b);self.assertTrue(torch.isfinite(a).all() and torch.isfinite(c).all())
 def test08_CS_gradient_finite_nonzero(self):
  p,y=sample();p.requires_grad_();g=torch.autograd.grad(output_parts(p,y)['local_cs'],p)[0];self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0.)
 def test09_FP64_gradcheck(self):
  torch.manual_seed(80);p=(torch.randn(1,16,1,dtype=torch.double)*.3).requires_grad_();y=torch.randn_like(p);self.assertTrue(torch.autograd.gradcheck(lambda a:output_parts(a,y)['local_cs'],(p,),atol=2e-5,rtol=2e-4))
 def test10_original_D_loss_recovered_when_RBF_restored(self):
  torch.manual_seed(2);p=torch.randn(1,720,7,requires_grad=True);y=torch.randn_like(p);head=torch.rand(7,180,180,requires_grad=True);old,_=D_terms(p,y,head,'none');new,_=training_terms(p,y,head,old_builder)
  self.assertEqual(tuple(old),tuple(new))
  for name in old:torch.testing.assert_close(old[name],new[name],atol=1e-7,rtol=1e-7)
  a=torch.autograd.grad(sum(old.values()),(p,head));b=torch.autograd.grad(sum(new.values()),(p,head));
  for x,z in zip(a,b):torch.testing.assert_close(x,z,atol=1e-7,rtol=2e-5)
 def test11_only_output_term_changes(self):
  torch.manual_seed(4);p=torch.randn(1,720,7);y=torch.randn_like(p);head=torch.rand(7,180,180);a,_=D_terms(p,y,head,'none');b,detail=training_terms(p,y,head)
  for name in TERM_NAMES[:3]:self.assertTrue(torch.equal(a[name],b[name]),name)
  self.assertNotEqual(float(a['output_local_cs']),float(b['output_local_cs']));self.assertEqual(len(detail['output']['phase_meta']),4)
  for meta in detail['output']['phase_meta']:self.assertEqual(meta['local_windows_per_matrix'],841);self.assertEqual(meta['stabilizer_C'],.0009);self.assertTrue(meta['diagonal_excluded'] and meta['target_flat_windows_included'])
 def test12_initialization_optimizer_exact_D(self):
  with tempfile.TemporaryDirectory(dir=R/'tmp') as folder:
   m,c=fresh(Path(folder));self.assertEqual((sum(p.numel() for p in m.parameters()),len(list(m.parameters())),len(m.state_dict()),len(list(m.buffers()))),(906416,227,259,32));D=R/read(PLAN)['baselines']['h720_d']['output'];self.assertEqual(groups_audit(m,optimizer_groups(m)),read(D/'optimizer_groups.json'))
 def test13_real_model_four_raw_gradients_and_norm_flow(self):
  with tempfile.TemporaryDirectory(dir=R/'tmp') as folder:
   m,c=fresh(Path(folder));m.eval();ds,_,_,_=data_for();x,y=ds[0];x=torch.from_numpy(x[None]);y=torch.from_numpy(y[None]);fw,h=hook_calls(m);p=fw(x);terms,_=training_terms(p,y,m.predicted_gram);raw=gradient_summary(terms,m);self.assertEqual(len(raw['norms']),4);self.assertEqual(len(raw['cosines']),6);sum(terms.values()).backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()));h.remove()
 def test14_scaler_and_sampler_exact_D(self):
  train,val,sc,data=data_for();D=R/read(PLAN)['baselines']['h720_d']['output'];d=read(D/'data_manifest.json');self.assertEqual((len(train),len(val)),(33745,10801));np.testing.assert_array_equal(sc.mean_,d['scaler_mean']);np.testing.assert_array_equal(sc.scale_,d['scaler_scale']);order=hashlib.sha256(np.asarray(batches(train,1).dataset.indices,dtype=np.int64).tobytes()).hexdigest();self.assertEqual(order,read(D/'curves.json')[0]['order_sha256'])
 def test15_diagnostic_single_forward_CPU_training_windows_only(self):
  with tempfile.TemporaryDirectory(dir=R/'tmp') as folder:
   folder=Path(folder);m,c=fresh(folder/'logs');ds,_,_,_=data_for();v=evaluate(m,torch.utils.data.Subset(ds,[0,8000,16000]),'cpu',folder/'saved');p,y=[torch.from_numpy(np.load(folder/'saved'/(n+'.npy'))) for n in ['prediction','truth']];self.assertAlmostEqual(v['amp_kernel_local_cs'],float(output_parts(p,y)['local_cs']),places=6);self.assertEqual(len(v['amp_kernel_phase_local_cs']),4)
 def test16_no_kernel_parameters_and_protocol(self):
  plan=read(PLAN);self.assertEqual(plan['training']['applied_weights'],WEIGHTS);self.assertTrue(plan['kernel']['extra_diagonal_MSE_training']==False);self.assertEqual(plan['kernel']['trainable_kernel_parameters'],0);self.assertTrue(plan['no_extra_small_fit']);self.assertFalse(plan['kernel']['ordinary_MMD_added'])
if __name__=='__main__':unittest.main(verbosity=2)
