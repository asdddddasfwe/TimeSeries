import unittest,torch,numpy as np,ast
from tests.test_original_wideband_20261026 import Kernel
from tmp.mafirst_dmodel24_mmdlambda5_distdfopt_h720_20261027_common import *
from tmp.mafirst_dmodel24_mmdlambda5_distdfopt_h720_20261027_factory import fresh,groups_audit,original
from tmp.mafirst_dmodel24_lambda5_contract_20261027 import make_optimizer,make_scheduler,training_loader
from tmp.mafirst_wideband_kernel_20261026 import parts
from tmp.mafirst_dmodel24_lambda5_objective_20261027 import combine,MMD_WEIGHT
torch.set_num_threads(2)
class Contract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m,cls.cfg=fresh(G/'cpu_tests/logs')
 def test_11_exact_scratch_and_model(self):
  ref=torch.load(G/'reference_initial.pt',map_location='cpu',weights_only=True)
  self.assertEqual(len(ref),236);self.assertEqual(sum(p.numel() for p in self.m.parameters()),read(G/'expected_model_card.json')['parameters'])
  self.assertTrue(all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in self.m.state_dict().items()))
  self.assertFalse(any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in self.m.modules()))
 def test_12_official_optimizer(self):
  opt=make_optimizer(self.m);self.assertIsInstance(opt,torch.optim.Adam);self.assertEqual(opt.defaults['lr'],.0001);self.assertEqual(opt.defaults['weight_decay'],0.);self.assertEqual(opt.defaults['betas'],(.9,.999))
  groups_audit(self.m,opt.param_groups);self.assertEqual(len(opt.param_groups),1)
 def test_13_scheduler_exact_trace(self):
  p=torch.nn.Parameter(torch.zeros(1));q=torch.nn.Parameter(torch.zeros(1));a=torch.optim.Adam([p],lr=.0001);b=torch.optim.Adam([q],lr=.0001)
  s=make_scheduler(a,263);t=torch.optim.lr_scheduler.OneCycleLR(b,max_lr=.0001,epochs=100,steps_per_epoch=263,pct_start=.4)
  self.assertAlmostEqual(a.param_groups[0]['lr'],.0001/25,places=16)
  for i in range(26300):
   self.assertEqual(a.param_groups[0]['lr'],b.param_groups[0]['lr']);self.assertEqual(a.param_groups[0]['betas'],b.param_groups[0]['betas']);a.step();b.step();s.step(verbose=False);t.step()
  self.assertEqual(s.scheduler.state_dict(),t.state_dict())
 def test_14_effective128_drop_and_shuffle(self):
  a=training_loader(torch.arange(33745),1);b=training_loader(torch.arange(33745),1)
  self.assertEqual(len(a),263);self.assertTrue(a.drop_last);self.assertEqual(a.dataset.indices,b.dataset.indices);self.assertEqual(len(list(a))*128,33664)
 def test_15_accumulation_gradient_equivalence_without_dropout(self):
  self.m.eval();g=torch.Generator().manual_seed(10);x=torch.randn(4,96,7,generator=g);y=torch.randn(4,720,7,generator=g);ps=list(self.m.parameters())
  self.m.zero_grad(set_to_none=True);p=self.m(x);loss=combine((p-y).square().mean(),parts(p,y)[0]);loss.backward();full=[p.grad.detach().clone() for p in ps]
  self.m.zero_grad(set_to_none=True)
  for i in range(4):
   p=self.m(x[i:i+1]);loss=combine((p-y[i:i+1]).square().mean(),parts(p,y[i:i+1])[0])/4;loss.backward()
  for a,p in zip(full,ps):torch.testing.assert_close(a,p.grad,atol=2e-5,rtol=2e-4)
 def test_16_loss_and_architecture_preserved(self):
  self.m.train();original.shared.seed_all(123);p=self.m(torch.randn(2,96,7));y=torch.randn(2,720,7)
  for z in ((p-y).square().mean(),parts(p,y)[0]):
   grads=torch.autograd.grad(z,list(self.m.parameters()),retain_graph=True);self.assertTrue(all(torch.isfinite(a).all() for a in grads))
 def test_17_data_exact(self):
  ds=original.dataset(self.cfg,'val');a=np.load(BASE/'time_position_mmd/val/history.npy',mmap_mode='r');b=np.load(BASE/'time_position_mmd/val/truth.npy',mmap_mode='r')
  for i in (0,len(ds)-1):
   x,y,_,_=ds[i];np.testing.assert_array_equal(x.astype('float32'),a[i]);np.testing.assert_array_equal(y[-720:].astype('float32'),b[i])
 def test_18_official_script_contract(self):
  s=(G/'official_sources/Fredformer.sh').read_text().split('dst=ETTm1',1)[1].split('dst=ETTm2',1)[0]
  for fragment in ('train_epochs=100','patience=10','720) lr=0.0001 batch_size=128 lradj=TST','--pct_start 0.4'):self.assertIn(fragment,s)
 def test_19_official_ast_unchanged(self):
  for snapshot,src,cls,name in [('tools.py','utils/tools.py',None,'Scheduler'),('exp_basic.py','exp/exp_basic.py','Exp_Basic','_select_optimizer')]:
   def node(path):
    t=ast.parse(path.read_text())
    if cls:t=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
   self.assertEqual(ast.dump(node(G/'official_sources'/snapshot)),ast.dump(node(R/'third_party/DistDF'/src)))
 def test_20_same_D24_configuration(self):
  baseline=read(PEER/'model_config.json');current=vars(self.cfg)
  self.assertEqual({k for k in baseline if baseline[k]!=current[k] and k not in ('log_dir','root_path')},set())
  self.assertEqual((self.cfg.d_model,self.cfg.d_ff),(24,64));self.assertFalse(hasattr(self.cfg,'cf_dim'))
 def test_21_all236_D24_initial_exact_two_instances(self):
  a,_=fresh(G/'cpu_tests/second');ref=torch.load(PEERG/'reference_initial.pt',map_location='cpu',weights_only=True)
  for n,v in self.m.state_dict().items():
   self.assertTrue(torch.equal(v,ref[n]));self.assertTrue(torch.equal(v,a.state_dict()[n]));self.assertNotEqual(v.data_ptr(),a.state_dict()[n].data_ptr())
 def test_22_fixed_lambda5_loss_and_targetdetach(self):
  g=torch.Generator().manual_seed(456);p=torch.randn(2,32,3,dtype=torch.float64,generator=g,requires_grad=True);y=torch.randn(2,32,3,dtype=torch.float64,generator=g,requires_grad=True)
  point=(p-y.detach()).square().mean();raw=parts(p,y)[0];loss=combine(point,raw)
  self.assertEqual(MMD_WEIGHT,5.);torch.testing.assert_close(loss,point+5.*raw,rtol=0,atol=0)
  gp=torch.autograd.grad(point,p,retain_graph=True)[0];gm=torch.autograd.grad(raw,p,retain_graph=True)[0];gl,gy=torch.autograd.grad(loss,(p,y),allow_unused=True)
  torch.testing.assert_close(gl,gp+5.*gm,atol=1e-14,rtol=1e-12);self.assertIsNone(gy)
 def test_23_all204_weighted_gradients_finite(self):
  original.shared.seed_all(543);self.m.train();p=self.m(torch.randn(2,96,7));y=torch.randn(2,720,7);a=(p-y.detach()).square().mean();b=parts(p,y)[0];ps=list(self.m.parameters())
  ga=torch.autograd.grad(a,ps,retain_graph=True);gb=torch.autograd.grad(b,ps,retain_graph=True);gl=torch.autograd.grad(combine(a,b),ps)
  self.assertEqual(len(gl),204)
  for x,y,z in zip(ga,gb,gl):
   self.assertTrue(torch.isfinite(z).all());torch.testing.assert_close(z,x+5.*y,atol=2e-5,rtol=2e-4)
 def test_24_runner_only_coefficient_math_change(self):
  a=(R/'tmp/run_mafirst_wideband_dmodel24_distdfopt_h720_20261027.py').read_text();b=(R/'tmp/run_mafirst_dmodel24_mmdlambda5_distdfopt_h720_20261027.py').read_text()
  for fragment in ('2021000000+epoch*10000+(step-1)*4+micro','(loss/4).backward()','sched.step(verbose=False)','improved=measured[\'mse\']<=best','if stale>=10:break'):
   self.assertIn(fragment,a);self.assertIn(fragment,b)
  self.assertNotIn('loss=mse+mmd',b);self.assertEqual(b.count('loss=combine(mse,mmd)'),2)
if __name__=='__main__':unittest.main(verbosity=2)
