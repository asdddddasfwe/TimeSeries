import unittest,torch,numpy as np,ast
from tests.test_original_wideband_20261026 import Kernel
from tmp.mafirst_wideband_distdfopt_h720_20261026_common import *
from tmp.mafirst_wideband_distdfopt_h720_20261026_factory import fresh,groups_audit,original
from tmp.mafirst_distdfopt_contract_20261026 import make_optimizer,make_scheduler,training_loader
from tmp.mafirst_wideband_kernel_20261026 import parts
torch.set_num_threads(2)
class Contract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m,cls.cfg=fresh(G/'cpu_tests/logs')
 def test_11_exact_scratch_and_model(self):
  ref=torch.load(G/'reference_initial.pt',map_location='cpu',weights_only=True)
  self.assertEqual(len(ref),236);self.assertEqual(sum(p.numel() for p in self.m.parameters()),320450)
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
  self.m.zero_grad(set_to_none=True);p=self.m(x);loss=(p-y).square().mean()+parts(p,y)[0];loss.backward();full=[p.grad.detach().clone() for p in ps]
  self.m.zero_grad(set_to_none=True)
  for i in range(4):
   p=self.m(x[i:i+1]);loss=((p-y[i:i+1]).square().mean()+parts(p,y[i:i+1])[0])/4;loss.backward()
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
if __name__=='__main__':unittest.main(verbosity=2)
