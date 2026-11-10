import unittest,torch,numpy as np,ast
from tests.test_original_wideband_20261026 import Kernel
from tmp.mafirst_season_sliding_distdfopt_h720_20261027_common import *
from tmp.mafirst_season_sliding_distdfopt_h720_20261027_factory import fresh,groups_audit,original
from tmp.mafirst_season_sliding_contract_20261027 import make_optimizer,make_scheduler,training_loader
from tmp.mafirst_wideband_kernel_20261026 import parts
torch.set_num_threads(2)
class Contract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m,cls.cfg=fresh(G/'cpu_tests/logs')
 def test_11_exact_scratch_and_model(self):
  ref=torch.load(G/'reference_initial.pt',map_location='cpu',weights_only=True)
  self.assertEqual(len(ref),read(G/'expected_model_card.json')['states']);self.assertEqual(sum(p.numel() for p in self.m.parameters()),read(G/'expected_model_card.json')['parameters'])
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
 def test_20_only_seasonal_encoder_changes(self):
  self.assertEqual(self.cfg.d_model,32);self.assertEqual(self.cfg.d_ff,64);self.assertFalse(hasattr(self.cfg,'cf_dim'))
  baseline=read(BASE/'protocol.json')['model_config'];current=vars(self.cfg)
  differences={k for k in baseline if baseline[k]!=current[k] and k not in ('log_dir','root_path')}
  self.assertEqual(differences,{'model'})
  source=original.initialize_pair(self.cfg)['time_position_mmd']
  candidate=self.m.state_dict();old=source.state_dict()
  removed=set(old)-set(candidate);added=set(candidate)-set(old)
  self.assertTrue(removed and all('.blocks.' in n for n in removed))
  self.assertTrue(added and all(n.startswith('season_patch_encoders.') for n in added))
  for n in set(old)&set(candidate):self.assertTrue(torch.equal(old[n],candidate[n]),n)
  self.assertTrue(all(not hasattr(e,'blocks') for e in self.m.seasonals))
 def test_21_patch_alignment_overlap_and_endpoints(self):
  for length,count,codec in zip((96,48,24,12),(47,23,11,5),self.m.season_patch_encoders):
   x=torch.arange(length*32,dtype=torch.float32).reshape(1,length,32)
   patches,padded=codec.patchify(x)
   self.assertEqual(patches.shape,(1,count,4,32));self.assertTrue(torch.equal(patches[0,1],x[0,2:6]))
   restored=codec.overlap_average(patches,padded,length)
   self.assertTrue(torch.equal(x,restored));self.assertTrue(torch.equal(x[:,-1],restored[:,-1]))
   self.assertEqual(codec.attention.self_attn.num_heads,4)
 def test_22_no_FFT_and_native_shapes(self):
  from unittest.mock import patch
  self.m.eval()
  with patch('torch.fft.rfft',side_effect=AssertionError('Seasonal FFT still active')):
   for t,codec in zip((96,48,24,12),self.m.season_patch_encoders):
    x=torch.randn(2,t,32);self.assertEqual(codec(x).shape,x.shape)
   p=self.m(torch.randn(2,96,7));self.assertEqual(p.shape,(2,720,7))
 def test_23_trend_and_downstream_objects_unchanged(self):
  self.assertEqual(type(self.m).predict_components,original.CLASSES['time_position_mmd'].predict_components)
  self.assertEqual(type(self.m).encode_multiscale,original.CLASSES['time_position_mmd'].encode_multiscale)
  source=original.initialize_pair(self.cfg)['time_position_mmd']
  for name in ('pdm_blocks','feature_trend_heads','initial_decomposition'):
   self.assertEqual(type(getattr(self.m,name)),type(getattr(source,name)))
   a=getattr(self.m,name).state_dict();b=getattr(source,name).state_dict()
   self.assertEqual(a.keys(),b.keys());self.assertTrue(all(torch.equal(v,b[k]) for k,v in a.items()))
if __name__=='__main__':unittest.main(verbosity=2)
