"""Official H192/H336 configs, upstream math, horizon boundaries and fresh state."""
import ast,unittest
import torch,numpy as np
from tmp.distdf_h192_h336_20261026_common import *
from tmp.distdf_h192_h336_20261026_factory import *
torch.set_num_threads(2)
class HorizonTests(unittest.TestCase):
 def test_01_official_width_and_weights(self):
  s=(R/'third_party/DistDF/scripts/Fredformer.sh').read_text().split('dst=ETTm1',1)[1].split('dst=ETTm2',1)[0]
  for arm,h in HORIZONS.items():
   c=config(arm);a=read(PLAN)['arms'][arm]
   self.assertEqual((c.seq_len,c.pred_len,c.d_model,c.cf_dim,c.cf_depth,c.cf_heads,c.cf_mlp,c.cf_head_dim),(96,h,24,128,2,8,96,32))
   self.assertIn(str(h)+') cf_dim=128',s);self.assertIn('alpha='+('.4' if h==192 else '.2'),s.replace('alpha=0.','alpha=.'))
   self.assertEqual(a['DistDF_weight'],.4 if h==192 else .2)
 def test_02_upstream_functions_exact(self):
  a=ast.parse((R/'third_party/DistDF/utils/fft_ot.py').read_text());b=ast.parse((R/'utils/distdf_joint_wasserstein.py').read_text())
  for name in ('batch_empirical_bures_wasserstein_distance','batch_upper_bound_distance'):
   aa=next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name==name);bb=next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name==name);self.assertEqual(ast.dump(aa),ast.dump(bb))
 def test_03_direct_upstream_value_gradient(self):
  tree=ast.parse((R/'third_party/DistDF/utils/fft_ot.py').read_text());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('batch_empirical_bures_wasserstein_distance','batch_upper_bound_distance')];ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream','exec'),ns)
  torch.manual_seed(31);p=torch.randn(3,12,2,dtype=torch.float64,requires_grad=True);y=torch.randn_like(p,requires_grad=True);h=torch.randn(3,8,2,dtype=torch.float64,requires_grad=True)
  v=distdf_loss(p,y,h);a=torch.cat((h.detach(),p),1).permute(2,0,1);b=torch.cat((h.detach(),y.detach()),1).permute(2,0,1);w=torch.ones(2,3,1,dtype=p.dtype)/3
  z=ns['batch_empirical_bures_wasserstein_distance'](a,b,reg=.005,ws=w,wt=w,var_weight=.002,mean_weight=1.,eps=1e-8)
  self.assertTrue(torch.equal(v,z));g=torch.autograd.grad(v,[p,y,h],retain_graph=True,allow_unused=True);self.assertTrue(torch.equal(g[0],torch.autograd.grad(z,p)[0]));self.assertIsNone(g[1]);self.assertIsNone(g[2])
 def test_04_combination_per_horizon(self):
  import tmp.run_distdf_h192_h336_20261026 as run
  for arm in HORIZONS:
   run.ARM=arm;a=read(PLAN)['arms'][arm];self.assertEqual(run.combined(2.,3.),a['point_mse_weight']*2+a['DistDF_weight']*3)
 def test_05_fresh_shape_gradient_both(self):
  for arm,h in HORIZONS.items():
   source.seed(2021);m,c=fresh(arm,G/'cpu_tests/logs');groups_audit(arm,m,optimizer_groups(m));x=torch.randn(2,96,7);y=torch.randn(2,h,7,requires_grad=True);p=m(x);a=read(PLAN)['arms'][arm];v=a['point_mse_weight']*(p-y.detach()).square().mean()+a['DistDF_weight']*distdf_loss(p,y,x)
   self.assertEqual(tuple(p.shape),(2,h,7));self.assertEqual(sum(p.numel() for p in m.parameters()),a['parameters']);gs=torch.autograd.grad(v,list(m.parameters())+[y],allow_unused=True);self.assertIsNone(gs[-1]);self.assertEqual([n for (n,p),g in zip(m.named_parameters(),gs[:-1]) if g is None],read(G/'reference_initial'/(arm+'_inactive.json')));self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gs))
 def test_06_data_boundaries_and_baseline_exact(self):
  for arm,h in HORIZONS.items():
   c=config(arm);tr=dataset(c,'train');val=dataset(c,'val');test=dataset(config(arm,R/'dataset/ETT-small'),'test');a=read(PLAN)['arms'][arm]
   self.assertEqual((len(tr),len(val),len(test)),(a['train_windows'],a['val_windows'],a['test_windows']))
   for ds,split in [(val,'val'),(test,'test')]:
    for i in (0,len(ds)-1):
     x,y,_,_=ds[i];base=BASES[arm]/'time_position_mmd'/split
     np.testing.assert_array_equal(x.astype('float32'),np.load(base/'history.npy',mmap_mode='r')[i]);np.testing.assert_array_equal(y[-h:].astype('float32'),np.load(base/'truth.npy',mmap_mode='r')[i])
 def test_07_independent_untrained_references(self):
  for arm in HORIZONS:
   source.seed(2021);m,c=fresh(arm,G/'cpu_tests/logs');ref=torch.load(G/'reference_initial'/(arm+'.pt'),map_location='cpu',weights_only=True);self.assertEqual(len(ref),72);self.assertTrue(all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in m.state_dict().items()))
 def test_08_protocol_unique_test_no_retry(self):
  p=read(PLAN);t=p['training'];self.assertEqual((t['batch_size'],t['epochs_max'],t['patience'],t['lr'],t['seed']),(32,30,5,.001,2021));self.assertFalse(t['formal_resume']);self.assertEqual(t['formal_retries'],0);self.assertTrue(t['test_once_per_run']);self.assertEqual(p['sequence'],['h192','h336'])
if __name__=='__main__':unittest.main(verbosity=2)
