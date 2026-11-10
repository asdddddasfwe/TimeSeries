"""Official H720 config, direct upstream value/gradient and fresh-state audits."""
import unittest,ast
import numpy as np,torch
from tmp.distdf_officialwidth_h720_20261026_common import *
from tmp.distdf_officialwidth_h720_20261026_factory import *
torch.set_num_threads(2)
class OfficialWidthTests(unittest.TestCase):
 def test_01_official_H720_config(self):
  c=config();self.assertEqual((c.seq_len,c.pred_len,c.enc_in,c.d_model,c.cf_dim,c.cf_depth,c.cf_heads,c.cf_mlp,c.cf_head_dim),(96,720,7,48,164,2,8,96,32))
  s=(R/'third_party/DistDF/scripts/Fredformer.sh').read_text().split('dst=ETTm1',1)[1].split('dst=ETTm2',1)[0]
  self.assertIn('720) cf_dim=164',s);self.assertIn('d_model=48',s);self.assertIn('alpha=0.2 var_weight=0.002',s)
  self.assertEqual((c.patch_len,c.stride,c.revin,c.affine,c.use_nys,c.dropout,c.fc_dropout),(4,4,1,0,0,.2,.2))
 def test_02_upstream_functions_unchanged(self):
  a=ast.parse((R/'third_party/DistDF/utils/fft_ot.py').read_text());b=ast.parse((R/'utils/distdf_joint_wasserstein.py').read_text())
  for n in ('batch_empirical_bures_wasserstein_distance','batch_upper_bound_distance'):
   x=next(v for v in a.body if isinstance(v,ast.FunctionDef) and v.name==n);y=next(v for v in b.body if isinstance(v,ast.FunctionDef) and v.name==n)
   self.assertEqual(ast.dump(x),ast.dump(y))
 def test_03_direct_official_value_gradient(self):
  from utils.distdf_joint_wasserstein import distdf_loss
  tree=ast.parse((R/'third_party/DistDF/utils/fft_ot.py').read_text());nodes=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in ('batch_empirical_bures_wasserstein_distance','batch_upper_bound_distance')];ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream','exec'),ns)
  torch.manual_seed(27);p=torch.randn(3,12,2,dtype=torch.float64,requires_grad=True);y=torch.randn_like(p,requires_grad=True);h=torch.randn(3,8,2,dtype=torch.float64,requires_grad=True)
  v=distdf_loss(p,y,h);a=torch.cat((h.detach(),p),1).permute(2,0,1);b=torch.cat((h.detach(),y.detach()),1).permute(2,0,1);w=torch.ones(2,3,1,dtype=p.dtype)/3
  q=ns['batch_empirical_bures_wasserstein_distance'](a,b,reg=.005,ws=w,wt=w,var_weight=.002,mean_weight=1.,eps=1e-8)
  self.assertTrue(torch.equal(v,q));gv=torch.autograd.grad(v,[p,y,h],retain_graph=True,allow_unused=True);gq=torch.autograd.grad(q,p)[0];self.assertTrue(torch.equal(gv[0],gq));self.assertIsNone(gv[1]);self.assertIsNone(gv[2])
 def test_04_combined_weights(self):
  from tmp.run_distdf_officialwidth_h720_20261026 import combined
  import tmp.run_distdf_officialwidth_h720_20261026 as run
  run.ARM='distdf';self.assertEqual(combined(2.,3.),.8*2+.2*3)
 def test_05_fresh_full_shape_active_gradient(self):
  source.seed(2021);m,c=fresh(G/'cpu_tests/logs');groups_audit(m,optimizer_groups(m))
  self.assertEqual(len(m.state_dict()),72);self.assertEqual(sum(p.numel() for p in m.parameters()),read(PLAN)['distdf']['parameters'])
  x=torch.randn(2,96,7);y=torch.randn(2,720,7,requires_grad=True);p=m(x);loss=.8*(p-y.detach()).square().mean()+.2*distdf_loss(p,y,x)
  self.assertEqual(tuple(p.shape),(2,720,7));ps=list(m.parameters());gs=torch.autograd.grad(loss,ps+[y],allow_unused=True)
  self.assertIsNone(gs[-1]);self.assertEqual([n for (n,p),g in zip(m.named_parameters(),gs[:-1]) if g is None],read(G/'reference_initial/distdf_inactive.json'))
  self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gs))
 def test_06_data_exact_saved_baselines(self):
  from scripts import run_mafirst_mmd_ettm1_h720_seed2021 as original
  c=config();tr=original.dataset(c,'train');v=original.dataset(c,'val');self.assertEqual((len(tr),len(v)),(33745,10801))
  x,y,_,_=v[0]
  for root in (BASE/'time_position_mmd/val',PREVIOUS/'selected_val'):
   np.testing.assert_array_equal(x.astype('float32'),np.load(root/'history.npy',mmap_mode='r')[0]);np.testing.assert_array_equal(y[-720:].astype('float32'),np.load(root/'truth.npy',mmap_mode='r')[0])
 def test_07_reference_independent_no_trained_weights(self):
  source.seed(2021);m,c=fresh(G/'cpu_tests/logs');ref=torch.load(G/'reference_initial/distdf.pt',map_location='cpu',weights_only=True)
  self.assertTrue(all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in m.state_dict().items()))
 def test_08_validation_selection_and_test_contract(self):
  p=read(PLAN)['training'];self.assertEqual((p['batch_size'],p['epochs_max'],p['patience'],p['lr'],p['seed']),(32,30,5,.001,2021))
  self.assertFalse(p['formal_resume']);self.assertEqual(p['formal_retries'],0);self.assertTrue(p['test_once_per_run'])
if __name__=='__main__':unittest.main(verbosity=2)
