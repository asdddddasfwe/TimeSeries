"""Variable-horizon adaptation checks, fresh models only, no old test evaluation."""
import os,ast,importlib,json,tempfile,math,unittest,gc
from pathlib import Path
import numpy as np
import torch
from utils.patch_covariance_loss import parts as oldD
from utils.time_position_patch_mmd import parts as mmd
from utils.mafirst_mmd_horizon_audit import numpy_relations,metrics
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from scripts import run_mafirst_mmd_original_recipe_seed2021_earlystop5 as base
H=int(os.environ['MAFIRST_AUDIT_HORIZON']);r=importlib.import_module(f'scripts.run_mafirst_mmd_ettm1_h{H}_seed2021');torch.set_num_threads(2)
class HorizonTests(unittest.TestCase):
 def test_unchanged_computational_function_AST(self):
  def functions(module):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
  a,b=functions(base),functions(r)
  for n in ('arm_config','scratch_reference','initialize_pair','control_reference','references','mmd_gradient_probe','loader','verify_policy'):self.assertEqual(a[n],b[n],n)
 def test_fresh_H_state_regeneration_and_component_shape_cpu_cuda(self):
  cfg=r.config(r.ROOT/'dataset/ETT-small');a=r.initialize_pair(cfg)['time_position_mmd'];b=r.control_reference(cfg);self.assertEqual(len(a.state_dict()),236);self.assertEqual(len(list(a.parameters())),204);self.assertEqual(sum(v.numel() for v in a.parameters()),r.COUNTS['time_position_mmd']);self.assertTrue(all(torch.equal(v,b.state_dict()[n]) and v.data_ptr()!=b.state_dict()[n].data_ptr() for n,v in a.state_dict().items()))
  for i in range(4):
   w=a.seasonals[i].decoder.readout.output_projection.weight;self.assertTrue(all(torch.equal(w[0],w[j]) for j in range(1,4)));self.assertEqual(torch.count_nonzero(a.feature_trend_heads[i].readout.weight).item(),0)
  for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
   a.to(device).eval();b.to(device).eval();x=torch.linspace(-1,1,2*96*7).reshape(2,96,7).to(device)
   with torch.no_grad():pa=a.predict_components(x);pb=b.predict_components(x)
   self.assertEqual(pa.shape,(2,H,7,8));self.assertTrue(torch.equal(pa,pb));self.assertTrue(torch.isfinite(pa).all())
  del a,b;gc.collect();torch.cuda.empty_cache()
 def test_full_batch32_training_gradients_optimizer_and_memory(self):
  if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
  torch.cuda.set_per_process_memory_fraction(.6 if H==720 else .4);torch.cuda.reset_peak_memory_stats();cfg=r.config(r.ROOT/'dataset/ETT-small');model=r.initialize_pair(cfg)['time_position_mmd'].cuda().train();opt=torch.optim.Adam(norm_bias_no_decay_groups(model,1e-4),lr=.001)
  names={id(p):n for n,p in model.named_parameters()};oldpolicy={n:g['weight_decay'] for g in json.loads((r.ROOT/'tmp/mafirst_mmd_original_recipe_20261016_seed2021_max30_patience5/optimizer_groups.json').read_text())['time_position_mmd'] for n in g['names']};policy={names[id(p)]:g['weight_decay'] for g in opt.param_groups for p in g['params']};self.assertEqual(oldpolicy,policy)
  x=torch.linspace(-1,1,32*96*7,device='cuda').reshape(32,96,7);y=torch.sin(torch.linspace(0,5,32*H*7,device='cuda')).reshape(32,H,7);r.shared.seed_all(2021000000+10000);p=model(x);lossM=(p-y).square().mean();D,_=oldD(p,y);M,diag=mmd(p,y);self.assertEqual(diag['patch_count'],H//4);r.mmd_gradient_probe(model,lossM,M);r.gradient_probe(model,lossM,D);r.output_gradient_probe(p,lossM,M,D);(lossM+M).backward();self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()));opt.step();self.assertTrue(all(torch.isfinite(v).all() for v in model.parameters()));peak=torch.cuda.max_memory_allocated();print('CUDA fullbatch peak MiB',peak/2**20,flush=True)
  del model,opt,p,lossM,D,M;gc.collect();torch.cuda.empty_cache()
 def test_data_windows_and_shared_raw_prefix(self):
  cfg=r.config(r.ROOT/'dataset/ETT-small')
  for split in ('train','val'):
   d=r.dataset(cfg,split);self.assertEqual(len(d),34465-H if split=='train' else 11521-H);self.assertEqual(d[0][0].shape,(96,7));self.assertEqual(d[0][1].shape,(H,7))
   from data_provider.data_loader import Dataset_ETT_minute
   ref=Dataset_ETT_minute(cfg,str(r.ROOT/'dataset/ETT-small'),flag=split,size=[96,0,96],features='M',data_path='ETTm1.csv',timeenc=1,freq='h')
   for i in (0,len(d)//2,len(d)-1):self.assertTrue(np.array_equal(d[i][0],ref[i][0]));self.assertTrue(np.array_equal(d[i][1][:96],ref[i][1]))
 def test_MMD_D_NumPy_batch_separability_and_metrics(self):
  p=torch.linspace(-.8,.9,2*H*3).reshape(2,H,3).requires_grad_(True);y=p.detach().flip(1);M,_=mmd(p,y);D,_=oldD(p,y);n=numpy_relations(p.detach().numpy(),y.numpy());self.assertAlmostEqual(float(M.detach()),n['time_position_mmd'],places=7);self.assertAlmostEqual(float(D.detach()),n['patch_covariance'],places=7);self.assertAlmostEqual(float(M.detach()),sum(float(mmd(p[i:i+1],y[i:i+1])[0].detach()) for i in range(2))/2,places=7);s=metrics(p.detach().numpy(),y.numpy());self.assertAlmostEqual(s['mse'],s['low4_mse']+s['remaining_mse'],places=12);self.assertEqual(s['remaining_mode_count'],H-4);self.assertEqual(s['increments']['boundary']['count'],H//4-1);self.assertEqual(s['increments']['interior']['count'],H-H//4)
 def test_saved_array_capture_audit_H(self):
  rng=np.random.default_rng(42);c=rng.normal(0,.1,(4,H,7,8)).astype('float32');mean=np.zeros((4,1,7),dtype='float32');p=c[...,:4].sum(-1,dtype=np.float32)+c[...,4:].sum(-1,dtype=np.float32);y=rng.normal(0,.2,p.shape).astype('float32');history=rng.normal(0,.1,(4,96,7)).astype('float32');M,md=mmd(torch.from_numpy(p),torch.from_numpy(y));D,dd=oldD(torch.from_numpy(p),torch.from_numpy(y));measured={**md,**dd,**{k:metrics(p,y)[k] for k in ('mse','mae')}}
  with tempfile.TemporaryDirectory(dir=r.ROOT/'tmp') as folder:
   out=Path(folder);(out/'val').mkdir()
   for n,v in [('prediction',p),('truth',y),('history',history),('components',c),('history_mean',mean)]:np.save(out/'val'/(n+'.npy'),v)
   a=r.audit_capture(out,'val',measured);self.assertEqual(a['windows'],4);self.assertEqual(a['prediction_length'],H)
 def test_LR_sampling_and_fullval_strict_stop_policy(self):
  for e in range(1,31):self.assertEqual(r.shared.learning_rate(e),.001*.5*(1+math.cos(math.pi*(e-1)/30)))
  from torch.utils.data import TensorDataset
  data=TensorDataset(torch.arange(65));batch=r.loader(data,3);g=torch.Generator().manual_seed(2021*1000+3);self.assertEqual(batch.dataset.indices,torch.randperm(65,generator=g).tolist());self.assertEqual([len(v[0]) for v in batch],[32,32,1]);self.assertEqual(math.ceil(r.TRAIN_WINDOWS/32),json.loads(r.PROTOCOL.read_text())['training']['updates_per_epoch'])
if __name__=='__main__':unittest.main()
