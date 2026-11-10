"""New-seed equivalence tests; no old training or test-set evaluations."""
import ast,importlib,os,tempfile,unittest
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from scripts import run_distdf_fredformer_official_recipe as ref
from test_distdf_fredformer_matched import original_class
import test_distdf_fredformer_official_recipe as official_tests
SEED=int(os.environ['DISTDF_AUDIT_SEED'])
r=importlib.import_module(f'scripts.run_distdf_fredformer_official_seed{SEED}')
torch.set_num_threads(1)
class SeedTests(unittest.TestCase):
 def test_all_computation_AST_only_seed_namespace_changed(self):
  old=ast.parse(Path(ref.__file__).read_text());new=Path(r.__file__).read_text().replace(f'seed({SEED})','seed(2023)').replace(f'seed={SEED}','seed=2023').replace(f'seed{SEED}','seed2023').replace(f'scripts/monitor_distdf_fredformer_official_seed2023.py','scripts/monitor_distdf_fredformer_official_recipe.py')
  a={x.name:ast.dump(x,include_attributes=False) for x in old.body if isinstance(x,ast.FunctionDef)};b={x.name:ast.dump(x,include_attributes=False) for x in ast.parse(new).body if isinstance(x,ast.FunctionDef)};self.assertEqual(a,b)
 def test_same_seed_official_fresh_function_gradient_Adam_cpu_cuda(self):
  cfg=r.config(r.ROOT/'dataset/ETT-small');original=original_class()
  for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
   r.seed(SEED);a=original(cfg).to(device);r.seed(SEED);b=r.Model(cfg).to(device)
   self.assertEqual(sum(x.numel() for x in b.parameters()),8591951);self.assertEqual(len(b.state_dict()),72)
   self.assertTrue(all(torch.equal(v,b.state_dict()[n]) and v.data_ptr()!=b.state_dict()[n].data_ptr() for n,v in a.state_dict().items()))
   r.seed(2023);old=r.Model(cfg);self.assertTrue(any(not torch.equal(v.cpu(),old.state_dict()[n]) for n,v in b.state_dict().items()));del old
   x=torch.linspace(-1,1,2*96*7).reshape(2,96,7).to(device);y=x.flip(1);outs=[];grads=[]
   for m in (a,b):
    r.seed(SEED+9000);m.train();pred=m(x);loss=.5*nn.functional.mse_loss(pred,y)+.5*r.distdf_loss(pred,y,x);loss.backward();outs.append(pred.detach());grads.append({n:None if v.grad is None else v.grad.clone() for n,v in m.named_parameters()})
   self.assertTrue(torch.equal(*outs))
   for n,g in grads[0].items():
    if g is None:self.assertIsNone(grads[1][n])
    else:self.assertTrue(torch.equal(g,grads[1][n]),n)
   oa=torch.optim.Adam(a.parameters(),lr=.0001);ob=torch.optim.Adam(r.optimizer_groups(b),lr=.0001);sa=ref.make_scheduler(oa,100,268);sb=r.make_scheduler(ob,100,268);oa.step();ob.step();sa.step();sb.step();self.assertTrue(all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()));self.assertEqual(len(ob.param_groups),1);self.assertEqual(ob.param_groups[0]['weight_decay'],0.)
   del a,b,oa,ob,sa,sb,outs,grads
   if device=='cuda':torch.cuda.empty_cache()
 def test_continuous_global_sampler_new_seed(self):
  data=TensorDataset(torch.arange(300))
  for purpose in ('train','selection_val'):
   r.seed(SEED);a=torch.cat([x[0] for x in DataLoader(data,batch_size=128,shuffle=True,drop_last=True,num_workers=10)]);rng=torch.get_rng_state();r.seed(SEED);b=torch.cat([x[0] for x in r.loader(data,purpose)]);self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(len(b),256)
 def test_full26800schedule_and_tie_patience10(self):
  official_tests.RecipeTests().test_official_horizon_and_early_stop_ties()
if __name__=='__main__':unittest.main()
