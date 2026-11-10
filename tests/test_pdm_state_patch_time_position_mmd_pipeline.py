import unittest,torch
from scripts import run_timemix_pdm_state_patch_time_position_mmd as run
from utils.time_position_patch_mmd import parts as mmd_parts
from utils.patch_covariance_loss import parts
from utils.pdm_state_patch_ma_linear_diagnostics import FeatureStatistics

class PipelineTests(unittest.TestCase):
 def check(self,device):
  torch.set_num_threads(2);torch.manual_seed(2021)
  models={k:v.to(device) for k,v in run.initialize_pair(run.config(run.ROOT/'dataset/ETT-small')).items()};model=models['time_position_mmd'];audit=run.initialization_audit(models);self.assertTrue(audit['common236_exact'])
  x=torch.randn(2,96,7,device=device);y=torch.randn_like(x,requires_grad=True);stats=FeatureStatistics(model);p=model(x);old,_=parts(p,y);new,_=mmd_parts(p,y);mse=(p-y.detach()).square().mean();params=list(model.parameters());expected=torch.autograd.grad(mse+new,params,retain_graph=True)
  rng=torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()
  scopes=run.mmd_gradient_probe(model,mse,new);oldscopes=run.gradient_probe(model,mse,old);output=run.output_gradient_probe(p,mse,new,old)
  self.assertEqual(scopes['all']['parameter_tensors'],204);self.assertEqual(len(stats.summary()),8);self.assertEqual(set(output),{'all','low4','remaining92','level'});stats.close()
  self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()));(mse+new).backward();self.assertIsNone(y.grad)
  for parameter,g in zip(params,expected):torch.testing.assert_close(parameter.grad,g,rtol=0,atol=0)
  rows=[dict(epoch=e,arm='time_position_mmd',learning_rate=run.shared.learning_rate(e),train_loss=.8,train_mse=.7,train_mmd=.1,train_d=.9,val=dict(mse=1.)) for e in range(1,12)]
  last=dict(epoch=11,curves=rows,best=dict(time_position_mmd=1.),best_epoch=dict(time_position_mmd=1),stale=10,updates_per_arm=11,optimizers=dict(time_position_mmd=dict(state={0:dict(step=torch.tensor(11.))})))
  self.assertEqual(run.verify_policy(last,32,11)['single_stale_diagnostic'],10)
 def test_cpu(self):self.check('cpu')
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
 def test_cuda(self):self.check('cuda')
if __name__=='__main__':unittest.main()
