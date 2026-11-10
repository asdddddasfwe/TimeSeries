"""Fixed matched budget and same-forward encoder diagnostics."""
import unittest,torch
from scripts import run_timemix_pdm_state_patch_trend_temporal_conv as run
from utils.pdm_state_patch_trend_temporal_conv_diagnostics import FeatureStatistics,gradient_probe
from utils.patch_covariance_loss import parts
class PipelineTests(unittest.TestCase):
 def test_fixed11_and_sameforward_no_rng_or_gradient_change(self):
  torch.set_num_threads(2);torch.manual_seed(2021)
  model=run.initialize_pair(run.config(run.ROOT/'dataset/ETT-small'))['trend_temporal_conv']
  run.initialization_audit(dict(trend_temporal_conv=model))
  x=torch.randn(2,96,7);y=torch.randn_like(x,requires_grad=True);stats=FeatureStatistics(model)
  p=model(x);d,_=parts(p,y);mse=(p-y.detach()).square().mean();params=list(model.parameters());expected=torch.autograd.grad(mse+d,params,retain_graph=True);rng=torch.get_rng_state().clone()
  scopes=gradient_probe(model,mse,d);self.assertEqual(scopes['trend_temporal_encoder']['parameter_tensors'],16);self.assertEqual(scopes['trend_temporal_depthwise']['parameter_tensors'],12);self.assertEqual(scopes['trend_temporal_output']['parameter_tensors'],4)
  self.assertEqual(len(stats.summary()),8);self.assertEqual(len(stats.encoder_summary()),4)
  for v in stats.encoder_summary().values():self.assertEqual(v['residual_rms'],0.)
  stats.close();self.assertTrue(torch.equal(rng,torch.get_rng_state()));(mse+d).backward();self.assertIsNone(y.grad)
  for p,g in zip(params,expected):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
  rows=[dict(epoch=e,arm='trend_temporal_conv',learning_rate=run.shared.learning_rate(e),train_loss=.8,train_mse=.7,train_d=.1,val=dict(mse=1.)) for e in range(1,12)]
  last=dict(epoch=11,curves=rows,best=dict(trend_temporal_conv=1.),best_epoch=dict(trend_temporal_conv=1),stale=10,updates_per_arm=11,optimizers=dict(trend_temporal_conv=dict(state={0:dict(step=torch.tensor(11.))})))
  self.assertEqual(run.verify_policy(last,32,11)['single_stale_diagnostic'],10)
if __name__=='__main__':unittest.main()
