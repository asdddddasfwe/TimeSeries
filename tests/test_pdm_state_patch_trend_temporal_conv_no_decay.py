"""Single-factor no-decay policy, exact initialization, diagnostics and fixed budget."""
import copy,unittest
import torch
from scripts import run_timemix_pdm_state_patch_trend_temporal_conv_no_decay as run
from utils.pdm_state_patch_trend_temporal_conv_no_decay_diagnostics import FeatureStatistics,gradient_probe,weight_statistics
from utils.patch_covariance_loss import parts

class NoDecayTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  cls.cfg=run.config(run.ROOT/'dataset/ETT-small')
 def devices(self):return ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
 def test_all252_initial_function_rng_and_data_gradients(self):
  saved=torch.load(run.EXPERIMENT_OLD/'initial.pt',map_location='cpu',weights_only=True)['trend_temporal_conv']
  for device in self.devices():
   model=run.initialize_pair(self.cfg)[run.ARMS[0]].to(device)
   audit=run.initialization_audit({run.ARMS[0]:model});self.assertTrue(audit['all252_original_tcn_initial_exact'])
   other=run.TrendTemporalConv(self.cfg).to(device);other.load_state_dict(saved)
   for training in [False,True]:
    outs=[];grads=[];rng=[]
    x=torch.randn(2,96,7,generator=torch.Generator().manual_seed(111)).to(device);y=torch.randn(2,96,7,generator=torch.Generator().manual_seed(112)).to(device)
    for m in [other,model]:
     m.train(training);run.shared.seed_all(122)
     c=m.predict_components(x);p=m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm');d,_=parts(p,y);loss=(p-y).square().mean()+d
     outs.append(c.detach());grads.append(torch.autograd.grad(loss,list(m.parameters())));rng.append(torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
    self.assertTrue(torch.equal(outs[0],outs[1]));self.assertTrue(torch.equal(rng[0],rng[1]))
    for g,h in zip(*grads):self.assertTrue(torch.equal(g,h))
 def test_only16_decay_changes_first_update_and_later_data_learning(self):
  for device in self.devices():
   model=run.initialize_pair(self.cfg)[run.ARMS[0]].to(device);other=copy.deepcopy(model)
   new_groups=run.optimizer_groups(model);old_groups=run.norm_bias_no_decay_groups(other,1e-4)
   def policy(m,groups):
    names={id(p):n for n,p in m.named_parameters()};return {names[id(p)]:g['weight_decay'] for g in groups for p in g['params']}
   pn,po=policy(model,new_groups),policy(other,old_groups)
   changed={n for n in pn if pn[n]!=po[n]};self.assertEqual(len(changed),16);self.assertTrue(all(n.startswith('trend_temporal_encoders.') and pn[n]==0 and po[n]==1e-4 for n in changed))
   new=torch.optim.Adam(new_groups,lr=.001);old=torch.optim.Adam(old_groups,lr=.001)
   original={n:p.detach().clone() for n,p in model.named_parameters()}
   x=torch.randn(2,96,7,generator=torch.Generator().manual_seed(121)).to(device);y=torch.randn(2,96,7,generator=torch.Generator().manual_seed(122)).to(device)
   for m,opt in [(model,new),(other,old)]:
    run.shared.seed_all(155);opt.zero_grad(set_to_none=True);pred=m(x);d,_=parts(pred,y);((pred-y).square().mean()+d).backward();opt.step()
   for (n,p),(_,q) in zip(model.named_parameters(),other.named_parameters()):
    if n not in changed:self.assertTrue(torch.equal(p,q),n)
    elif '.depthwise.' in n:
     self.assertTrue(torch.equal(p,original[n]));self.assertFalse(torch.equal(q,original[n]))
   learned_dw=learned_out=False
   for step in range(6):
    run.shared.seed_all(160+step);new.zero_grad(set_to_none=True);pred=model(x);d,_=parts(pred,y);((pred-y).square().mean()+d).backward()
    for n,p in model.named_parameters():
     self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all(),n)
     if n.startswith('trend_temporal_encoders.'):
      if '.depthwise.' in n:learned_dw|=bool(p.grad.abs().max()>0)
      if '.output.' in n:learned_out|=bool(p.grad.abs().max()>0)
    new.step()
   self.assertTrue(learned_dw and learned_out)
 def test_sameforward_probes_and_fixed11_stale10(self):
  model=run.initialize_pair(self.cfg)[run.ARMS[0]];x=torch.randn(2,96,7);y=torch.randn_like(x,requires_grad=True);stats=FeatureStatistics(model);p=model(x);d,_=parts(p,y);mse=(p-y.detach()).square().mean();params=list(model.parameters());expected=torch.autograd.grad(mse+d,params,retain_graph=True);rng=torch.get_rng_state().clone()
  scopes=gradient_probe(model,mse,d);weights=weight_statistics(model)
  self.assertEqual(len(weights),16);self.assertEqual(len([n for n in scopes if n.startswith('encoder_scale')]),16)
  self.assertEqual(len(stats.encoder_summary()),4);stats.close();self.assertTrue(torch.equal(rng,torch.get_rng_state()));(mse+d).backward();self.assertIsNone(y.grad)
  for p,g in zip(params,expected):self.assertTrue(torch.equal(p.grad,g))
  arm=run.ARMS[0];rows=[dict(epoch=e,arm=arm,learning_rate=run.shared.learning_rate(e),train_loss=.8,train_mse=.7,train_d=.1,val=dict(mse=1.)) for e in range(1,12)]
  last=dict(epoch=11,curves=rows,best={arm:1.},best_epoch={arm:1},stale=10,updates_per_arm=11,optimizers={arm:dict(state={0:dict(step=torch.tensor(11.))})})
  self.assertEqual(run.verify_policy(last,32,11)['single_stale_diagnostic'],10)
if __name__=='__main__':unittest.main()
