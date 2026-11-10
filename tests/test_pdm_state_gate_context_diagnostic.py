"""Mechanism tests for gate-context routing; no forecasting data or trained fitting."""
import json,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from models.OscillatorPDMStatePatchDriveDiagnostic import Model as Legacy,phase_drives
from models.OscillatorPDMStatePatchGateContextDiagnostic import Model,gate_context
R=Path(__file__).resolve().parents[1]
class GateContextTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);cls.cfg=json.loads((R/'tmp/mafirst_drive_mean_ettm1_fold0_h720_20261031_group/plan.json').read_text())['models']['mafirst_season']['config']
 def build(self,policy):
  torch.manual_seed(2021);return Model(SimpleNamespace(**(self.cfg|dict(gate_context_policy=policy,state_drive_policy='mean',state_readout_policy='substate'))))
 def activate_gate(self,d):
  with torch.random.fork_rng(),torch.no_grad():torch.manual_seed(937);d.future_gate.projection.weight.normal_(0,.1)
 def test_constructor_states_counts_and_external_rng(self):
  a=self.build('sum');rng=torch.get_rng_state().clone();b=self.build('season');self.assertTrue(torch.equal(rng,torch.get_rng_state()));torch.manual_seed(2021);c=Legacy(SimpleNamespace(**self.cfg));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  self.assertEqual(sum(p.numel() for p in a.parameters()),345786);self.assertEqual((len(list(a.parameters())),len(a.state_dict()),len(list(a.buffers()))),(212,244,32))
  for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]) and torch.equal(v,c.state_dict()[k]))
 def test_season_route_matches_original_output_and_gradients(self):
  a=self.build('season');torch.manual_seed(2021);b=Legacy(SimpleNamespace(**self.cfg));a.train();b.train();x=torch.randn(2,96,7);y=torch.randn(2,720,7);rng=torch.get_rng_state();pa=a(x);end=torch.get_rng_state();torch.set_rng_state(rng);pb=b(x);self.assertTrue(torch.equal(pa,pb));self.assertTrue(torch.equal(end,torch.get_rng_state()))
  (pa-y).square().mean().backward();(pb-y).square().mean().backward()
  for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):self.assertEqual(n,m);self.assertTrue(torch.equal(p.grad,q.grad),n)
 def test_zero_trend_preserves_drive_and_input_gradients(self):
  d=self.build('sum').seasonals[0].decoder;self.activate_gate(d);d.train();h=torch.randn(2,8,32,requires_grad=True);t=torch.zeros_like(h);z=torch.randn(2,5,4,32,requires_grad=True);rng=torch.get_rng_state();a=phase_drives(d,z,gate_context(h,t,'season'),'mean');end=torch.get_rng_state();torch.set_rng_state(rng);b=phase_drives(d,z,gate_context(h,t,'sum'),'mean');self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(end,torch.get_rng_state()))
  ga=torch.autograd.grad(a.square().sum(),(z,h));gb=torch.autograd.grad(b.square().sum(),(z,h))
  for x,y in zip(ga,gb):self.assertTrue(torch.equal(x,y))
 def test_real_gate_receives_actual_final_pdm_sum(self):
  m=self.build('sum').eval();stored={};handles=[]
  def pdm(mod,args,out):stored['s']=[x.detach().clone() for x in out[0]];stored['t']=[x.detach().clone() for x in out[1]]
  handles.append(m.pdm_blocks[-1].register_forward_hook(pdm))
  for i,e in enumerate(m.seasonals):
   def hook(mod,args,index=i):stored[index]=args[1].detach().clone()
   handles.append(e.decoder.future_gate.register_forward_pre_hook(hook))
  with torch.no_grad():m(torch.randn(2,96,7))
  for i in range(4):self.assertTrue(torch.equal(stored[i],stored['s'][i]+stored['t'][i]))
  self.assertTrue(all(float(d['gate_context_matches_expected_max_abs'])==0 for d in m.last_state_readout_diagnostics))
  for h in handles:h.remove()
 def test_other_forward_routes_unchanged_and_zero_init_function_same(self):
  a=self.build('season').eval();b=self.build('sum').eval();x=torch.randn(2,96,7);captured=[]
  for m in [a,b]:
   rows=[];hs=[]
   for e in m.seasonals:
    hs.append(e.temporal_projection.register_forward_pre_hook(lambda mod,args:rows.append(args[0].detach().clone())))
   with torch.no_grad():p=m(x)
   captured.append((rows,p))
   for h in hs:h.remove()
  self.assertTrue(torch.equal(captured[0][1],captured[1][1]))
  for u,v in zip(captured[0][0],captured[1][0]):self.assertTrue(torch.equal(u,v))
  self.assertTrue(all(d['original_block_clock'] and d['shared_block_dropout_mask'] and d['selected_endpoint_bitwise_exact'] and float(d['projected_drive_phase_contrast_energy'].max())<1e-16 for d in b.last_state_readout_diagnostics))
 def test_trend_can_affect_existing_gate_and_get_gradients(self):
  d=self.build('sum').seasonals[0].decoder;self.activate_gate(d);d.eval();h=torch.randn(2,8,32);t=torch.randn_like(h,requires_grad=True);z=torch.randn(2,4,4,32);a=phase_drives(d,z,gate_context(h,t,'season'),'mean');b=phase_drives(d,z,gate_context(h,t,'sum'),'mean')
  self.assertGreater(float((a-b).abs().max()),1e-5);grad=torch.autograd.grad(b.square().mean(),t)[0];self.assertGreater(float(grad.abs().max()),1e-8)
 def test_invalid_policy_shape_and_no_new_persistent_state(self):
  h=torch.randn(2,8,32)
  with self.assertRaises(ValueError):gate_context(h,h,'bad')
  with self.assertRaises(ValueError):gate_context(h,h[:,:4],'sum')
  m=self.build('sum').eval();before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():m(torch.randn(2,96,7))
  self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.state_dict().items()));self.assertFalse(any('correction_head' in n or 'gate_context' in n for n in m.state_dict()))
 def test_full_MMD_gradients_random_stream_and_learning_signal(self):
  a=self.build('season');b=self.build('sum');a.train();b.train();x=torch.randn(2,96,7);y=torch.randn(2,720,7);rng=torch.get_rng_state();pa=a(x);end=torch.get_rng_state();torch.set_rng_state(rng);pb=b(x);self.assertTrue(torch.equal(end,torch.get_rng_state()));self.assertTrue(torch.equal(pa,pb))
  from tmp.mafirst_fullbatch_mmd_recompute_20261028 import parts
  for m,p in [(a,pa),(b,pb)]:
   before=torch.get_rng_state().clone();loss=(p-y).square().mean()+parts(p,y)[0];loss.backward();self.assertTrue(torch.equal(before,torch.get_rng_state()))
   self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()));self.assertEqual(list(m.initial_decomposition.kernels.values()),[9]*4)
  ga=a.seasonals[0].decoder.future_gate.projection.weight.grad;gb=b.seasonals[0].decoder.future_gate.projection.weight.grad;self.assertGreater(float((ga[:,32:]-gb[:,32:]).abs().max()),1e-9)
if __name__=='__main__':unittest.main()
