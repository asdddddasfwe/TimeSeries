"""Entire periods, not scalar points or fixed four-point patches, are tokens."""
import copy,unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from layers.LagARFormer import LagAxialBlock,sinusoidal_positions
from models.OscillatorPDMStatePatchSeasonPeriodToken import PeriodCodec,PeriodAttention,period_token_features
from scripts import run_timemix_pdm_state_patch_season_period_token as run
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts

class PeriodTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

 def explicit(self,device):
  length,width=12,4
  codec=PeriodCodec(length,width).to(device).double()
  block=PeriodAttention(LagAxialBlock(width,8,0.)).to(device).double()
  with torch.no_grad():block.column.gamma.fill_(.3)
  for period in (2,5,12):
   cycles=(length+period-1)//period
   x=torch.randn(2,length,1,device=device,dtype=torch.float64,requires_grad=True)
   raw=F.pad(x.squeeze(-1),(0,cycles*period-length)).reshape(2,cycles,period)
   actual=codec(raw,block)
   # Independent gather + phase-vector projection, explicit unscaled softmax.
   vectors=torch.stack([torch.stack([x[n,c*period:(c+1)*period,0].new_tensor(0.) if j>=period or c*period+j>=length else x[n,c*period+j,0] for j in range(length)]) for n in range(2) for c in range(cycles)]).reshape(2,cycles,length)
   tokens=F.linear(vectors,codec.token_projection.weight)+sinusoidal_positions(cycles,width).to(x)
   column=block.column
   q=F.linear(tokens,column.query_conv.weight[:,:,0,0],column.query_conv.bias)
   k=F.linear(tokens,column.key_conv.weight[:,:,0,0],column.key_conv.bias)
   v=F.linear(tokens,column.value_conv.weight[:,:,0,0],column.value_conv.bias)
   update=(q@k.transpose(-1,-2)).softmax(-1)@v
   encoded=F.layer_norm(tokens+column.gamma*update,(width,),block.norm.weight,block.norm.bias,block.norm.eps)
   expected=F.linear(encoded,codec.period_readout.weight).reshape(2,cycles,length,width)[:,:,:period]
   torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
   params=[x,*codec.parameters(),*block.parameters()]
   a=torch.autograd.grad(actual.square().sum(),params,retain_graph=True)
   b=torch.autograd.grad(expected.square().sum(),params)
   for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=1e-9,atol=1e-10)
   self.assertEqual(actual.shape,(2,cycles,period,width))
  # Multiple routes including non-divisible P, duplicates, and P=T.
  class Encoder:pass
  encoder=Encoder();encoder.blocks=[block];encoder.period_top_k=3
  periods=torch.tensor([[5,2,5],[12,5,2]],device=device)
  weights=torch.tensor([[.2,.5,.3],[.2,.4,.4]],device=device,dtype=torch.float64)
  x=torch.randn(2,12,1,device=device,dtype=torch.float64,requires_grad=True)
  with patch('models.OscillatorPDMStatePatchSeasonPeriodToken.select_seasonal_feature_periods',return_value=(periods,weights)):
   actual=period_token_features(encoder,x,codec)
  expected=[]
  for n in range(2):
   result=x.new_zeros(length,width)
   for p,w in zip(periods[n].tolist(),weights[n]):
    c=(length+p-1)//p
    raw=F.pad(x[n,:,0],(0,c*p-length)).reshape(1,c,p)
    result=result+w*codec(raw,block).reshape(c*p,width)[:length]
   expected.append(result)
  expected=torch.stack(expected)
  torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
  aa=torch.autograd.grad(actual.square().sum(),[x,*codec.parameters(),*block.parameters()],retain_graph=True)
  bb=torch.autograd.grad(expected.square().sum(),[x,*codec.parameters(),*block.parameters()])
  for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)

 def test_explicit_cpu(self):self.explicit('cpu')
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
 def test_explicit_cuda(self):self.explicit('cuda')

 def network(self,device):
  models=run.initialize_pair(run.config(run.ROOT/'dataset/ETT-small'));m=models['season_period_token'].to(device)
  audit=run.initialization_audit(models);self.assertTrue(audit['common200_exact'])
  self.assertEqual(sum(p.numel() for p in m.parameters()),309694)
  self.assertTrue(all(not hasattr(e,'value_embedding') and not hasattr(e.blocks[0],'row') for e in m.seasonals))
  self.assertEqual([c.length for c in m.season_period_codecs],[96,48,24,12])
  ref=run.control_reference(run.config(run.ROOT/'dataset/ETT-small')).to(device)
  # Actual trend histories remain bitwise the original path before PDM.
  x=torch.randn(2,96,7,device=device);native=m.downsample(x)
  normalized=[m.normalize(native[0],'norm')]+[n(t,'norm') for n,t in zip(m.coarse_normalizers,native[1:])]
  for e,embedding,z in zip(ref.seasonals,m.trend_embeddings,normalized):
   observed=z.transpose(1,2).reshape(14,e.length,1);_,t=m.initial_decomposition(observed)
   torch.testing.assert_close(embedding(t),e.value_embedding(t),rtol=0,atol=0)
  optimizer=torch.optim.Adam(norm_bias_no_decay_groups(m,1e-4),lr=.001)
  seen=set();m.train();initial={n:p.detach().clone() for n,p in m.named_parameters()}
  for step in range(5):
   optimizer.zero_grad(set_to_none=True);p=m(x);target=torch.randn_like(p);d,_=parts(p,target);loss=(p-target).square().mean()+d;loss.backward()
   for n,p in m.named_parameters():
    self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
    if torch.count_nonzero(p.grad):seen.add(n)
   optimizer.step()
  for label,pattern in [('tokenizer','token_projection'),('decoder','period_readout'),('gamma','column.gamma'),('QKV','column.query_conv'),('condition','trend_condition')]:
   self.assertTrue(any(pattern in n for n in seen),label)
   self.assertTrue(any(pattern in n and not torch.equal(initial[n],p) for n,p in m.named_parameters()),label)
  # Original trend->season modulation remains, with no TCN added.
  self.assertEqual(sum('trend_condition.weight' in n for n,_ in m.named_parameters()),4)
  self.assertFalse(hasattr(m,'trend_temporal_encoders'))
 def test_network_cpu(self):self.network('cpu')
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
 def test_network_cuda(self):self.network('cuda')
if __name__=='__main__':unittest.main()
