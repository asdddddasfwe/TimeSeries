"""Trend residual encoder: explicit causal formula and full-network identity."""
import json,unittest
from types import SimpleNamespace
from pathlib import Path
import torch
from torch.nn import functional as F
from models.OscillatorPDMStatePatchTrendTemporalConv import Model,TrendTemporalResidual
from models.OscillatorPDMStatePatchMAFirst import Model as Base
from models.OscillatorPDMStatePatchSeparateMLP import Model as SeparateMLP
from scripts.train_timemix_pdm_state_patch import initialize_pair
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts
ROOT=Path(__file__).resolve().parents[1]

def cfg():
    values=json.loads((ROOT/'tmp/lag_study_timemix_pdm_state_patch_20261011_seed2021_earlystop5/training/protocol.json').read_text())['model_args']
    return SimpleNamespace(**values)

def fresh(cls):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021);source=initialize_pair(cfg())['patch_covariance'];reference=SeparateMLP(cfg())
        state=reference.state_dict();state.update({n:v.clone() for n,v in source.state_dict().items()});reference.load_state_dict(state);reference.clone_trend_mlps()
        torch.manual_seed(20211015);model=cls(cfg());state=model.state_dict();state.update({n:v.clone() for n,v in reference.state_dict().items() if n in state});model.load_state_dict(state)
    return model

class TrendTemporalTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(2021)
    def test_explicit_formula_value_gradient_causality_and_receptive_field(self):
        module=TrendTemporalResidual(4).double()
        with torch.no_grad():module.output.weight.normal_(0,.1)
        x=torch.randn(2,32,4,dtype=torch.float64,requires_grad=True)
        z=x
        for conv,d in zip(module.depthwise,module.dilations):
            indices=(torch.arange(32)[:,None]-2*d+torch.arange(3)[None]*d).clamp_min(0)
            z=F.gelu(torch.einsum('ntkd,dk->ntd',z[:,indices],conv.weight[:,0,:]))
        expected=x+F.linear(z,module.output.weight[:,:,0]);actual=module(x)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        params=[x,*module.parameters()];a=torch.autograd.grad(actual.square().sum(),params,retain_graph=True);b=torch.autograd.grad(expected.square().sum(),params)
        for u,v in zip(a,b):torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-11)
        with torch.no_grad():
            changed=x.detach().clone();changed[:,21:]+=3
            torch.testing.assert_close(module(changed)[:,:21],actual.detach()[:,:21],rtol=0,atol=0)
            changed=x.detach().clone();changed[:,5]+=4
            torch.testing.assert_close(module(changed)[:,20],actual.detach()[:,20],rtol=0,atol=0)
        for t in (96,48,24,12):
            identity=TrendTemporalResidual(4).double();h=torch.randn(2,t,4,dtype=torch.float64)
            torch.testing.assert_close(identity(h),h,rtol=0,atol=0)
    def full_identity(self,device):
        base,model=fresh(Base).to(device),fresh(Model).to(device)
        common=base.state_dict();state=model.state_dict();self.assertEqual(len(common),236);self.assertEqual(len(state),252)
        self.assertTrue(all(torch.equal(v,state[n]) and v.data_ptr()!=state[n].data_ptr() for n,v in common.items()))
        self.assertEqual(sum(p.numel() for p in model.parameters()),145986)
        x=torch.randn(2,96,7,device=device);y=torch.randn_like(x)
        for training in (False,True):
            outputs=[];rng=[]
            for m in (base,model):
                m.train(training);torch.manual_seed(175);outputs.append(m.predict_components(x));rng.append(torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
            torch.testing.assert_close(*outputs,rtol=0,atol=0);self.assertTrue(torch.equal(*rng))
        gradients=[]
        for m in (base,model):
            m.zero_grad(set_to_none=True);torch.manual_seed(175);p=m(x);d,_=parts(p,y);((p-y).square().mean()+d).backward();gradients.append({n:p.grad for n,p in m.named_parameters()})
        for n in gradients[0]:torch.testing.assert_close(gradients[0][n],gradients[1][n],rtol=0,atol=0,msg=n)
        for n,g in gradients[1].items():
            if n.startswith('trend_temporal_encoders.'):self.assertIsNotNone(g);self.assertEqual(float(g.abs().max()),0.)
        opt=torch.optim.Adam(norm_bias_no_decay_groups(model,1e-4),lr=.001)
        for _ in range(4):
            opt.zero_grad(set_to_none=True);p=model(x);d,_=parts(p,y);((p-y).square().mean()+d).backward()
            for n,p in model.named_parameters():self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
            opt.step()
        for block in model.trend_temporal_encoders:
            self.assertGreater(float(block.output.weight.detach().norm()),0.)
            self.assertGreater(sum(float(p.grad.square().sum()) for p in block.depthwise.parameters()),0.)
    def test_cpu_initial_function_rng_commongrad_and_later_learning(self):self.full_identity('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_initial_function_rng_commongrad_and_later_learning(self):self.full_identity('cuda')

if __name__=='__main__':unittest.main()
