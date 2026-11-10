"""Raw patch order, double-axis equations, independent paths and learning."""
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from test_pdm_state_patch import cfg
from layers.LagARFormer import LagAxialBlock
from models.OscillatorPDMStatePatchPreDecomp import PatchCodec
from scripts.run_timemix_pdm_state_patch_predecomp_experiment import initialize_pair, COUNTS
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts
from utils.pdm_state_patch_predecomp_diagnostics import FeatureStatistics, gradient_probe


class PreDecompTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def axial(self,device):
        block=LagAxialBlock(32,64,0.).to(device).double()
        with torch.no_grad():
            block.row.gamma.fill_(.4);block.column.gamma.fill_(-.2)
        x=torch.randn(2,32,3,5,device=device,dtype=torch.float64,requires_grad=True)
        actual=block(x)
        # Upstream PyPOTS row and column each includes an input residual;
        # adapter sums updates and keeps that residual exactly once.
        mixed=block.row(x)+block.column(x)-x
        expected=block.norm(mixed.permute(0,2,3,1)).permute(0,3,1,2)
        torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
        probe=torch.randn_like(x)
        inputs=[x,*block.parameters()]
        ga=torch.autograd.grad((actual*probe).sum(),inputs,retain_graph=True)
        gb=torch.autograd.grad((expected*probe).sum(),inputs)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)

    def test_cpu_time_mixer_pp_row_column_equations(self):self.axial('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_time_mixer_pp_row_column_equations(self):self.axial('cuda')

    def test_sliding_patch_before_projection_overlap_formula(self):
        codec=PatchCodec(32).double()
        x=torch.randn(2,12,1,dtype=torch.float64,requires_grad=True)
        actual=codec.sliding(x)
        windows=torch.stack([x[:,i:i+4,0] for i in range(0,9,2)],1)
        tokens=F.linear(windows,codec.patch_projection.weight,codec.patch_projection.bias)
        decoded=F.linear(tokens,codec.patch_readout.weight,codec.patch_readout.bias).reshape(2,5,4,32)
        positions=torch.arange(5)[:,None]*2+torch.arange(4)[None,:]
        values=torch.zeros(2,12,32,dtype=x.dtype).index_add(1,positions.flatten(),decoded.flatten(1,2))
        counts=torch.zeros(12,dtype=x.dtype).index_add(0,positions.flatten(),torch.ones(20,dtype=x.dtype))
        expected=values/counts[None,:,None]
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        probe=torch.randn_like(actual);args=[x,*codec.parameters()]
        a=torch.autograd.grad((actual*probe).sum(),args,retain_graph=True)
        b=torch.autograd.grad((expected*probe).sum(),args)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=1e-12,atol=1e-12)

    def test_all_raw_ma_then_fft_and_no_pointwise_embedding(self):
        models=initialize_pair(cfg())
        self.assertEqual({a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},COUNTS)
        import models.OscillatorPDMStatePatchPreDecomp as module
        original=module.select_seasonal_feature_periods
        widths=[];inputs=[]
        def select(x,k):
            self.assertEqual(x.shape[-1],1);inputs.append(x.detach().clone())
            return original(x,k)
        for arm,model in models.items():
            self.assertFalse(any(isinstance(m,torch.nn.Linear) and m.in_features==1 and m.out_features==32 for m in model.modules()))
            handle=model.initial_decomposition.register_forward_pre_hook(lambda m,args:widths.append(args[0].shape[-1]))
            model.eval()
            with patch.object(module,'select_seasonal_feature_periods',side_effect=select):
                with torch.no_grad():c=model.predict_components(torch.randn(2,96,7))
            handle.remove();self.assertEqual(tuple(c.shape),(2,96,7,8));self.assertTrue(torch.isfinite(c).all())
        self.assertEqual(widths,[1]*12);self.assertEqual(len(inputs),12)

    def test_independent_storage_and_branch_intervention(self):
        models=initialize_pair(cfg());base=models['position_only'];ind=models['decoupled'];third=models['season_to_trend']
        for sc,tc,bc in zip(ind.season_patch_codecs,ind.trend_patch_codecs,base.shared_patch_codecs):
            for n,p in sc.named_parameters():
                self.assertTrue(torch.equal(p,dict(bc.named_parameters())[n]));self.assertNotEqual(p.data_ptr(),dict(tc.named_parameters())[n].data_ptr())
            self.assertEqual({id(p) for p in sc.parameters()}&{id(p) for p in tc.parameters()},set())
        self.assertFalse(any('trend_condition' in n or 'season_condition' in n for n,_ in ind.named_parameters()))
        ind.eval();x=torch.randn(2,96,7)
        with torch.no_grad():
            a=ind.predict_components(x)
            ind.trend_patch_codecs[0].patch_projection.weight.add_(.5)
            ind.feature_trend_heads[0].readout.weight.fill_(.1)
            b=ind.predict_components(x)
        torch.testing.assert_close(a[...,:4],b[...,:4],rtol=0,atol=0)
        self.assertGreater(float((a[...,4:]-b[...,4:]).abs().max()),0.)
        for e,h in zip(third.seasonals,third.feature_trend_heads):
            for n,p in e.decoder.named_parameters():
                if n in dict(h.decoder.named_parameters()):self.assertNotEqual(p.data_ptr(),dict(h.decoder.named_parameters())[n].data_ptr())
            self.assertEqual(float(h.decoder.season_condition.weight.abs().max()),0.)
            self.assertEqual(float(h.decoder.readout.output_projection.weight.abs().max()),0.)

    def test_historical_season_changes_trend_dynamics(self):
        model=initialize_pair(cfg())['season_to_trend'].double()
        decoder=model.feature_trend_heads[0].decoder
        history=torch.randn(2,96,32,dtype=torch.float64)
        season=torch.randn_like(history,requires_grad=True)
        with torch.no_grad():decoder.season_condition.weight.normal_(0,.03)
        a=decoder.dynamics(history,season)[0]
        b=decoder.dynamics(history,torch.zeros_like(season))[0]
        self.assertGreater(float((a-b).detach().abs().max()),1e-6)
        g=torch.autograd.grad(a.square().sum(),season)[0]
        self.assertGreater(float(g.norm()),0.)

    def test_cuda_learning_policy_and_sameforward_diagnostics(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        models={a:m.to(device) for a,m in initialize_pair(cfg()).items()}
        x=torch.randn(2,96,7,device=device);target=torch.randn_like(x,requires_grad=True)
        for arm,model in models.items():
            groups=norm_bias_no_decay_groups(model,1e-4);opt=torch.optim.Adam(groups,lr=.001)
            names={id(p):n for n,p in model.named_parameters()}
            policy={names[id(p)]:g['weight_decay'] for g in groups for p in g['params']}
            for n,p in model.named_parameters():
                if '.season_condition.' in n or '.trend_condition.' in n:self.assertEqual(policy[n],0.)
                elif ('patch_' in n) and n.endswith('weight') and '.norm' not in n and 'output_norm' not in n:self.assertEqual(policy[n],1e-4)
            for step in range(3):
                model.train();opt.zero_grad(set_to_none=True);torch.manual_seed(74)
                stats=FeatureStatistics(model);pred=model(x);d,_=parts(pred,target);mse=(pred-target.detach()).square().mean()
                probe=gradient_probe(model,mse,d);stats.close();self.assertEqual(len(stats.summary()),8)
                self.assertTrue(all(torch.isfinite(torch.tensor(s['mse_norm'])) for s in probe.values()))
                (mse+d).backward()
                for n,p in model.named_parameters():self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
                if step==2 and arm=='season_to_trend':
                    self.assertGreater(sum(float(p.grad.square().sum()) for n,p in model.named_parameters() if '.season_condition.' in n),0.)
                    self.assertGreater(sum(float(p.grad.square().sum()) for n,p in model.named_parameters() if n.startswith('trend_patch_encoders.')),0.)
                opt.step()
            self.assertIsNone(target.grad)


if __name__=='__main__':unittest.main()
