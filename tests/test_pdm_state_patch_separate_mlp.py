"""Untied feature MLP formula, exact initial function, and gradient accounting."""
import unittest
import torch
from torch.nn import functional as F
from test_pdm_state_patch import cfg
from scripts.run_timemix_pdm_state_patch_separate_mlp_experiment import initialize_pair, COUNTS
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts


class SeparateMLPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def formula(self, device):
        block = initialize_pair(cfg())['separate_mlp'].pdm_blocks[0].to(device).double()
        s = [torch.randn(2,t,32,device=device,dtype=torch.float64,requires_grad=True) for t in (96,48,24,12)]
        t = [torch.randn_like(v,requires_grad=True) for v in s]
        actual = block(s,t)
        ms = block.mixing_multi_scale_season([v.transpose(1,2) for v in s])
        mt = block.mixing_multi_scale_trend([v.transpose(1,2) for v in t])
        def explicit(x, mlp):
            return F.linear(F.gelu(F.linear(x,mlp[0].weight,mlp[0].bias)),mlp[2].weight,mlp[2].bias)
        expected = ([v+explicit(m,block.out_cross_layer) for v,m in zip(s,ms)],
                    [v+explicit(m,block.trend_out_cross_layer) for v,m in zip(t,mt)])
        for a,b in zip(sum(actual,[]),sum(expected,[])):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        inputs=s+t+list(block.parameters())
        probes=[torch.randn_like(v) for v in sum(actual,[])]
        aa=torch.autograd.grad(sum((v*p).sum() for v,p in zip(sum(actual,[]),probes)),inputs,retain_graph=True)
        bb=torch.autograd.grad(sum((v*p).sum() for v,p in zip(sum(expected,[]),probes)),inputs)
        for a,b in zip(aa,bb): torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_cpu_formula_gradient(self): self.formula('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_formula_gradient(self): self.formula('cuda')

    def test_scratch_components_rng_independent_storage(self):
        models=initialize_pair(cfg())
        self.assertEqual({a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},COUNTS)
        control, changed=models.values()
        for n,v in control.state_dict().items():
            self.assertTrue(torch.equal(v,changed.state_dict()[n]),n)
            self.assertNotEqual(v.data_ptr(),changed.state_dict()[n].data_ptr())
        block=changed.pdm_blocks[0]
        for a,b in zip(block.out_cross_layer.parameters(),block.trend_out_cross_layer.parameters()):
            self.assertTrue(torch.equal(a,b));self.assertNotEqual(a.data_ptr(),b.data_ptr())
        x=torch.randn(2,96,7)
        for training in (False,True):
            cs=[];rng=[]
            for model in models.values():
                model.train(training);torch.manual_seed(74)
                cs.append(model.predict_components(x));rng.append(torch.get_rng_state())
            torch.testing.assert_close(*cs,rtol=0,atol=0)
            self.assertTrue(torch.equal(*rng))

    def test_block_intervention_is_path_specific(self):
        block=initialize_pair(cfg())['separate_mlp'].pdm_blocks[0]
        s=[torch.randn(2,t,32) for t in (96,48,24,12)]
        t=[torch.randn_like(v) for v in s]
        a=block(s,t)
        with torch.no_grad(): block.trend_out_cross_layer[2].bias.add_(1)
        b=block(s,t)
        for aa,bb in zip(a[0],b[0]): torch.testing.assert_close(aa,bb,rtol=0,atol=0)
        for aa,bb in zip(a[1],b[1]): torch.testing.assert_close(bb-aa,torch.ones_like(aa),rtol=1e-6,atol=1e-6)

    def test_cuda_shared_gradient_sum_and_learning(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        models={a:m.to(device) for a,m in initialize_pair(cfg()).items()}
        x=torch.randn(2,96,7,device=device);target=torch.randn_like(x,requires_grad=True)
        for model in models.values():
            model.train();torch.manual_seed(74)
            pred=model(x);d,_=parts(pred,target)
            ((pred-target.detach()).square().mean()+d).backward()
        self.assertIsNone(target.grad)
        base,changed=[dict(m.named_parameters()) for m in models.values()]
        for n,p in base.items():
            expected=changed[n].grad
            if '.out_cross_layer.' in n:
                expected=expected+changed[n.replace('.out_cross_layer.','.trend_out_cross_layer.')].grad
            torch.testing.assert_close(p.grad,expected,rtol=2e-5,atol=2e-7,msg=n)
        for model in models.values():
            groups=norm_bias_no_decay_groups(model,1e-4)
            policy={id(p):g['weight_decay'] for g in groups for p in g['params']}
            before={n:p.detach().clone() for n,p in model.named_parameters() if '.out_cross_layer.' in n or '.trend_out_cross_layer.' in n}
            for n,p in model.named_parameters():
                self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
                if n in before:
                    self.assertEqual(policy[id(p)],0. if n.endswith('.bias') else 1e-4)
            opt=torch.optim.Adam(groups,lr=.001)
            opt.step()
            # Original trend readout/conditioning initialization blocks some
            # upstream trend gradients on step one. Check learning after it opens.
            for _ in range(2):
                opt.zero_grad(set_to_none=True);torch.manual_seed(74)
                pred=model(x);d,_=parts(pred,target)
                ((pred-target.detach()).square().mean()+d).backward()
                for n,p in model.named_parameters():
                    self.assertTrue(torch.isfinite(p.grad).all(),n)
                opt.step()
            for n,p in model.named_parameters():
                if n in before:
                    self.assertGreater(float(p.grad.norm()),0.,n)
                    self.assertFalse(torch.equal(p,before[n]),n)


if __name__=='__main__': unittest.main()
