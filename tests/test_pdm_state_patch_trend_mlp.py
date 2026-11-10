"""Four-point trend layout, formula, common initialization and optimizer checks."""
import unittest
import torch
from torch.nn import functional as F
from test_pdm_state_patch import cfg
from models.OscillatorPDMStatePatchTrendMLP import PatchMLPTrendHead
from scripts.run_timemix_pdm_state_patch_trend_mlp_experiment import initialize_pair, COUNTS
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts


class TrendMLPTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2021)

    def check_formula(self, device):
        head=PatchMLPTrendHead(48,96,32).to(device).double()
        x=torch.randn(2,48,32,device=device,dtype=torch.float64,requires_grad=True)
        output=head(x)
        future=F.linear(F.gelu(F.linear(x.transpose(1,2),head.temporal[0].weight,head.temporal[0].bias)),head.temporal[2].weight,head.temporal[2].bias).transpose(1,2)
        blocks=F.linear(F.gelu(F.linear(future,head.readout[0].weight,head.readout[0].bias)),head.readout[2].weight,head.readout[2].bias)
        expected=torch.stack([blocks[:,k,j] for k in range(24) for j in range(4)],dim=1)
        torch.testing.assert_close(output,expected,rtol=0,atol=0)
        v=torch.randn_like(output)
        params=[x,*head.parameters()]
        a=torch.autograd.grad((output*v).sum(),params,retain_graph=True)
        b=torch.autograd.grad((expected*v).sum(),params)
        for aa,bb in zip(a,b): torch.testing.assert_close(aa,bb,rtol=1e-12,atol=1e-12)

    def test_cpu_formula_gradient(self): self.check_formula('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_formula_gradient(self): self.check_formula('cuda')

    def test_independent_four_point_outputs(self):
        head=PatchMLPTrendHead(96,96,32)
        with torch.no_grad():
            head.readout[2].weight.zero_()
            head.readout[2].bias.copy_(torch.arange(4.))
        y=head(torch.randn(3,96,32))
        torch.testing.assert_close(y,torch.arange(4.).repeat(24).expand(3,-1),rtol=0,atol=0)
        with self.assertRaises(ValueError): PatchMLPTrendHead(96,95,32)

    def test_common_scratch_and_seasonal_outputs(self):
        models=initialize_pair(cfg())
        self.assertEqual({a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},COUNTS)
        control, changed=models.values()
        common={n:v for n,v in control.state_dict().items() if not n.startswith('feature_trend_heads.')}
        for n,v in common.items():
            self.assertTrue(torch.equal(v,changed.state_dict()[n]),n)
            self.assertNotEqual(v.data_ptr(),changed.state_dict()[n].data_ptr())
        x=torch.randn(2,96,7)
        for training in (False,True):
            cs=[];rng=[]
            for model in models.values():
                model.train(training);torch.manual_seed(74)
                cs.append(model.predict_components(x));rng.append(torch.get_rng_state())
            torch.testing.assert_close(cs[0][...,:4],cs[1][...,:4],rtol=0,atol=0)
            self.assertGreater(float((cs[0][...,4:]-cs[1][...,4:]).detach().abs().max()),0.)
            self.assertTrue(torch.equal(*rng))
        self.assertFalse(any(n.endswith('temporal.weight') for n in changed.state_dict() if n.startswith('feature_trend_heads.')))

    def test_full_cuda_gradients_learning_optimizer_targetdetach(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        models={a:m.to(device) for a,m in initialize_pair(cfg()).items()}
        x=torch.randn(2,96,7,device=device);target=torch.randn_like(x,requires_grad=True)
        for arm,model in models.items():
            groups=norm_bias_no_decay_groups(model,1e-4)
            policy={id(p):g['weight_decay'] for g in groups for p in g['params']}
            before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('feature_trend_heads.')}
            for n,p in model.named_parameters():
                if n in before:self.assertEqual(policy[id(p)],0. if n.endswith('.bias') else 1e-4)
            opt=torch.optim.Adam(groups,lr=.001)
            pred=model(x);self.assertEqual(pred.shape,(2,96,7))
            d,_=parts(pred,target)
            loss=(pred-target.detach()).square().mean()+d
            loss.backward();self.assertIsNone(target.grad)
            for n,p in model.named_parameters():
                self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
                if n in before and arm=='trend_mlp':self.assertGreater(float(p.grad.norm()),0.,n)
            opt.step()
            for n,p in model.named_parameters():
                if n in before and arm=='trend_mlp':self.assertFalse(torch.equal(p,before[n]),n)


if __name__=='__main__': unittest.main()
