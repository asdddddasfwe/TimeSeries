"""Interpolation references, same initial function, and correction learning."""
import copy
import math
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from test_pdm_state_patch import cfg
from models.OscillatorPDMSeasonalHead import LinearFeatureTrendHead
from models.OscillatorPDMStatePatchTrendInterp import InterpolatedTrendCorrection,interpolation_matrix
from scripts.run_timemix_pdm_state_patch_trend_interp_experiment import initialize_pair,COUNTS
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.forecast_low_frequency_loss import moving_average_forecast,low_frequency_mse
from utils.patch_covariance_loss import parts


class TrendInterpTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(2021)

    def formula(self,device):
        head=InterpolatedTrendCorrection(LinearFeatureTrendHead(12,96,32),12,96,32).to(device).double()
        with torch.no_grad():head.coefficient_net[2].weight.normal_();head.coefficient_net[2].bias.normal_()
        x=torch.randn(2,12,32,device=device,dtype=torch.float64,requires_grad=True)
        actual=head(x)
        pooled=torch.stack([x[:,i*12//8:math.ceil((i+1)*12/8)].mean(1) for i in range(8)],dim=1)
        context=torch.cat([pooled.flatten(1),x[:,-1],x[:,-1]-x[:,0]],dim=-1)
        q=F.linear(F.gelu(F.linear(context,head.coefficient_net[0].weight,head.coefficient_net[0].bias)),head.coefficient_net[2].weight,head.coefficient_net[2].bias)
        expected_correction=[]
        # Match the stored FP32 fixed basis values when operating head in FP64.
        for step in range(96):
            u=step*7/95;left=min(int(u),6);fraction=u-left
            w0=float(np.float32(1-fraction));w1=float(np.float32(fraction))
            expected_correction.append(q[:,left]*w0+q[:,left+1]*w1)
        temporal=F.linear(x.transpose(1,2),head.temporal.weight,head.temporal.bias).transpose(1,2)
        base=F.linear(temporal,head.readout.weight,head.readout.bias).squeeze(-1)
        expected=base+torch.stack(expected_correction,1)
        torch.testing.assert_close(actual,expected,rtol=2e-7,atol=1e-7)
        v=torch.randn_like(actual);params=[x,*head.parameters()]
        a=torch.autograd.grad((actual*v).sum(),params,retain_graph=True)
        b=torch.autograd.grad((expected*v).sum(),params)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=2e-6,atol=2e-6)

    def test_cpu_explicit_formula_gradients(self):self.formula('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_explicit_formula_gradients(self):self.formula('cuda')

    def test_basis_endpoints_constant_linear_curves(self):
        b=interpolation_matrix(96)
        self.assertTrue((b>=0).all());torch.testing.assert_close(b.sum(1),torch.ones(96),rtol=0,atol=1e-7)
        torch.testing.assert_close(b[0],F.one_hot(torch.tensor(0),8).float(),rtol=0,atol=0)
        torch.testing.assert_close(b[-1],F.one_hot(torch.tensor(7),8).float(),rtol=0,atol=0)
        torch.testing.assert_close(b@torch.ones(8)*2,torch.full((96,),2.),rtol=0,atol=2e-7)
        torch.testing.assert_close(b@torch.linspace(-1,1,8),torch.linspace(-1,1,96),rtol=0,atol=2e-7)
        with self.assertRaises(ValueError):interpolation_matrix(4,8)

    def test_same_initial_predictions_rng_common_gradients_storage(self):
        models=initialize_pair(cfg());self.assertEqual({k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},COUNTS)
        common=models['control'].state_dict()
        for key in ['trend_interp','trend_interp_lowfreq']:
            for n,v in common.items():
                self.assertTrue(torch.equal(v,models[key].state_dict()[n]),n)
                self.assertNotEqual(v.data_ptr(),models[key].state_dict()[n].data_ptr())
        for n,v in models['trend_interp'].state_dict().items():
            self.assertTrue(torch.equal(v,models['trend_interp_lowfreq'].state_dict()[n]),n)
            self.assertNotEqual(v.data_ptr(),models['trend_interp_lowfreq'].state_dict()[n].data_ptr())
        x=torch.randn(2,96,7);y=torch.randn_like(x)
        for training in (False,True):
            preds=[];rngs=[];grads=[]
            for model in models.values():
                model.train(training);torch.manual_seed(82);model.zero_grad()
                pred=model(x);preds.append(pred);rngs.append(torch.get_rng_state())
                d,_=parts(pred,y);((pred-y).square().mean()+d).backward()
                grads.append({n:p.grad.clone() for n,p in model.named_parameters() if n in common})
            for i in [1,2]:
                torch.testing.assert_close(preds[0],preds[i],rtol=0,atol=0)
                self.assertTrue(torch.equal(rngs[0],rngs[i]))
                for n,g in grads[0].items():torch.testing.assert_close(g,grads[i][n],rtol=0,atol=0)

    def test_zero_last_layer_then_both_layers_learn_and_optimizer(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=initialize_pair(cfg())['trend_interp_lowfreq'].to(device)
        groups=norm_bias_no_decay_groups(model,1e-4)
        policy={id(p):g['weight_decay'] for g in groups for p in g['params']}
        opt=torch.optim.Adam(groups,lr=.001)
        x=torch.randn(2,96,7,device=device);y=torch.randn_like(x,requires_grad=True)
        before={n:p.detach().clone() for n,p in model.named_parameters() if '.coefficient_net.' in n}
        for step in range(2):
            opt.zero_grad();pred=model(x);d,_=parts(pred,y)
            loss=(pred-y.detach()).square().mean()+d+low_frequency_mse(pred,y);loss.backward()
            self.assertIsNone(y.grad)
            for n,p in model.named_parameters():
                self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
                if n in before:
                    self.assertEqual(policy[id(p)],0. if n.endswith('.bias') else 1e-4)
                    if step==0 and '.coefficient_net.0.' in n:self.assertEqual(float(p.grad.norm()),0.)
                    else:self.assertGreater(float(p.grad.norm()),0.,n)
            opt.step()
        for n,p in model.named_parameters():
            if n in before:self.assertFalse(torch.equal(p,before[n]),n)


class LowFrequencyLossTests(unittest.TestCase):
    def test_numpy_value_and_analytic_gradient_cpu_cuda(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            p=torch.randn(2,96,7,device=device,dtype=torch.float64,requires_grad=True)
            y=torch.randn_like(p,requires_grad=True)
            pa,ya=p.detach().cpu().numpy(),y.detach().cpu().numpy()
            def average(a):return np.lib.stride_tricks.sliding_window_view(np.pad(a,((0,0),(12,12),(0,0)),mode='edge'),25,axis=1).mean(-1)
            e=average(pa)-average(ya);expected=float((e**2).mean())
            loss=low_frequency_mse(p,y);self.assertAlmostEqual(float(loss.detach()),expected,places=12)
            loss.backward();self.assertIsNone(y.grad)
            grad=np.zeros_like(pa)
            for t in range(96):
                for offset in range(-12,13):grad[:,min(max(t+offset,0),95)]+=2*e[:,t]/(25*e.size)
            np.testing.assert_allclose(p.grad.cpu().numpy(),grad,rtol=1e-11,atol=1e-12)

    def test_dc_and_target_detach(self):
        p=torch.full((2,96,7),2.,requires_grad=True);y=torch.zeros_like(p,requires_grad=True)
        torch.testing.assert_close(moving_average_forecast(p),p,rtol=0,atol=5e-7)
        loss=low_frequency_mse(p,y);self.assertAlmostEqual(float(loss.detach()),4.,places=5)
        loss.backward();self.assertIsNone(y.grad);self.assertGreater(float(p.grad.norm()),0.)


if __name__=='__main__':unittest.main()
