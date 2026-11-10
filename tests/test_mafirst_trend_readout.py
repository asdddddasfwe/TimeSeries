import unittest
from pathlib import Path
import torch
from torch import nn
from models.OscillatorPDMSeasonalHead import LinearFeatureTrendHead
from models.OscillatorPDMStatePatchMAFirstTrendReadout import FlexibleFeatureTrendHead
from tmp.run_mafirst_trend_readout_20261020 import fresh,optimizer_groups,R

class TrendReadoutTests(unittest.TestCase):
    def test_initial_complete_forecast_and_recurrent_mean_exact(self):
        from tmp.run_mafirst_level_head_e2e_h192_seed2021_20261017 import control_reference
        model,cfg=fresh(720,'flexible_trend_readout','/tmp/mafirst_trend_readout_unit')
        ref=control_reference(cfg)
        model.eval();ref.eval();torch.manual_seed(73);x=torch.randn(2,96,7)
        with torch.no_grad():
            expected=ref.predict_components(x);actual=model.predict_components(x)
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(actual[...,:4].mean(1),expected[...,:4].mean(1)))
            self.assertGreater(float(actual[...,:4].mean(1).abs().max()),1e-6)
            self.assertTrue(torch.equal(model(x),ref(x)))
            self.assertEqual(tuple(model(x).shape),(2,720,7))

    def test_parameter_count_storage_and_optimizer_coverage(self):
        model,cfg=fresh(720,'flexible_trend_readout','/tmp/mafirst_trend_readout_unit')
        original=torch.load(R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5/initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
        self.assertEqual(len(original),236)
        for name,value in original.items():
            self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
            self.assertNotEqual(value.data_ptr(),model.state_dict()[name].data_ptr())
        self.assertEqual(sum(p.numel() for p in model.parameters()),329158)
        self.assertEqual(len(model.state_dict()),252)
        self.assertEqual(len(list(model.parameters())),220)
        self.assertEqual(len(list(model.buffers())),32)
        groups=optimizer_groups(model)
        actual=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(actual),len(set(actual)));self.assertEqual(set(actual),{id(p) for p in model.parameters()})

    def test_new_branch_learns_after_zero_start_and_full_gradients_finite(self):
        model,cfg=fresh(720,'flexible_trend_readout','/tmp/mafirst_trend_readout_unit')
        model.train();torch.manual_seed(99);x=torch.randn(2,96,7);target=torch.randn(2,720,7)
        opt=torch.optim.Adam(optimizer_groups(model),lr=.001)
        for step in range(2):
            opt.zero_grad();loss=(model(x)-target).square().mean();loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            for head in model.feature_trend_heads:
                if step==0:
                    self.assertEqual(float(head.nonlinear_readout[0].weight.grad.abs().max()),0.)
                    self.assertGreater(float(head.nonlinear_readout[-1].weight.grad.abs().max()),0.)
                else:self.assertGreater(float(head.nonlinear_readout[0].weight.grad.abs().max()),0.)
            opt.step()

    def test_added_head_can_express_non_affine_feature_readout(self):
        old=LinearFeatureTrendHead(1,1,1).double();head=FlexibleFeatureTrendHead(old,0).double()
        with torch.no_grad():
            head.temporal.weight.fill_(1);head.temporal.bias.zero_()
            head.readout.weight.zero_();head.readout.bias.zero_()
            head.nonlinear_readout[0].weight.zero_();head.nonlinear_readout[0].bias.zero_()
            head.nonlinear_readout[0].weight[0,0]=1
            head.nonlinear_readout[-1].weight.zero_();head.nonlinear_readout[-1].weight[0,0]=1
        vals=head(torch.tensor([[[-1.]],[[0.]],[[1.]]],dtype=torch.float64)).flatten()
        self.assertGreater(float(abs(vals[0]+vals[2]-2*vals[1])),.1)
if __name__=='__main__':unittest.main()
