import unittest
from pathlib import Path
import torch
from torch import nn
from models.OscillatorPDMSeasonalHead import LinearFeatureTrendHead
from models.OscillatorPDMStatePatchMAFirstTrendHistoryPatch import HistoryPatchTrendHead
from tmp.run_mafirst_trend_history_patch_20261020 import fresh,optimizer_groups,R

INITIAL_PARITY={}

class TrendReadoutTests(unittest.TestCase):
    def test_initial_complete_forecast_and_recurrent_mean_exact(self):
        from tmp.run_mafirst_level_head_e2e_h192_seed2021_20261017 import control_reference
        model,cfg=fresh(720,'flexible_trend_history_patch','/tmp/mafirst_trend_history_patch_unit')
        ref=control_reference(cfg)
        model.eval();ref.eval();torch.manual_seed(73);x=torch.randn(2,96,7)
        with torch.no_grad():
            expected=ref.predict_components(x);actual=model.predict_components(x)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
            self.assertTrue(torch.equal(actual[...,:4],expected[...,:4]))
            self.assertTrue(torch.equal(actual[...,:4].mean(1),expected[...,:4].mean(1)))
            self.assertGreater(float(actual[...,:4].mean(1).abs().max()),1e-6)
            torch.testing.assert_close(model(x),ref(x),rtol=2e-5,atol=2e-5)
            self.assertEqual(tuple(model(x).shape),(2,720,7))
            INITIAL_PARITY.update(component_max_abs=float((actual-expected).abs().max()),
                prediction_max_abs=float((model(x)-ref(x)).abs().max()),
                recurrent_components_bitwise_exact=torch.equal(actual[...,:4],expected[...,:4]),
                tolerance_atol=2e-5,tolerance_rtol=2e-5,
                note='Original affine head is algebraically reordered; not claimed whole-output bitwise equality')

    def test_parameter_count_storage_and_optimizer_coverage(self):
        model,cfg=fresh(720,'flexible_trend_history_patch','/tmp/mafirst_trend_history_patch_unit')
        original=torch.load(R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5/initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
        self.assertEqual(len(original),236)
        for name,value in original.items():
            self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
            self.assertNotEqual(value.data_ptr(),model.state_dict()[name].data_ptr())
        self.assertEqual(sum(p.numel() for p in model.parameters()),616578)
        self.assertEqual(len(model.state_dict()),260)
        self.assertEqual(len(list(model.parameters())),228)
        self.assertEqual(len(list(model.buffers())),32)
        groups=optimizer_groups(model)
        actual=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(actual),len(set(actual)));self.assertEqual(set(actual),{id(p) for p in model.parameters()})

    def test_new_branch_learns_after_zero_start_and_full_gradients_finite(self):
        model,cfg=fresh(720,'flexible_trend_history_patch','/tmp/mafirst_trend_history_patch_unit')
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
                else:
                    self.assertGreater(float(head.nonlinear_readout[0].weight.grad.abs().max()),0.)
                    self.assertGreater(float(head.segment_encoder[0].weight.grad.abs().max()),0.)
            opt.step()

    def test_historical_segment_features_joint_outputs_and_no_future_features(self):
        old=LinearFeatureTrendHead(4,3,2).double();head=HistoryPatchTrendHead(old,0).double()
        with torch.no_grad():
            head.readout.weight.zero_();head.readout.bias.zero_()
            head.segment_encoder[0].weight.zero_();head.segment_encoder[0].bias.zero_()
            head.segment_encoder[0].weight[0,0::2]=.25
            head.nonlinear_readout[0].weight.zero_();head.nonlinear_readout[0].bias.zero_()
            head.nonlinear_readout[0].weight[0,0]=1.
            head.nonlinear_readout[-1].weight.zero_();head.nonlinear_readout[-1].bias.zero_()
            head.nonlinear_readout[-1].weight[:,0]=torch.tensor([1.,2.,3.],dtype=torch.float64)
        x=torch.zeros(3,4,2,dtype=torch.float64,requires_grad=True)
        with torch.no_grad():x[:,:,0]=torch.tensor([-1.,0.,1.],dtype=torch.float64)[:,None]
        observed=[];hook=head.segment_encoder.register_forward_hook(lambda m,a,o:observed.append(tuple(o.shape)))
        y=head(x);hook.remove()
        expected=torch.nn.functional.gelu(torch.nn.functional.gelu(x[:,:,0].mean(1)))[:,None]*torch.tensor([1.,2.,3.],dtype=torch.float64)
        torch.testing.assert_close(y,expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(tuple(y.shape),(3,3));self.assertEqual(observed,[(3,1,2)])
        self.assertGreater(float(abs(y[0,0]+y[2,0]-2*y[1,0]).detach()),.1)
        grad=torch.autograd.grad(y[2].sum(),x)[0]
        self.assertTrue(torch.all(grad[2,:,0].abs()>0))
if __name__=='__main__':unittest.main()
