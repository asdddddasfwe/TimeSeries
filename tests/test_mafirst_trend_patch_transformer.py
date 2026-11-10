import unittest
from pathlib import Path
import torch
from torch import nn
from models.OscillatorPDMSeasonalHead import LinearFeatureTrendHead
from models.OscillatorPDMStatePatchMAFirstTrendPatchTransformer import PatchTransformerTrendHead
from tmp.run_mafirst_trend_patch_transformer_20261020 import fresh,optimizer_groups,R

INITIAL_PARITY={}

class TrendReadoutTests(unittest.TestCase):
    def test_initial_complete_forecast_and_recurrent_mean_exact(self):
        from tmp.run_mafirst_level_head_e2e_h192_seed2021_20261017 import control_reference
        model,cfg=fresh(720,'flexible_trend_patch_transformer','/tmp/mafirst_trend_patch_transformer_unit')
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
        model,cfg=fresh(720,'flexible_trend_patch_transformer','/tmp/mafirst_trend_patch_transformer_unit')
        original=torch.load(R/'tmp/mafirst_mmd_ettm1_20261016_h720_seed2021_max30_patience5/initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
        self.assertEqual(len(original),236)
        for name,value in original.items():
            self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
            self.assertNotEqual(value.data_ptr(),model.state_dict()[name].data_ptr())
        self.assertEqual(sum(p.numel() for p in model.parameters()),734722)
        self.assertEqual(len(model.state_dict()),312)
        self.assertEqual(len(list(model.parameters())),276)
        self.assertEqual(len(list(model.buffers())),36)
        groups=optimizer_groups(model)
        actual=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(actual),len(set(actual)));self.assertEqual(set(actual),{id(p) for p in model.parameters()})

    def test_new_branch_learns_after_zero_start_and_full_gradients_finite(self):
        model,cfg=fresh(720,'flexible_trend_patch_transformer','/tmp/mafirst_trend_patch_transformer_unit')
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
                    self.assertGreater(float(head.patch_projection.weight.grad.abs().max()),0.)
            opt.step()

    def test_overlap_patch_order_and_affine_equivalence_without_future_tokens(self):
        torch.manual_seed(7)
        old=LinearFeatureTrendHead(8,6,4).double()
        head=PatchTransformerTrendHead(old,0).double().eval()
        with torch.no_grad():
            old.temporal.weight.normal_();old.temporal.bias.normal_()
            old.readout.weight.normal_();old.readout.bias.normal_()
        x=torch.randn(2,8,4,dtype=torch.float64)
        actual_patches=[];future_calls=[]
        h1=head.patch_projection.register_forward_pre_hook(lambda m,args:actual_patches.append(args[0].detach()))
        h2=head.temporal.register_forward_hook(lambda m,args,out:future_calls.append(tuple(out.shape)))
        actual=head(x);h1.remove();h2.remove()
        expected_patches=torch.stack([x[:,i:i+4,:].reshape(2,-1) for i in (0,2,4)],1)
        self.assertTrue(torch.equal(actual_patches[0],expected_patches))
        self.assertEqual(future_calls,[])
        torch.testing.assert_close(actual,old(x),rtol=1e-11,atol=1e-11)
        self.assertEqual(tuple(actual.shape),(2,6))
        self.assertEqual(tuple(head.patch_positions.shape),(1,3,4))
        self.assertEqual(head.transformer.layers[0].self_attn.dropout,0.)

    def test_attention_encodes_other_historical_patches(self):
        torch.manual_seed(17)
        old=LinearFeatureTrendHead(12,6,4).double()
        head=PatchTransformerTrendHead(old,0).double().eval()
        x=torch.randn(2,12,4,dtype=torch.float64,requires_grad=True)
        encoded=head.encode_history(x)
        self.assertEqual(tuple(encoded.shape),(2,5,4))
        grad=torch.autograd.grad(encoded[0,0,0],x)[0]
        self.assertGreater(float(grad[0,-1].abs().max()),1e-9)
if __name__=='__main__':unittest.main()
