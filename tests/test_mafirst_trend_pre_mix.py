import unittest
from pathlib import Path
import torch
from torch import nn
from models.OscillatorPDMSeasonalHead import LinearFeatureTrendHead
from models.OscillatorPDMStatePatchMAFirstTrendPreMix import PreMixTrendHead,restore_patch_residual
from tmp.run_mafirst_trend_pre_mix_20261020 import fresh,optimizer_groups,R

INITIAL_PARITY={}

class TrendReadoutTests(unittest.TestCase):
    def test_initial_state_matches_post_mix_and_original_predictions(self):
        from tmp.run_mafirst_trend_patch_transformer_20261020 import fresh as fresh_after
        from tmp.run_mafirst_level_head_e2e_h192_seed2021_20261017 import control_reference
        before,cfg=fresh(720,'flexible_trend_pre_mix','/tmp/premix_unit')
        after,_=fresh_after(720,'flexible_trend_patch_transformer','/tmp/premix_unit_after')
        for k,v in before.state_dict().items():self.assertTrue(torch.equal(v,after.state_dict()[k]),k)
        from tmp.mafirst_trend_pre_mix_common_20261020 import read,PLAN
        predecessor=read(PLAN)['predecessor']
        saved=torch.load(R/predecessor['output']/'initial.pt',map_location='cpu',weights_only=True)
        self.assertEqual(len(saved),312)
        for k,v in before.state_dict().items():self.assertTrue(torch.equal(v,saved[k]),k)
        INITIAL_PARITY['actual_predecessor_initial312_exact']=True
        original=control_reference(cfg);before.eval();after.eval();original.eval()
        torch.manual_seed(73);x=torch.randn(2,96,7)
        with torch.no_grad():
            expected=original.predict_components(x);actual=before.predict_components(x)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
            self.assertTrue(torch.equal(actual[...,:4],expected[...,:4]))
            torch.testing.assert_close(before(x),after(x),rtol=2e-5,atol=2e-5)
            INITIAL_PARITY.update(prediction_max_abs=float((before(x)-original(x)).abs().max()),all312_initial_states_match_post_mix=True,recurrent_components_bitwise_equal=True)
    def test_parameter_count_storage_and_optimizer_coverage(self):
        model,cfg=fresh(720,'flexible_trend_pre_mix','/tmp/mafirst_trend_pre_mix_unit')
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
        model,cfg=fresh(720,'flexible_trend_pre_mix','/tmp/mafirst_trend_pre_mix_unit')
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

    def test_overlap_restoration_matches_independent_loop_and_gradients(self):
        torch.manual_seed(19)
        delta=torch.randn(2,5,4,dtype=torch.float64,requires_grad=True)
        actual=restore_patch_residual(delta,12)
        expected=torch.stack([torch.stack([delta[:,j] for j in range(5) if 2*j<=t<2*j+4],1).mean(1) for t in range(12)],1)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        g1=torch.autograd.grad(actual.square().sum(),delta,retain_graph=True)[0]
        g2=torch.autograd.grad(expected.square().sum(),delta)[0]
        torch.testing.assert_close(g1,g2,rtol=1e-11,atol=1e-12)

    def test_transformer_before_mixing_mlp_after_mixing(self):
        model,_=fresh(720,'flexible_trend_pre_mix','/tmp/premix_order')
        model.eval();events=[];handles=[]
        for h in model.feature_trend_heads:
            handles.append(h.transformer.register_forward_hook(lambda m,a,o:events.append('transformer')))
            handles.append(h.nonlinear_readout.register_forward_hook(lambda m,a,o:events.append('mlp')))
        for b in model.pdm_blocks:handles.append(b.register_forward_hook(lambda m,a,o:events.append('pdm')))
        with torch.no_grad():result=model(torch.randn(2,96,7))
        for h in handles:h.remove()
        self.assertEqual(tuple(result.shape),(2,720,7))
        self.assertEqual(events[:4],['transformer']*4)
        self.assertTrue(all(v=='pdm' for v in events[4:-4]))
        self.assertGreater(len(events[4:-4]),0)
        self.assertEqual(events[-4:],['mlp']*4)
if __name__=='__main__':unittest.main()
