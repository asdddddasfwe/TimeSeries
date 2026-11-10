import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from test_oscillator_mixer import config
from scripts.train_timemix_pdm_trend_prelinear import initialize_pair as old_pair
from scripts.train_timemix_pdm_trend_smooth_residual import initialize_pair, parameter_groups, dataset, training_losses
from models.OscillatorPDMTrendSmoothResidual import Model, SmoothTrendCorrection
from utils.pdm_trend_smooth_residual_diagnostics import SelectedValidationCapture


class SmoothResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,e_layers=2,dropout=.2)
        self.cfg.seasonal_normalize=False

    def test_shared_initial_states_rng_and_parameter_counts(self):
        torch.manual_seed(2021);old=old_pair(self.cfg);rng=torch.get_rng_state()
        torch.manual_seed(2021);new=initialize_pair(self.cfg)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for arm,m in old.items():
            for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,new[arm].state_dict()[k]))
        for k,v in new['trend_prelinear'].state_dict().items():
            self.assertTrue(torch.equal(v,new['smooth_residual'].state_dict()[k]))
        self.assertEqual(sum(p.numel() for p in new['trend_prelinear'].parameters()),sum(p.numel() for p in new['smooth_residual'].parameters()))
        for length in [96,64]:
            cfg=config(seq_len=length,pred_len=16,moving_avg=25,e_layers=2);cfg.seasonal_normalize=False
            m=Model(cfg)
            for block in m.pdm_blocks:
                self.assertEqual([f.kernel for f in block.mixing_multi_scale_trend.correction_filters],[25,13,7,3])

    def test_explicit_filter_axis_endpoints_and_mixing_order(self):
        pair=initialize_pair(self.cfg)
        old=pair['seasonal_trend'].pdm_blocks[0].mixing_multi_scale_trend
        mix=pair['smooth_residual'].pdm_blocks[0].mixing_multi_scale_trend
        data=[torch.randn(3,4,n) for n in mix.lengths];transformed=[]
        with torch.no_grad():
            for i,(x,l,f) in enumerate(zip(data,mix.pre_linears,mix.correction_filters)):
                l.weight.copy_(torch.eye(l.in_features).roll(1,0)*(i+1))
                l.bias.copy_(torch.arange(l.in_features)%2)
                delta=l(x)-x
                ref=x+F.avg_pool1d(F.pad(delta,(f.kernel//2,f.kernel//2),mode='replicate'),f.kernel,stride=1)
                torch.testing.assert_close(f(x,l(x)),ref,rtol=0,atol=0)
                transformed.append(ref)
            expected=old(transformed);actual=mix(data)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaises(ValueError):mix(data[:-1])
        with self.assertRaises(ValueError):mix([x.transpose(1,2) for x in data])

    def test_identity_train_eval_components_and_future_ignored(self):
        pair=initialize_pair(self.cfg);x=torch.randn(2,96,2)
        for training in [False,True]:
            out=[];rng=[]
            for m in pair.values():
                m.train(training);torch.manual_seed(97)
                with torch.no_grad():out.append(m(x))
                rng.append(torch.get_rng_state())
            for p,r in zip(out[1:],rng[1:]):
                torch.testing.assert_close(p,out[0],rtol=0,atol=2e-6)
                self.assertTrue(torch.equal(r,rng[0]))
        m=pair['smooth_residual'].eval();nan=torch.full_like(x,float('nan'))
        with torch.no_grad():
            p=m(x);torch.testing.assert_close(p,m(x,nan,nan,nan),rtol=0,atol=0)
            self.assertEqual(m.predict_components(x).shape[-1],12)

    def test_optimizer_gradients_update_and_pure_mse(self):
        m=initialize_pair(self.cfg)['smooth_residual'];groups=parameter_groups(m,'smooth_residual')
        params=[p for g in groups for p in g['params']]
        self.assertEqual(len(params),len(set(map(id,params))))
        self.assertEqual(set(map(id,params)),set(map(id,m.parameters())))
        rates={id(p):g['weight_decay'] for g in groups for p in g['params']}
        x=torch.randn(2,96,2);y=torch.randn_like(x);p=m(x)
        loss=training_losses(p,y,'smooth_residual',{'offsets':[0,1,2,4,8,12,24,48,96],'variance_floor':.01,'jitter':.01},x)
        torch.testing.assert_close(loss['loss'],(p-y).square().mean(),rtol=0,atol=0)
        self.assertEqual(loss['auxiliary_kl'].item(),0);loss['loss'].backward()
        for name,v in m.named_parameters():
            if 'pre_linears' in name:
                self.assertEqual(rates[id(v)],0 if name.endswith('bias') else 1e-4)
                self.assertTrue(torch.isfinite(v.grad).all());self.assertGreater(v.grad.abs().sum().item(),0)
        torch.optim.Adam(groups,lr=.001).step()
        for b in m.pdm_blocks:
            for l in b.mixing_multi_scale_trend.pre_linears:self.assertFalse(torch.equal(l.weight,torch.eye(l.in_features)))
        with self.assertRaises(ValueError):dataset(self.cfg,'test',0)

    def test_capture_actual_filtered_trend_and_cleanup(self):
        m=initialize_pair(self.cfg)['smooth_residual'].eval();x=torch.randn(3,96,2)
        with torch.no_grad():
            for b in m.pdm_blocks:
                for l in b.mixing_multi_scale_trend.pre_linears:l.bias.copy_((torch.arange(l.in_features)%2).float())
        seen=[];smoother=m.pdm_blocks[0].mixing_multi_scale_trend.correction_filters[0]
        h=smoother.register_forward_hook(lambda mod,args,out:seen.append((args[0].detach().double(),out.detach().double())))
        with tempfile.TemporaryDirectory() as folder:
            with SelectedValidationCapture(m,Path(folder),'smooth_residual',3) as cap:
                with torch.no_grad():p=m(x).numpy()
            h.remove();report=cap.report(p,np.zeros_like(p))
            np.testing.assert_allclose(cap.components.sum(-1)+cap.means,p,rtol=0,atol=2e-5)
        self.assertEqual(len(seen),1);self.assertEqual(len(report['smooth_residual_transforms']),8)
        stats=report['smooth_residual_transforms']['layer0_length96']
        self.assertAlmostEqual(stats['output_rms'],seen[0][1].square().mean().sqrt().item(),places=12)
        self.assertLess(stats['filtered_correction_second_difference_rms'],stats['raw_correction_second_difference_rms'])
        self.assertIsNone(cap.model);self.assertFalse(any(mod._forward_hooks for mod in m.modules()))

    def test_correction_preserves_identity_and_does_not_smooth_input(self):
        x=torch.randn(2,3,96);f=SmoothTrendCorrection(25)
        torch.testing.assert_close(f(x,x),x,rtol=0,atol=0)
        torch.testing.assert_close(f(x,x+2),x+2,rtol=0,atol=1e-6)


if __name__=='__main__':unittest.main()
