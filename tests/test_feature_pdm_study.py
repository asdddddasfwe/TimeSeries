import copy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from test_oscillator_mixer import config
from models.OscillatorLowRankTimeMix import Model as Original
from models.OscillatorFeaturePDMTimeMix import Model, encode_one_fft_layer
from layers.LagARFormer import select_fft_periods
from scripts.train_timemix_feature_pdm import initialize_pair, training_losses, loss_configuration, selection_score, dataset
from models.TimeMixer import PastDecomposableMixing


class FeaturePDMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2021)
        self.cfg=config(seq_len=96,pred_len=96,dropout=.2)
        self.cfg.e_layers=2;self.cfg.seasonal_normalize=False

    def test_control_is_exact_original(self):
        a=Original(self.cfg);b=Model(self.cfg,enabled=False);b.load_state_dict(a.state_dict(),strict=True)
        x=torch.randn(2,96,2)
        for mode in (False,True):
            a.train(mode);b.train(mode);torch.manual_seed(73);expected=a(x);rng=torch.get_rng_state();torch.manual_seed(73);actual=b(x)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0);self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_common_initial_parameters_equal(self):
        m=initialize_pair(self.cfg);a=m['control'].state_dict();b=m['feature_pdm'].state_dict()
        for name in a.keys()&b.keys():torch.testing.assert_close(a[name],b[name],atol=0,rtol=0)
        self.assertTrue(any(k.startswith('pdm_blocks.') for k in b.keys()-a.keys()))
        self.assertFalse(hasattr(m['feature_pdm'],'trend_heads'))
        self.assertFalse(hasattr(m['feature_pdm'],'season_mix'))
        self.assertTrue(all(not hasattr(e,'trend_embedding') for e in m['feature_pdm'].seasonals))

    def test_one_fft_layer_matches_existing_encoder(self):
        m=Model(self.cfg,enabled=True);e=m.seasonals[0]
        x=torch.randn(3,96);route=select_fft_periods(x,e.period_top_k)
        tokens=e.value_embedding(x[...,None])+e.positions
        for i in range(2):
            ref=copy.deepcopy(e);ref.blocks=nn.ModuleList([copy.deepcopy(e.blocks[i])])
            torch.manual_seed(77);expected=ref.encode_periods(tokens,*route);rng=torch.get_rng_state()
            torch.manual_seed(77);actual=encode_one_fft_layer(e,tokens,*route,i)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0);self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_pdm_reuses_timemixer_equations(self):
        m=Model(self.cfg,enabled=True).eval();block=m.pdm_blocks[0]
        self.assertIsInstance(block,PastDecomposableMixing)
        z=[torch.randn(3,t,self.cfg.d_model) for t in [96,48,24,12]]
        seasons=[];trends=[]
        for x in z:
            s,t=block.decompsition(x);torch.testing.assert_close(s+t,x);seasons.append(s.transpose(1,2));trends.append(t.transpose(1,2))
        s=block.mixing_multi_scale_season(seasons);t=block.mixing_multi_scale_trend(trends)
        expected=[x+block.out_cross_layer(a+b) for x,a,b in zip(z,s,t)]
        for a,b in zip(block(z),expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertFalse(hasattr(block,'layer_norm'))

    def test_decomposition_runs_on_features_each_layer_each_scale(self):
        m=Model(self.cfg,enabled=True).eval();shapes=[];handles=[]
        for block in m.pdm_blocks:
            handles.append(block.decompsition.register_forward_pre_hook(lambda mod,args:shapes.append(tuple(args[0].shape))))
        y=m(torch.randn(2,96,2))
        for h in handles:h.remove()
        self.assertEqual(shapes,[(4,t,self.cfg.d_model) for t in [96,48,24,12]]*2)
        self.assertEqual(tuple(y.shape),(2,96,2));self.assertTrue(torch.isfinite(y).all())

    def test_one_input_normalization_per_scale_and_one_inverse(self):
        m=Model(self.cfg,enabled=True).eval();calls=[];handles=[]
        for i,norm in enumerate([m.normalize,*m.coarse_normalizers]):
            handles.append(norm.register_forward_pre_hook(lambda mod,args,i=i:calls.append((i,args[1],args[0].shape[1]))))
        m(torch.randn(2,96,2))
        for h in handles:h.remove()
        self.assertEqual(calls,[(0,'norm',96),(1,'norm',48),(2,'norm',24),(3,'norm',12),(0,'denorm',96)])
        self.assertFalse(hasattr(m,'seasonal_normalize'));self.assertFalse(hasattr(m,'decomposition'))

    def test_all_parameters_receive_finite_gradients(self):
        m=Model(self.cfg,enabled=True);x=torch.randn(2,96,2);y=torch.randn_like(x)
        (m(x)-y).square().mean().backward()
        missing=[n for n,p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        self.assertEqual(missing,[])

    def test_constant_and_mixed_scale_inputs_finite(self):
        m=Model(self.cfg,enabled=True).eval()
        for x in (torch.ones(2,96,2)*4,torch.randn(2,96,2)*torch.tensor([.001,100.])):
            self.assertTrue(torch.isfinite(m(x)).all())

    def test_no_future_dependency_and_batch_crosstalk(self):
        m=Model(self.cfg,enabled=True).eval();x=torch.randn(2,96,2)
        a=m(x);other=x.clone();other[1]*=100;b=m(other)
        torch.testing.assert_close(a[0],b[0],rtol=1e-5,atol=1e-5)
        c=m(x,None,torch.randn(2,96,2),None)
        torch.testing.assert_close(a,c,rtol=0,atol=0)

    def test_pure_mse_both_arms_and_selection(self):
        cfg=loss_configuration(SimpleNamespace(variance_floor=.01,kl_jitter=.01))
        for arm in ('control','feature_pdm'):
            x=torch.randn(2,96,2,requires_grad=True);y=torch.randn_like(x,requires_grad=True);h=torch.randn_like(x,requires_grad=True)
            parts=training_losses(x,y,arm,cfg,h);parts['loss'].backward()
            torch.testing.assert_close(x.grad,2*(x.detach()-y.detach())/x.numel());self.assertIsNone(y.grad);self.assertIsNone(h.grad)
            self.assertEqual(float(parts['auxiliary_kl']),0);torch.testing.assert_close(parts['loss'],parts['mse'])
        self.assertEqual(selection_score(dict(neural_mse=.2,neural_mae=.1,kl=10)),.2)
        with self.assertRaises(ValueError):dataset(None,'test',0)

    def test_reject_seasonal_second_normalization(self):
        cfg=copy.copy(self.cfg);cfg.seasonal_normalize=True
        with self.assertRaises(ValueError):Model(cfg,enabled=True)

    def test_strict_restore(self):
        m=Model(self.cfg,enabled=True).eval();other=Model(self.cfg,enabled=True).eval();other.load_state_dict(m.state_dict(),strict=True)
        x=torch.randn(2,96,2);torch.testing.assert_close(m(x),other(x),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
