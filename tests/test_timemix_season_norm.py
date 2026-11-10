import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from test_oscillator_mixer import config
from models.OscillatorLowRankTimeMix import Model as Original
from models.OscillatorSeasonNormTimeMix import Model, normalize_season_history
from scripts.train_timemix_season_norm import (ARMS, NORMALIZED, initialize_pair, training_losses, loss_configuration, selection_score, dataset)
from scripts.run_timemix_season_norm_experiment import check_test_result
from utils.joint_gaussian_autocov_loss import joint_gaussian_autocov_kl_loss
from utils.season_norm_study_metrics import shape_metrics, factorial_contrasts

class SeasonNormStudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2021)
        self.cfg=config(seq_len=96,pred_len=96,dropout=.2)
        self.loss_cfg=loss_configuration(SimpleNamespace(variance_floor=.01,kl_jitter=.01))

    def test_history_transform_inverse_and_moments(self):
        x=(torch.randn(3,96,2,dtype=torch.double)*2+3).requires_grad_()
        u,m,s=normalize_season_history(x)
        torch.testing.assert_close(u*s+m,x,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(u.mean(1),torch.zeros(3,2,dtype=torch.double),atol=1e-12,rtol=0)
        torch.testing.assert_close(u.var(1,unbiased=False),torch.ones(3,2,dtype=torch.double))
        self.assertFalse(m.requires_grad);self.assertFalse(s.requires_grad)

    def test_constant_history_is_finite_and_reversible(self):
        x=torch.full((2,96,2),5.)
        u,m,s=normalize_season_history(x)
        self.assertTrue(torch.equal(u,torch.zeros_like(u)))
        torch.testing.assert_close(s,torch.full_like(s,.1))
        torch.testing.assert_close(u*s+m,x)

    def test_no_batch_cross_talk(self):
        x=torch.randn(2,96,2);a=normalize_season_history(x)
        x[1]*=100;b=normalize_season_history(x)
        for av,bv in zip(a,b):torch.testing.assert_close(av[0],bv[0],atol=0,rtol=0)

    def test_affine_transform_does_not_whiten_acf(self):
        x=torch.randn(2,96,2,dtype=torch.double)*3+4;u,_,_=normalize_season_history(x)
        def acf(a):
            a=a-a.mean(1,keepdim=True)
            return (a[:,4:]*a[:,:-4]).sum(1)/a.square().sum(1)
        torch.testing.assert_close(acf(x),acf(u),rtol=1e-12,atol=1e-12)

    def test_invalid_floor(self):
        for f in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):normalize_season_history(torch.randn(2,96,2),f)

    def test_disabled_exact_original_predictions_and_rng(self):
        original=Original(self.cfg);model=Model(self.cfg,enabled=False)
        model.load_state_dict(original.state_dict(),strict=True);x=torch.randn(2,96,2)
        for mode in [False,True]:
            original.train(mode);model.train(mode);torch.manual_seed(9);a=original(x);rng=torch.get_rng_state();torch.manual_seed(9);b=model(x)
            torch.testing.assert_close(a,b,atol=0,rtol=0);self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_same_parameters_buffers_and_within_pair_predictions(self):
        models=initialize_pair(self.cfg);state=models['control'].state_dict();x=torch.randn(2,96,2)
        for name,m in models.items():
            self.assertEqual(m.seasonal_normalize,NORMALIZED[name]);self.assertEqual(state.keys(),m.state_dict().keys())
            for k,v in state.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
        for mode in [False,True]:
            outputs=[];rng=[]
            for m in models.values():
                m.train(mode);torch.manual_seed(9);outputs.append(m(x));rng.append(torch.get_rng_state())
            for a,b in [(0,1),(2,3)]:
                torch.testing.assert_close(outputs[a],outputs[b],atol=0,rtol=0);self.assertTrue(torch.equal(rng[a],rng[b]))
            self.assertGreater((outputs[0]-outputs[2]).abs().max().item(),0)

    def test_season_restore_mean_once_trend_not_rescaled(self):
        m=Model(self.cfg,enabled=True).eval();season=torch.randn(2,96,2)*2+3;trend=torch.randn_like(season)
        _,mu,sigma=normalize_season_history(season)
        trends=m.prepare_trend_scales(trend)
        trend_prediction=sum(h(t.transpose(1,2)).transpose(1,2) for h,t in zip(m.trend_heads,trends))
        with patch('models.OscillatorSeasonNormTimeMix.calibrated_seasonal_forecast',side_effect=lambda *a,**k:torch.ones(2,96,2)):
            p=m.predict_scales(season,trend).sum(-1)
        torch.testing.assert_close(p,4*sigma+mu+trend_prediction)

    def test_normalized_branches_receive_finite_gradients(self):
        m=Model(self.cfg,enabled=True);x=torch.randn(2,96,2);y=torch.randn_like(x)
        loss=training_losses(m(x),y,'season_norm_kl',self.loss_cfg,x)['loss'];loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_both_mse_arms_have_exact_mse_gradient(self):
        for arm in ['control','season_norm_mse']:
            p=torch.randn(2,96,2,requires_grad=True);y=torch.randn_like(p,requires_grad=True);h=torch.randn_like(p,requires_grad=True)
            z=training_losses(p,y,arm,self.loss_cfg,h);z['loss'].backward()
            torch.testing.assert_close(p.grad,2*(p.detach()-y.detach())/p.numel());self.assertIsNone(y.grad);self.assertIsNone(h.grad);self.assertEqual(z['auxiliary_kl'],0)

    def test_both_kl_arms_use_same_original_unit_loss_no_weight(self):
        for arm in ['original_kl','season_norm_kl']:
            p=torch.randn(2,96,2,requires_grad=True);y=torch.randn_like(p,requires_grad=True);h=torch.randn_like(p,requires_grad=True)
            z=training_losses(p,y,arm,self.loss_cfg,h);expected=(p-y.detach()).square().mean()+joint_gaussian_autocov_kl_loss(h,p,y)
            torch.testing.assert_close(z['loss'],expected);torch.testing.assert_close(z['auxiliary_kl'],z['kl'],atol=0,rtol=0)
            z['loss'].backward();self.assertIsNone(h.grad);self.assertIsNone(y.grad)

    def test_selection_and_no_test_dataset(self):
        self.assertEqual(selection_score(dict(neural_mse=2,neural_mae=1,kl=0)),2)
        with self.assertRaises(ValueError):dataset(None,'test',0)

    def test_perfect_predictions_have_zero_acf_error(self):
        x=np.random.default_rng(4).normal(size=(3,96,2));m=shape_metrics(x,x)
        for v in m['future_acf'].values():self.assertEqual(v['mse'],0)
        self.assertEqual(m['phase']['robust_shift_fraction'],0)

    def test_constant_prediction_not_dropped_from_acf_error(self):
        x=np.cumsum(np.random.default_rng(4).normal(size=(3,96,2)),1);m=shape_metrics(np.zeros_like(x),x)
        self.assertEqual(m['valid_acf_pairs'],6);self.assertGreater(m['future_acf']['1']['mse'],0)

    def test_factorial_interaction_sign(self):
        rows=[]
        for name,x in zip(ARMS,[1,1.3,.9,1.0]):rows.append(dict(variant=name,selected_validation={k:x for k in ['mse','mae','low4_mse','window_offset_mse']}))
        z=factorial_contrasts(rows,'val')['mse'];self.assertAlmostEqual(z['interaction'],-.2);self.assertAlmostEqual(z['normalization_effect_mse'],-.1)

    def test_freeze_receipt_rejects_wrong_config(self):
        result=dict(status='completed',split='test',smoke=False,windows=11425,official_windows=11425,checkpoint_sha256='a',model_config_sha256='b',source_hashes_verified=True)
        check_test_result(result,dict(checkpoint_sha256='a'),dict(checkpoint_sha256='a',model_config_sha256='b'))
        with self.assertRaises(RuntimeError):check_test_result(result,dict(checkpoint_sha256='a'),dict(checkpoint_sha256='a',model_config_sha256='wrong'))

if __name__=='__main__':unittest.main()
