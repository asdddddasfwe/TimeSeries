import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from test_oscillator_mixer import config
from scripts import train_timemix_pdm_trend_decay as train
from scripts.pdm_trend_decay_study import targeted,parameter_groups,gradient_audit,TrendDecayCapture
from scripts.run_timemix_pdm_trend_decay_experiment import decision

class TrendDecayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);train.seed_all(2021)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,dropout=.2)
        self.cfg.e_layers=2;self.cfg.seasonal_normalize=False
    def test_initial_models_predictions_rng_identical(self):
        models=train.initialize_pair(self.cfg);a,b=models.values()
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        x=torch.randn(2,96,2)
        for mode in [False,True]:
            a.train(mode);b.train(mode);torch.manual_seed(31);p=a(x);rng=torch.get_rng_state()
            torch.manual_seed(31);q=b(x)
            torch.testing.assert_close(p,q,rtol=0,atol=0);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_only_twelve_declared_weights_change_decay(self):
        m=train.initialize_pair(self.cfg)['baseline_decay']
        maps=[]
        for arm in train.ARMS:
            groups=parameter_groups(m,arm);ids=[id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.parameters()})
            by={id(p):g['weight_decay'] for g in groups for p in g['params']}
            maps.append({n:by[id(p)] for n,p in m.named_parameters()})
        changed=[n for n in maps[0] if maps[0][n]!=maps[1][n]]
        self.assertEqual(len(changed),12);self.assertTrue(all(targeted(n) for n in changed))
        self.assertTrue(all(maps[0][n]==1e-4 and maps[1][n]==0 for n in changed))
    def test_raw_gradients_equal_before_optimizer_and_untargeted_first_update_equal(self):
        models=train.initialize_pair(self.cfg);x=torch.randn(2,96,2);y=torch.randn_like(x)
        opts={}
        grads={}
        for arm,m in models.items():
            groups=parameter_groups(m,arm);opts[arm]=train.build_optimizer(groups,arm)
            train.seed_all(42);((m(x)-y)**2).mean().backward()
            grads[arm]={n:p.grad.clone() for n,p in m.named_parameters()}
            audit=gradient_audit(m,{id(p):g['weight_decay'] for g in groups for p in g['params']})
            self.assertEqual(len(audit),12)
        a,b=models.values()
        for n in grads['baseline_decay']:torch.testing.assert_close(grads['baseline_decay'][n],grads['trend_no_decay'][n],rtol=0,atol=0)
        for opt in opts.values():opt.step()
        diffs=[]
        for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if not targeted(n):torch.testing.assert_close(p,q,rtol=0,atol=0)
            elif not torch.equal(p,q):diffs.append(n)
        self.assertTrue(diffs)
    def test_capture_preserves_forecast_rng_and_releases_handles(self):
        m=train.initialize_pair(self.cfg)['baseline_decay'].eval();x=torch.randn(2,96,2)
        with torch.no_grad():expected=m(x)
        state=torch.get_rng_state()
        with tempfile.TemporaryDirectory(dir=Path(train.ROOT)/'tmp') as tmp:
            with TrendDecayCapture(m,tmp,'check',4) as cap:
                with torch.no_grad():
                    for _ in range(2):actual=m(x)
                self.assertEqual(cap.cache,[])
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            report=cap.report(np.tile(actual.numpy(),(2,1,1)),np.zeros((4,96,2)))
            self.assertEqual(len(report['trend_path_directional_gain']),8)
            self.assertLess(report['reconstruction_max_abs_error'],2e-5)
            self.assertIsNone(cap.model)
            self.assertTrue(all(not z._forward_hooks and not z._forward_pre_hooks for z in m.modules()))
    def test_confirmation_requires_all_preregistered_metrics(self):
        base={'variant':'baseline_decay','validation_mse':1.,'mechanism_diagnostics':{'trend_path_directional_gain':{'layer0_scale2':1.,'layer1_scale2':1.},'per_channel_waveform_mse':[1.]*7,'coarse_mse_excess_per_channel':[1.]*7}}
        new=copy.deepcopy(base);new.update(variant='trend_no_decay',validation_mse=.9)
        self.assertFalse(decision([base,new])['passed'])
        new['mechanism_diagnostics']['trend_path_directional_gain']={k:2. for k in base['mechanism_diagnostics']['trend_path_directional_gain']}
        new['mechanism_diagnostics']['per_channel_waveform_mse']=[.9]*7
        new['mechanism_diagnostics']['coarse_mse_excess_per_channel']=[.9]*7
        self.assertTrue(decision([base,new])['passed'])
if __name__=='__main__':unittest.main()
