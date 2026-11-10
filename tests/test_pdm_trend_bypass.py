"""Regression contracts for the raw-trend bypass and paired early stopping."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import TensorDataset
from test_oscillator_mixer import config
from models.OscillatorFeaturePDMTimeMix import Model as PreviousPDM
from models.OscillatorLowRankTimeMix import Model as Original
from models.OscillatorFeaturePDMTrendBypass import Model
from scripts import train_timemix_pdm_trend_bypass as train
from scripts.train_timemix_feature_pdm import initialize_pair as previous_pair


class TrendBypassTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2021)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,dropout=.2)
        self.cfg.e_layers=2
        self.cfg.seasonal_normalize=False

    def test_baseline_exact_previous_pdm_forward_and_rng(self):
        old=PreviousPDM(self.cfg,enabled=True)
        new=Model(self.cfg,enabled=False)
        new.load_state_dict(old.state_dict(),strict=True)
        x=torch.randn(2,96,2)
        for mode in [False,True]:
            old.train(mode);new.train(mode)
            torch.manual_seed(17);a=old(x);rng=torch.get_rng_state()
            torch.manual_seed(17);b=new(x)
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_initialization_matches_previous_and_common_weights(self):
        torch.manual_seed(2021);ref=previous_pair(self.cfg)['feature_pdm']
        torch.manual_seed(2021);pair=train.initialize_pair(self.cfg)
        for name,tensor in ref.state_dict().items():
            for m in pair.values():torch.testing.assert_close(tensor,m.state_dict()[name],rtol=0,atol=0)
        extra=set(pair['pdm_trend_bypass'].state_dict())-set(pair['pdm'].state_dict())
        self.assertTrue(extra)
        self.assertTrue(all(n.startswith(('raw_trend_mix.','raw_trend_heads.')) for n in extra))

    def test_bypass_equals_original_trend_equations(self):
        old=Original(self.cfg).eval();new=Model(self.cfg,enabled=True).eval()
        new.raw_trend_mix.load_state_dict(old.trend_mix.state_dict())
        new.raw_trend_heads.load_state_dict(old.trend_heads.state_dict())
        x=torch.randn(2,96,2);normalized=old.normalize(x,'norm');_,trend=old.decomposition(normalized)
        expected=torch.stack([head(t.transpose(1,2)).transpose(1,2) for head,t in zip(old.trend_heads,old.prepare_trend_scales(trend))],-1).sum(-1)
        torch.testing.assert_close(new.predict_raw_trend(normalized),expected,atol=0,rtol=0)

    def test_zero_heads_recovers_pdm_forward_rng_and_single_inverse(self):
        pair=train.initialize_pair(self.cfg);base=pair['pdm'];new=pair['pdm_trend_bypass']
        for h in new.raw_trend_heads:torch.nn.init.zeros_(h.weight);torch.nn.init.zeros_(h.bias)
        calls=[];handles=[norm.register_forward_pre_hook(lambda m,a:calls.append(a[1])) for norm in [new.normalize,*new.coarse_normalizers]]
        x=torch.randn(2,96,2)
        for mode in [False,True]:
            base.train(mode);new.train(mode);torch.manual_seed(31);expected=base(x);rng=torch.get_rng_state();calls.clear();torch.manual_seed(31);actual=new(x)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(calls,['norm']*4+['denorm'])
        for h in handles:h.remove()

    def test_all_parameters_finite_gradients_and_no_future_dependency(self):
        m=Model(self.cfg,enabled=True);x=torch.randn(2,96,2);y=torch.randn_like(x)
        (m(x)-y).square().mean().backward()
        self.assertEqual([n for n,p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()],[])
        m.eval();a=m(x);b=m(x,None,torch.randn_like(x),None);torch.testing.assert_close(a,b,rtol=0,atol=0)
        other=x.clone();other[1]*=10;torch.testing.assert_close(a[0],m(other)[0],rtol=1e-5,atol=1e-5)
        restored=Model(self.cfg,enabled=True).eval();restored.load_state_dict(m.state_dict(),strict=True);torch.testing.assert_close(a,restored(x),rtol=0,atol=0)

    def test_joint_stale_reset_and_stop_boundary(self):
        stale=0
        for improved in [True,False,False,True,False,False,False,False]:stale=train.joint_stale_count(stale,improved)
        self.assertEqual(stale,4)
        self.assertEqual(train.joint_stale_count(stale,False),5)
        self.assertEqual(train.joint_stale_count(stale,True),0)

    def test_driver_stops_and_preserves_best_before_max_epochs(self):
        cfg=copy.copy(self.cfg);cfg.enc_in=cfg.c_out=2
        data=TensorDataset(torch.randn(4,96,2),torch.randn(4,96,2),torch.zeros(4,96,1),torch.zeros(4,96,1))
        framework=train.framework_args
        def framework_config(args):
            actual=framework(args)
            vars(actual).update(vars(cfg))
            return actual
        vals=iter([.1,.1,.2,.2,.3,.3])
        def validation(*args):
            v=next(vals);return dict(neural_mse=v,neural_mae=v)
        def selected(*args,**kwargs):
            p=np.full((4,96,2),np.sqrt(.1));y=np.zeros_like(p)
            return dict(mse=.1),p,y
        with tempfile.TemporaryDirectory(prefix='pdm_earlystop_',dir=Path(train.ROOT)/'tmp') as tmp:
            out=Path(tmp)/'training'
            argv=['train','--output',str(out),'--device','cpu','--epochs','8','--patience','2','--train-limit','4','--val-limit','4']
            with patch('sys.argv',argv),patch.object(train,'framework_args',side_effect=framework_config),patch.object(train,'dataset',return_value=data),patch.object(train,'evaluate_validation',side_effect=validation),patch.object(train,'evaluate',side_effect=selected):train.main()
            p=json.loads((out/'2021/progress.json').read_text());report=json.loads((out/'report.json').read_text())
            self.assertEqual(p['epoch'],3);self.assertEqual(p['stale'],2)
            self.assertEqual(p['best_epoch'],{'pdm':1,'pdm_trend_bypass':1})
            self.assertTrue(all(v['stop_reason']=='joint_patience' and v['trained_epochs']==3 for v in report['runs']))
            ck=torch.load(out/'2021/last.pt',weights_only=True)
            self.assertEqual(ck['completed_updates'],3)


if __name__=='__main__':unittest.main()
