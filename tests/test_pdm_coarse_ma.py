import copy
import gc
import weakref
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from torch.utils.data import TensorDataset
from test_oscillator_mixer import config
from models.OscillatorFeaturePDMTrendBypass import Model as Previous
from models.OscillatorPDMCoarseMA import Model, ScaleSelectiveDecomposition
from layers.Autoformer_EncDec import series_decomp
from scripts import train_timemix_pdm_coarse_ma as train
from scripts.train_timemix_pdm_trend_bypass import initialize_pair as previous_pair
from utils.pdm_coarse_ma_diagnostics import SelectedValidationCapture, forecast_module


class CoarseMATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2021)
        self.cfg=config(seq_len=96,pred_len=96,moving_avg=25,dropout=.2)
        self.cfg.e_layers=2;self.cfg.seasonal_normalize=False

    def test_shared_initialization_and_original_baseline_exact(self):
        torch.manual_seed(2021);ref=previous_pair(self.cfg)['pdm_trend_bypass']
        torch.manual_seed(2021);models=train.initialize_pair(self.cfg)
        for model in models.values():
            self.assertEqual(set(ref.state_dict()),set(model.state_dict()))
            self.assertEqual(sum(p.numel() for p in ref.parameters()),sum(p.numel() for p in model.parameters()))
            for key,value in ref.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        x=torch.randn(2,96,2)
        for mode in [False,True]:
            ref.train(mode);torch.manual_seed(31);expected=ref(x);state=torch.get_rng_state()
            model=models['ma25'].train(mode);torch.manual_seed(31)
            torch.testing.assert_close(expected,model(x),rtol=0,atol=0)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_only_coarse_decomposition_changes_and_identity_holds(self):
        original=series_decomp(25);candidate=ScaleSelectiveDecomposition(original,3)
        for length in [96,48,24,12]:
            x=torch.randn(3,length,7,dtype=torch.float64)
            s,t=candidate(x);torch.testing.assert_close(s+t,x,atol=1e-14,rtol=0)
            expected=series_decomp(3 if length==12 else 25)(x)
            for a,b in zip((s,t),expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        x=torch.eye(12,dtype=torch.float64).unsqueeze(-1)
        old=original(x)[1].squeeze(-1).T;new=candidate(x)[1].squeeze(-1).T
        self.assertEqual(int(torch.linalg.matrix_rank(old,atol=1e-12,rtol=0)),2)
        self.assertGreater(int(torch.linalg.matrix_rank(new,atol=1e-12,rtol=0)),2)
        self.assertGreater(float(new.diff(n=2,dim=0).abs().max()),.1)

    def test_finite_gradients_and_unchanged_trend_path(self):
        models=train.initialize_pair(self.cfg);x=torch.randn(2,96,2)
        for model in models.values():
            loss=(model(x)-torch.randn_like(x)).square().mean();loss.backward()
            self.assertEqual([n for n,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()],[])
        for model in models.values():model.eval()
        torch.testing.assert_close(models['ma25'].predict_raw_trend(x),models['coarse_ma3'].predict_raw_trend(x),atol=0,rtol=0)

    def test_serialization_and_no_future_inputs(self):
        m=Model(self.cfg,variant='coarse_ma3').eval();x=torch.randn(2,96,2)
        cfg=copy.copy(self.cfg);cfg.coarse_ma_variant='coarse_ma3';other=Model(cfg).eval();other.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(m(x),other(x),rtol=0,atol=0)
        torch.testing.assert_close(m(x),m(x,None,torch.randn_like(x),None),rtol=0,atol=0)

    def test_diagnostic_capture_preserves_prediction_rng_and_reconstructs(self):
        models=train.initialize_pair(self.cfg);x=torch.randn(2,96,2)
        with tempfile.TemporaryDirectory(dir=Path(train.ROOT)/'tmp') as tmp:
            for name,model in models.items():
                model.eval();torch.manual_seed(31)
                with torch.no_grad():expected=model(x)
                state=torch.get_rng_state();torch.manual_seed(31)
                with SelectedValidationCapture(model,Path(tmp),name,8) as capture:
                    for _ in range(4):
                        with torch.no_grad():actual=model(x)
                        self.assertEqual(capture.cache,[])
                        self.assertEqual(capture.heads,{})
                torch.testing.assert_close(actual,expected,rtol=0,atol=0);self.assertTrue(torch.equal(state,torch.get_rng_state()))
                saved=np.tile(actual.numpy(),(4,1,1))
                audit=capture.report(saved,np.zeros_like(saved))
                self.assertLess(audit['reconstruction_max_abs_error'],2e-5)
                if name=='ma25':self.assertLess(max(v['second_difference_rms'] for v in audit['coarse_decomposition'].values()),1e-5)
                else:self.assertGreater(min(v['second_difference_rms'] for v in audit['coarse_decomposition'].values()),1e-3)

    def test_capture_does_not_retain_forecast_arguments_across_batches(self):
        # Exercise the actual wrapper without running a forecasting model or GPU.
        model=torch.nn.Module();model.enc_in=2;model.pdm_blocks=[];model.raw_trend_heads=[]
        references=[]
        def forecast(*args, **kwargs):
            return torch.zeros(1,96,2)
        with tempfile.TemporaryDirectory(dir=Path(train.ROOT)/'tmp') as tmp:
            with patch.object(forecast_module,'calibrated_seasonal_forecast',new=forecast):
                with SelectedValidationCapture(model,tmp,'retention',0) as capture:
                    for _ in range(400):
                        capture.before(model,())
                        history=torch.randn(1,12,2)
                        references.append(weakref.ref(history))
                        forecast_module.calibrated_seasonal_forecast(encoded_history=history)
                        del history
                    self.assertTrue(all(ref() is None for ref in references))
                self.assertIs(forecast_module.calibrated_seasonal_forecast,forecast)
                self.assertEqual(capture.cache,[]);self.assertEqual(capture.heads,{})
                self.assertIsNone(capture.model);self.assertIsNone(capture.patch)
        ref=weakref.ref(model);del model;gc.collect()
        self.assertIsNone(ref())

    def test_capture_cleans_up_when_forward_fails(self):
        model=Model(self.cfg,variant='coarse_ma3').eval()
        original=forecast_module.calibrated_seasonal_forecast
        with tempfile.TemporaryDirectory(dir=Path(train.ROOT)/'tmp') as tmp:
            with self.assertRaisesRegex(RuntimeError,'simulated failure'):
                with SelectedValidationCapture(model,tmp,'failure',2) as capture:
                    capture.cache.append(torch.ones(1))
                    capture.heads[0]=torch.ones(1)
                    raise RuntimeError('simulated failure')
            self.assertIs(forecast_module.calibrated_seasonal_forecast,original)
            self.assertEqual(capture.cache,[]);self.assertEqual(capture.heads,{})
            self.assertIsNone(capture.model);self.assertIsNone(capture.patch)
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_memory_fraction_rejects_invalid_values_before_creating_output(self):
        for value in ['0','-0.1','1.1','nan','inf']:
            with self.subTest(value=value),patch('sys.argv',['train','--gpu-memory-fraction',value]):
                with self.assertRaises(SystemExit):train.main()

    def test_pair_driver_joint_stop(self):
        cfg=copy.copy(self.cfg);cfg.enc_in=cfg.c_out=2
        data=TensorDataset(torch.randn(4,96,2),torch.randn(4,96,2),torch.zeros(4,96,1),torch.zeros(4,96,1))
        framework=train.framework_args
        def settings(args):
            actual=framework(args);vars(actual).update(vars(cfg));return actual
        vals=iter([.1]*2+[.2]*2+[.3]*2)
        def validation(*args):
            v=next(vals);return dict(neural_mse=v,neural_mae=v)
        def selected(*args,**kwargs):
            p=np.full((4,96,2),np.sqrt(.1));return dict(mse=.1),p,np.zeros_like(p)
        dummy=MagicMock();dummy.__enter__.return_value=dummy;dummy.report.return_value={}
        with tempfile.TemporaryDirectory(prefix='coarse_ma_stop_',dir=Path(train.ROOT)/'tmp') as tmp:
            out=Path(tmp)/'training';argv=['train','--output',str(out),'--device','cpu','--epochs','8','--patience','2','--train-limit','4','--val-limit','4']
            with patch('sys.argv',argv),patch.object(train,'framework_args',side_effect=settings),patch.object(train,'dataset',return_value=data),patch.object(train,'evaluate_validation',side_effect=validation),patch.object(train,'evaluate',side_effect=selected),patch.object(train,'SelectedValidationCapture',return_value=dummy):train.main()
            p=json.loads((out/'2021/progress.json').read_text());report=json.loads((out/'report.json').read_text())
            self.assertEqual(p['epoch'],3);self.assertEqual(p['stale'],2)
            protocol=json.loads((out/'protocol.json').read_text())
            self.assertEqual(protocol['options']['gpu_memory_fraction'],1.0)
            self.assertEqual(p['best_epoch'],{name:1 for name in train.ARMS})
            self.assertEqual(len(report['runs']),2)
            self.assertTrue(all(v['stop_reason']=='joint_patience' for v in report['runs']))
            ck=torch.load(out/'2021/last.pt',weights_only=True);self.assertEqual(ck['completed_updates'],3)
