"""完整未来递推、低秩初始化、冻结适配器及测试复用身份核验。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from test_oscillator_mixer import config
from models.OscillatorTimeMix import Model as Dense
from models.OscillatorLowRankTimeMix import Model as Rank
from models.OscillatorLowRankLocalTimeMix import Model as Local
from models.OscillatorLowRankLinearBlend import Model as Adapter
from layers.LinearForecastBlend import LinearForecastBlend
from layers.OscillatorTimeMix import complex_oscillator_scan
from scripts.run_timemix_recurrent_rank16_experiment import try_reuse_control_test,sha
from scripts.train_timemix_recurrent_rank16 import parameter_groups


class RecurrentRankTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2021)
        self.cfg=config(seq_len=96,pred_len=96,dropout=.2,is_training=0)

    def test_full_future_scan_matches_sequential_recurrence(self):
        model=Rank(self.cfg).eval()
        records=[]
        def capture(a,b,z):
            result=complex_oscillator_scan(a,b,z)
            records.append((a.detach(),b.detach(),z.detach(),result.detach()))
            return result
        with patch('models.OscillatorTimeMix.complex_oscillator_scan',side_effect=capture):
            with torch.no_grad():model(torch.randn(2,96,2))
        self.assertEqual(len(records),4)
        a,b,z,result=records[0]
        self.assertEqual(b.shape[1],96)
        ac=torch.view_as_complex(a.contiguous());bc=torch.view_as_complex(b.contiguous())
        state=torch.view_as_complex(z.contiguous());expected=[]
        for h in range(96):
            state=ac[:,h]*state+bc[:,h];expected.append(state)
        torch.testing.assert_close(result,torch.view_as_real(torch.stack(expected,1)),atol=1e-5,rtol=3e-5)
        local=complex_oscillator_scan(a,torch.zeros_like(b),z)+b
        self.assertGreater(float((local-result).abs().max()),1e-4)

    def test_svd_shared_initialization_and_equivalent_dense_map(self):
        dense=Dense(self.cfg).eval();rank=Rank(self.cfg).eval();rank.initialize_from_control(dense)
        first,second=rank.seasonals[0].temporal_projection
        u,s,v=torch.linalg.svd(dense.seasonals[0].temporal_projection.weight.double(),full_matrices=False)
        torch.testing.assert_close(second.weight@first.weight,((u[:,:16]*s[:16])@v[:16]).float(),atol=2e-7,rtol=2e-6)
        with torch.no_grad():
            dense.seasonals[0].temporal_projection.weight.copy_(second.weight@first.weight)
            dense.seasonals[0].temporal_projection.bias.copy_(second.weight@first.bias+second.bias)
            x=torch.randn(2,96,2)
            torch.testing.assert_close(dense(x),rank(x),atol=3e-6,rtol=2e-5)
        self.assertEqual(sum(p.numel() for p in dense.parameters())-sum(p.numel() for p in rank.parameters()),6128)

    def test_train_gradients_and_initial_rng_pairing(self):
        dense=Dense(self.cfg).train();rank=Rank(self.cfg).train();rank.initialize_from_control(dense)
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        torch.manual_seed(99);dense(x);rng=torch.get_rng_state()
        torch.manual_seed(99);pred=rank(x)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        opt=torch.optim.Adam(parameter_groups(rank,'combined'),lr=.001)
        before=rank.seasonals[0].temporal_projection[0].weight.detach().clone()
        (pred-y).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in rank.parameters()))
        opt.step();self.assertFalse(torch.equal(before,rank.seasonals[0].temporal_projection[0].weight))

    def test_frozen_adapter_uses_full_recurrence(self):
        neural=Rank(self.cfg).eval()
        direct=LinearForecastBlend(neural,torch.randn(2,97,96)*.01).eval()
        adapter=Adapter(self.cfg);adapter.load_state_dict(copy.deepcopy(direct.state_dict()))
        x=torch.randn(2,96,2)
        with torch.no_grad():
            torch.testing.assert_close(adapter(x),direct(x),atol=3e-6,rtol=1e-5)
            local=Local(self.cfg).eval();local.load_state_dict(neural.state_dict())
            self.assertGreater(float((local(x)-neural(x)).abs().max()),1e-5)
        with self.assertRaises(ValueError):adapter.train()

    def test_test_reuse_requires_weights_config_and_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            src=root/'source.py';src.write_text('pass')
            cfg=dict(model='OscillatorLinearBlend',seq_len=96,log_dir='old',tensorboard_dir='old')
            for d in (old,new):
                (d/'checkpoint.pth').write_bytes(b'same frozen weights')
                (d/'model_config.json').write_text(json.dumps(cfg))
            ch=sha(old/'checkpoint.pth');cf=sha(old/'model_config.json')
            manifest=dict(checkpoint_sha256=ch,model_config_sha256=cf,adapter_source_sha256={str(src):sha(src)})
            (old/'manifest.json').write_text(json.dumps(manifest))
            report=root/'report.json';report.write_text(json.dumps(dict(status='completed',split='test',windows=11425,
                checkpoint_sha256=ch,model_config_sha256=cf,source_hashes_verified=True)))
            (old/'test_result.json').write_text(json.dumps(dict(checkpoint_sha256=ch,report=str(report))))
            cfg['seq_len']=192;(new/'model_config.json').write_text(json.dumps(cfg))
            self.assertIsNone(try_reuse_control_test(new,old))
            cfg['seq_len']=96;cfg['log_dir']='new';(new/'model_config.json').write_text(json.dumps(cfg))
            src.write_text('changed')
            self.assertIsNone(try_reuse_control_test(new,old))
            src.write_text('pass');(new/'checkpoint.pth').write_bytes(b'different')
            self.assertIsNone(try_reuse_control_test(new,old))
            (new/'checkpoint.pth').write_bytes(b'same frozen weights')
            self.assertTrue(try_reuse_control_test(new,old)['reused'])


if __name__=='__main__':unittest.main()
