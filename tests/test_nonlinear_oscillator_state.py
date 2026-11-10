"""Independent serial equivalence, differentiation, causality, and integration."""
import copy
import unittest
import torch
from test_oscillator_mixer import config
from layers.OscillatorTimeMix import complex_oscillator_scan
from layers.NonlinearOscillatorScan import (
    HistoryStateCorrection, sequential_nonlinear_scan, parallel_nonlinear_scan,
    recurrence_residual,
)
from models.OscillatorPDMTrendCondition import Model as Baseline
from models.OscillatorPDMNonlinearState import Model
from utils.timemix_decay_study import parameter_groups


class NonlinearScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
    def inputs(self, horizon=5):
        torch.manual_seed(23)
        a = (torch.randn(2,horizon,2,2,dtype=torch.double)*.12).requires_grad_()
        u = torch.randn_like(a,requires_grad=True)
        z = torch.randn(2,2,2,dtype=torch.double,requires_grad=True)
        c = torch.randn(2,6,dtype=torch.double,requires_grad=True)
        f = HistoryStateCorrection(2,6,7).double()
        with torch.no_grad():f.output.weight.normal_(0,.08);f.output.bias.normal_(0,.02)
        return a,u,z,c,f
    def test_full_horizon_rounds_match_nonzero_serial_values_and_all_gradients(self):
        for horizon in (1,2,5,8):
            a,u,z,c,f = self.inputs(horizon)
            serial=sequential_nonlinear_scan(a,u,z,c,f)
            parallel=parallel_nonlinear_scan(a,u,z,c,f,horizon)
            torch.testing.assert_close(parallel,serial,atol=2e-12,rtol=2e-12)
            weights=torch.randn_like(serial)
            variables=(a,u,z,c,*f.parameters())
            left=torch.autograd.grad((parallel*weights).sum(),variables,retain_graph=True)
            right=torch.autograd.grad((serial*weights).sum(),variables)
            for x,y in zip(left,right):torch.testing.assert_close(x,y,atol=2e-11,rtol=2e-11)
            self.assertLess(recurrence_residual(parallel,a,u,z,c,f).abs().max().item(),2e-12)
    def test_gradcheck_fixed_round_solver(self):
        a,u,z,c,f=self.inputs(3)
        self.assertTrue(torch.autograd.gradcheck(lambda aa,uu,zz,cc:parallel_nonlinear_scan(aa,uu,zz,cc,f,2),
                                               (a,u,z,c),eps=1e-6,atol=1e-5,rtol=1e-4,fast_mode=True))
    def test_zero_initialization_exact_affine_and_first_step_gradient(self):
        a,u,z,c,f=self.inputs()
        with torch.no_grad():f.output.weight.zero_();f.output.bias.zero_()
        expected=complex_oscillator_scan(a,u,z)
        actual=parallel_nonlinear_scan(a,u,z,c,f,4)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        actual.square().sum().backward()
        self.assertGreater(f.output.weight.grad.norm().item(),0.)
        self.assertEqual(f.input.weight.grad.norm().item(),0.)
    def test_few_rounds_are_approximate_and_history_participates(self):
        a,u,z,c,f=self.inputs(8)
        reference=sequential_nonlinear_scan(a,u,z,c,f)
        approximate=parallel_nonlinear_scan(a,u,z,c,f,1)
        self.assertGreater((reference-approximate).abs().max().item(),1e-5)
        self.assertGreater(recurrence_residual(approximate,a,u,z,c,f).abs().max().item(),1e-5)
        self.assertGreater(torch.autograd.grad(approximate[:,-1].square().sum(),c)[0].norm().item(),0.)
        # Isolate the direct historical entrance at every position, independent of z0.
        delta=f(torch.zeros_like(u),torch.zeros_like(u),c)
        for h in range(8):
            g=torch.autograd.grad(delta[:,h].square().sum(),c,retain_graph=True)[0]
            self.assertGreater(g.norm().item(),0.)
    def test_future_candidates_do_not_change_earlier_states(self):
        a,u,z,c,f=self.inputs(8)
        for rounds in (1,3,8):
            p=parallel_nonlinear_scan(a,u,z,c,f,rounds)
            grad=torch.autograd.grad(p[:,3].square().sum(),u)[0]
            self.assertEqual(grad[:,4:].abs().max().item(),0.)
            changed=u.detach().clone();changed[:,4:]+=100
            q=parallel_nonlinear_scan(a,changed,z,c,f,rounds)
            torch.testing.assert_close(p[:,:4],q[:,:4],atol=0,rtol=0)
    def test_invalid_inputs_fail_explicitly(self):
        a,u,z,c,f=self.inputs()
        for n in (-1,True,1.5):
            with self.assertRaises(ValueError):parallel_nonlinear_scan(a,u,z,c,f,n)
        with self.assertRaises(ValueError):sequential_nonlinear_scan(a,u,z,c[:1],f)
        with self.assertRaises(ValueError):parallel_nonlinear_scan(a.half(),u.half(),z.half(),c.half(),f,2)


class NonlinearModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def models(self):
        cfg=config(seq_len=32,pred_len=16,moving_avg=5,e_layers=1,dropout=.2,
                   down_sampling_layers=3,down_sampling_window=2,
                   nonlinear_solver='parallel',nonlinear_iterations=16,nonlinear_hidden=8)
        cfg.seasonal_normalize=False
        torch.manual_seed(2021);old=Baseline(cfg);old_rng=torch.get_rng_state()
        torch.manual_seed(2021);new=Model(cfg)
        self.assertTrue(torch.equal(old_rng,torch.get_rng_state()))
        return cfg,old,new
    def test_baseline_initial_function_common_state_rng_and_normalization(self):
        cfg,old,new=self.models();source=old.state_dict()
        for name,value in new.state_dict().items():
            if name in source:torch.testing.assert_close(value,source[name],atol=0,rtol=0)
            else:self.assertIn('.state_correction.',name)
        x=torch.randn(2,32,2)
        for train in (False,True):
            values=[];rng=[]
            for model in (old,new):
                model.train(train);torch.manual_seed(17);values.append(model(x));rng.append(torch.get_rng_state())
            torch.testing.assert_close(*values,atol=0,rtol=0)
            self.assertTrue(torch.equal(*rng))
        for model in (old,new):
            normalized=model.normalize(x,'norm')
            torch.testing.assert_close(model.normalize(normalized,'denorm'),x,atol=2e-6,rtol=2e-6)
    def test_optimizer_training_restore_components_and_nonzero_serial_equivalence(self):
        cfg,old,new=self.models()
        groups=parameter_groups(new,'adam_norm_bias');ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in new.parameters()})
        policies=[]
        for model in (old,new):
            wd={id(p):g['weight_decay'] for g in parameter_groups(model,'adam_norm_bias') for p in g['params']}
            policies.append({n:wd[id(p)] for n,p in model.named_parameters()})
        for n,v in policies[0].items():self.assertEqual(v,policies[1][n])
        for n,v in policies[1].items():
            if '.state_correction.' in n:self.assertEqual(v,0. if n.endswith('bias') else 1e-4)
        x,y=torch.randn(2,32,2),torch.randn(2,16,2)
        optimizer=torch.optim.Adam(groups,lr=.001)
        for _ in range(2):
            optimizer.zero_grad();loss=(new(x)-y).square().mean();loss.backward()
            for n,p in new.named_parameters():
                self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
            optimizer.step()
        self.assertTrue(all(e.decoder.state_correction.output.weight.norm()>0 for e in new.seasonals))
        new.eval();restored=Model(copy.deepcopy(cfg)).eval();restored.load_state_dict(new.state_dict(),strict=True)
        with torch.no_grad():
            p=new(x);torch.testing.assert_close(p,restored(x),atol=0,rtol=0)
            restored.nonlinear_solver='sequential';q=restored(x)
            torch.testing.assert_close(p,q,atol=3e-6,rtol=3e-6)
            parts=new.predict_components(x)
            self.assertEqual(parts.shape,(2,16,2,8))
            expected=new.normalize(parts[...,:4].sum(-1)+parts[...,4:].sum(-1),'denorm')
            torch.testing.assert_close(p,expected,atol=0,rtol=0)

if __name__=='__main__':unittest.main()
