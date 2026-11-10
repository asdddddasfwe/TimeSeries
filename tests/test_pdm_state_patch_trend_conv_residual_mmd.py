"""Independent internal-skip formula, unchanged initialization/policy, objectives and probes."""
import copy
import unittest

import torch
from torch.nn import functional as F
from models.OscillatorPDMStatePatchTrendTemporalInnerResidual import TrendTemporalInnerResidual
from scripts import run_timemix_pdm_state_patch_trend_conv_residual_mmd as run


class ResidualMMDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def devices(self):
        return ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

    def test_independent_internal_formula_gradients_causality_receptive_field(self):
        for device in self.devices():
            block = TrendTemporalInnerResidual(2).double().to(device)
            with torch.no_grad():
                for layer in block.depthwise:
                    layer.weight.fill_(.15)
                block.output.weight.copy_(torch.eye(2, dtype=torch.float64, device=device)[:, :, None]*.2)
            x = torch.full((1,24,2), .4, dtype=torch.float64, device=device, requires_grad=True)
            actual = block(x)
            manual = x.transpose(1,2)
            for layer, dilation in zip(block.depthwise,(1,2,4)):
                # Explicit per-tap gather, independent of Conv1d and F.pad.
                times = torch.arange(24,device=device)
                values = sum(manual[:, :, (times-(2-tap)*dilation).clamp_min(0)]*
                             layer.weight[:,0,tap][None,:,None] for tap in range(3))
                manual = manual+F.gelu(values)
            expected = x+torch.einsum('nct,oc->nto',manual,block.output.weight[:,:,0])
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            params = [x]+list(block.parameters())
            ga = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
            ge = torch.autograd.grad(expected.square().sum(), params, retain_graph=True)
            for a,b in zip(ga,ge):
                torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
            changed = x.detach().clone(); changed[:,16:] += 3
            torch.testing.assert_close(block(changed)[:,:16], actual[:,:16],rtol=0,atol=0)
            g14 = torch.autograd.grad(actual[:,14].sum(),x,retain_graph=True)[0]
            g15 = torch.autograd.grad(actual[:,15].sum(),x)[0]
            self.assertGreater(float(g14[:,0].abs().max()),0)
            self.assertEqual(float(g15[:,0].abs().max()),0)

    def test_all252_initial_function_rng_common_gradients_and_policy(self):
        saved = torch.load(run.NO_DECAY_OLD/'initial.pt',map_location='cpu',weights_only=True)['trend_temporal_conv_no_decay']
        policy = {n:g['weight_decay'] for g in run.shared.read(run.NO_DECAY_OLD/'optimizer_groups.json')['trend_temporal_conv_no_decay'] for n in g['names']}
        for device in self.devices():
            for arm in run.ALL_ARMS:
                run.configure(arm)
                cfg = run.config(run.ROOT/'dataset/ETT-small')
                model = run.initialize_pair(cfg)[arm].to(device)
                self.assertTrue(run.initialization_audit({arm:model})[arm]['all252_saved_no_decay_initial_exact'])
                other = run.TrendTemporalConv(cfg).to(device)
                other.load_state_dict(saved)
                names = {id(p):n for n,p in model.named_parameters()}
                groups = run.optimizer_groups(model)
                actual_policy = {names[id(p)]:g['weight_decay'] for g in groups for p in g['params']}
                expected_policy={n:(run.conv_weight_decay(arm) if n.startswith('trend_temporal_encoders.') else v) for n,v in policy.items()}
                self.assertEqual(expected_policy,actual_policy)
                x = torch.randn(2,96,7,generator=torch.Generator().manual_seed(111)).to(device)
                y = torch.randn_like(x)
                for training in (False,True):
                    outputs,grads,rng = [],[],[]
                    for m in (other,model):
                        m.train(training); run.shared.seed_all(122)
                        c = m.predict_components(x)
                        p = m.normalize(c[...,:4].sum(-1)+c[...,4:].sum(-1),'denorm')
                        d,_ = run.parts(p,y)
                        outputs.append(c.detach())
                        grads.append(torch.autograd.grad((p-y).square().mean()+d,list(m.parameters())))
                        rng.append(torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
                    self.assertTrue(torch.equal(outputs[0],outputs[1]))
                    self.assertTrue(torch.equal(rng[0],rng[1]))
                    for g,h in zip(*grads):
                        self.assertTrue(torch.equal(g,h))

    def test_both_objective_probes_no_interference_and_later_layer_learning(self):
        for device in self.devices():
            for arm in run.ALL_ARMS:
                run.configure(arm)
                model = run.initialize_pair(run.config(run.ROOT/'dataset/ETT-small'))[arm].to(device)
                opt = torch.optim.Adam(run.optimizer_groups(model),lr=.001)
                x = torch.randn(2,96,7,generator=torch.Generator().manual_seed(121)).to(device)
                y = torch.randn(2,96,7,generator=torch.Generator().manual_seed(122)).to(device).requires_grad_()
                learned = set()
                for step in range(7):
                    run.shared.seed_all(160+step); opt.zero_grad(set_to_none=True)
                    stats = run.FeatureStatistics(model)
                    p = model(x); d,_ = run.parts(p,y); mmd,_ = run.mmd_parts(p,y)
                    mse = (p-y.detach()).square().mean()
                    loss = mse+(mmd if run.uses_mmd(arm) else d)
                    params = list(model.parameters())
                    expected = torch.autograd.grad(loss,params,retain_graph=True)
                    rng = torch.cuda.get_rng_state().clone() if device=='cuda' else torch.get_rng_state().clone()
                    scopes = run.gradient_probe(model,mse,d)
                    mscopes = run.mmd_gradient_probe(model,mse,mmd)
                    output = run.output_gradient_probe(p,mse,mmd,d)
                    self.assertEqual(len([n for n in scopes if n.startswith('encoder_scale')]),16)
                    self.assertEqual(set(scopes),set(mscopes))
                    self.assertEqual(set(output),{'all','low4','remaining92','level'})
                    self.assertEqual(len(run.weight_statistics(model)),16)
                    self.assertEqual(len(stats.encoder_summary()),4); stats.close()
                    current = torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()
                    self.assertTrue(torch.equal(rng,current))
                    loss.backward(); self.assertIsNone(y.grad)
                    for (n,p),g in zip(model.named_parameters(),expected):
                        self.assertTrue(torch.equal(p.grad,g),n)
                        self.assertTrue(torch.isfinite(p.grad).all(),n)
                        if n.startswith('trend_temporal_encoders.') and p.grad.abs().max()>0:
                            learned.add(n)
                    opt.step()
                self.assertEqual(len(learned),16,(arm,learned))

    def test_independent_fixed11_selection_and_loss_accounting(self):
        for arm in run.ALL_ARMS:
            run.configure(arm)
            rows = [dict(epoch=e,arm=arm,learning_rate=run.shared.learning_rate(e),
                train_loss=.8,train_mse=.7,train_d=.1 if not run.uses_mmd(arm) else .2,
                train_mmd=.1 if run.uses_mmd(arm) else .2,val=dict(mse=1.)) for e in range(1,12)]
            last = dict(epoch=11,curves=rows,best={arm:1.},best_epoch={arm:1},stale=10,
                updates_per_arm=11,optimizers={arm:dict(state={0:dict(step=torch.tensor(11.))})})
            self.assertEqual(run.verify_policy(last,32,11)['single_stale_diagnostic'],10)
            wrong = copy.deepcopy(last); wrong['curves'][0]['train_loss'] += .01
            with self.assertRaises(AssertionError):
                run.verify_policy(wrong,32,11)


if __name__ == '__main__':
    unittest.main()
