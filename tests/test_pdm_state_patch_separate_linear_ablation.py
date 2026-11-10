"""Single new linear-ablation arm with saved-control budget and no old reruns."""
import unittest,math
import torch
from scripts import run_timemix_pdm_state_patch_separate_linear_ablation as run
from utils.pdm_state_patch_ma_linear_diagnostics import FeatureStatistics,gradient_probe
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts

class LinearAblationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(2021)
    def identity(self,device):
        cfg=run.config(run.ROOT/'dataset/ETT-small');base=run.control_reference(cfg).to(device);split=run.initialize_pair(cfg)['separate_linear'].to(device)
        x=torch.randn(2,96,7,device=device);target=torch.randn_like(x,requires_grad=True)
        for training in (False,True):
            outputs=[];rng=[]
            for model in (base,split):
                model.train(training);run.shared.seed_all(174);outputs.append(model.predict_components(x));rng.append(torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
            torch.testing.assert_close(*outputs,rtol=0,atol=0);self.assertTrue(torch.equal(*rng))
        for model in (base,split):
            model.zero_grad(set_to_none=True);run.shared.seed_all(174);p=model(x);d,_=parts(p,target);((p-target.detach()).square().mean()+d).backward()
        names=dict(split.named_parameters())
        for n,p in base.named_parameters():
            if '.value_embedding.weight' in n:expected=names[n].grad+names[f"trend_embeddings.{n.split('.')[1]}.weight"].grad
            elif '.value_embedding.bias' in n:expected=names[f"trend_embeddings.{n.split('.')[1]}.bias"].grad
            else:expected=names[n].grad
            torch.testing.assert_close(p.grad,expected,rtol=3e-5,atol=3e-7,msg=n)
        self.assertIsNone(target.grad)
    def test_cpu_initial_function_rng_gradient_mapping(self):self.identity('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_initial_function_rng_gradient_mapping(self):self.identity('cuda')
    def test_only_linear_storage_policy_cross_retained_later_learning(self):
        cfg=run.config(run.ROOT/'dataset/ETT-small');model=run.initialize_pair(cfg)['separate_linear'];a=run.initialization_audit(dict(separate_linear=model))
        self.assertEqual(a['states'],240);self.assertTrue(a['common232_exact']);self.assertTrue(a['all_trend_to_season_modulation_retained'])
        dec=model.seasonals[0].decoder;own=torch.randn(2,96,32);other=torch.randn_like(own)
        with torch.no_grad():dec.trend_condition.weight.normal_(0,.03)
        self.assertGreater(float((dec.dynamics(own,other)[0]-dec.dynamics(own,other*2+.5)[0]).detach().abs().max()),0.)
        model=run.initialize_pair(cfg)['separate_linear'];groups=norm_bias_no_decay_groups(model,1e-4);policy={id(p):g['weight_decay'] for g in groups for p in g['params']}
        for e in model.trend_embeddings:self.assertEqual(policy[id(e.weight)],1e-4);self.assertEqual(policy[id(e.bias)],0.)
        opt=torch.optim.Adam(groups,lr=.001);x=torch.randn(2,96,7);y=torch.randn_like(x)
        for _ in range(3):
            opt.zero_grad(set_to_none=True);p=model(x);d,_=parts(p,y);((p-y).square().mean()+d).backward()
            for n,p in model.named_parameters():self.assertIsNotNone(p.grad,n);self.assertTrue(torch.isfinite(p.grad).all(),n)
            opt.step()
        self.assertGreater(sum(float(e.weight.grad.square().sum()) for e in model.trend_embeddings),0.)
        self.assertGreater(sum(float((e.value_embedding.weight-t.weight).detach().square().sum()) for e,t in zip(model.seasonals,model.trend_embeddings)),0.)
    def test_sameforward_diagnostics_and_fixed11_no_patience_exit(self):
        cfg=run.config(run.ROOT/'dataset/ETT-small');model=run.initialize_pair(cfg)['separate_linear'];x=torch.randn(2,96,7);y=torch.randn_like(x,requires_grad=True)
        stats=FeatureStatistics(model);p=model(x);d,_=parts(p,y);m=(p-y.detach()).square().mean();params=list(model.parameters());expected=torch.autograd.grad(m+d,params,retain_graph=True);rng=torch.get_rng_state().clone()
        scopes=gradient_probe(model,m,d);self.assertEqual(scopes['shared_pointwise_embedding']['parameter_tensors'],0);self.assertEqual(scopes['season_pointwise_embedding']['parameter_tensors'],4);self.assertEqual(scopes['trend_pointwise_embedding']['parameter_tensors'],8);self.assertEqual(scopes['trend_to_season_condition']['parameter_tensors'],4)
        self.assertEqual(len(stats.summary()),8);stats.close();self.assertTrue(torch.equal(rng,torch.get_rng_state()));(m+d).backward()
        for p,g in zip(params,expected):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        self.assertIsNone(y.grad)
        curves=[dict(epoch=e,arm='separate_linear',learning_rate=run.shared.learning_rate(e),train_loss=.8,train_mse=.7,train_d=.1,val=dict(mse=1.)) for e in range(1,12)]
        last=dict(epoch=11,curves=curves,best=dict(separate_linear=1.),best_epoch=dict(separate_linear=1),stale=10,updates_per_arm=11,optimizers=dict(separate_linear=dict(state={0:dict(step=torch.tensor(11.))})))
        self.assertEqual(run.verify_policy(last,32,11)['single_stale_diagnostic'],10)
        with self.assertRaises(AssertionError):run.verify_policy({**last,'epoch':10},32,11)

if __name__=='__main__':unittest.main()
