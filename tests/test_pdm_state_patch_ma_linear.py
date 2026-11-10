"""Independent pointwise embedding only: initial function/gradient identity."""
import unittest
from pathlib import Path
import torch
from test_pdm_state_patch import cfg
from scripts import run_timemix_pdm_state_patch_ma_linear_experiment as run
from utils.pdm_state_patch_ma_linear_diagnostics import gradient_probe,FeatureStatistics
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts


class SeparateLinearTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(2021)

    def shared_gradient(self,device):
        models=run.initialize_pair(cfg());base=models['position_only'].to(device);split=models['separate_linear'].to(device)
        x=torch.randn(2,96,7,device=device);target=torch.randn_like(x,requires_grad=True)
        for training in (False,True):
            outputs=[];rng=[]
            for model in (base,split):
                model.train(training);model.zero_grad(set_to_none=True);torch.manual_seed(174)
                outputs.append(model.predict_components(x));rng.append(torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
            torch.testing.assert_close(*outputs,rtol=0,atol=0);self.assertTrue(torch.equal(*rng))
        for model in (base,split):
            model.zero_grad(set_to_none=True);torch.manual_seed(174);p=model(x);d,_=parts(p,target);((p-target.detach()).square().mean()+d).backward()
        names=dict(split.named_parameters())
        for name,p in base.named_parameters():
            if '.value_embedding.weight' in name:
                i=name.split('.')[1];expected=names[name].grad+names[f'trend_embeddings.{i}.weight'].grad
            elif '.value_embedding.bias' in name:
                i=name.split('.')[1];expected=names[f'trend_embeddings.{i}.bias'].grad
            else:expected=names[name].grad
            torch.testing.assert_close(p.grad,expected,rtol=3e-5,atol=3e-7,msg=name)
        self.assertIsNone(target.grad)

    def test_cpu_initial_components_rng_shared_gradient_sum(self):self.shared_gradient('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_initial_components_rng_shared_gradient_sum(self):self.shared_gradient('cuda')

    def test_only_embedding_keys_changed_and_first_third_unchanged(self):
        models=run.initialize_pair(cfg());audit=run.initialization_audit(models)
        self.assertEqual(audit['parameter_counts'],dict(position_only=140738,separate_linear=140866,season_to_trend=235628))
        base,split=models['position_only'],models['separate_linear'];self.assertEqual(len(split.state_dict()),240)
        removed=set(base.state_dict())-set(split.state_dict());added=set(split.state_dict())-set(base.state_dict())
        self.assertEqual(removed,{f'seasonals.{i}.value_embedding.bias' for i in range(4)})
        self.assertEqual(added,{f'trend_embeddings.{i}.{k}' for i in range(4) for k in ('weight','bias')})
        for n in set(base.state_dict())&set(split.state_dict()):self.assertTrue(torch.equal(base.state_dict()[n],split.state_dict()[n]),n)
        for i,(e,t) in enumerate(zip(split.seasonals,split.trend_embeddings)):
            self.assertNotEqual(e.value_embedding.weight.data_ptr(),t.weight.data_ptr())
            self.assertTrue(torch.equal(e.value_embedding.weight,t.weight));self.assertIsNone(e.value_embedding.bias)
            self.assertTrue(hasattr(e.decoder,'trend_condition'))
        prior=torch.load(Path('tmp/lag_study_timemix_pdm_state_patch_ma_first_20261014_seed2021_earlystop5/initial.pt'),map_location='cpu',weights_only=True)
        for arm in ('position_only','season_to_trend'):
            self.assertEqual(set(models[arm].state_dict()),set(prior[arm]))
            for n,v in models[arm].state_dict().items():self.assertTrue(torch.equal(v,prior[arm][n]),arm+'/'+n)

    def test_projection_intervention_optimizer_and_later_learning(self):
        model=run.initialize_pair(cfg())['separate_linear']
        se,te=model.seasonals[0].value_embedding,model.trend_embeddings[0]
        x=torch.randn(2,96,1)
        a=se(x).detach().clone()
        with torch.no_grad():te.weight.add_(.1)
        torch.testing.assert_close(se(x),a,rtol=0,atol=0)
        # Existing historical cross condition is still active when its zero
        # initialization opens: independent projections are not disconnected.
        decoder=model.seasonals[0].decoder;own=torch.randn(2,96,32);other=torch.randn_like(own)
        with torch.no_grad():decoder.trend_condition.weight.normal_(0,.03)
        self.assertGreater(float((decoder.dynamics(own,other)[0]-decoder.dynamics(own,other*2+.5)[0]).detach().abs().max()),0.)
        model=run.initialize_pair(cfg())['separate_linear']
        groups=norm_bias_no_decay_groups(model,1e-4);policy={id(p):g['weight_decay'] for g in groups for p in g['params']}
        for embedding in model.trend_embeddings:
            self.assertEqual(policy[id(embedding.weight)],1e-4);self.assertEqual(policy[id(embedding.bias)],0.)
        opt=torch.optim.Adam(groups,lr=.001);x=torch.randn(2,96,7);target=torch.randn_like(x)
        for _ in range(3):
            opt.zero_grad(set_to_none=True);p=model(x);d,_=parts(p,target);((p-target.detach()).square().mean()+d).backward()
            for name,param in model.named_parameters():self.assertIsNotNone(param.grad,name);self.assertTrue(torch.isfinite(param.grad).all(),name)
            opt.step()
        self.assertGreater(sum(float(e.weight.grad.square().sum()) for e in model.trend_embeddings),0.)
        self.assertGreater(sum(float((e.value_embedding.weight-t.weight).detach().square().sum()) for e,t in zip(model.seasonals,model.trend_embeddings)),0.)

    def test_sameforward_diagnostics_and_joint_three_arm_stopping(self):
        model=run.initialize_pair(cfg())['separate_linear'];x=torch.randn(2,96,7);y=torch.randn_like(x,requires_grad=True)
        stats=FeatureStatistics(model);p=model(x);d,_=parts(p,y);mse=(p-y.detach()).square().mean();loss=mse+d;params=list(model.parameters());expected=torch.autograd.grad(loss,params,retain_graph=True);rng=torch.get_rng_state().clone()
        scopes=gradient_probe(model,mse,d);self.assertEqual(scopes['shared_pointwise_embedding']['parameter_tensors'],0);self.assertEqual(scopes['season_pointwise_embedding']['parameter_tensors'],4);self.assertEqual(scopes['trend_pointwise_embedding']['parameter_tensors'],8)
        self.assertEqual(len(stats.summary()),8);stats.close();self.assertTrue(torch.equal(rng,torch.get_rng_state()));loss.backward()
        for param,grad in zip(params,expected):torch.testing.assert_close(param.grad,grad,rtol=0,atol=0)
        self.assertIsNone(y.grad)
        curves=[dict(epoch=e,arm=a,learning_rate=run.shared.learning_rate(e),train_loss=.8,train_mse=.7,train_d=.1,val=dict(mse=1.)) for e in range(1,7) for a in run.ARMS]
        last=dict(epoch=6,curves=curves,best={a:1. for a in run.ARMS},best_epoch={a:1 for a in run.ARMS},stale=5,updates_per_arm=6,optimizers={a:dict(state={0:dict(step=torch.tensor(6.))}) for a in run.ARMS})
        self.assertEqual(run.verify_policy(last,32,30)['joint_stale'],5)


if __name__=='__main__':unittest.main()
