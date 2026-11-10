"""Initial equality and absence of learned edges between the two branches."""
import unittest
from pathlib import Path
import torch
from test_pdm_state_patch import cfg
from scripts import run_timemix_pdm_state_patch_ma_decoupled_experiment as run
from utils.pdm_state_patch_ma_decoupled_diagnostics import gradient_probe,FeatureStatistics
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups
from utils.patch_covariance_loss import parts


def branch_parameters(model):
    season=[*model.seasonals.parameters(),*model.parameters(recurse=False)]
    trend=[*model.trend_embeddings.parameters(),*model.feature_trend_heads.parameters()]
    for block in model.pdm_blocks:
        season += [*block.mixing_multi_scale_season.parameters(),*block.out_cross_layer.parameters()]
        trend += [*block.mixing_multi_scale_trend.parameters(),*block.trend_out_cross_layer.parameters()]
    return season,trend


class FullyDecoupledTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(2021)

    def initial_gradient(self,device):
        models=run.initialize_pair(cfg());base=models['position_only'].to(device);split=models['fully_decoupled'].to(device)
        x=torch.randn(2,96,7,device=device);target=torch.randn_like(x,requires_grad=True)
        for training in (False,True):
            outputs=[];rng=[]
            for model in (base,split):
                model.train(training);torch.manual_seed(174);outputs.append(model.predict_components(x));rng.append(torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
            torch.testing.assert_close(*outputs,rtol=0,atol=0);self.assertTrue(torch.equal(*rng))
        for model in (base,split):
            model.zero_grad(set_to_none=True);torch.manual_seed(174);p=model(x);d,_=parts(p,target);((p-target.detach()).square().mean()+d).backward()
        named=dict(split.named_parameters())
        for name,p in base.named_parameters():
            if '.trend_condition.' in name:continue
            if '.value_embedding.weight' in name:
                i=name.split('.')[1];expected=named[name].grad+named[f'trend_embeddings.{i}.weight'].grad
            elif '.value_embedding.bias' in name:
                i=name.split('.')[1];expected=named[f'trend_embeddings.{i}.bias'].grad
            else:expected=named[name].grad
            torch.testing.assert_close(p.grad,expected,rtol=3e-5,atol=3e-7,msg=name)
        self.assertIsNone(target.grad)

    def test_cpu_initial_components_rng_common_gradients(self):self.initial_gradient('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_initial_components_rng_common_gradients(self):self.initial_gradient('cuda')

    def test_complete_parameter_partition_and_bidirectional_intervention(self):
        models=run.initialize_pair(cfg());model=models['fully_decoupled'];model.eval()
        self.assertEqual({a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},run.COUNTS)
        self.assertEqual(sum(p.numel() for p in model.parameters()),136770);self.assertEqual(len(model.state_dict()),236)
        base=models['position_only'].state_dict();removed=set(base)-set(model.state_dict());added=set(model.state_dict())-set(base)
        self.assertEqual(removed,{f'seasonals.{i}.{key}' for i in range(4) for key in ('value_embedding.bias','decoder.trend_condition.weight')})
        self.assertEqual(added,{f'trend_embeddings.{i}.{key}' for i in range(4) for key in ('weight','bias')})
        self.assertFalse(any('trend_condition' in n or 'season_condition' in n for n,_ in model.named_parameters()))
        self.assertEqual({n for n,_ in model.named_parameters(recurse=False)},{'history_gain_offset','drive_gain_offset'})
        sp,tp=branch_parameters(model);si,ti={id(p) for p in sp},{id(p) for p in tp}
        self.assertFalse(si&ti);self.assertEqual(si|ti,{id(p) for p in model.parameters()})
        for e,t in zip(model.seasonals,model.trend_embeddings):self.assertNotEqual(e.value_embedding.weight.data_ptr(),t.weight.data_ptr())
        previous=torch.load(Path('tmp/lag_study_timemix_pdm_state_patch_ma_linear_20261014_seed2021_earlystop5/initial.pt'),map_location='cpu',weights_only=True)
        for arm in ('position_only','season_to_trend'):
            self.assertEqual(set(models[arm].state_dict()),set(previous[arm]))
            for n,v in models[arm].state_dict().items():self.assertTrue(torch.equal(v,previous[arm][n]),arm+'/'+n)
        x=torch.randn(2,96,7)
        with torch.no_grad():
            model.feature_trend_heads[0].readout.weight.fill_(.1)
            a=model.predict_components(x)
            model.trend_embeddings[0].weight[0].add_(.5)
            b=model.predict_components(x)
            torch.testing.assert_close(a[...,:4],b[...,:4],rtol=0,atol=0)
            self.assertGreater(float((a[...,4:]-b[...,4:]).abs().max()),0.)
            model.seasonals[0].decoder.readout.output_projection.weight.mul_(1.2)
            c=model.predict_components(x)
            torch.testing.assert_close(b[...,4:],c[...,4:],rtol=0,atol=0)
            self.assertGreater(float((b[...,:4]-c[...,:4]).abs().max()),0.)
        # Capture BEFORE final stack/sum, so slicing a stack cannot create
        # synthetic zero-gradient paths to otherwise unrelated heads.
        raw_s,raw_t=[],[];handles=[]
        for e in model.seasonals:handles.append(e.decoder.readout.output_projection.register_forward_hook(lambda m,args,out:raw_s.append(out)))
        for h in model.feature_trend_heads:handles.append(h.register_forward_hook(lambda m,args,out:raw_t.append(out)))
        model.predict_components(x)
        for h in handles:h.remove()
        sg=torch.autograd.grad(sum(v.sum() for v in raw_s),tp,retain_graph=True,allow_unused=True)
        tg=torch.autograd.grad(sum(v.sum() for v in raw_t),sp,allow_unused=True)
        self.assertTrue(all(g is None for g in sg));self.assertTrue(all(g is None for g in tg))

    def test_optimizer_finite_gradients_and_later_trend_learning(self):
        model=run.initialize_pair(cfg())['fully_decoupled'];groups=norm_bias_no_decay_groups(model,1e-4);policy={id(p):g['weight_decay'] for g in groups for p in g['params']}
        for e in model.trend_embeddings:self.assertEqual(policy[id(e.weight)],1e-4);self.assertEqual(policy[id(e.bias)],0.)
        opt=torch.optim.Adam(groups,lr=.001);x=torch.randn(2,96,7);target=torch.randn_like(x)
        before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('trend_embeddings.')}
        for _ in range(3):
            opt.zero_grad(set_to_none=True);p=model(x);d,_=parts(p,target);((p-target.detach()).square().mean()+d).backward()
            for n,param in model.named_parameters():self.assertIsNotNone(param.grad,n);self.assertTrue(torch.isfinite(param.grad).all(),n)
            opt.step()
        self.assertGreater(sum(float(e.weight.grad.square().sum()) for e in model.trend_embeddings),0.)
        self.assertTrue(any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before))

    def test_sameforward_scopes_diagnostics_and_joint_three_arm_policy(self):
        model=run.initialize_pair(cfg())['fully_decoupled'];x=torch.randn(2,96,7);y=torch.randn_like(x,requires_grad=True)
        stats=FeatureStatistics(model);p=model(x);d,_=parts(p,y);mse=(p-y.detach()).square().mean();loss=mse+d;params=list(model.parameters());expected=torch.autograd.grad(loss,params,retain_graph=True);rng=torch.get_rng_state().clone()
        scopes=gradient_probe(model,mse,d);self.assertEqual(scopes['trend_condition']['parameter_tensors'],0);self.assertEqual(scopes['season_to_trend_condition']['parameter_tensors'],0);self.assertEqual(scopes['season_pointwise_embedding']['parameter_tensors'],4);self.assertEqual(scopes['trend_pointwise_embedding']['parameter_tensors'],8)
        self.assertEqual(len(stats.summary()),8);stats.close();self.assertTrue(torch.equal(rng,torch.get_rng_state()));loss.backward()
        for param,grad in zip(params,expected):torch.testing.assert_close(param.grad,grad,rtol=0,atol=0)
        self.assertIsNone(y.grad)
        curves=[dict(epoch=e,arm=a,learning_rate=run.shared.learning_rate(e),train_loss=.8,train_mse=.7,train_d=.1,val=dict(mse=1.)) for e in range(1,7) for a in run.ARMS]
        last=dict(epoch=6,curves=curves,best={a:1. for a in run.ARMS},best_epoch={a:1 for a in run.ARMS},stale=5,updates_per_arm=6,optimizers={a:dict(state={0:dict(step=torch.tensor(6.))}) for a in run.ARMS})
        self.assertEqual(run.verify_policy(last,32,30)['joint_stale'],5)


if __name__=='__main__':unittest.main()
