"""Strict first-arm MA/Linear exchange: state, function, gradients and order."""
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from test_pdm_state_patch import cfg
from scripts import run_timemix_pdm_state_patch_ma_first_experiment as run
from utils.pdm_state_patch_ma_first_diagnostics import gradient_probe,FeatureStatistics
from utils.patch_covariance_loss import parts


class MAFirstTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(2021)

    def equivalence(self,device):
        old=run.scratch_reference(cfg()).to(device)
        new=run.initialize_pair(cfg())['position_only'].to(device)
        self.assertEqual(sum(p.numel() for p in new.parameters()),140738)
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        self.assertEqual(len(new.state_dict()),236)
        for n,v in old.state_dict().items():
            self.assertTrue(torch.equal(v,new.state_dict()[n]),n)
            self.assertNotEqual(v.data_ptr(),new.state_dict()[n].data_ptr())
        self.assertFalse(any('patch_codec' in n or 'patch_encoder' in n for n,_ in new.named_parameters()))
        x=torch.randn(2,96,7,device=device);target=torch.randn_like(x,requires_grad=True)
        for training in (False,True):
            outputs=[];grads=[];rng=[]
            for model in (old,new):
                model.train(training);torch.manual_seed(174)
                p=model(x);outputs.append(p);d,_=parts(p,target);loss=(p-target.detach()).square().mean()+d
                grads.append(torch.autograd.grad(loss,list(model.parameters())))
                rng.append(torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
            torch.testing.assert_close(*outputs,rtol=3e-5,atol=3e-6)
            self.assertTrue(torch.equal(*rng))
            for (name,_),a,b in zip(old.named_parameters(),*grads):
                torch.testing.assert_close(a,b,rtol=5e-4,atol=3e-6,msg=name)
        self.assertIsNone(target.grad)

    def test_cpu_original0318_function_and_gradient(self):self.equivalence('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_original0318_function_and_gradient(self):self.equivalence('cuda')

    def test_raw_ma_before_existing_embedding_fft_after_embedding(self):
        new=run.initialize_pair(cfg())['position_only'];new.eval();order=[]
        import models.OscillatorPDMSegmentPE as module
        select=module.select_seasonal_feature_periods
        h=new.initial_decomposition.register_forward_pre_hook(lambda m,args:order.append(('ma',args[0].shape[-1])))
        handles=[e.value_embedding.register_forward_pre_hook(lambda m,args:order.append(('linear',args[0].shape[-1]))) for e in new.seasonals]
        def checked(x,k):order.append(('fft',x.shape[-1]));return select(x,k)
        with patch.object(module,'select_seasonal_feature_periods',side_effect=checked):
            with torch.no_grad():new(torch.randn(2,96,7))
        h.remove()
        for handle in handles:handle.remove()
        self.assertEqual(order[:8],[('ma',1),('linear',1)]*4)
        self.assertEqual(order[8:],[('fft',32)]*4)

    def test_retained_patch_arms_and_initial_audit(self):
        models=run.initialize_pair(cfg());audit=run.initialization_audit(models)
        self.assertTrue(audit['first_arm_all_236_states_exact_old0318_initial'])
        self.assertEqual(audit['parameter_counts'],run.COUNTS)
        previous=Path('tmp/lag_study_timemix_pdm_state_patch_predecomp_20261014_seed2021_earlystop5/initial.pt')
        state=torch.load(previous,map_location='cpu',weights_only=True)
        for arm in ('decoupled','season_to_trend'):
            self.assertEqual(set(models[arm].state_dict()),set(state[arm]))
            for n,v in models[arm].state_dict().items():self.assertTrue(torch.equal(v,state[arm][n]),arm+'/'+n)

    def test_probe_preserves_first_arm_gradients_and_three_arm_policy(self):
        model=run.initialize_pair(cfg())['position_only'];x=torch.randn(2,96,7);y=torch.randn_like(x,requires_grad=True)
        stats=FeatureStatistics(model);p=model(x);d,_=parts(p,y);mse=(p-y.detach()).square().mean();loss=mse+d
        params=list(model.parameters());expected=torch.autograd.grad(loss,params,retain_graph=True);rng=torch.get_rng_state().clone()
        gradient_probe(model,mse,d);self.assertEqual(len(stats.summary()),8);stats.close();self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        loss.backward()
        for p,g in zip(params,expected):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        curves=[dict(epoch=e,arm=a,learning_rate=run.shared.learning_rate(e),train_loss=.8,train_mse=.7,train_d=.1,val=dict(mse=1.)) for e in range(1,7) for a in run.ARMS]
        last=dict(epoch=6,curves=curves,best={a:1. for a in run.ARMS},best_epoch={a:1 for a in run.ARMS},stale=5,updates_per_arm=6,optimizers={a:dict(state={0:dict(step=torch.tensor(6.))}) for a in run.ARMS})
        self.assertEqual(run.verify_policy(last,32,30)['joint_stale'],5)


if __name__=='__main__':unittest.main()
