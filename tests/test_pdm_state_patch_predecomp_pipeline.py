"""Three-arm stopping accounting and diagnostics without gradient mutation."""
import unittest
import torch
from test_pdm_state_patch import cfg
from scripts.run_timemix_pdm_state_patch_predecomp_experiment import initialize_pair,verify_policy,ARMS,shared
from utils.pdm_state_patch_predecomp_diagnostics import gradient_probe,FeatureStatistics
from utils.patch_covariance_loss import parts


class PipelineTests(unittest.TestCase):
    def test_three_arm_joint_patience_accounting(self):
        curves=[]
        for epoch in range(1,9):
            for arm in ARMS:
                value=.9 if arm=='season_to_trend' and epoch>=3 else 1.
                curves.append(dict(epoch=epoch,arm=arm,learning_rate=shared.learning_rate(epoch),
                    train_loss=.8,train_mse=.7,train_d=.1,val=dict(mse=value)))
        last=dict(epoch=8,curves=curves,best={a:.9 if a=='season_to_trend' else 1. for a in ARMS},
            best_epoch={a:3 if a=='season_to_trend' else 1 for a in ARMS},stale=5,updates_per_arm=8,
            optimizers={a:dict(state={0:dict(step=torch.tensor(8.))}) for a in ARMS})
        self.assertEqual(verify_policy(last,32,30)['joint_stale'],5)
        last['stale']=4
        with self.assertRaises(AssertionError):verify_policy(last,32,30)
        last['stale']=5;curves[-1]['train_loss']=1.
        with self.assertRaises(AssertionError):verify_policy(last,32,30)

    def test_sameforward_diagnostics_preserve_gradients_and_rng(self):
        torch.set_num_threads(2)
        model=initialize_pair(cfg())['season_to_trend']
        x=torch.randn(2,96,7);y=torch.randn_like(x,requires_grad=True)
        stats=FeatureStatistics(model);p=model(x);d,_=parts(p,y);mse=(p-y.detach()).square().mean();loss=mse+d
        params=list(model.parameters())
        expected=torch.autograd.grad(loss,params,retain_graph=True)
        rng=torch.get_rng_state().clone()
        gradient_probe(model,mse,d);summary=stats.summary();stats.close()
        self.assertEqual(len(summary),8)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in params));loss.backward()
        for actual,wanted in zip(params,expected):torch.testing.assert_close(actual.grad,wanted,rtol=0,atol=0)
        self.assertIsNone(y.grad)


if __name__=='__main__':unittest.main()
