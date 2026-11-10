"""Verify LR-only pairing, objective, optimizer, and fixed cosine schedule."""
import unittest
from types import SimpleNamespace
import torch
from test_oscillator_mixer import config
from scripts import train_timemix_pdm_learning_rate as study

class LearningRateTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.cfg = config(seq_len=96, pred_len=96, moving_avg=25, e_layers=1,
                          dropout=.2, down_sampling_layers=3, down_sampling_window=2)
        self.cfg.seasonal_normalize = False
        torch.manual_seed(2021)
        self.models = study.initialize_pair(self.cfg)
    def tearDown(self):
        torch.set_num_threads(self.threads)
    def test_identical_state_prediction_and_restore(self):
        a, b = self.models.values()
        self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
        for k, v in a.state_dict().items():
            torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
        x = torch.randn(2,96,2)
        for training in (False, True):
            values=[]
            for model in self.models.values():
                model.train(training);torch.manual_seed(41)
                values.append(model(x))
            torch.testing.assert_close(*values, rtol=0, atol=0)
        for arm, model in self.models.items():
            restored=study.VARIANTS[arm](study.arm_config(self.cfg,arm))
            restored.load_state_dict(model.state_dict(),strict=True)
            self.assertEqual(study.arm_config(self.cfg,arm).learning_rate,study.LEARNING_RATES[arm])
    def test_optimizer_only_lr_differs(self):
        optimizers={arm:study.build_optimizer(study.parameter_groups(m,arm),arm) for arm,m in self.models.items()}
        a,b=[o.state_dict()['param_groups'] for o in optimizers.values()]
        for ga,gb in zip(a,b):
            self.assertEqual(ga['lr'],.001);self.assertEqual(gb['lr'],.0003)
            self.assertEqual({k:v for k,v in ga.items() if k!='lr'},{k:v for k,v in gb.items() if k!='lr'})
        # Identical synthetic gradients must produce equal Adam moments with distinct updates.
        before={arm:{k:p.detach().clone() for k,p in m.named_parameters()} for arm,m in self.models.items()}
        for arm,m in self.models.items():
            for p in m.parameters():p.grad=torch.ones_like(p)*.125
            optimizers[arm].step()
        for (name,a),(other,b) in zip(self.models['lr1e3'].named_parameters(),self.models['lr3e4'].named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(before['lr1e3'][name]-a,(before['lr3e4'][name]-b)/.3,rtol=3e-3,atol=3e-7)
    def test_pure_mse_gradient_and_finite_backward(self):
        x,y=torch.randn(2,96,2),torch.randn(2,96,2)
        cfg=study.loss_configuration(SimpleNamespace(variance_floor=.01,kl_jitter=.01))
        for arm,m in self.models.items():
            p=m(x);parts=study.training_losses(p,y,arm,cfg,x)
            reference=(p-y).square().mean()
            torch.testing.assert_close(parts['loss'],reference,rtol=0,atol=0)
            a=torch.autograd.grad(parts['loss'],p,retain_graph=True)[0]
            b=torch.autograd.grad(reference,p,retain_graph=True)[0]
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(parts['auxiliary_kl'].item(),0.)
            parts['loss'].backward()
            for name,param in m.named_parameters():
                self.assertIsNotNone(param.grad,name)
                self.assertTrue(torch.isfinite(param.grad).all(),name)
    def test_fixed_cosine_and_joint_patience(self):
        for e in range(30):
            self.assertAlmostEqual(study.learning_rate('lr3e4',e,30)/study.learning_rate('lr1e3',e,30),.3)
        for arm in study.ARMS:
            self.assertEqual(study.learning_rate(arm,0,30),study.LEARNING_RATES[arm])
            self.assertEqual(study.learning_rate(arm,30,30),0.)
        self.assertEqual(study.joint_stale_count(4,True),0)
        self.assertEqual(study.joint_stale_count(4,False),5)

if __name__=='__main__':unittest.main()
