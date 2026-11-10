import copy,math,unittest
import torch
from scripts.train_timemix_pdm_schedule import SchedulePolicy,training_losses,loss_configuration
from types import SimpleNamespace

class ScheduleTests(unittest.TestCase):
    def test_plateau_matches_pytorch_and_resume(self):
        values=[1.,.9,.9,.91,.92,.93,.94,.95,.96,.97]
        p=torch.nn.Parameter(torch.ones(()));opt=torch.optim.Adam([p],lr=.001)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.3,patience=2,threshold=0.,threshold_mode='abs',cooldown=0,min_lr=3e-5,eps=1e-8)
        policy=SchedulePolicy('plateau'); resumed=None
        for i,value in enumerate(values):
            policy.step(value);scheduler.step(value)
            self.assertEqual(policy.lr,opt.param_groups[0]['lr'])
            if i==3:
                resumed=SchedulePolicy('plateau');resumed.load_state_dict(policy.state_dict())
            elif i>3:
                resumed.step(value);self.assertEqual(policy.state_dict(),resumed.state_dict())
        self.assertEqual(policy.epoch,10);self.assertTrue(policy.stopped)
        self.assertEqual([r['after_epoch'] for r in policy.reductions],[5,8])
        self.assertAlmostEqual(policy.lr,.00009)
    def test_independent_stopping(self):
        policies={k:SchedulePolicy(k) for k in ('cosine','plateau')};epochs=[]
        for epoch in range(1,31):
            for k,p in policies.items():
                if not p.stopped:p.step(1. if epoch==1 else .9)
            epochs.append({k:p.epoch for k,p in policies.items()})
            if all(p.stopped for p in policies.values()):break
        self.assertEqual(policies['cosine'].epoch,7)
        self.assertEqual(policies['plateau'].epoch,10)
        self.assertEqual(epochs[-1],{'cosine':7,'plateau':10})
    def test_cosine_exact_schedule_and_strict_selection(self):
        p=SchedulePolicy('cosine')
        for epoch,v in enumerate([1.,.8,.8,.79],1):
            self.assertEqual(p.lr,.001*.5*(1+math.cos(math.pi*(epoch-1)/30)))
            p.step(v)
        self.assertEqual(p.best_epoch,4);self.assertEqual(p.stale,0)
    def test_plateau_floor_and_max_epoch(self):
        p=SchedulePolicy('plateau')
        # Reset early-stop streak by a strict improvement before 8 bad epochs.
        for i in range(30):p.step(1.-.01*(i//7))
        self.assertTrue(p.stopped);self.assertEqual(p.stop_reason,'max_epochs')
        self.assertEqual(p.lr,3e-5)
    def test_mse_gradient_and_target_detach(self):
        for arm in ('cosine','plateau'):
            p=torch.tensor([[[1.,2.],[3.,4.]]],requires_grad=True)
            y=torch.zeros_like(p,requires_grad=True)
            loss=training_losses(p,y,arm,loss_configuration(SimpleNamespace()),None)
            self.assertEqual(float(loss['loss']),7.5)
            loss['loss'].backward();self.assertTrue(torch.equal(p.grad,2*p.detach()/p.numel()));self.assertIsNone(y.grad)
            self.assertEqual(float(loss['relation']),0.)
    def test_invalid_resume_or_score(self):
        p=SchedulePolicy('cosine')
        with self.assertRaises(ValueError):p.step(float('nan'))
        with self.assertRaises(ValueError):p.load_state_dict(SchedulePolicy('plateau').state_dict())
if __name__=='__main__':unittest.main()
