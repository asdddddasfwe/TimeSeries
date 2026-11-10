"""Official TST LR/beta1 trajectory, independent formula, and checkpoint continuation."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from scripts import run_distdf_fredformer_tst_matched as r

class TSTTests(unittest.TestCase):
    def optimizer(self):
        return torch.optim.Adam([{'params':[torch.nn.Parameter(torch.ones(2))],'weight_decay':1e-4},
                                 {'params':[torch.nn.Parameter(torch.ones(2))],'weight_decay':0.}],lr=.001)
    def official(self, optimizer, epochs, steps):
        p=r.ROOT/'third_party/DistDF/utils/tools.py'
        node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
        ns=dict(optim=torch.optim,np=np,math=math)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
        args=SimpleNamespace(lradj='TST',step_size=1,lr_decay=.5,min_lr=1e-6,mode=0,train_epochs=epochs,pct_start=.4)
        return ns['Scheduler'](optimizer,args,steps)
    def test_official_all11825_and_independent_formula(self):
        a,b=self.optimizer(),self.optimizer();sa=r.make_scheduler(a,11,1075);sb=self.official(b,11,1075)
        total=11825;end1=.4*total-1
        for step in range(total+1):
            for ga,gb in zip(a.param_groups,b.param_groups):
                self.assertEqual(ga['lr'],gb['lr']);self.assertEqual(ga['betas'],gb['betas'])
            if step<=end1:
                pct=step/end1;lo,hi=.001/25,.001;mb,me=.95,.85
            else:
                pct=(step-end1)/(total-1-end1);lo,hi=.001,.001/25/10000;mb,me=.85,.95
            expected=hi+(lo-hi)/2*(math.cos(math.pi*pct)+1)
            beta=me+(mb-me)/2*(math.cos(math.pi*pct)+1)
            self.assertAlmostEqual(a.param_groups[0]['lr'],expected,places=16)
            self.assertAlmostEqual(a.param_groups[0]['betas'][0],beta,places=16)
            if step<total:
                a.step();b.step();sa.step();sb.step(verbose=False)
        self.assertEqual(sa.last_epoch,11825)
    def test_scheduler_optimizer_resume_exact(self):
        a=self.optimizer();sa=r.make_scheduler(a,3,2)
        for _ in range(2):
            for g in a.param_groups:
                for p in g['params']:p.grad=torch.ones_like(p)
            a.step();sa.step()
        import copy
        state=copy.deepcopy(a.state_dict());ss=copy.deepcopy(sa.state_dict())
        b=self.optimizer();sb=r.make_scheduler(b,3,2);b.load_state_dict(state);sb.load_state_dict(ss)
        for _ in range(4):
            self.assertEqual(sa.state_dict(),sb.state_dict())
            for ga,gb in zip(a.param_groups,b.param_groups):
                self.assertEqual(ga['lr'],gb['lr']);self.assertEqual(ga['betas'],gb['betas'])
            a.step();b.step();sa.step();sb.step()
        self.assertEqual(sa.last_epoch,6)

if __name__=='__main__':unittest.main()
