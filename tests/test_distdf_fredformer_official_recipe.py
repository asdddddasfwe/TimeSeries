"""Official recipe: seed2023 function/Adam, global-RNG batching, validation and early stopping."""
import ast,math,os,tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scripts import run_distdf_fredformer_official_recipe as r
from test_distdf_fredformer_matched import original_class
import test_distdf_tst_matched as tst_reference

torch.set_num_threads(1)

class RecipeTests(unittest.TestCase):
    def test_official2023_model_gradient_one_adam_step_cpu_cuda(self):
        official=original_class();cfg=r.config(r.ROOT/'dataset/ETT-small')
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            r.seed(2023);a=official(cfg).to(device);r.seed(2023);b=r.Model(cfg).to(device)
            self.assertEqual(set(a.state_dict()),set(b.state_dict()))
            self.assertTrue(all(torch.equal(v,b.state_dict()[n]) and v.data_ptr()!=b.state_dict()[n].data_ptr() for n,v in a.state_dict().items()))
            x=torch.linspace(-1,1,2*96*7).reshape(2,96,7).to(device);y=x.flip(1)
            outputs=[];rngs=[];grads=[]
            for m in (a,b):
                m.train();r.seed(9001);p=m(x);loss=.5*nn.functional.mse_loss(p,y)+.5*r.distdf_loss(p,y,x);loss.backward();outputs.append(p.detach());rngs.append(r.rng_snapshot())
                grads.append({n:None if v.grad is None else v.grad.clone() for n,v in m.named_parameters()})
            self.assertTrue(torch.equal(outputs[0],outputs[1]))
            self.assertTrue(torch.equal(rngs[0]['torch_cpu'],rngs[1]['torch_cpu']))
            for n,g in grads[0].items():
                if g is None:self.assertIsNone(grads[1][n])
                else:self.assertTrue(torch.equal(g,grads[1][n]),n)
            oa=torch.optim.Adam(a.parameters(),lr=.0001);ob=torch.optim.Adam(r.optimizer_groups(b),lr=.0001)
            sa=r.make_scheduler(oa,100,268);sb=r.make_scheduler(ob,100,268);oa.step();ob.step();sa.step();sb.step()
            self.assertTrue(all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()))
            self.assertEqual(len(ob.param_groups),1);self.assertEqual(ob.param_groups[0]['weight_decay'],0.)
            del a,b,oa,ob
            if device=='cuda':torch.cuda.empty_cache()
    def test_official_global_rng_sampler_and_validation(self):
        data=TensorDataset(torch.arange(300))
        for purpose in ('train','selection_val'):
            r.seed(2023);expected=torch.cat([x[0] for x in DataLoader(data,batch_size=128,shuffle=True,drop_last=True,num_workers=10)]);rng=torch.get_rng_state()
            r.seed(2023);actual=torch.cat([x[0] for x in r.loader(data,purpose)]);self.assertTrue(torch.equal(expected,actual));self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(len(actual),256)
        cfg=r.config(r.ROOT/'dataset/ETT-small');val=r.dataset(cfg,'val',256)
        class Identity(nn.Module):
            def forward(self,x):return x
        r.seed(309);losses=[];ids=[]
        for ii,x,y,_,_ in r.loader(val,'selection_val'):
            losses.append(nn.functional.mse_loss(x.float(),y[:,-96:].float()));ids.extend(ii.tolist())
        expected=torch.mean(torch.stack(losses)).item()
        with tempfile.TemporaryDirectory(dir=r.ROOT/'tmp') as d:
            out=Path(d);(out/'selection_validation').mkdir();r.seed(309);actual=r.selection_validation(Identity(),val,'cpu',out,1)
            self.assertEqual(expected,actual);self.assertEqual(ids,r.read(out/'selection_validation/epoch001.json')['indices'])
    def test_official_horizon_and_early_stop_ties(self):
        a=torch.optim.Adam([nn.Parameter(torch.ones(1))],lr=.0001);b=torch.optim.Adam([nn.Parameter(torch.ones(1))],lr=.0001)
        sa=r.make_scheduler(a,100,268);sb=tst_reference.TSTTests().official(b,100,268)
        for _ in range(26800):
            self.assertEqual(a.param_groups[0]['lr'],b.param_groups[0]['lr']);self.assertEqual(a.param_groups[0]['betas'],b.param_groups[0]['betas']);a.step();b.step();sa.step();sb.step(verbose=False)
        self.assertEqual(sa.last_epoch,26800)
        tree=ast.parse((r.ROOT/'third_party/DistDF/utils/tools.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EarlyStopping');ns=dict(np=SimpleNamespace(Inf=np.inf,isnan=np.isnan,isinf=np.isinf),torch=torch,os=os);exec(compile(ast.Module(body=[node],type_ignores=[]),'officialEarlyStopping','exec'),ns)
        with tempfile.TemporaryDirectory(dir=r.ROOT/'tmp') as d:
            original=ns['EarlyStopping'](patience=10);best,chosen,stale=math.inf,0,0;m=nn.Linear(1,1)
            for epoch,value in enumerate([1.,.9,.9]+[.95]*10,1):
                improved=original(value,m,d);best,chosen,stale,ours=r.early_update(value,best,chosen,stale,epoch)
                self.assertEqual(improved,ours);self.assertEqual(stale,original.counter);self.assertEqual(stale>=10,original.early_stop)
            self.assertEqual(chosen,3)
    def test_official_dataset_values_and_marks_train_val(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from utils.timefeatures import time_features
        source=r.ROOT/'third_party/DistDF/data_provider/data_loader.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Dataset_ETT_minute')
        ns=dict(Dataset=torch.utils.data.Dataset,StandardScaler=StandardScaler,pd=pd,np=np,torch=torch,os=os,time_features=time_features)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        cfg=r.config(r.ROOT/'dataset/ETT-small')
        for split in ('train','val'):
            a=ns['Dataset_ETT_minute'](str(r.ROOT/'dataset/ETT-small'),flag=split,size=[96,48,96],features='M',data_path='ETTm1.csv',timeenc=1,freq='h',cycle=96)
            b=r.dataset(cfg,split);self.assertEqual(len(a),len(b));self.assertTrue(np.array_equal(a.data_x,b.data_x));self.assertTrue(np.array_equal(a.data_y,b.data_y))
            for i in [0,len(a)//2,len(a)-1]:
                for x,y in zip(a[i][:4],b[i]):self.assertTrue(np.array_equal(x,y))

if __name__=='__main__':unittest.main()
