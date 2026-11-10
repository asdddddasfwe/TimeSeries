"""New MAFirst+MMD matched-recipe initialization/function/optimizer/data/selection checks."""
import math,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scripts import run_mafirst_mmd_official_recipe_repeat2 as r
from scripts import run_distdf_fredformer_official_recipe as official_recipe
from utils.time_position_patch_mmd import parts as existing_mmd

torch.set_num_threads(1)

class MatchedRecipeTests(unittest.TestCase):
    def test_preserve_scratch_recipe_and_fresh2023_cpu_cuda(self):
        cfg=r.config(r.ROOT/'dataset/ETT-small')
        r.seed(2021);oldseed=r.initialize_model(cfg)
        saved=torch.load(r.BASELINE/'initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
        self.assertEqual(len(saved),236)
        self.assertTrue(all(torch.equal(v,saved[n]) and v.data_ptr()!=saved[n].data_ptr() for n,v in oldseed.state_dict().items()))
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            r.seed(2023);a=r.initialize_model(cfg).to(device);r.seed(2023);b=r.initialize_model(cfg).to(device)
            control=torch.load(r.ROOT/'tmp/mafirst_mmd_official_recipe_20261015_seed2023_tst100_patience10/initial.pt',map_location='cpu',weights_only=True)
            self.assertTrue(all(torch.equal(v.cpu(),control[n]) and v.data_ptr()!=control[n].data_ptr() for n,v in a.state_dict().items()))
            self.assertEqual(sum(p.numel() for p in a.parameters()),140738)
            self.assertTrue(all(torch.equal(v,b.state_dict()[n]) and v.data_ptr()!=b.state_dict()[n].data_ptr() for n,v in a.state_dict().items()))
            self.assertTrue(any(not torch.equal(v.cpu(),saved[n]) for n,v in a.state_dict().items()))
            x=torch.linspace(-1,1,4*96*7).reshape(4,96,7).to(device);y=x.flip(1)
            results=[]
            for model,probe in [(a,True),(b,False)]:
                model.train();r.seed(9001);p=model(x);mse=nn.functional.mse_loss(p,y);mmd,_=existing_mmd(p,y);loss=mse+mmd
                if probe:
                    old,_=r.covariance_parts(p,y);rng_before=r.rng_snapshot();r.output_probe(p,mse,mmd,old);r.gradient_probe(model,mse,mmd)
                    self.assertTrue(torch.equal(rng_before['torch_cpu'],torch.get_rng_state()))
                loss.backward();results.append((p.detach(),{n:v.grad.detach().clone() for n,v in model.named_parameters()}))
            self.assertTrue(torch.equal(results[0][0],results[1][0]))
            self.assertTrue(all(torch.equal(g,results[1][1][n]) for n,g in results[0][1].items()))
            oa=torch.optim.Adam(r.optimizer_groups(a),lr=.0001);ob=torch.optim.Adam(b.parameters(),lr=.0001)
            sa=r.make_scheduler(oa,100,268);sb=official_recipe.make_scheduler(ob,100,268)
            oa.step();ob.step();sa.step();sb.step()
            self.assertEqual(len(oa.param_groups),1);self.assertEqual(oa.param_groups[0]['weight_decay'],0.)
            self.assertTrue(all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()))
            del a,b,oa,ob
            if device=='cuda':torch.cuda.empty_cache()

    def test_exact_repeat_source_functions(self):
        import ast
        a=(r.ROOT/'scripts/run_mafirst_mmd_official_recipe.py').read_text()
        b=(r.ROOT/'scripts/run_mafirst_mmd_official_recipe_repeat2.py').read_text()
        fa={n.name:n for n in ast.parse(a).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        fb={n.name:n for n in ast.parse(b).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(set(fa),set(fb))
        for name in fa:
            aa,bb=ast.get_source_segment(a,fa[name]),ast.get_source_segment(b,fb[name])
            if name=='main':aa=aa.replace('monitor_mafirst_mmd_official_recipe.py','monitor_mafirst_mmd_official_recipe_repeat2.py')
            self.assertEqual(ast.dump(ast.parse(aa),include_attributes=False),ast.dump(ast.parse(bb),include_attributes=False),name)

    def test_global_rng_sampler_data_and_selection(self):
        data=TensorDataset(torch.arange(300))
        for purpose in ('train','selection_val'):
            r.seed(2023);expected=torch.cat([x[0] for x in DataLoader(data,batch_size=128,shuffle=True,drop_last=True,num_workers=10)]);rng=torch.get_rng_state()
            r.seed(2023);actual=torch.cat([x[0] for x in r.loader(data,purpose)])
            self.assertTrue(torch.equal(expected,actual));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        cfg=r.config(r.ROOT/'dataset/ETT-small');val=r.dataset(cfg,'val',256)
        self.assertEqual(cfg.label_len,48)
        # Label prefix has no effect on future target or input, model consumes history only.
        from data_provider.data_loader import Dataset_ETT_minute
        legacy=Dataset_ETT_minute(cfg,cfg.root_path,flag='val',size=[96,0,96],features='M',data_path='ETTm1.csv',timeenc=1,freq='h')
        full=r.dataset(cfg,'val')
        for i in (0,5712,11424):
            self.assertTrue(np.array_equal(full[i][0],legacy[i][0]));self.assertTrue(np.array_equal(full[i][1][-96:],legacy[i][1][-96:]))
        class Identity(nn.Module):
            def forward(self,x):return x
        r.seed(309);losses=[];ids=[]
        for ii,x,y,_,_ in r.loader(val,'selection_val'):
            losses.append(nn.functional.mse_loss(x.float(),y[:,-96:].float()));ids.extend(ii.tolist())
        expected=torch.mean(torch.stack(losses)).item()
        with tempfile.TemporaryDirectory(dir=r.ROOT/'tmp') as d:
            out=Path(d);(out/'selection_validation').mkdir();r.seed(309);actual=r.selection_validation(Identity(),val,'cpu',out,1)
            self.assertEqual(actual,expected);self.assertEqual(ids,r.read(out/'selection_validation/epoch001.json')['indices'])

    def test_scheduler_full_horizon_and_tie(self):
        a=torch.optim.Adam([nn.Parameter(torch.ones(1))],lr=.0001);sch=r.make_scheduler(a,100,268)
        for update in range(26800):
            if update<=10719:
                pct=update/10719;start,end=4e-6,1e-4;bs,be=.95,.85
            else:pct=(update-10719)/(26800-1-10719);start,end=1e-4,4e-10;bs,be=.85,.95
            value=end+(start-end)/2*(math.cos(math.pi*pct)+1)
            beta=be+(bs-be)/2*(math.cos(math.pi*pct)+1)
            self.assertAlmostEqual(value,a.param_groups[0]['lr'],places=16);self.assertAlmostEqual(beta,a.param_groups[0]['betas'][0],places=15)
            a.step();sch.step()
        best,chosen,stale=math.inf,0,0
        for epoch,value in enumerate([1.,.9,.9]+[.95]*10,1):best,chosen,stale,_=r.early_update(value,best,chosen,stale,epoch)
        self.assertEqual((best,chosen,stale),(.9,3,10))

    def test_independent_saved_kernel_formulas(self):
        p=np.random.default_rng(78).normal(size=(5,96,7)).astype('float32');y=np.random.default_rng(79).normal(size=p.shape).astype('float32')
        md=r.numpy_mmd_parts(p,y);od=r.numpy_covariance_parts(p,y)
        _,a=r.mmd_parts(torch.from_numpy(p),torch.from_numpy(y));_,b=r.covariance_parts(torch.from_numpy(p),torch.from_numpy(y))
        for k,v in md.items():self.assertAlmostEqual(v,a[k],places=7)
        for k,v in od.items():self.assertAlmostEqual(v,b[k],places=7)

if __name__=='__main__':unittest.main()
