"""Original recipe independent-seed initialization, function, optimizer and batch-policy checks."""
import ast,math,unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from scripts import run_mafirst_mmd_original_recipe_seed2022 as r

torch.set_num_threads(2)
class OriginalRecipeSeedTests(unittest.TestCase):
    def test_original_computational_functions_unchanged(self):
        old=(r.ROOT/'scripts/run_timemix_pdm_state_patch_time_position_mmd.py').read_text()
        expected=old.replace('shared.seed_all(2021)','shared.seed_all(SEED)').replace('shared.seed_all(2021*1000000+epoch*10000+step-1)','shared.seed_all(SEED*1000000+epoch*10000+step-1)').replace('monitor_timemix_pdm_state_patch_time_position_mmd.py','monitor_mafirst_mmd_original_recipe_seed2022.py')
        a={f.name:ast.dump(f,include_attributes=False) for f in ast.parse(expected).body if isinstance(f,ast.FunctionDef)}
        b={f.name:ast.dump(f,include_attributes=False) for f in ast.parse(Path(r.__file__).read_text()).body if isinstance(f,ast.FunctionDef)}
        for name in ['main','evaluate','verify_policy','references','mmd_gradient_probe','output_gradient_probe','audit_capture','arm_config','initialize_pair']:
            self.assertEqual(a[name],b[name],name)
    def test_fresh_seed2021_parity_seed2022_distinct_storage(self):
        cfg=r.config(r.ROOT/'dataset/ETT-small');oldseed=r.SEED
        try:
            r.SEED=2021;a=r.initialize_pair(cfg)['time_position_mmd'];saved=torch.load(r.ROOT/'tmp/lag_study_timemix_pdm_state_patch_time_position_mmd_20261015_seed2021_fixed11/initial.pt',map_location='cpu',weights_only=True)['time_position_mmd']
            self.assertTrue(all(torch.equal(v,saved[n]) and v.data_ptr()!=saved[n].data_ptr() for n,v in a.state_dict().items()))
            r.SEED=2022;b=r.initialize_pair(cfg)['time_position_mmd'];c=r.control_reference(cfg)
            self.assertEqual(len(b.state_dict()),236);self.assertEqual(len(list(b.parameters())),204);self.assertEqual(sum(v.numel() for v in b.parameters()),140738)
            self.assertTrue(all(torch.equal(v,c.state_dict()[n]) and v.data_ptr()!=c.state_dict()[n].data_ptr() for n,v in b.state_dict().items()))
            self.assertTrue(any(not torch.equal(v,saved[n]) for n,v in b.named_parameters()))
            audit=r.initialization_audit({'time_position_mmd':b});self.assertTrue(audit['fresh236_regeneration_exact']);self.assertEqual(audit['fresh_seed'],2022)
        finally:r.SEED=oldseed
    def test_same_forward_gradients_adam_policy_cpu_cuda(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            cfg=r.config(r.ROOT/'dataset/ETT-small');a=r.initialize_pair(cfg)['time_position_mmd'].to(device);b=r.control_reference(cfg).to(device)
            results=[]
            for model,probe in [(a,True),(b,False)]:
                r.shared.seed_all(2022*1000000+10000);model.train();x=torch.linspace(-1,1,4*96*7,device=device).reshape(4,96,7);y=x.flip(1);p=model(x);mse=(p-y).square().mean();mmd,_=r.mmd_parts(p,y)
                if probe:
                    old,_=r.parts(p,y);r.mmd_gradient_probe(model,mse,mmd);r.output_gradient_probe(p,mse,mmd,old)
                (mse+mmd).backward();results.append((p.detach(),{n:v.grad.clone() for n,v in model.named_parameters()}))
            self.assertTrue(torch.equal(results[0][0],results[1][0]));self.assertTrue(all(torch.equal(g,results[1][1][n]) for n,g in results[0][1].items()))
            opts=[torch.optim.Adam(r.norm_bias_no_decay_groups(m,1e-4),lr=.001) for m in (a,b)]
            saved=r.shared.read(r.ROOT/'tmp/lag_study_timemix_pdm_state_patch_time_position_mmd_20261015_seed2021_fixed11/optimizer_groups.json')['time_position_mmd'];policy={n:g['weight_decay'] for g in saved for n in g['names']}
            names={id(v):n for n,v in a.named_parameters()};self.assertEqual({names[id(v)]:g['weight_decay'] for g in opts[0].param_groups for v in g['params']},policy)
            for o in opts:o.step()
            self.assertTrue(all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()))
            del a,b,opts
            if device=='cuda':torch.cuda.empty_cache()
    def test_original_batch_seeds_cosine_fullval_policy(self):
        data=TensorDataset(torch.arange(34369));epoch=4
        expected=torch.randperm(34369,generator=torch.Generator().manual_seed(2022*1000+epoch));actual=torch.cat([x[0] for x in r.loader(data,epoch)])
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(len(list(r.loader(data,epoch))),1075);self.assertEqual(len(list(r.loader(data))[-1][0]),1)
        for e in range(1,12):self.assertEqual(r.shared.learning_rate(e),.001*.5*(1+math.cos(math.pi*(e-1)/30)))
        cfg=r.config(r.ROOT/'dataset/ETT-small');self.assertEqual(cfg.label_len,0);self.assertEqual(cfg.seed,2022)
    def test_independent_mmd_numpy_and_batch_separability(self):
        p=np.random.default_rng(810).normal(size=(8,96,7)).astype('float32');y=np.random.default_rng(811).normal(size=p.shape).astype('float32');md=r.numpy_mmd_parts(p,y);loss,d=r.mmd_parts(torch.from_numpy(p),torch.from_numpy(y))
        for k,v in md.items():self.assertAlmostEqual(v,d[k],places=7)
        parts=[r.mmd_parts(torch.from_numpy(p[i:i+2]),torch.from_numpy(y[i:i+2]))[0].item() for i in range(0,8,2)]
        self.assertAlmostEqual(float(loss),sum(parts)/4,places=7)
if __name__=='__main__':unittest.main()
