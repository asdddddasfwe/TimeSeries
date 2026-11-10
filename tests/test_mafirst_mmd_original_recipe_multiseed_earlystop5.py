"""One isolated test process per requested seed; no old-model forward/test."""
import ast
import importlib
import math
import os
from pathlib import Path
import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset

SEED = int(os.environ['MAFIRST_AUDIT_SEED'])
assert SEED in (2021, 2023, 2024)
r = importlib.import_module(f'scripts.run_mafirst_mmd_original_recipe_seed{SEED}_earlystop5')
torch.set_num_threads(2)


class OriginalRecipeEarlyStopTests(unittest.TestCase):
    def test_computational_functions_and_stop_location(self):
        old = (r.ROOT/'scripts/run_mafirst_mmd_original_recipe_seed2022.py').read_text().replace(
            'monitor_mafirst_mmd_original_recipe_seed2022.py',
            f'monitor_mafirst_mmd_original_recipe_seed{SEED}_earlystop5.py')
        new = Path(r.__file__).read_text()
        functions = lambda t: {x.name: ast.dump(x, include_attributes=False)
                              for x in ast.parse(t).body if isinstance(x, ast.FunctionDef)}
        a, b = functions(old), functions(new)
        for name in a.keys()-{'main', 'verify_policy', 'publish'}:
            self.assertEqual(a[name], b[name], name)
        self.assertIn("parser.add_argument('--epochs', type=int, default=30)", new)
        self.assertIn('if not smoke and args.epochs != 30:', new)
        self.assertEqual(new.count('if stale >= 5:\n                break'), 1)
        self.assertLess(new.index("shared.atomic_json(out/'progress.json'"),
                        new.index('if stale >= 5:\n                break'))
        self.assertLess(new.index('if stale >= 5:\n                break'),
                        new.index("shared.phase(out, 'freezing'"))

    def test_patience5_ties_cap_and_no_continuing_after_stop(self):
        def last(values):
            best, epoch, stale = math.inf, 0, 0
            rows = []
            for e, value in enumerate(values, 1):
                if value < best:
                    best, epoch, stale = value, e, 0
                else:
                    stale += 1
                rows.append(dict(epoch=e, arm='time_position_mmd', val=dict(mse=value),
                                 learning_rate=r.shared.learning_rate(e),
                                 train_loss=2., train_mse=1., train_mmd=1.))
            return dict(epoch=len(values), curves=rows, best=dict(time_position_mmd=best),
                        best_epoch=dict(time_position_mmd=epoch), stale=stale,
                        updates_per_arm=2*len(values), optimizers=dict(
                            time_position_mmd=dict(state={0:dict(step=2*len(values))})))
        value = r.verify_policy(last([1.]*6), 64, 30)
        self.assertEqual(value['epochs'], 6)
        self.assertEqual(value['selected_epochs']['time_position_mmd'], 1)
        self.assertEqual(value['single_stale_diagnostic'], 5)
        self.assertEqual(r.verify_policy(last([31.-e for e in range(30)]), 64, 30)['epochs'], 30)
        with self.assertRaises(AssertionError):
            r.verify_policy(last([1.]*5), 64, 30)
        with self.assertRaises(AssertionError):
            r.verify_policy(last([1.]*6+[.5]), 64, 30)

    def test_fresh_initial236_exact_independent_and_seed_specific(self):
        cfg = r.config(r.ROOT/'dataset/ETT-small')
        a = r.initialize_pair(cfg)['time_position_mmd']
        b = r.control_reference(cfg)
        self.assertEqual(len(a.state_dict()), 236)
        self.assertEqual(len(list(a.parameters())), 204)
        self.assertEqual(sum(p.numel() for p in a.parameters()), 140738)
        self.assertTrue(all(torch.equal(v, b.state_dict()[n]) and
                            v.data_ptr() != b.state_dict()[n].data_ptr()
                            for n, v in a.state_dict().items()))
        old = torch.load(r.ROOT/'tmp/lag_study_timemix_pdm_state_patch_time_position_mmd_20261015_seed2021_fixed11/initial.pt',
                         map_location='cpu', weights_only=True)['time_position_mmd']
        exact = all(torch.equal(v, old[n]) for n, v in a.state_dict().items())
        self.assertEqual(exact, SEED == 2021)
        audit = r.initialization_audit({'time_position_mmd': a})
        self.assertEqual(audit['fresh_seed'], SEED)
        self.assertTrue(audit['fresh236_regeneration_exact'])

    def test_same_forward_gradient_and_adam_cpu_cuda(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            cfg = r.config(r.ROOT/'dataset/ETT-small')
            models = [r.initialize_pair(cfg)['time_position_mmd'].to(device),
                      r.control_reference(cfg).to(device)]
            results = []
            for model, probe in zip(models, (True, False)):
                r.shared.seed_all(SEED*1000000+10000)
                model.train()
                x = torch.linspace(-1, 1, 4*96*7, device=device).reshape(4,96,7)
                y = x.flip(1)
                p = model(x)
                mse = (p-y).square().mean()
                mmd, _ = r.mmd_parts(p,y)
                if probe:
                    d, _ = r.parts(p,y)
                    r.mmd_gradient_probe(model,mse,mmd)
                    r.output_gradient_probe(p,mse,mmd,d)
                (mse+mmd).backward()
                results.append((p.detach(), {n:v.grad.clone() for n,v in model.named_parameters()}))
            self.assertTrue(torch.equal(results[0][0], results[1][0]))
            self.assertTrue(all(torch.equal(g, results[1][1][n]) for n,g in results[0][1].items()))
            opts = [torch.optim.Adam(r.norm_bias_no_decay_groups(m,1e-4),lr=.001) for m in models]
            names = {id(v):n for n,v in models[0].named_parameters()}
            policy = {n:g['weight_decay'] for g in r.shared.read(r.ROOT/'tmp/lag_study_timemix_pdm_state_patch_time_position_mmd_20261015_seed2021_fixed11/optimizer_groups.json')['time_position_mmd'] for n in g['names']}
            self.assertEqual({names[id(v)]:g['weight_decay'] for g in opts[0].param_groups for v in g['params']},policy)
            for opt in opts:
                opt.step()
            self.assertTrue(all(torch.equal(v,models[1].state_dict()[n]) for n,v in models[0].state_dict().items()))
            del models,opts,results
            if device == 'cuda':
                torch.cuda.empty_cache()

    def test_original_batch_seed_cosine_and_full_validation(self):
        data = TensorDataset(torch.arange(34369))
        expected = torch.randperm(34369,generator=torch.Generator().manual_seed(SEED*1000+4))
        actual = torch.cat([x[0] for x in r.loader(data,4)])
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(len(list(r.loader(data,4))),1075)
        self.assertEqual(len(list(r.loader(data))[-1][0]),1)
        for epoch in range(1,31):
            self.assertEqual(r.shared.learning_rate(epoch),.001*.5*(1+math.cos(math.pi*(epoch-1)/30)))
        cfg = r.config(r.ROOT/'dataset/ETT-small')
        self.assertEqual(cfg.seed,SEED)
        self.assertEqual(cfg.label_len,0)
        self.assertEqual(cfg.patience,5)

    def test_mmd_numpy_and_batch_separability(self):
        p = np.random.default_rng(810).normal(size=(8,96,7)).astype('float32')
        y = np.random.default_rng(811).normal(size=p.shape).astype('float32')
        loss, d = r.mmd_parts(torch.from_numpy(p),torch.from_numpy(y))
        for k,v in r.numpy_mmd_parts(p,y).items():
            self.assertAlmostEqual(v,d[k],places=7)
        partial = [r.mmd_parts(torch.from_numpy(p[i:i+2]),torch.from_numpy(y[i:i+2]))[0].item()
                   for i in range(0,8,2)]
        self.assertAlmostEqual(float(loss),sum(partial)/4,places=7)


if __name__ == '__main__':
    unittest.main()
